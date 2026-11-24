# Copyright 2026 Arlow Constellations. All rights reserved.
# Written and edited by IRIS.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn
from torch.nn import LayerNorm

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...generation import GenerationMixin
from ...image_utils import ImageInput
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
    MultimodalBaseModelOutputWithPast,
    MultimodalCausalLMOutputWithPast,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, RopeParameters, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import accepts_precomputed_kwargs, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ...video_utils import VideoInput


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="yuchenxie/ArlowGPT-3B-Foundational-Preview")
@strict
class ArlowVisionConfig(PreTrainedConfig):
    r"""
    use_deformable_attention (`bool`, *optional*, defaults to `False`):
        Whether to apply deformable attention bias for high-resolution regions.
    use_progressive_patches (`bool`, *optional*, defaults to `False`):
        Whether to inject progressive multi-scale patch features.
    deformable_attention_window (`float`, *optional*, defaults to 0.25):
        Normalized radius (in THW coordinates) used to compute deformable attention locality bias.
    deformable_attention_strength (`float`, *optional*, defaults to 4.0):
        Scaling factor applied to deformable attention bias.
    deepstack_visual_indexes (`list[int]`, *optional*):
        Vision block indexes from which to capture DeepStack skip features. Defaults to a
        depth-aware spread when unset (empty when depth < 6).
    mrope_sections (`list[int]`, *optional*):
        Multimodal RoPE allocation across [temporal, height, width] (sum must equal head dimension).
    """

    model_type = "arlow"
    base_config_key = "vision_config"
    attribute_map = {"num_attention_heads": "num_heads", "num_hidden_layers": "depth"}

    depth: int = 2
    embed_dim: int = 32
    hidden_size: int = 64
    hidden_act: str = "gelu_pytorch_tanh"
    mlp_ratio: int = 4
    num_heads: int = 4
    in_channels: int = 3
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    use_deformable_attention: bool = False
    use_progressive_patches: bool = False
    deformable_attention_window: float = 0.25
    deformable_attention_strength: float = 4.0
    deepstack_visual_indexes: list[int] | None = None
    mrope_sections: list[int] | None = None
    initializer_range: float = 0.02
    max_position_embeddings: int | None = None

    def __post_init__(self, **kwargs):
        self.validate_architecture()
        if self.max_position_embeddings is None:
            self.max_position_embeddings = 32768

        if self.deepstack_visual_indexes is None:
            if self.depth >= 6:
                approx = {max(0, self.depth // 4), max(0, self.depth // 2), max(0, self.depth - 4)}
                self.deepstack_visual_indexes = sorted(idx for idx in approx if 0 <= idx < self.depth)
            else:
                self.deepstack_visual_indexes = []
        else:
            self.deepstack_visual_indexes = sorted(
                {idx for idx in self.deepstack_visual_indexes if 0 <= idx < self.depth}
            )

        head_dim = self.embed_dim // self.num_heads
        if self.mrope_sections is None:
            t = max(1, head_dim // 3)
            h = max(1, (head_dim - t) // 2)
            self.mrope_sections = [t, h, head_dim - t - h]
        elif sum(self.mrope_sections) != head_dim:
            raise ValueError(
                f"Sum of mrope_sections {self.mrope_sections} (={sum(self.mrope_sections)}) must equal head_dim ({head_dim})."
            )

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads}).")


@auto_docstring(checkpoint="yuchenxie/ArlowGPT-3B-Foundational-Preview")
@strict
class ArlowTextConfig(PreTrainedConfig):
    r"""
    resid_dropout (`float`, *optional*, defaults to 0.0):
        Dropout ratio for residual connections.
    mlp_dropout (`float`, *optional*, defaults to 0.0):
        Dropout ratio for MLP layers.
    mrope_sections (`list[int]`, *optional*):
        Multimodal RoPE allocation across [temporal, height, width] (sum must equal `head_dim`). Defaults to an
        even split of `head_dim`.

    Example:

    ```python
    >>> from transformers import ArlowTextConfig, ArlowTextModel

    >>> # Initializing a text-only Arlow configuration
    >>> configuration = ArlowTextConfig()

    >>> # Initializing a model from the configuration
    >>> model = ArlowTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "arlow_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["cache"]
    default_theta = 100000.0
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    ignore_keys_at_rope_validation = {"mrope_sections"}

    vocab_size: int = 131074
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 32
    num_attention_heads: int = 24
    num_key_value_heads: int | None = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    resid_dropout: float | int = 0.0
    mlp_dropout: float | int = 0.0
    head_dim: int | None = None
    use_sliding_window: bool = False
    sliding_window: int | None = 4096
    max_window_layers: int = 28
    layer_types: list[str] | None = None
    mrope_sections: list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads "
                    f"({self.num_attention_heads}) when head_dim is not set."
                )
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.mrope_sections is None:
            t = max(1, self.head_dim // 3)
            h = max(1, (self.head_dim - t) // 2)
            self.mrope_sections = [t, h, self.head_dim - t - h]
        self.validate_architecture()

        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive.")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})."
            )
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive.")
        if sum(self.mrope_sections) != self.head_dim:
            raise ValueError(
                f"Sum of mrope_sections {self.mrope_sections} (={sum(self.mrope_sections)}) "
                f"must equal head_dim ({self.head_dim})."
            )


@auto_docstring(checkpoint="yuchenxie/ArlowGPT-3B-Foundational-Preview")
@strict
class ArlowConfig(PreTrainedConfig):
    r"""
    use_gated_cross_attention (`bool`, *optional*, defaults to `False`):
        Whether to use gated cross-attention in upper layers.
    gated_cross_attention_start_layer (`int`, *optional*):
        Layer index to start gated cross-attention (if enabled).
    text_deepstack_injection_layers (`list[int]`, *optional*):
        Text decoder layer indexes where captured DeepStack visual features are injected. When unset, the
        injection layers are spread across the text decoder according to the number of captured vision layers.
    vision_start_token_id (`int`, *optional*, defaults to 3):
        Token ID marking start of vision input.
    vision_end_token_id (`int`, *optional*, defaults to 4):
        Token ID marking end of vision input.
    frame_separator_token_id (`int`, *optional*):
        Token ID for separating video frames.

    Text backbone arguments (`vocab_size`, `hidden_size`, `num_hidden_layers`, ...) can also be passed at the top
    level, in which case they are used to build `text_config`.

    Example:

    ```python
    >>> from transformers import ArlowConfig, ArlowForConditionalGeneration

    >>> # Initializing an Arlow style configuration
    >>> configuration = ArlowConfig()

    >>> # Initializing a model from the configuration
    >>> model = ArlowForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "arlow"
    sub_configs_defaults = {"vision_config": ArlowVisionConfig, "text_config": ArlowTextConfig}
    keys_to_ignore_at_inference = ["cache"]
    ignore_keys_at_rope_validation = {"mrope_sections"}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    use_gated_cross_attention: bool = False
    gated_cross_attention_start_layer: int | None = None
    text_deepstack_injection_layers: list[int] | None = None
    image_token_id: int = 131072
    video_token_id: int = 131073
    vision_start_token_id: int = 3
    vision_end_token_id: int = 4
    frame_separator_token_id: int | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        # Flat hub configs carry the text backbone arguments at the top level
        text_keys = [field.name for field in fields(ArlowTextConfig)] + ["rope_scaling", "rope_theta"]
        text_kwargs = {key: kwargs.pop(key) for key in text_keys if key in kwargs}
        if self.text_config is None:
            self.text_config = {**text_kwargs, "tie_word_embeddings": self.tie_word_embeddings}
        elif isinstance(self.text_config, PreTrainedConfig):
            self._apply_text_config_overrides(self.text_config, text_kwargs)

        super().__post_init__(**kwargs)

        text_config = self.text_config
        if not self.tie_word_embeddings and text_config.tie_word_embeddings:
            self.tie_word_embeddings = True
        self.vision_config.hidden_size = text_config.hidden_size

        text_deepstack_injection_layers = self.text_deepstack_injection_layers
        visual_deepstack_count = len(getattr(self.vision_config, "deepstack_visual_indexes", []))
        if text_deepstack_injection_layers is None:
            if visual_deepstack_count > 0:
                text_deepstack_injection_layers = [
                    min(
                        text_config.num_hidden_layers - 1,
                        max(0, ((slot + 1) * text_config.num_hidden_layers) // (visual_deepstack_count + 1)),
                    )
                    for slot in range(visual_deepstack_count)
                ]
            else:
                text_deepstack_injection_layers = []
        else:
            text_deepstack_injection_layers = list(text_deepstack_injection_layers)
            if len(set(text_deepstack_injection_layers)) != len(text_deepstack_injection_layers):
                raise ValueError("text_deepstack_injection_layers must not contain duplicate layer indexes.")
            invalid_layers = [
                idx for idx in text_deepstack_injection_layers if idx < 0 or idx >= text_config.num_hidden_layers
            ]
            if invalid_layers:
                raise ValueError(
                    "text_deepstack_injection_layers contains indexes outside the text decoder depth: "
                    f"{invalid_layers}."
                )
        if len(text_deepstack_injection_layers) != visual_deepstack_count:
            raise ValueError(
                "text_deepstack_injection_layers must have the same length as "
                f"vision_config.deepstack_visual_indexes ({visual_deepstack_count}), got "
                f"{len(text_deepstack_injection_layers)}."
            )
        self.text_deepstack_injection_layers = text_deepstack_injection_layers

        # The decoder reads the text backbone settings directly off the composite config
        for key in (
            "vocab_size",
            "max_position_embeddings",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "hidden_act",
            "initializer_range",
            "rms_norm_eps",
            "use_cache",
            "rope_parameters",
            "attention_bias",
            "attention_dropout",
            "resid_dropout",
            "mlp_dropout",
            "head_dim",
            "layer_types",
            "use_sliding_window",
            "sliding_window",
            "max_window_layers",
            "mrope_sections",
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
        ):
            setattr(self, key, getattr(text_config, key))

        if sum(self.mrope_sections) != self.head_dim:
            raise ValueError(
                f"Sum of mrope_sections {self.mrope_sections} (={sum(self.mrope_sections)}) "
                f"must equal head_dim ({self.head_dim})"
            )

        self._mrope_ratio = [section / self.head_dim for section in self.mrope_sections]
        self.validate_architecture()
        vision_head_dim = self.vision_config.embed_dim // self.vision_config.num_heads
        self.vision_config.mrope_sections = self._scale_mrope_sections_from_ratio(vision_head_dim, self._mrope_ratio)

    @staticmethod
    def _apply_text_config_overrides(text_config: "ArlowTextConfig", text_kwargs: dict) -> None:
        rope_parameters = text_kwargs.get("rope_scaling") or text_kwargs.get("rope_parameters")
        if rope_parameters is not None:
            rope_parameters = dict(rope_parameters)
            rope_parameters.setdefault("rope_theta", text_kwargs.get("rope_theta", text_config.default_theta))
            text_config.rope_parameters = rope_parameters
            text_config.standardize_rope_params()
        for key in ("head_dim", "mrope_sections", "layer_types", "max_window_layers"):
            if text_kwargs.get(key) is not None:
                setattr(text_config, key, text_kwargs[key])
        if text_kwargs.get("use_sliding_window") is not None:
            text_config.use_sliding_window = text_kwargs["use_sliding_window"]
            text_config.sliding_window = (
                text_kwargs.get("sliding_window", 4096) if text_config.use_sliding_window else None
            )

    def to_dict(self):
        output = super().to_dict()
        dtype = output.get("dtype")
        if dtype is not None:
            for sub_config_key in ("text_config", "vision_config"):
                sub_config = output.get(sub_config_key)
                if isinstance(sub_config, dict) and sub_config.get("dtype") is None:
                    sub_config["dtype"] = dtype
        return output

    @staticmethod
    def _scale_mrope_sections_from_ratio(head_dim: int, ratio: list[float]) -> list[int]:
        if head_dim <= 0:
            raise ValueError("head_dim must be positive when computing M-ROPE sections.")
        if len(ratio) != 3:
            raise ValueError("ratio must contain three entries for [temporal, height, width].")

        raw = [r * head_dim for r in ratio]
        base = [math.floor(val) for val in raw]
        remainders = [val - floor for val, floor in zip(raw, base)]

        # Guarantee a minimum allocation of 1 per dimension
        for i in range(len(base)):
            if base[i] < 1:
                base[i] = 1
                remainders[i] = 0.0

        total = sum(base)

        # Distribute remaining capacity based on largest remainders
        while total < head_dim:
            idx = max(range(len(base)), key=lambda i: remainders[i])
            base[idx] += 1
            remainders[idx] = 0.0
            total += 1

        # Trim excess while keeping each section >= 1
        while total > head_dim:
            candidates = [i for i in range(len(base)) if base[i] > 1]
            if not candidates:
                break
            idx = min(candidates, key=lambda i: remainders[i])
            base[idx] -= 1
            total -= 1

        return base

    def validate_architecture(self):
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive.")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})."
            )
        if self.vision_config.num_heads <= 0:
            raise ValueError("vision_config.num_heads must be positive.")
        if self.vision_config.embed_dim % self.vision_config.num_heads != 0:
            raise ValueError(
                f"vision_config.embed_dim ({self.vision_config.embed_dim}) must be divisible by "
                f"vision_config.num_heads ({self.vision_config.num_heads})."
            )


# Inspired by transformers.models.qwen3_vl.modeling_qwen3_vl.BaseModelOutputWithDeepstackFeatures
@auto_docstring
@dataclass
class BaseModelOutputWithDeepstackFeatures(BaseModelOutputWithPooling):
    r"""
    deepstack_features (`List[torch.Tensor]`, *optional*):
        List of hidden-states (feature maps) from deepstack layers.
    """

    deepstack_features: list[torch.Tensor] | None = None


@use_kernel_forward_from_hub("RMSNorm")
# Inspired by transformers.models.gemma.modeling_gemma.GemmaRMSNorm
class ArlowRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states * self.weight.to(torch.float32)
        return hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Inspired by transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLRotaryEmbedding
class ArlowTextRotaryEmbedding(nn.Module):
    def __init__(self, config: ArlowConfig, device=None):
        super().__init__()
        rope_config = getattr(config, "rope_parameters", None)
        if rope_config is not None and not isinstance(rope_config, dict):
            raise ValueError("rope_parameters must be a dict if provided")
        if isinstance(rope_config, dict):
            self.rope_type = rope_config.get("rope_type", rope_config.get("type"))
            if self.rope_type is None:
                logger.warning("rope_parameters provided without 'rope_type'/'type'; defaulting to 'default'.")
                self.rope_type = "default"
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        # Handle default rope type
        rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: ArlowConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
        **rope_kwargs,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies for default RoPE (no scaling).
        """
        if config is not None:
            base = config.rope_parameters["rope_theta"]
            head_dim = config.head_dim
            if head_dim is None:
                head_dim = config.hidden_size // config.num_attention_heads
        else:
            base = rope_kwargs.get("base", 10000.0)
            head_dim = rope_kwargs.get("dim")

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(device) / head_dim))
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor):
        """
        Compute rotary cos/sin. Supports both 1D text RoPE (batch, seq) and
        multimodal 3D M-ROPE (3, batch, seq) Qwen3-style.
        """
        device = x.device

        # Standard 1D RoPE path (text-only)
        if position_ids.ndim == 2:
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(device)
            position_ids_expanded = position_ids[:, None, :].float()
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

        # M-ROPE path (3, batch, seq): interleaved temporal/height/width
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError(
                f"Expected position_ids of shape (3, batch, seq_len) for M-ROPE, got {tuple(position_ids.shape)}"
            )

        batch_size = position_ids.shape[1]
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, batch_size, -1, 1).to(device)
        position_ids_expanded = position_ids[:, :, None, :].float()

        # (3, batch, seq, head_dim//2)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        # Interleave per mrope_sections
        freqs = apply_interleaved_mrope(freqs, self.config.mrope_sections)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Inspired by transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the hidden states using the standard half-split RoPE layout."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
    """Apply interleaved M-ROPE layout to 3D rotary frequencies.

    Converts the chunked layout `[TTT...HHH...WWW]` into `[THTHWHTH...TT]` while preserving
    per-dimension frequency continuity. Mirrors the helper used in Qwen-VL models.

    Args:
        freqs (`torch.Tensor`): Tensor of shape `(3, batch, seq, head_dim // 2)`.
        mrope_section (`list[int]`): Allocation per [temporal, height, width].

    Returns:
        `torch.Tensor`: Interleaved tensor of shape `(batch, seq, head_dim // 2)`.
    """
    freqs_t = freqs[0]
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        if length <= offset:
            continue
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


# Inspired by transformers.models.gemma.modeling_gemma.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Inspired by transformers.models.qwen2_vl.modeling_qwen2_vl.apply_rotary_pos_emb_vision
def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to vision transformer Q/K tensors."""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def get_arlow_vision_position_ids(grid_thw: torch.Tensor) -> torch.Tensor:
    """
    Compute M-ROPE position ids for packed vision tokens.

    Follows `transformers.vision_utils.get_vision_position_ids`, but keeps the plain (t, h, w)
    row-major patch order the Arlow vision encoder packs tokens in, and returns the
    `(3, total_patches)` layout the interleaved M-ROPE expects.

    Args:
        grid_thw (`torch.LongTensor` of shape `(num_items, 3)`):
            Temporal, height and width of each visual item, in patches.

    Returns:
        `torch.Tensor`: position ids of shape `(3, total_patches)` — temporal, height and
        width indices per patch.
    """
    device = grid_thw.device
    position_ids = []
    for t, h, w in grid_thw.tolist():
        if t <= 0 or h <= 0 or w <= 0:
            continue
        t_index = torch.arange(t, device=device, dtype=torch.float32).view(-1, 1, 1).expand(t, h, w).reshape(-1)
        h_index = torch.arange(h, device=device, dtype=torch.float32).view(1, -1, 1).expand(t, h, w).reshape(-1)
        w_index = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, -1).expand(t, h, w).reshape(-1)
        position_ids.append(torch.stack([t_index, h_index, w_index], dim=0))
    if not position_ids:
        return torch.zeros((3, 0), device=device, dtype=torch.float32)
    return torch.cat(position_ids, dim=1)


# Inspired by transformers.models.qwen2_vl.modeling_qwen2_vl.VisionRotaryEmbedding
class ArlowVLRotaryEmbedding(nn.Module):
    """Grid-aware rotary position embeddings for vision transformer (THW)."""

    def __init__(
        self,
        dim: int | None = None,
        theta: float = 100000.0,
        config: ArlowVisionConfig | ArlowConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        # Determine head dimension and rope parameters
        rope_theta = theta
        self.rope_type = "default"
        if config is not None:
            if isinstance(config, ArlowVisionConfig):
                dim = config.embed_dim // config.num_heads
            else:
                # Fallback path if text config passed -- E264
                dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
            # Extract rope params if available
            if hasattr(config, "rope_parameters") and getattr(config, "rope_parameters", None):
                rope_params = config.rope_parameters
                self.rope_type = rope_params.get("rope_type", rope_params.get("type", "default"))
                rope_theta = rope_params.get("rope_theta", getattr(config, "rope_theta", rope_theta))
            else:
                rope_theta = getattr(config, "rope_theta", rope_theta)
        # Cache original sequence length if available (used by dynamic/yarn tests) -- E264
        self.max_seq_len_cached = getattr(config, "max_position_embeddings", 0) if config is not None else 0
        self.original_max_seq_len = self.max_seq_len_cached
        # Initialize inverse frequencies (default or scaled if non-default rope type) -- E264
        if dim is None:
            raise ValueError("ArlowVLRotaryEmbedding requires `dim` or `config` to infer head dimension.")
        if self.rope_type != "default" and self.config is not None:
            # Initialize according to scaling type -- E264
            rope_init_fn: Callable = ROPE_INIT_FUNCTIONS[self.rope_type]
            inv_freq, self.attention_scaling = rope_init_fn(self.config, None, seq_len=None)
        else:
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
            self.attention_scaling = 1.0
        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute vision M-ROPE frequencies from precomputed position ids.

        Args:
            position_ids (`torch.Tensor` of shape `(3, total_patches)`):
                Temporal, height and width indices per packed patch, as returned by
                [`get_arlow_vision_position_ids`].

        Returns:
            `torch.Tensor`: rotary frequencies of shape `(total_patches, head_dim // 2)`.
        """
        device = self.inv_freq.device
        dtype = self.inv_freq.dtype

        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        position_ids = position_ids.to(device=device, dtype=torch.float32)
        freqs = torch.einsum("ds,f->dsf", position_ids, inv_freq)
        freqs = freqs[:, None, :, :]
        freqs = apply_interleaved_mrope(freqs, self.config.mrope_sections)
        return freqs.squeeze(0).to(dtype=dtype)


# Inspired by transformers.models.gemma.modeling_gemma.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Inspired by transformers.models.gemma.modeling_gemma.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[FlashAttentionKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Inspired by transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLAttention and transformers.models.gemma.modeling_gemma.GemmaAttention
class ArlowAttention(nn.Module):
    def __init__(self, config: ArlowConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or (self.hidden_size // self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        # Sliding window support
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        # Mark this attention as causal for backend dispatchers expecting this flag
        self.is_causal = True

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=self.config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=self.config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=self.config.attention_bias,
        )
        # Always keep output projection bias disabled regardless of attention_bias.
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        cache: Cache | None = None,
        token_coords: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        q_shape = (*input_shape, self.num_heads, self.head_dim)
        kv_shape = (*input_shape, self.num_kv_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(q_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(kv_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if cache is not None:
            key_states, value_states = cache.update(key_states, value_states, self.layer_idx, kwargs=kwargs)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        # Fold heads to last dim before output projection
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# Inspired by transformers.models.gemma.modeling_gemma.GemmaMLP
class ArlowVLMLP(nn.Module):
    def __init__(self, config: ArlowConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        output = self.dropout(x)
        return output


# Inspired by transformers.models.gemma.modeling_gemma.GemmaDecoderLayer and transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLDecoderLayer
class ArlowDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: ArlowConfig, layer_idx: int):
        super().__init__()
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = ArlowAttention(config=config, layer_idx=layer_idx)
        self.mlp = ArlowVLMLP(config)
        self.input_layernorm = ArlowRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ArlowRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            cache=cache,
            **kwargs,
        )
        hidden_states = residual + self.resid_dropout(attn_out)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = residual + self.resid_dropout(mlp_out)
        return hidden_states


# Inspired by transformers.models.qwen2_vl.modeling_qwen2_vl.PatchEmbed
class ArlowVLPatchEmbed(nn.Module):
    """Convert images/videos to patch embeddings."""

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1280,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # 3D convolution for video, 2D for images
        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states:
                - `(total_patches, patch_dim)` packed flattened patches from the processor,
                - `(batch, seq_len, patch_dim)` padded flattened patches from the processor, or
                - `(batch, channels, temporal, height, width)` raw clips/images.
        Returns:
            embeddings:
                - packed input -> `(total_patches, embed_dim)`
                - padded input -> `(batch, seq_len, embed_dim)`
                - raw input -> `(batch, num_patches, embed_dim)`
        """
        expected_dim = self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size

        if hidden_states.dim() in (2, 3):
            if hidden_states.shape[-1] != expected_dim:
                raise ValueError(
                    f"Expected flattened patch dimension {expected_dim}, but received {hidden_states.shape[-1]}."
                )
            weight = self.proj.weight.flatten(1)
            return F.linear(hidden_states, weight)

        if hidden_states.dim() == 4:
            # Images without explicit temporal dimension
            hidden_states = hidden_states.unsqueeze(2)
        if hidden_states.dim() != 5:
            raise ValueError(
                "ArlowVLPatchEmbed expects flattened patches with 2/3 dims or image/video tensors with "
                f"4/5 dims. Received shape: {tuple(hidden_states.shape)}"
            )

        frames = hidden_states.shape[2]
        remainder = frames % self.temporal_patch_size
        if remainder:
            pad = self.temporal_patch_size - remainder
            hidden_states = torch.cat([hidden_states, hidden_states[:, :, -1:].repeat(1, 1, pad, 1, 1)], dim=2)

        projected = self.proj(hidden_states)
        batch_size = projected.shape[0]
        projected = projected.reshape(batch_size, self.embed_dim, -1).transpose(1, 2)
        return projected.contiguous()


# Inspired by transformers.models.qwen2_vl.modeling_qwen2_vl.PatchMerger
class ArlowVLPatchMerger(nn.Module):
    """Merge vision patches and project to text model dimension."""

    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        # LayerNorm should normalize the merged dimension, not context_dim
        self.ln_q = LayerNorm(self.hidden_size, eps=1e-6)
        # Kept as nn.Sequential to preserve the `mlp.0.*` / `mlp.2.*` keys of existing
        # checkpoints; renaming to explicit submodules would require a conversion mapping.
        self.mlp = nn.Sequential(  # trf-ignore: TRF036
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: vision embeddings of shape (total_patches, embed_dim)
        Returns:
            merged: (total_merged_patches, hidden_size)
        """
        x = self.ln_q(x)
        # Reshape for spatial merging (simplified version - actual impl may vary)
        # For now, apply MLP directly
        output = self.mlp(x)
        return output


# Inspired by transformers.models.qwen2_vl.modeling_qwen2_vl.VisionAttention
class ArlowVLAttention(nn.Module):
    """Vision self-attention with RoPE."""

    def __init__(self, config: ArlowVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=True)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        # For backend-dispatched attention APIs
        self.num_key_value_groups = 1
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.Tensor | None = None,
        token_coords: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (total_tokens, embed_dim)
            position_embeddings: (cos, sin) for RoPE
        """
        input_shape = hidden_states.shape
        input_ndim = hidden_states.dim()
        hidden_states = hidden_states.reshape(-1, self.embed_dim)
        total_tokens = hidden_states.shape[0]

        # Compute Q, K, V
        qkv = self.qkv(hidden_states).reshape(total_tokens, 3, self.num_heads, self.head_dim)
        query_states, key_states, value_states = qkv.unbind(1)

        # Apply RoPE if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            if cos.shape[0] != total_tokens:
                raise ValueError(
                    f"Vision RoPE length mismatch: cos has {cos.shape[0]} tokens, "
                    f"hidden_states has {total_tokens} tokens."
                )
            query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # Reshape for attention: (seq, heads, head_dim) -> (1, seq, heads, head_dim) -> (1, heads, seq, head_dim)
        query_states = query_states.unsqueeze(0).transpose(1, 2)
        key_states = key_states.unsqueeze(0).transpose(1, 2)
        value_states = value_states.unsqueeze(0).transpose(1, 2)

        if cu_seqlens is None:
            cu_seqlens = torch.tensor([0, total_tokens], device=hidden_states.device, dtype=torch.int32)
        else:
            cu_seqlens = cu_seqlens.to(device=hidden_states.device, dtype=torch.int32)

        # Dispatch to attention backends (FA2/SDPA/eager)
        # Use the same attribute name as text model if set on the vision config via parent
        attn_impl = getattr(self, "_attn_implementation", None)
        if attn_impl is None:
            attn_impl = getattr(self.proj, "_attn_implementation", None)
        # Fall back to config if available
        if attn_impl is None:
            attn_impl = getattr(getattr(self, "config", None), "_attn_implementation", "eager")
        if getattr(self.config, "use_deformable_attention", False) or attn_impl == "flash_attention_2":
            attn_impl = "eager"
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(attn_impl, eager_attention_forward)

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)]
        token_coord_splits = (
            torch.split(token_coords, lengths.tolist(), dim=0)
            if getattr(self.config, "use_deformable_attention", False) and token_coords is not None
            else None
        )

        outputs: list[torch.Tensor] = []
        for segment_idx, (q, k, v) in enumerate(zip(*splits)):
            if q.shape[2] == 0:
                continue

            attention_mask = None
            if token_coord_splits is not None:
                coords = token_coord_splits[segment_idx].to(device=q.device, dtype=torch.float32)
                strength = getattr(self.config, "deformable_attention_strength", 4.0)
                radius = getattr(self.config, "deformable_attention_window", 0.25)
                penalty = strength * 100.0

                dist = torch.cdist(coords, coords, p=2)
                bias = -dist * strength
                if radius > 0:
                    bias = torch.where(dist <= radius, bias, bias - penalty)
                attention_mask = bias[None, None, :, :].to(dtype=q.dtype)

            out, _ = attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=attention_mask,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )
            outputs.append(out)

        if len(outputs) == 0:
            attn_output = hidden_states.new_zeros((0, self.embed_dim))
        else:
            # Local eager/SDPA interfaces return [batch, seq, heads, head_dim].
            attn_output = torch.cat(outputs, dim=1).reshape(total_tokens, self.embed_dim)
        attn_output = self.proj(attn_output)

        if input_ndim == 3:
            attn_output = attn_output.view(input_shape)

        return attn_output


# Inspired by transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLVisionBlock
class ArlowVLBlock(GradientCheckpointingLayer):
    """Vision transformer block with attention and MLP."""

    def __init__(self, config: ArlowVisionConfig):
        super().__init__()
        self.norm1 = LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = LayerNorm(config.embed_dim, eps=1e-6)
        self.attn = ArlowVLAttention(config)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)
        # Inline MLP instead of separate class
        self.fc1 = nn.Linear(config.embed_dim, mlp_hidden_dim)
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(mlp_hidden_dim, config.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.Tensor | None = None,
        token_coords: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Self-attention with residual
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            token_coords=token_coords,
            **kwargs,
        )
        # MLP with residual (inline)
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.fc2(self.act(self.fc1(hidden_states)))
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class ArlowPreTrainedModel(PreTrainedModel):
    config: ArlowConfig
    base_model_prefix = "model"
    input_modalities = ("image", "video", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["ArlowDecoderLayer"]
    _skip_keys_device_placement = ["cache", "rotary_emb.inv_freq"]
    is_loaded_in_8bit = False
    is_loaded_in_4bit = False
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": ArlowDecoderLayer,
        "attentions": ArlowAttention,
    }

    def _init_weights(self, module: nn.Module):
        # The vision rotary embedding has no `compute_default_rope_parameters` for the base class to call
        if isinstance(module, ArlowVLRotaryEmbedding) and module.rope_type == "default":
            head_dim = module.inv_freq.shape[0] * 2
            rope_theta = getattr(module.config, "rope_theta", 100000.0)
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
            init.copy_(module.inv_freq, inv_freq)
            init.copy_(module.original_inv_freq, inv_freq)
        else:
            super()._init_weights(module)


@auto_docstring
class ArlowTextPreTrainedModel(ArlowPreTrainedModel):
    config: ArlowTextConfig
    input_modalities = ("text",)


# Inspired by transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLVisionModel
@auto_docstring(
    custom_intro="""
    Vision encoder for Arlow vision-language models. It processes images and videos into continuous embeddings that
    are forwarded to the language model, and is used internally by [`ArlowModel`].
    """
)
class ArlowVLVisionModel(ArlowPreTrainedModel):
    config: ArlowVisionConfig
    input_modalities = ("image", "video")
    _no_split_modules = ["ArlowVLBlock"]
    _can_record_outputs = {
        "hidden_states": ArlowVLBlock,
        "attentions": ArlowVLAttention,
    }

    def __init__(self, config: ArlowVisionConfig):
        super().__init__(config)
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        # Patch embedding
        self.patch_embed = ArlowVLPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        # Rotary position embeddings for vision (dim=head_dim)
        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = ArlowVLRotaryEmbedding(head_dim, config=config)

        # Transformer blocks
        self.blocks = nn.ModuleList([ArlowVLBlock(config) for _ in range(config.depth)])

        # Patch merger to project to text dimension
        self.merger = ArlowVLPatchMerger(
            dim=config.hidden_size,
            context_dim=config.embed_dim,
            spatial_merge_size=config.spatial_merge_size,
        )

        self.gradient_checkpointing = False
        self.use_progressive_patches = config.use_progressive_patches
        if self.use_progressive_patches:
            self.progressive_pool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), ceil_mode=True)
            self.progressive_proj = nn.Linear(config.embed_dim, config.embed_dim)

        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self.enable_deepstack = len(self.deepstack_visual_indexes) > 0
        if self.enable_deepstack:
            self.deepstack_mergers = nn.ModuleList(
                [
                    ArlowVLPatchMerger(
                        dim=config.hidden_size,
                        context_dim=config.embed_dim,
                        spatial_merge_size=config.spatial_merge_size,
                    )
                    for _ in self.deepstack_visual_indexes
                ]
            )
        else:
            self.deepstack_mergers = None

        self.post_init()

    def get_dtype(self) -> torch.dtype:
        return self.blocks[0].fc2.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks[0].fc2.weight.device

    def _resolve_grid_thw(
        self,
        grid_thw: torch.LongTensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.LongTensor | None:
        """Normalize grid metadata to the actual patch-token sequence length."""
        if grid_thw is None:
            return None

        if hidden_states.dim() == 2:
            token_count = hidden_states.shape[0]
        elif hidden_states.dim() == 3:
            token_count = hidden_states.shape[0] * hidden_states.shape[1]
        else:
            return grid_thw

        expected = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
        if expected == token_count:
            return grid_thw

        temporal_patch = getattr(self.patch_embed, "temporal_patch_size", 1)
        patched_grid = grid_thw.clone()
        patched_grid[:, 0] = (patched_grid[:, 0] + temporal_patch - 1) // temporal_patch
        patched_expected = int((patched_grid[:, 0] * patched_grid[:, 1] * patched_grid[:, 2]).sum().item())
        if patched_expected == token_count:
            return patched_grid

        return grid_thw

    def _pack_hidden_states(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.LongTensor | None,
    ) -> torch.Tensor:
        """Convert padded vision states to a packed token sequence."""
        if hidden_states.dim() == 2:
            return hidden_states

        if hidden_states.dim() != 3:
            raise ValueError(f"Expected 2D or 3D hidden_states, got {tuple(hidden_states.shape)}.")

        if grid_thw is None:
            return hidden_states.reshape(-1, hidden_states.shape[-1])

        pieces: list[torch.Tensor] = []
        for batch_idx, (t, h, w) in enumerate(grid_thw.tolist()):
            n = int(t) * int(h) * int(w)
            pieces.append(hidden_states[batch_idx, :n])

        if len(pieces) == 0:
            return hidden_states.new_zeros((0, hidden_states.shape[-1]))

        return torch.cat(pieces, dim=0)

    def _make_cu_seqlens(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.LongTensor | None,
    ) -> torch.Tensor:
        """Build packed-sequence boundaries for independent image/video attention."""
        if grid_thw is None:
            return torch.tensor([0, hidden_states.shape[0]], device=hidden_states.device, dtype=torch.int32)

        lengths = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).to(
            device=hidden_states.device,
            dtype=torch.int32,
        )
        total = int(lengths.sum().item())
        if total != hidden_states.shape[0]:
            raise ValueError(
                f"grid_thw describes {total} vision tokens, but patch embedding produced "
                f"{hidden_states.shape[0]} tokens. grid_thw={grid_thw.tolist()}"
            )

        return F.pad(torch.cumsum(lengths, dim=0), (1, 0))

    def _make_token_coords(
        self,
        grid_thw: torch.LongTensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Packed THW coordinates matching row-major processor order."""
        if grid_thw is None:
            return None

        coords: list[torch.Tensor] = []
        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            if t <= 0 or h <= 0 or w <= 0:
                continue

            t_range = torch.linspace(0, 1, steps=t, device=device) if t > 1 else torch.zeros(1, device=device)
            h_range = torch.linspace(0, 1, steps=h, device=device) if h > 1 else torch.zeros(1, device=device)
            w_range = torch.linspace(0, 1, steps=w, device=device) if w > 1 else torch.zeros(1, device=device)

            tt, hh, ww = torch.meshgrid(t_range, h_range, w_range, indexing="ij")
            coords.append(torch.stack([tt, hh, ww], dim=-1).reshape(-1, 3))

        if len(coords) == 0:
            return None

        return torch.cat(coords, dim=0)

    def _apply_progressive_patches(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.LongTensor | None,
    ) -> torch.Tensor:
        """Progressive patch refinement for packed row-major THW tokens."""
        if not self.use_progressive_patches or grid_thw is None:
            return hidden_states

        pieces: list[torch.Tensor] = []
        start = 0
        embed_dim = hidden_states.shape[-1]

        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            n = t * h * w
            tokens = hidden_states[start : start + n]
            start += n

            if n == 0:
                pieces.append(tokens)
                continue

            view_tensor = tokens.view(t, h, w, embed_dim).permute(3, 0, 1, 2).contiguous()
            pooled = self.progressive_pool(view_tensor.unsqueeze(0))
            upsampled = F.interpolate(
                pooled,
                size=(t, h, w),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)

            refined = self.progressive_proj(upsampled.permute(1, 2, 3, 0).reshape(n, embed_dim)).to(tokens.dtype)
            pieces.append(tokens + refined)

        if start != hidden_states.shape[0]:
            raise ValueError(
                f"grid_thw consumed {start} tokens, but hidden_states has {hidden_states.shape[0]} tokens."
            )

        return torch.cat(pieces, dim=0) if pieces else hidden_states

    def _reshape_for_merger(self, hidden_states: torch.Tensor, grid_thw: torch.LongTensor | None) -> torch.Tensor:
        """
        Convert unmerged row-major THW patch tokens to spatially merged tokens.
        """
        hidden_states = self._pack_hidden_states(hidden_states, grid_thw)
        spatial_merge_size = self.config.spatial_merge_size
        embed_dim = hidden_states.shape[-1]
        merge_unit = spatial_merge_size * spatial_merge_size

        if grid_thw is None:
            if hidden_states.shape[0] % merge_unit != 0:
                raise ValueError(
                    f"Cannot spatial-merge {hidden_states.shape[0]} tokens by merge unit {merge_unit} without grid_thw."
                )
            return hidden_states.reshape(-1, merge_unit * embed_dim)

        merged: list[torch.Tensor] = []
        start = 0
        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)

            if h % spatial_merge_size != 0 or w % spatial_merge_size != 0:
                raise ValueError(
                    f"Grid height/width must be divisible by spatial_merge_size={spatial_merge_size}; "
                    f"got grid (t={t}, h={h}, w={w})."
                )

            n = t * h * w
            item = hidden_states[start : start + n]
            start += n

            if item.shape[0] != n:
                raise ValueError(f"Expected {n} tokens for grid {(t, h, w)}, got {item.shape[0]}.")

            item = item.view(t, h, w, embed_dim)
            item = item.view(
                t,
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
                embed_dim,
            )
            item = item.permute(0, 1, 3, 2, 4, 5).contiguous()
            item = item.view(
                t * (h // spatial_merge_size) * (w // spatial_merge_size),
                merge_unit * embed_dim,
            )
            merged.append(item)

        if start != hidden_states.shape[0]:
            raise ValueError(
                f"grid_thw consumed {start} tokens, but hidden_states has {hidden_states.shape[0]} tokens."
            )

        return torch.cat(merged, dim=0) if merged else hidden_states.new_zeros((0, merge_unit * embed_dim))

    @capture_outputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
        r"""
        pixel_values (`torch.Tensor`):
            Packed flattened patches, padded flattened patches, or raw image/video tensors.
        grid_thw (`torch.LongTensor` of shape `(num_items, 3)`, *optional*):
            Unmerged patch grid per visual item in `[temporal, height, width]` format.
        """
        hidden_states = self.patch_embed(pixel_values)

        if grid_thw is not None:
            grid_thw = grid_thw.to(device=hidden_states.device)
            grid_thw = self._resolve_grid_thw(grid_thw, hidden_states)

        hidden_states = self._pack_hidden_states(hidden_states, grid_thw)

        cu_seqlens = self._make_cu_seqlens(hidden_states, grid_thw)
        token_coords = self._make_token_coords(grid_thw, hidden_states.device)
        hidden_states = self._apply_progressive_patches(hidden_states, grid_thw)

        if grid_thw is not None:
            position_ids = get_arlow_vision_position_ids(grid_thw)
        else:
            # 1D fallback: a flat ramp over the packed patches, replicated on all three M-ROPE axes
            seq = torch.arange(
                hidden_states.shape[0],
                device=hidden_states.device,
                dtype=self.rotary_pos_emb.inv_freq.dtype,
            )
            position_ids = torch.stack([seq, seq, seq])
        rotary_pos_emb = self.rotary_pos_emb(position_ids)
        if rotary_pos_emb.shape[0] != hidden_states.shape[0]:
            # Progressive patching can change the packed token count; fall back to a flat ramp.
            seq = torch.arange(
                hidden_states.shape[0],
                device=hidden_states.device,
                dtype=self.rotary_pos_emb.inv_freq.dtype,
            )
            rotary_pos_emb = torch.outer(seq, self.rotary_pos_emb.inv_freq.to(seq.device))
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos().to(hidden_states.dtype)
        sin = emb.sin().to(hidden_states.dtype)
        position_embeddings = (cos, sin)

        deepstack_feature_lists: list[torch.Tensor] = []
        deepstack_idx = 0

        # Apply transformer blocks
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                token_coords=token_coords,
            )
            if (
                self.enable_deepstack
                and deepstack_idx < len(self.deepstack_visual_indexes)
                and layer_idx == self.deepstack_visual_indexes[deepstack_idx]
            ):
                reshaped = self._reshape_for_merger(hidden_states, grid_thw)
                deepstack_feature_lists.append(self.deepstack_mergers[deepstack_idx](reshaped))
                deepstack_idx += 1

        merged_tokens = self._reshape_for_merger(hidden_states, grid_thw)
        vision_embeddings = self.merger(merged_tokens)

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=vision_embeddings,
            deepstack_features=deepstack_feature_lists if self.enable_deepstack else None,
        )


# Inspired by transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLTextModel
@auto_docstring(
    custom_intro="""
    Text-only decoder of Arlow, used internally by the multimodal [`ArlowModel`] and standalone in
    [`ArlowForCausalLM`] for text-only tasks.
    """
)
class ArlowTextModel(ArlowTextPreTrainedModel):
    config: ArlowTextConfig

    def __init__(
        self,
        config: ArlowTextConfig,
        vision_config: ArlowVisionConfig | None = None,
        use_gated_cross_attention: bool | None = None,
        gated_cross_attention_start_layer: int | None = None,
    ):
        """
        Args:
            vision_config (`ArlowVisionConfig`, *optional*):
                Vision encoder configuration, provided when the decoder runs as part of the multimodal
                [`ArlowModel`] so layers can attend to visual features.
            use_gated_cross_attention (`bool`, *optional*):
                Whether decoder layers cross-attend to the vision encoder hidden states. Defaults to the text
                config's `use_gated_cross_attention` value.
            gated_cross_attention_start_layer (`int`, *optional*):
                First layer that performs gated cross-attention. Defaults to the text config's
                `gated_cross_attention_start_layer` value.
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.has_sliding_layers = "sliding_attention" in config.layer_types
        self.vision_config = vision_config
        self.use_gated_cross_attention = (
            getattr(config, "use_gated_cross_attention", False)
            if use_gated_cross_attention is None
            else use_gated_cross_attention
        )
        self.gated_cross_attention_start_layer = (
            getattr(config, "gated_cross_attention_start_layer", None)
            if gated_cross_attention_start_layer is None
            else gated_cross_attention_start_layer
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ArlowDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = ArlowRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = ArlowTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.visual_gates: torch.nn.Parameter | None = None
        self._gated_cross_attention_start_layer: int | None = None
        if self.use_gated_cross_attention:
            deepstack_indexes: list[int] = []
            if self.vision_config is not None:
                deepstack_indexes = getattr(self.vision_config, "deepstack_visual_indexes", [])

            start_layer = self.gated_cross_attention_start_layer
            if start_layer is None:
                start_layer = min(deepstack_indexes) if len(deepstack_indexes) > 0 else config.num_hidden_layers - 1
            self._gated_cross_attention_start_layer = max(0, min(start_layer, config.num_hidden_layers - 1))
            self.visual_gates = nn.Parameter(torch.zeros(config.num_hidden_layers))

        self.post_init()

    # emb getters
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if visual_embeds.numel() == 0:
            return hidden_states
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        updated = hidden_states.clone()
        updated[visual_pos_masks, :] = updated[visual_pos_masks, :] + visual_embeds
        return updated

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        position_embeddings (`tuple[torch.Tensor, torch.Tensor]`, *optional*):
            Precomputed `(cos, sin)` rotary embeddings, e.g. the multimodal M-RoPE embeddings computed by
            [`ArlowModel`]. Computed from `position_ids` when not provided.
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Boolean mask marking the visual token positions that receive DeepStack features.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            Visual features captured from the vision encoder, one entry per decoder layer that receives them.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        # When both are provided, prefer inputs_embeds (this is expected during generation)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and cache is None:
            cache = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = cache.get_seq_length() if cache is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # Determine text position ids. Support both classic 2D text positions and FA2-style packed positions.
        text_position_ids = None
        # FA2-style packed positions: [4, batch, seq] => [text_pos; 3D mrope]
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        # Classic text-only packed positions: [batch, seq]
        elif position_ids.ndim == 2:
            text_position_ids = position_ids

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache": cache,
                # Pass 2D text position ids so the new attention mask API can infer packed sequences
                "position_ids": text_position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # The text row of packed 4D positions only serves the masks, RoPE uses the M-RoPE rows
        if position_embeddings is None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[layer_idx]],
                position_ids=text_position_ids,
                cache=cache,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if (
                visual_pos_masks is not None
                and deepstack_visual_embeds is not None
                and layer_idx < len(deepstack_visual_embeds)
            ):
                deepstack_layer = deepstack_visual_embeds[layer_idx]
                if deepstack_layer is not None and deepstack_layer.numel() > 0:
                    deepstack_layer = deepstack_layer.to(hidden_states.device, hidden_states.dtype)
                    apply_layer = True
                    if self.use_gated_cross_attention:
                        start_layer = self._gated_cross_attention_start_layer or 0
                        apply_layer = layer_idx >= start_layer
                    if apply_layer:
                        if self.visual_gates is not None:
                            gate = torch.sigmoid(self.visual_gates[layer_idx]).to(
                                dtype=hidden_states.dtype, device=hidden_states.device
                            )
                            deepstack_layer = deepstack_layer * gate
                        hidden_states = self._deepstack_process(hidden_states, visual_pos_masks, deepstack_layer)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            cache=cache if use_cache else None,
        )


# Inspired by transformers.models.gemma.modeling_gemma.GemmaForCausalLM
@auto_docstring(
    custom_intro="""
    Arlow model for causal language modeling (text-only, no vision). For multimodal (vision + text) tasks, use
    [`ArlowForConditionalGeneration`] instead.
    """
)
class ArlowForCausalLM(ArlowTextPreTrainedModel, GenerationMixin):
    config: ArlowTextConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: ArlowTextConfig):
        super().__init__(config)
        text_config = config.text_config if hasattr(config, "text_config") else config
        self.model = ArlowTextModel(text_config)
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

        self.post_init()
        self._tp_plan["lm_head"] = "colwise_gather_output"
        self._pp_plan["lm_head"] = (["hidden_states"], ["logits"])

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int):
            slice_indices = slice(None) if logits_to_keep <= 0 else slice(-logits_to_keep, None)
        else:
            slice_indices = logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            labels_window = labels[:, slice_indices]
            loss = self.loss_function(logits=logits, labels=labels_window, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Inspired by transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration
@auto_docstring(
    custom_intro="""
    The Arlow vision-language model, combining the vision encoder and the text decoder. For text-only tasks, use
    [`ArlowTextModel`] or [`ArlowForCausalLM`] instead.
    """
)
class ArlowModel(ArlowPreTrainedModel):
    config: ArlowConfig

    base_model_prefix = "model"
    accepts_loss_kwargs = False

    def __init__(self, config: ArlowConfig):
        super().__init__(config)

        # Vision encoder
        self.visual = ArlowVLVisionModel._from_config(config.vision_config)

        # Text model (language decoder)
        self.language_model = ArlowTextModel._from_config(
            config.text_config,
            vision_config=config.vision_config,
            use_gated_cross_attention=config.use_gated_cross_attention,
            gated_cross_attention_start_layer=config.gated_cross_attention_start_layer,
        )
        self.rope_deltas = None  # cache rope_deltas here

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def _get_visual_split_sizes(self, grid_thw: torch.LongTensor | None, num_tokens: int) -> list[int]:
        """Number of merged vision tokens of each image/video, used to split the packed encoder outputs."""
        if grid_thw is None:
            return [num_tokens]

        merge_size = self.config.vision_config.spatial_merge_size
        grid_thw = grid_thw.to("cpu")
        split_sizes = (grid_thw[:, 0] * (grid_thw[:, 1] // merge_size) * (grid_thw[:, 2] // merge_size)).tolist()
        if sum(split_sizes) != num_tokens:
            # The vision encoder also accepts grids that count frames instead of temporal patches
            temporal_patch_size = self.config.vision_config.temporal_patch_size
            patched_t = (grid_thw[:, 0] + temporal_patch_size - 1) // temporal_patch_size
            split_sizes = (patched_t * (grid_thw[:, 1] // merge_size) * (grid_thw[:, 2] // merge_size)).tolist()
            if sum(split_sizes) != num_tokens:
                raise ValueError(
                    f"Visual feature split mismatch: the vision encoder returned {num_tokens} merged tokens, which "
                    f"does not match grid_thw={grid_thw.tolist()}."
                )
        return split_sizes

    def _encode_visual(
        self, pixel_values: torch.Tensor, grid_thw: torch.LongTensor | None, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutputWithDeepstackFeatures:
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output: BaseModelOutputWithDeepstackFeatures = self.visual(
            pixel_values, grid_thw=grid_thw, return_dict=True, **kwargs
        )
        split_sizes = self._get_visual_split_sizes(grid_thw, vision_output.pooler_output.shape[0])
        vision_output.pooler_output = torch.split(vision_output.pooler_output, split_sizes)
        if vision_output.deepstack_features is not None:
            vision_output.deepstack_features = [
                torch.split(feature, split_sizes) for feature in vision_output.deepstack_features
            ]
        return vision_output

    @accepts_precomputed_kwargs(modality="image")
    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
        r"""
        pixel_values (`torch.FloatTensor`):
            Flattened image patch tensor produced by the image processor.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Image grid metadata in `[temporal, height, width]` format for each image.
        """
        return self._encode_visual(pixel_values, image_grid_thw, **kwargs)

    @accepts_precomputed_kwargs(modality="video")
    @can_return_tuple
    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
        r"""
        pixel_values_videos (`torch.FloatTensor`):
            Flattened video patch tensor produced by the video processor.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Video grid metadata in `[temporal, height, width]` format for each video.
        """
        return self._encode_visual(pixel_values_videos, video_grid_thw, **kwargs)

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that
        the placeholder token count is equal to the length of multimodal features.
        """
        if input_ids is None:
            if self.config.image_token_id is not None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.full((), self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                special_image_mask = special_image_mask.all(-1)
            else:
                special_image_mask = torch.zeros(
                    inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device
                )

            if self.config.video_token_id is not None:
                special_video_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.full((), self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                special_video_mask = special_video_mask.all(-1)
            else:
                special_video_mask = torch.zeros(
                    inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device
                )
        else:
            if self.config.image_token_id is not None:
                special_image_mask = input_ids == self.config.image_token_id
            else:
                special_image_mask = torch.zeros_like(input_ids, dtype=torch.bool)

            if self.config.video_token_id is not None:
                special_video_mask = input_ids == self.config.video_token_id
            else:
                special_video_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []

        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i].to(input_ids.device) == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                # Handle case when there are no vision tokens or only one
                if vision_start_indices.numel() == 0:
                    vision_tokens = torch.tensor([], dtype=input_ids.dtype, device=input_ids.device)
                else:
                    if vision_start_indices.dim() == 0:
                        vision_start_indices = vision_start_indices.unsqueeze(0)
                    vision_tokens = input_ids[vision_start_indices + 1]

                # Handle scalar case when vision_tokens is a single element
                if vision_tokens.numel() == 0:
                    image_nums = 0
                    video_nums = 0
                elif vision_tokens.numel() == 1:
                    # Single token case - need to handle as tensor
                    image_nums = 1 if vision_tokens.item() == image_token_id else 0
                    video_nums = 1 if vision_tokens.item() == video_token_id else 0
                else:
                    image_nums = (vision_tokens == image_token_id).sum()
                    video_nums = (vision_tokens == video_token_id).sum()

                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cache: Cache | None = None,
    ) -> torch.Tensor | None:
        cache_length = 0 if cache is None else cache.get_seq_length()
        if input_ids is not None and cache_length == 0:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        # Use the rope deltas of the prefill to infer the 3D position ids during incremental decoding
        elif self.rope_deltas is not None and cache_length > 0:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(cache_length, cache_length + seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
            position_ids = position_ids + delta.to(device=inputs_embeds.device)
        else:
            # Can't build correct 3D positions. Let the model infer it
            position_ids = None
        return position_ids

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MultimodalBaseModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Image grid dimensions as `[temporal=1, height, width]` for each image in the batch.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Video grid dimensions as `[temporal, height, width]` for each video in the batch.
        """
        if (pixel_values is not None or pixel_values_videos is not None) and mm_encoder_outputs is not None:
            raise ValueError(
                "You cannot specify both pixel_values/pixel_values_videos and mm_encoder_outputs at the same time"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True, **kwargs
            )
        if mm_encoder_outputs.get("video") is None and pixel_values_videos is not None:
            mm_encoder_outputs["video"] = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True, **kwargs
            )

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None
        if mm_encoder_outputs.get("image") is not None:
            image_embeds = torch.cat(mm_encoder_outputs["image"].pooler_output, dim=0)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            deepstack_image_embeds = mm_encoder_outputs["image"].deepstack_features
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if mm_encoder_outputs.get("video") is not None:
            video_embeds = torch.cat(mm_encoder_outputs["video"].pooler_output, dim=0)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            deepstack_video_embeds = mm_encoder_outputs["video"].deepstack_features
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            if deepstack_image_embeds is not None and deepstack_video_embeds is not None:
                deepstack_visual_embeds = []
                image_mask_joint = image_mask[visual_pos_masks]
                video_mask_joint = video_mask[visual_pos_masks]
                for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                    img_embed = torch.cat(img_embed, dim=0)
                    vid_embed = torch.cat(vid_embed, dim=0)
                    embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1])
                    embed_joint[image_mask_joint, :] = img_embed
                    embed_joint[video_mask_joint, :] = vid_embed
                    deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
            if deepstack_image_embeds is not None:
                deepstack_visual_embeds = [torch.cat(feature, dim=0) for feature in deepstack_image_embeds]
        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
            if deepstack_video_embeds is not None:
                deepstack_visual_embeds = [torch.cat(feature, dim=0) for feature in deepstack_video_embeds]

        if deepstack_visual_embeds is not None:
            # The text decoder expects one entry per layer, with the captured features at their injection layers
            layer_embeds: list[torch.Tensor | None] = [None] * self.config.text_config.num_hidden_layers
            for layer_idx, embeds in zip(self.config.text_deepstack_injection_layers, deepstack_visual_embeds):
                layer_embeds[layer_idx] = embeds
            deepstack_visual_embeds = layer_embeds

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                cache=cache,
            )

        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return MultimodalBaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_outputs=mm_encoder_outputs.get("image"),
            video_outputs=mm_encoder_outputs.get("video"),
        )


@auto_docstring
# Inspired by transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration
class ArlowForConditionalGeneration(ArlowPreTrainedModel, GenerationMixin):
    """
    Arlow model for conditional generation with vision-language inputs (VLM).

    This is the main model for multimodal vision-language tasks. It uses ArlowModel internally
    and adds a language modeling head for text generation conditioned on vision inputs.

    For text-only generation, use ArlowForCausalLM instead.

    Example:
        ```python
        >>> from transformers import AutoProcessor, ArlowForConditionalGeneration
        >>> model = ArlowForConditionalGeneration.from_pretrained("yuchenxie/arlow-vlm")
        >>> processor = AutoProcessor.from_pretrained("yuchenxie/arlow-vlm")
        >>>
        >>> messages = [{
        ...     "role": "user",
        ...     "content": [
        ...         {"type": "image", "image": "path/to/image.jpg"},
        ...         {"type": "text", "text": "Describe this image."},
        ...     ],
        ... }]
        >>>
        >>> inputs = processor(text=messages, images=..., return_tensors="pt")
        >>> outputs = model.generate(**inputs, max_new_tokens=100)
        >>> processor.batch_decode(outputs, skip_special_tokens=True)[0]
        ```
    """

    config: ArlowConfig
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}

    def __init__(self, config: ArlowConfig):
        super().__init__(config)
        self.model = ArlowModel(config)
        text_config = config.text_config
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Image grid dimensions as `[temporal=1, height, width]` for each image in the batch.
        """
        return self.model.get_image_features(pixel_values, image_grid_thw, **kwargs)

    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
        r"""
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Video grid dimensions as `[temporal, height, width]` for each video in the batch.
        """
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, **kwargs)

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MultimodalCausalLMOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Image grid dimensions as `[temporal=1, height, width]` for each image in the batch.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Video grid dimensions as `[temporal, height, width]` for each video in the batch.

        Example:
        ```python
        >>> from transformers import AutoProcessor, ArlowForConditionalGeneration
        >>> model = ArlowForConditionalGeneration.from_pretrained("yuchenxie/arlow-vlm")
        >>> processor = AutoProcessor.from_pretrained("yuchenxie/arlow-vlm")
        >>>
        >>> messages = [{
        ...     "role": "user",
        ...     "content": [
        ...         {"type": "image", "image": "path/to/image.jpg"},
        ...         {"type": "text", "text": "Describe this image."},
        ...     ],
        ... }]
        >>>
        >>> inputs = processor(text=messages, images=..., return_tensors="pt")
        >>> outputs = model.generate(**inputs, max_new_tokens=100)
        >>> processor.batch_decode(outputs, skip_special_tokens=True)[0]
        ```
        """
        outputs: MultimodalBaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            labels_window = labels[:, slice_indices]
            loss = self.loss_function(
                logits=logits, labels=labels_window, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return MultimodalCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_outputs=outputs.image_outputs,
            video_outputs=outputs.video_outputs,
        )

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        # Overwritten -- requires 3D position ids

        text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)

        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]
        is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]

        # When continuing generation from past kv, the stored `rope_deltas` come from an earlier call, so the 3D
        # positions only have to be recomputed if the uncached tokens hold new images or videos
        past_length = 0
        if (cache := model_kwargs.get("cache")) is not None:
            past_length = cache.get_seq_length()
        vision_token_ids = torch.tensor(
            [self.config.image_token_id, self.config.video_token_id], device=inputs_tensor.device
        )
        has_new_vision_tokens = is_input_ids and bool(
            torch.isin(inputs_tensor[:, past_length:], vision_token_ids).any()
        )

        if past_length != 0 and self.model.rope_deltas is not None and not has_new_vision_tokens:
            vision_positions = (text_positions + self.model.rope_deltas).unsqueeze(0).expand(3, -1, -1)
        elif is_input_ids and (
            model_kwargs.get("image_grid_thw") is not None or model_kwargs.get("video_grid_thw") is not None
        ):
            vision_positions, rope_deltas = self.model.get_rope_index(
                inputs_tensor,
                image_grid_thw=model_kwargs.get("image_grid_thw"),
                video_grid_thw=model_kwargs.get("video_grid_thw"),
                attention_mask=model_kwargs.get("attention_mask"),
            )
            self.model.rope_deltas = rope_deltas
        else:
            vision_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
            self.model.rope_deltas = torch.zeros(
                inputs_tensor.shape[0], 1, dtype=torch.long, device=inputs_tensor.device
            )

        # Concatenate "text + vision" positions into [4, bs, seq-len]
        return torch.cat([text_positions[None, ...], vision_positions.to(text_positions.device)], dim=0)


class ArlowProcessorKwargs(ProcessingKwargs, total=False):
    r"""
    timestamp_alignment (`bool`, *optional*):
        Whether to inject timestamp prompts for video frames. Overrides the processor default for this call.
    """

    timestamp_alignment: bool


# Inspired by transformers.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
@auto_docstring
class ArlowProcessor(ProcessorMixin):
    r"""
    timestamp_alignment (`bool`, *optional*, defaults to `False`):
        Whether to inject timestamp prompts for video frames.
    """

    text_kwargs = {"padding": False, "return_token_type_ids": False}
    videos_kwargs = {"return_metadata": True}
    attributes = ["image_processor", "tokenizer", "video_processor"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "ArlowTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        timestamp_alignment: bool | None = None,
        **kwargs,
    ):
        """
        Args:
            timestamp_alignment (`bool`, *optional*, defaults to `False`):
                Whether to inject timestamp prompts for video frames. Falls back to the video processor or
                tokenizer setting when not provided.
        """
        # multimodal special tokens
        self.image_token = "<image>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<video>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )

        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)

        self.vision_start_token = (
            "<|vision_start|>" if not hasattr(tokenizer, "vision_start_token") else tokenizer.vision_start_token
        )
        self.vision_end_token = (
            "<|vision_end|>" if not hasattr(tokenizer, "vision_end_token") else tokenizer.vision_end_token
        )
        self.vision_start_token_id = (
            tokenizer.vision_start_token_id
            if getattr(tokenizer, "vision_start_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_start_token)
        )
        self.vision_end_token_id = (
            tokenizer.vision_end_token_id
            if getattr(tokenizer, "vision_end_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_end_token)
        )

        if timestamp_alignment is None:
            timestamp_alignment = getattr(video_processor, "timestamp_alignment", None)
        if timestamp_alignment is None:
            timestamp_alignment = getattr(tokenizer, "timestamp_alignment", None)
        if timestamp_alignment is None and hasattr(tokenizer, "init_kwargs"):
            timestamp_alignment = tokenizer.init_kwargs.get("timestamp_alignment")
        self.timestamp_alignment = bool(timestamp_alignment) if timestamp_alignment is not None else False

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[ArlowProcessorKwargs],
    ) -> BatchFeature:
        override_timestamp_alignment = kwargs.pop("timestamp_alignment", None)

        output_kwargs = self._merge_kwargs(
            ArlowProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = {}
        video_inputs = {}
        image_grid_thw = None
        video_grid_thw = None
        image_num_crops: list[int] | None = None

        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs.get("image_grid_thw")
            raw_num_crops = image_inputs.pop("num_crops", None)
            if raw_num_crops is not None:
                if isinstance(raw_num_crops, torch.Tensor):
                    image_num_crops = raw_num_crops.tolist()
                elif isinstance(raw_num_crops, (list, tuple)):
                    image_num_crops = list(raw_num_crops)
                elif isinstance(raw_num_crops, int):
                    image_num_crops = [raw_num_crops]
                elif np.isscalar(raw_num_crops):
                    image_num_crops = [int(raw_num_crops)]
                else:
                    image_num_crops = list(raw_num_crops)

        videos_kwargs = output_kwargs["videos_kwargs"]
        videos_timestamp_alignment = videos_kwargs.pop("timestamp_alignment", None)

        if videos is not None:
            video_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = video_inputs.get("video_grid_thw")
            # preserve metadata for timestamp prompts if provided
            if "return_metadata" not in kwargs and "video_metadata" in video_inputs:
                video_metadata = video_inputs.pop("video_metadata")
            else:
                video_metadata = video_inputs.get("video_metadata")

        if override_timestamp_alignment is not None:
            timestamp_alignment_flag = bool(override_timestamp_alignment)
        elif videos_timestamp_alignment is not None:
            timestamp_alignment_flag = bool(videos_timestamp_alignment)
        else:
            timestamp_alignment_flag = self.timestamp_alignment

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # will edit in-place

        # Inject additional image placeholders for pan-and-scan crops
        if image_num_crops is not None:
            placeholder_idx = 0
            for i, prompt in enumerate(text):
                if prompt is None:
                    continue
                occurrences: list[int] = []
                start = 0
                while True:
                    found = prompt.find(self.image_token, start)
                    if found == -1:
                        break
                    occurrences.append(found)
                    start = found + len(self.image_token)

                for pos in reversed(occurrences):
                    if placeholder_idx >= len(image_num_crops):
                        raise ValueError("Mismatch between number of image placeholders in text and image inputs.")
                    crops = image_num_crops[placeholder_idx]
                    if crops > 0:
                        formatted = (
                            f"Here is the original image {self.image_token} and here are some crops to help you see better "
                            + " ".join([self.image_token] * crops)
                        )
                        prompt = prompt[:pos] + formatted + prompt[pos + len(self.image_token) :]
                    placeholder_idx += 1
                text[i] = prompt

            if placeholder_idx != len(image_num_crops):
                raise ValueError(
                    "Mismatch between number of processed images and placeholders after pan-and-scan expansion."
                )

        # Expand image placeholders by computed token budget
        if image_grid_thw is not None:
            merge_len = self.image_processor.merge_size**2 if hasattr(self.image_processor, "merge_size") else 1
            index = 0
            placeholder_idx = 0
            image_compound = f"{self.vision_start_token}{self.image_token}{self.vision_end_token}"
            for i in range(len(text)):
                while text[i] is not None and self.image_token in text[i]:
                    views = 1
                    if image_num_crops is not None:
                        if placeholder_idx >= len(image_num_crops):
                            raise ValueError(
                                "Mismatch between number of image placeholders and pan-and-scan metadata."
                            )
                        views += image_num_crops[placeholder_idx]
                    placeholder_tokens = ""
                    for _ in range(views):
                        if index >= len(image_grid_thw):
                            raise ValueError(
                                "Not enough image grid metadata to expand placeholders. "
                                "Check pan-and-scan configuration."
                            )
                        grid_entry = image_grid_thw[index]
                        if isinstance(grid_entry, torch.Tensor):
                            num_image_tokens = int(torch.prod(grid_entry).item()) // merge_len
                        else:
                            num_image_tokens = int(np.prod(grid_entry)) // merge_len
                        placeholder_tokens += (
                            self.vision_start_token + "<|placeholder|>" * num_image_tokens + self.vision_end_token
                        )
                        index += 1
                    if image_compound in text[i]:
                        text[i] = text[i].replace(image_compound, placeholder_tokens, 1)
                    else:
                        text[i] = text[i].replace(self.image_token, placeholder_tokens, 1)
                    placeholder_idx += 1
                if text[i] is not None:
                    text[i] = text[i].replace("<|placeholder|>", self.image_token)

            if image_num_crops is not None and placeholder_idx != len(image_num_crops):
                raise ValueError(
                    "Mismatch between number of placeholders processed and pan-and-scan metadata entries."
                )
            if index != len(image_grid_thw):
                logger.warning(
                    "Some image grid metadata entries were unused during placeholder expansion. "
                    "This may indicate mismatched prompts or preprocessing configuration."
                )

        # Expand video placeholders into per-frame segments with optional timestamps
        if video_grid_thw is not None:
            merge_len = self.video_processor.merge_size**2 if hasattr(self.video_processor, "merge_size") else 1
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    if index >= len(video_grid_thw):
                        raise ValueError(
                            "Not enough video grid metadata to expand placeholders. "
                            "Check video preprocessing and prompt placeholders."
                        )
                    # Build one contiguous visual span per video grid row so M-RoPE consumes one video item.
                    grid_entry = video_grid_thw[index]
                    if isinstance(grid_entry, torch.Tensor):
                        num_video_tokens = int(torch.prod(grid_entry).item()) // merge_len
                    else:
                        num_video_tokens = int(np.prod(grid_entry)) // merge_len
                    num_video_frames = int(grid_entry[0])

                    # compute timestamps if metadata exists, otherwise just omit
                    curr_timestamps = None
                    if timestamp_alignment_flag and video_metadata is not None:
                        if index >= len(video_metadata):
                            logger.warning_once(
                                "Video metadata entries are fewer than video placeholders; "
                                "skipping timestamp prompts for remaining videos."
                            )
                        else:
                            metadata = video_metadata[index]
                            if getattr(metadata, "fps", None) is None:
                                logger.warning_once(
                                    "Arlow requires video fps to build timestamp prompts; defaulting to fps=24."
                                )
                                metadata.fps = 24
                            curr_timestamps = self._calculate_timestamps(
                                metadata.frames_indices, metadata.fps, self.video_processor.merge_size
                            )
                            if len(curr_timestamps) < num_video_frames:
                                if len(curr_timestamps) == 0:
                                    logger.warning_once(
                                        "No frame indices available for timestamp alignment; omitting timestamp prompts."
                                    )
                                    curr_timestamps = None
                                else:
                                    logger.warning_once(
                                        "Timestamp count is smaller than temporal grid size; "
                                        "repeating the last timestamp."
                                    )
                                    curr_timestamps = curr_timestamps + [curr_timestamps[-1]] * (
                                        num_video_frames - len(curr_timestamps)
                                    )
                            elif len(curr_timestamps) > num_video_frames:
                                curr_timestamps = curr_timestamps[:num_video_frames]

                    timestamp_prefix = ""
                    if curr_timestamps is not None:
                        timestamp_prefix = "".join(f"<{curr_time:.1f} seconds>" for curr_time in curr_timestamps)
                    video_placeholder = (
                        timestamp_prefix
                        + self.vision_start_token
                        + "<|placeholder|>" * num_video_tokens
                        + self.vision_end_token
                    )

                    compound = f"{self.vision_start_token}{self.video_token}{self.vision_end_token}"
                    if compound in text[i]:
                        text[i] = text[i].replace(compound, video_placeholder, 1)
                    else:
                        text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    index += 1

                text[i] = text[i].replace("<|placeholder|>", self.video_token)
            if index != len(video_grid_thw):
                logger.warning(
                    "Some video grid metadata entries were unused during placeholder expansion. "
                    "This may indicate mismatched prompts or preprocessing configuration."
                )

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])  # type: ignore[arg-type]
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])  # type: ignore[arg-type]
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **video_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        vision_data = {}
        if image_sizes is not None and self.image_processor is not None:
            images_kwargs = dict(kwargs)
            merge_size = images_kwargs.get("merge_size") or getattr(self.image_processor, "merge_size", 1)
            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        if video_sizes is not None and self.video_processor is not None:
            videos_kwargs = {**self.videos_kwargs, **kwargs}
            num_video_patches = [
                self.video_processor.get_number_of_video_patches(*video_size, videos_kwargs)
                for video_size in video_sizes
            ]
            merge_size = getattr(self.video_processor, "merge_size", 1)
            num_video_tokens = [(num_patches // merge_size**2) for num_patches in num_video_patches]
            vision_data["num_video_tokens"] = num_video_tokens

        return MultiModalData(**vision_data)  # type: ignore[name-defined]

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def _calculate_timestamps(self, indices: list[int] | np.ndarray, video_fps: float, merge_size: int = 2):
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        timestamps = [idx / video_fps for idx in indices]
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps


@auto_docstring
class ArlowForSequenceClassification(GenericForSequenceClassification, ArlowPreTrainedModel):
    """
    Sequence classification head for `ArlowConfig`, built on the full [`ArlowModel`] backbone (vision encoder
    included), matching `Gemma3ForSequenceClassification`. For a text-only backbone, use
    [`ArlowTextForSequenceClassification`] with an [`ArlowTextConfig`].
    """

    config: ArlowConfig


class ArlowTextForSequenceClassification(GenericForSequenceClassification, ArlowTextPreTrainedModel):
    """
    Text-only sequence classification head that mirrors `GenericForSequenceClassification` for `ArlowTextConfig`.
    """

    config: ArlowTextConfig


@auto_docstring
class ArlowForQuestionAnswering(GenericForQuestionAnswering, ArlowPreTrainedModel):
    """
    Question answering head for `ArlowConfig`, built on the full [`ArlowModel`] backbone (vision encoder included).
    For a text-only backbone, use [`ArlowTextForQuestionAnswering`] with an [`ArlowTextConfig`].
    """

    config: ArlowConfig

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)


class ArlowTextForQuestionAnswering(GenericForQuestionAnswering, ArlowTextPreTrainedModel):
    """
    Text-only question answering head leveraging the generic QA mixin for `ArlowTextConfig`.
    """

    config: ArlowTextConfig


@auto_docstring
class ArlowForTokenClassification(GenericForTokenClassification, ArlowPreTrainedModel):
    """
    Token classification head for `ArlowConfig`, built on the full [`ArlowModel`] backbone (vision encoder included).
    For a text-only backbone, use [`ArlowTextForTokenClassification`] with an [`ArlowTextConfig`].
    """

    config: ArlowConfig


class ArlowTextForTokenClassification(GenericForTokenClassification, ArlowTextPreTrainedModel):
    """
    Text-only token classification head for `ArlowTextConfig`.
    """

    config: ArlowTextConfig


__all__ = [
    "ArlowConfig",
    "ArlowTextConfig",
    "ArlowVisionConfig",
    "ArlowPreTrainedModel",
    "ArlowTextModel",
    "ArlowModel",
    "ArlowForCausalLM",
    "ArlowForConditionalGeneration",
    "ArlowVLVisionModel",
    "ArlowProcessor",
    "ArlowForSequenceClassification",
    "ArlowTextForSequenceClassification",
    "ArlowForQuestionAnswering",
    "ArlowTextForQuestionAnswering",
    "ArlowForTokenClassification",
    "ArlowTextForTokenClassification",
]
