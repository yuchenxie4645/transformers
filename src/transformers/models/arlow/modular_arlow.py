from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from packaging import version
from torch import nn
from torch.nn import LayerNorm

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PretrainedConfig, layer_type_validation
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import is_flash_attn_available
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, ModelOutput
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update, rope_config_validation
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.import_utils import get_torch_version
from ...video_utils import VideoInput


if is_flash_attn_available():
    pass

logger = logging.get_logger(__name__)

class ArlowVisionConfig(PretrainedConfig):
    r"""
    Configuration for the vision transformer component of Arlow multimodal models.

    Args:
        depth (`int`, *optional*, defaults to 32):
            Number of hidden layers in the vision transformer.
        embed_dim (`int`, *optional*, defaults to 1280):
            Dimensionality of the vision encoder embeddings.
        hidden_size (`int`, *optional*, defaults to 3584):
            Dimensionality after vision projection to match text model.
        hidden_act (`str`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function in the vision encoder.
        mlp_ratio (`int`, *optional*, defaults to 4):
            Ratio of mlp hidden dim to embedding dim.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads in the vision transformer.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input image channels.
        patch_size (`int`, *optional*, defaults to 14):
            Size of image patches.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            Spatial merge factor for patch merging.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            Temporal patch size for video inputs.
        use_deformable_attention (`bool`, *optional*, defaults to False):
            Whether to use deformable attention for high-resolution regions.
        use_progressive_patches (`bool`, *optional*, defaults to False):
            Whether to use progressive patch embeddings for multi-scale.
        token_pruning_ratio (`float`, *optional*, defaults to 0.0):
            Ratio of tokens to prune per region (0.0 means no pruning).
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
    """

    model_type = "arlow_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth: int = 32,
        embed_dim: int = 1280,
        hidden_size: int = 3584,
        hidden_act: str = "gelu_pytorch_tanh",
        mlp_ratio: int = 4,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        use_deformable_attention: bool = False,
        use_progressive_patches: bool = False,
        token_pruning_ratio: float = 0.0,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.use_deformable_attention = use_deformable_attention
        self.use_progressive_patches = use_progressive_patches
        self.token_pruning_ratio = token_pruning_ratio
        self.initializer_range = initializer_range


class ArlowConfig(PretrainedConfig):
    r"""
    Configuration class for Arlow models (text-only and multimodal).

    Instantiating with defaults yields configuration similar to Arlow-Base
    [yuchenxie/ArlowGPT-Base](https://huggingface.co/yuchenxie/ArlowGPT-Base).

    Configuration objects inherit from [`PretrainedConfig`] and control model outputs.

    Args:
        vocab_size (`int`, *optional*, defaults to 131072):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 2304):
            Dimension of hidden representations.
        intermediate_size (`int`, *optional*, defaults to 9216):
            Dimension of MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 24):
            Number of attention heads per layer.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            Number of key_value heads for Grouped Query Attention.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Non-linear activation function.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            Maximum sequence length.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon for RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return past key/values for faster decoding.
        pad_token_id (`int`, *optional*):
            Padding token ID.
        bos_token_id (`int`, *optional*):
            Beginning of sequence token ID.
        eos_token_id (`int`, *optional*):
            End of sequence token ID.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input/output embeddings.
        rope_theta (`float`, *optional*, defaults to 100000.0):
            Base period of RoPE embeddings.
        rope_parameters (`Dict`, *optional*):
            RoPE scaling configuration. For backwards compatibility, `rope_scaling` is also accepted.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention projections.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio for attention probabilities.
        resid_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio for residual connections.
        mlp_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio for MLP layers.
        head_dim (`int`, *optional*):
            Attention head dimension. Defaults to hidden_size // num_attention_heads.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window size.
        max_window_layers (`int`, *optional*, defaults to 28):
            Number of layers using full attention before switching to sliding window.
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        vision_config (`Union[PretrainedConfig, dict]`, *optional*):
            Vision backbone configuration.
        mm_tokens_per_image (`int`, *optional*, defaults to 256):
            Number of tokens per image after vision projection.
        mm_tokens_per_video (`int`, *optional*, defaults to 128):
            Number of tokens per video after temporal resampling.
        video_max_frames (`int`, *optional*, defaults to 64):
            Maximum number of video frames to extract.
        video_sample_strategy (`str`, *optional*, defaults to "uniform"):
            Video frame sampling strategy: "uniform", "motion_adaptive", or "fps_based".
        dynamic_resolution (`bool`, *optional*, defaults to True):
            Whether to use dynamic resolution for images.
        pan_and_scan (`bool`, *optional*, defaults to False):
            Whether to use pan-and-scan to keep token budgets constant.
        timestamp_alignment (`bool`, *optional*, defaults to False):
            Whether to enable timestamp supervision for video grounding.
        use_gated_cross_attention (`bool`, *optional*, defaults to False):
            Whether to use gated cross-attention in upper layers.
        gated_cross_attention_start_layer (`int`, *optional*):
            Layer index to start gated cross-attention (if enabled).
        image_token_id (`int`, *optional*):
            Token ID for image placeholders.
        video_token_id (`int`, *optional*):
            Token ID for video placeholders.
        vision_start_token_id (`int`, *optional*):
            Token ID marking start of vision input.
        vision_end_token_id (`int`, *optional*):
            Token ID marking end of vision input.
        frame_separator_token_id (`int`, *optional*):
            Token ID for separating video frames.
        mrope_sections (`list`, *optional*):
            M-ROPE sections for [temporal, height, width] dimensions.
            Defaults to split based on head_dim.
        mrope_learnable_phases (`bool`, *optional*, defaults to False):
            Whether to use learnable phase shifts in M-ROPE.
        debug_vision (`bool`, *optional*, defaults to False):
            Enable debug logging for vision forward passes.
        debug_mrope (`bool`, *optional*, defaults to False):
            Enable debug logging for M-ROPE operations.
        debug_attention (`bool`, *optional*, defaults to False):
            Enable debug logging for attention patterns.
        debug_cache (`bool`, *optional*, defaults to False):
            Enable debug logging for KV cache operations.
    """

    model_type = "arlow"
    sub_configs = {"vision_config": ArlowVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=32,
        num_attention_heads=24,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=False,
        rope_theta=100000.0,
        rope_parameters=None,
        rope_scaling=None,  # Deprecated, use rope_parameters
        attention_bias=False,
        attention_dropout=0.0,
        resid_dropout=0.0,
        mlp_dropout=0.0,
        head_dim=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        # Multimodal parameters
        vision_config=None,
        mm_tokens_per_image=256,
        mm_tokens_per_video=128,
        video_max_frames=64,
        video_sample_strategy="uniform",
        dynamic_resolution=True,
        pan_and_scan=False,
        timestamp_alignment=False,
        use_gated_cross_attention=False,
        gated_cross_attention_start_layer=None,
        image_token_id=None,
        video_token_id=None,
        vision_start_token_id=None,
        vision_end_token_id=None,
        frame_separator_token_id=None,
        mrope_sections=None,
        mrope_learnable_phases=False,
        # Debug parameters (will be tested and removed)
        debug_vision=False,
        debug_mrope=False,
        debug_attention=False,
        debug_cache=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_parameters = rope_scaling or rope_parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.resid_dropout = resid_dropout
        self.mlp_dropout = mlp_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads

        # Multimodal configuration
        if isinstance(vision_config, dict):
            self.vision_config = ArlowVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = ArlowVisionConfig()
        else:
            self.vision_config = vision_config

        self.mm_tokens_per_image = mm_tokens_per_image
        self.mm_tokens_per_video = mm_tokens_per_video
        self.video_max_frames = video_max_frames
        self.video_sample_strategy = video_sample_strategy
        self.dynamic_resolution = dynamic_resolution
        self.pan_and_scan = pan_and_scan
        self.timestamp_alignment = timestamp_alignment
        self.use_gated_cross_attention = use_gated_cross_attention
        self.gated_cross_attention_start_layer = gated_cross_attention_start_layer
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.frame_separator_token_id = frame_separator_token_id

        # M-ROPE configuration: default sections based on head_dim
        if mrope_sections is None:
            # Split head_dim into temporal, height, width sections
            # Default: temporal=16, rest split between h/w
            remaining = self.head_dim - 16
            self.mrope_sections = [16, remaining // 2, remaining - remaining // 2]
        else:
            self.mrope_sections = mrope_sections

        self.mrope_learnable_phases = mrope_learnable_phases

        # Validate mrope_sections sum equals head_dim
        if sum(self.mrope_sections) != self.head_dim:
            raise ValueError(
                f"Sum of mrope_sections {self.mrope_sections} (={sum(self.mrope_sections)}) "
                f"must equal head_dim ({self.head_dim})"
            )

        # Debug flags (to be tested and removed)
        self.debug_vision = debug_vision
        self.debug_mrope = debug_mrope
        self.debug_attention = debug_attention
        self.debug_cache = debug_cache

        # Validate rope parameters (ignore M-ROPE specific keys since we use custom M-ROPE)
        if self.rope_parameters is not None and "type" in self.rope_parameters:
            self.rope_parameters["rope_type"] = self.rope_parameters["type"]
        rope_config_validation(self, ignore_keys={"mrope_sections", "mrope_learnable_phases"})

        # Layer types configuration
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Arlow multimodal model outputs, with hidden states and M-ROPE deltas.
    """
)
class ArlowMultimodalModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope for M-ROPE.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Arlow multimodal causal language model outputs.
    """
)
class ArlowMultimodalCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope for M-ROPE.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


if version.parse(get_torch_version()) >= version.parse("2.3.0"):

    class ArlowRMSNorm(nn.RMSNorm):
        def __init__(self, hidden_size: int, eps: float = 1e-6):
            super().__init__(normalized_shape=hidden_size, eps=eps, elementwise_affine=True)

else:

    @use_kernel_forward_from_hub("RMSNorm")
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
            return self.weight * hidden_states.to(input_dtype)

        def extra_repr(self):
            return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class ArlowRotaryEmbedding(nn.Module):
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
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[ArlowConfig] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
        **rope_kwargs,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies for default RoPE (no scaling).
        """
        if config is not None:
            # Access rope_theta from rope_parameters if available, otherwise fallback to config attribute
            if hasattr(config, "rope_parameters") and config.rope_parameters:
                base = config.rope_parameters.get("rope_theta", config.rope_theta)
            else:
                base = config.rope_theta
            head_dim = config.head_dim
            if head_dim is None:
                head_dim = config.hidden_size // config.num_attention_heads
        else:
            base = rope_kwargs.get("base", 10000.0)
            head_dim = rope_kwargs.get("dim")

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(device) / head_dim)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # Concatenate for standard RoPE (matches Llama implementation)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Interleave even/odd features: (-x_odd, x_even)
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1, debug=False):
    """
    Applies Multimodal Rotary Position Embedding (M-ROPE) to query and key tensors.
    
    M-ROPE extends standard RoPE for multimodal inputs by splitting the head dimension into
    sections for temporal, height, and width dimensions separately.
    
    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim)
        k: Key tensor of shape (batch, heads, seq_len, head_dim)
        cos: Cosine positional embeddings of shape (3, batch, seq_len, section_dim)
        sin: Sine positional embeddings of shape (3, batch, seq_len, section_dim)
        mrope_section: List of 3 integers [t_dim, h_dim, w_dim] summing to head_dim
        unsqueeze_dim: Dimension to unsqueeze for broadcasting (default: 1)
        debug: Whether to enable debug logging (default: False)
    
    Returns:
        Tuple of (q_embed, k_embed) with M-ROPE applied
    """
    print(f"[DEBUG M-ROPE] Input shapes: q={q.shape}, k={k.shape}, cos={cos.shape}, sin={sin.shape}")
    print(f"[DEBUG M-ROPE] Sections: {mrope_section}")
    
    # mrope_section is [t, h, w] dimensions, we need to double for cos/sin interleaving
    mrope_section = [s * 2 for s in mrope_section]
    
    # Split cos/sin into chunks and reassemble by interleaving temporal/height/width
    # cos/sin shape: (3, batch, seq_len, section_dim*2)
    cos_chunks = [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))]
    sin_chunks = [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))]
    
    cos = torch.cat(cos_chunks, dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat(sin_chunks, dim=-1).unsqueeze(unsqueeze_dim)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    print(f"[DEBUG M-ROPE] Output shapes: q_embed={q_embed.shape}, k_embed={k_embed.shape}")
    
    return q_embed, k_embed


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


class VisionRotaryEmbedding(nn.Module):
    """Rotary position embeddings for vision transformer."""

    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class ArlowMultimodalRotaryEmbedding(nn.Module):
    """
    M-ROPE with optional learnable phase shifts and damping per axis.
    Extends ArlowRotaryEmbedding to handle (temporal, height, width) position IDs.
    """

    def __init__(self, config: ArlowConfig, device=None):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.mrope_sections = config.mrope_sections
        
        # Validate sections
        if sum(self.mrope_sections) != self.head_dim:
            raise ValueError(
                f"Sum of mrope_sections {self.mrope_sections} must equal head_dim {self.head_dim}"
            )
        
        # Compute inverse frequencies for each section (t, h, w)
        self.rope_type = "default"
        if hasattr(config, "rope_parameters") and config.rope_parameters:
            self.rope_type = config.rope_parameters.get("rope_type", "default")
        
        rope_theta = config.rope_theta
        
        # Create separate inv_freq for each dimension
        inv_freqs = []
        for section_dim in self.mrope_sections:
            dim = section_dim
            inv_freq = 1.0 / (
                rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
            )
            inv_freqs.append(inv_freq)
        
        # Register as buffers
        for i, inv_freq in enumerate(inv_freqs):
            self.register_buffer(f"inv_freq_{i}", inv_freq, persistent=False)
        
        # Learnable phase shifts and damping if enabled
        if config.mrope_learnable_phases:
            self.phase_shift = nn.Parameter(torch.zeros(3))  # [t, h, w]
            self.damping = nn.Parameter(torch.ones(3))  # [t, h, w]
        else:
            self.phase_shift = None
            self.damping = None
        
        self.attention_scaling = 1.0
        self.debug = config.debug_mrope
        
        print(f"[DEBUG M-ROPE Init] sections={self.mrope_sections}, head_dim={self.head_dim}")
        print(f"[DEBUG M-ROPE Init] learnable_phases={config.mrope_learnable_phases}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor):
        """
        Args:
            x: Input tensor (used for device/dtype)
            position_ids: Shape (3, batch, seq_len) containing [t, h, w] position indices
        
        Returns:
            Tuple of (cos, sin) with shape (3, batch, seq_len, section_dim)
        """
        print(f"[DEBUG M-ROPE Forward] x.shape={x.shape}, position_ids.shape={position_ids.shape}")
        
        # position_ids shape: (3, batch, seq_len) for [temporal, height, width]
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        
        cos_list = []
        sin_list = []
        
        with torch.autocast(device_type=device_type, enabled=False):
            for i in range(3):  # temporal, height, width
                inv_freq = getattr(self, f"inv_freq_{i}")
                # inv_freq shape: (section_dim/2,)
                # position_ids[i] shape: (batch, seq_len)
                
                inv_freq_expanded = inv_freq[None, None, :, None].float().expand(1, position_ids.shape[1], -1, 1)
                position_ids_expanded = position_ids[i:i+1, :, None, :].float()  # (1, batch, 1, seq_len)
                
                freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)  # (1, batch, seq_len, section_dim/2)
                freqs = freqs.repeat_interleave(2, dim=-1)  # (1, batch, seq_len, section_dim)
                
                # Apply learnable phase shifts and damping if enabled
                if self.phase_shift is not None and self.damping is not None:
                    phase = self.phase_shift[i]
                    damp = self.damping[i]
                    freqs = freqs + phase
                    freqs = freqs * damp
                
                cos_i = (freqs.cos() * self.attention_scaling).squeeze(0)  # (batch, seq_len, section_dim)
                sin_i = (freqs.sin() * self.attention_scaling).squeeze(0)  # (batch, seq_len, section_dim)
                
                cos_list.append(cos_i)
                sin_list.append(sin_i)
        
        # Stack to get shape (3, batch, seq_len, section_dim)
        cos = torch.stack(cos_list, dim=0).to(dtype=x.dtype)
        sin = torch.stack(sin_list, dim=0).to(dtype=x.dtype)
        
        print(f"[DEBUG M-ROPE Forward] Output cos.shape={cos.shape}, sin.shape={sin.shape}")
        
        return cos, sin


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        q_shape = (*input_shape, self.num_heads, self.head_dim)
        kv_shape = (*input_shape, self.num_kv_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(q_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(kv_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # For SDPA, during incremental decoding (query_len == 1), we should pass None as the mask
        # to allow SDPA to use is_causal=False, avoiding shape issues with 4D masks
        if (
            getattr(self.config, "_attn_implementation", "eager") == "sdpa"
            and attention_mask is not None
            and query_states.shape[2] == 1  # q_len == 1 (incremental decoding)
        ):
            attention_mask = None

        # Dispatch to proper attention implementation
        attention_interface = eager_attention_forward
        if getattr(self.config, "_attn_implementation", "eager") != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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


class ArlowMLP(nn.Module):
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
        return self.dropout(x)


class ArlowDecoderLayer(nn.Module):
    def __init__(self, config: ArlowConfig, layer_idx: int):
        super().__init__()
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = ArlowAttention(config=config, layer_idx=layer_idx)
        self.mlp = ArlowMLP(config)
        self.input_layernorm = ArlowRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ArlowRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + self.resid_dropout(attn_out)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = residual + self.resid_dropout(mlp_out)

        output_attentions = kwargs.get("output_attentions", False)
        if output_attentions:
            return (hidden_states, attn_weights)

        return (hidden_states,)


class PatchEmbed(nn.Module):
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
            hidden_states: (batch * num_tiles, channels, temporal, height, width)
        Returns:
            embeddings: (batch * num_tiles, embed_dim, T, H, W) -> flattened to (total_tokens, embed_dim)
        """
        hidden_states = self.proj(hidden_states)
        # Flatten spatial dimensions: (B, C, T, H, W) -> (B, T*H*W, C)
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(batch_size, self.embed_dim, -1).transpose(1, 2)
        return hidden_states


class PatchMerger(nn.Module):
    """Merge vision patches and project to text model dimension."""
    
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
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
        return self.mlp(x)


class ArlowVisionAttention(nn.Module):
    """Vision self-attention with RoPE."""
    
    def __init__(self, config: ArlowVisionConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=True)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (total_tokens, embed_dim)
            position_embeddings: (cos, sin) for RoPE
        """
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1] if hidden_states.dim() > 2 else hidden_states.shape[0]
        
        # Compute Q, K, V
        qkv = self.qkv(hidden_states).reshape(-1, 3, self.num_heads, self.head_dim)
        query_states, key_states, value_states = qkv.unbind(1)
        
        # Apply RoPE if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        
        # Reshape for attention: (seq, heads, head_dim) -> (1, seq, heads, head_dim) -> (1, heads, seq, head_dim)
        query_states = query_states.unsqueeze(0).transpose(1, 2)
        key_states = key_states.unsqueeze(0).transpose(1, 2)
        value_states = value_states.unsqueeze(0).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).reshape(-1, self.embed_dim)
        attn_output = self.proj(attn_output)
        
        return attn_output


class ArlowVisionBlock(GradientCheckpointingLayer):
    """Vision transformer block with attention and MLP."""
    
    def __init__(self, config: ArlowVisionConfig):
        super().__init__()
        self.norm1 = LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = LayerNorm(config.embed_dim, eps=1e-6)
        self.attn = ArlowVisionAttention(config)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)
        # Inline MLP instead of separate class
        self.fc1 = nn.Linear(config.embed_dim, mlp_hidden_dim)
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(mlp_hidden_dim, config.embed_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Self-attention with residual
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            position_embeddings=position_embeddings,
        )
        # MLP with residual (inline)
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.fc2(self.act(self.fc1(hidden_states)))
        hidden_states = residual + hidden_states
        return hidden_states


class ArlowVisionTransformer(nn.Module):
    """Complete vision encoder for Arlow multimodal models."""
    
    def __init__(self, config: ArlowVisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )
        
        # Rotary position embeddings for vision
        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ArlowVisionBlock(config) for _ in range(config.depth)
        ])
        
        # Patch merger to project to text dimension
        self.merger = PatchMerger(
            dim=config.hidden_size,
            context_dim=config.embed_dim,
            spatial_merge_size=config.spatial_merge_size,
        )
        
        self.gradient_checkpointing = False
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch, channels, temporal, height, width)
            grid_thw: (num_images/videos, 3) containing [temporal, height, width] dimensions
        Returns:
            vision_embeddings: (total_tokens, hidden_size)
        """
        # Patch embedding
        hidden_states = self.patch_embed(pixel_values)  # (batch, num_patches, embed_dim)
        
        # Get RoPE embeddings
        # Simplified - in practice you'd compute based on grid_thw
        seq_len = hidden_states.shape[1]
        rotary_pos_emb = self.rotary_pos_emb(seq_len)
        # Convert to cos/sin
        cos = rotary_pos_emb.cos()
        sin = rotary_pos_emb.sin()
        position_embeddings = (cos, sin)
        
        # Apply transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = block._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    position_embeddings,
                )
            else:
                hidden_states = block(hidden_states, position_embeddings)
        
        # Merge patches and project to text dimension
        hidden_states = hidden_states.reshape(-1, self.config.embed_dim)
        vision_embeddings = self.merger(hidden_states)
        
        return vision_embeddings


class ArlowPreTrainedModel(PreTrainedModel):
    config_class = ArlowConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ArlowDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values", "rotary_emb.inv_freq"]
    is_loaded_in_8bit = False
    is_loaded_in_4bit = False
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_record_outputs = {}

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, ArlowRMSNorm):
            nn.init.ones_(module.weight)


@auto_docstring
class ArlowModel(ArlowPreTrainedModel):
    def __init__(self, config: ArlowConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.has_sliding_layers = "sliding_attention" in config.layer_types

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ArlowDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = ArlowRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = ArlowRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    # emb getters
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        # Handle input_ids vs inputs_embeds
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        # When both are provided, prefer inputs_embeds (this is expected during generation)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # Recompute cache_position if it doesn't match the input length
        if cache_position is not None and cache_position.shape[0] != inputs_embeds.shape[1]:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        elif cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Collect hidden states and attentions for output if requested
        output_hidden_states = kwargs.get("output_hidden_states", self.config.output_hidden_states)
        output_attentions = kwargs.get("output_attentions", self.config.output_attentions)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Ensure output flags are in kwargs for decoder layers
        kwargs["output_hidden_states"] = output_hidden_states
        kwargs["output_attentions"] = output_attentions

        # Gradient checkpointing support
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting use_cache=False."
                )
            use_cache = False

        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

            if all_self_attns is not None and isinstance(layer_outputs, tuple) and len(layer_outputs) > 1:
                all_self_attns += (layer_outputs[1],)

            if all_hidden_states is not None:
                all_hidden_states += (hidden_states,)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@auto_docstring
class ArlowForCausalLM(ArlowPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: ArlowConfig):
        super().__init__(config)
        self.model = ArlowModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Align labels to sliced logits window before shifting
            labels_window = labels[:, slice_indices]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_window[..., 1:].contiguous()
            ignore_index = self.config.pad_token_id if self.config.pad_token_id is not None else -100
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=ignore_index,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # Only use inputs_embeds for the first forward pass (when cache is empty)
        # After that, we use input_ids for subsequent generation steps
        # Check if cache has tokens, not just if cache object exists
        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if cache_length > 0:
            # We have cached tokens, so we're in a subsequent generation step
            # Special case: if input_ids is empty but inputs_embeds is provided,
            # we need to slice inputs_embeds to only the new tokens not in cache
            if input_ids is not None and input_ids.shape[1] == 0 and inputs_embeds is not None:
                # Slice inputs_embeds to only process tokens not yet in cache
                if inputs_embeds.shape[1] > cache_length:
                    inputs_embeds = inputs_embeds[:, cache_length:]
                    input_ids = None
            else:
                # For assisted/speculative decoding, we may need to process multiple new tokens
                # Use cache_position to determine how many tokens to keep
                if cache_position is not None and len(cache_position) > 1:
                    # Assisted generation: keep multiple tokens
                    input_ids = input_ids[:, cache_position[0] :]
                else:
                    # Normal generation: just the last token
                    input_ids = input_ids[:, -1:]
                inputs_embeds = None
        elif inputs_embeds is not None:
            # First step with inputs_embeds (cache is empty or doesn't exist) - don't pass input_ids
            input_ids = None

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "cache_position": cache_position,
            **kwargs,
        }

    # reorder cache (beam)
    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is not None and hasattr(past_key_values, "reorder_cache"):
            past_key_values.reorder_cache(beam_idx)
        return past_key_values


@auto_docstring
class ArlowMultimodalModel(ArlowPreTrainedModel):
    """
    Multimodal model combining vision encoder and text decoder.
    This model handles both image and video inputs alongside text.
    """
    
    def __init__(self, config: ArlowConfig):
        super().__init__(config)
        
        # Vision encoder
        self.visual = ArlowVisionTransformer(config.vision_config)
        
        # Text model (reuse the existing text-only model)
        self.model = ArlowModel(config)
        
        # Multimodal RoPE for combined vision-text sequences
        self.multimodal_rotary_emb = ArlowMultimodalRotaryEmbedding(config)
        
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
    
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """Extract image features from vision encoder."""
        return self.visual(pixel_values, image_grid_thw)
    
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """Extract video features from vision encoder (same as images with temporal dim)."""
        return self.visual(pixel_values_videos, video_grid_thw)
    
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ArlowMultimodalModelOutputWithPast:
        """
        Forward pass combining vision and text modalities.
        
        Args:
            input_ids: Text token IDs with image/video placeholder tokens
            pixel_values: Image tensor (batch, channels, height, width) 
            pixel_values_videos: Video tensor (batch, channels, frames, height, width)
            image_grid_thw: Image grid dimensions (num_images, 3) as [temporal=1, height, width]
            video_grid_thw: Video grid dimensions (num_videos, 3) as [temporal, height, width]
        """
        # Get text embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        # Process vision inputs if provided
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            # Merge image embeddings into text sequence at placeholder positions
            inputs_embeds = self._merge_vision_embeds(
                inputs_embeds, image_embeds, input_ids, self.config.image_token_id
            )
        
        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            # Merge video embeddings into text sequence at placeholder positions
            inputs_embeds = self._merge_vision_embeds(
                inputs_embeds, video_embeds, input_ids, self.config.video_token_id
            )
        
        # Forward through text model
        outputs = self.model(
            input_ids=None,  # We're using inputs_embeds instead
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs,
        )
        
        return ArlowMultimodalModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=None,  # TODO: Compute rope deltas for M-ROPE
        )
    
    def _merge_vision_embeds(
        self,
        text_embeds: torch.Tensor,
        vision_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        vision_token_id: int,
    ) -> torch.Tensor:
        """
        Merge vision embeddings into text sequence at placeholder token positions.
        
        Args:
            text_embeds: (batch, seq_len, hidden_size)
            vision_embeds: (total_vision_tokens, hidden_size)
            input_ids: (batch, seq_len) with vision_token_id placeholders
            vision_token_id: ID of vision placeholder token
        
        Returns:
            merged_embeds: (batch, seq_len, hidden_size) with vision embeds inserted
        """
        batch_size, seq_len, hidden_size = text_embeds.shape
        
        # Find positions of vision tokens
        vision_mask = (input_ids == vision_token_id)
        
        # Replace text embeddings at vision token positions with vision embeddings
        vision_idx = 0
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                if vision_mask[batch_idx, seq_idx]:
                    if vision_idx < vision_embeds.shape[0]:
                        text_embeds[batch_idx, seq_idx] = vision_embeds[vision_idx]
                        vision_idx += 1
        
        return text_embeds


@auto_docstring
class ArlowForConditionalGeneration(ArlowPreTrainedModel, GenerationMixin):
    """
    Arlow model for conditional generation (multimodal: text + images/videos).
    This is the main model for VLM tasks.
    """
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config: ArlowConfig):
        super().__init__(config)
        self.model = ArlowMultimodalModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
    
    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)
    
    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)
    
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ArlowMultimodalCausalLMOutputWithPast:
        """
        Forward pass for multimodal conditional generation.
        
        Args:
            image_grid_thw (`torch.LongTensor`, *optional*):
                Image grid dimensions of shape `(num_images, 3)` as `[temporal=1, height, width]`.
            video_grid_thw (`torch.LongTensor`, *optional*):
                Video grid dimensions of shape `(num_videos, 3)` as `[temporal, height, width]`.

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
        outputs: ArlowMultimodalModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        loss = None
        if labels is not None:
            labels_window = labels[:, slice_indices]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_window[..., 1:].contiguous()
            ignore_index = self.config.pad_token_id if self.config.pad_token_id is not None else -100
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=ignore_index,
            )
        
        return ArlowMultimodalCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """Prepare inputs for generation, handling both text-only and multimodal inputs."""
        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        
        # Only process vision on first forward pass
        if cache_length > 0:
            pixel_values = None
            pixel_values_videos = None
            image_grid_thw = None
            video_grid_thw = None
            
            if input_ids is not None and input_ids.shape[1] == 0 and inputs_embeds is not None:
                if inputs_embeds.shape[1] > cache_length:
                    inputs_embeds = inputs_embeds[:, cache_length:]
                    input_ids = None
            else:
                if cache_position is not None and len(cache_position) > 1:
                    input_ids = input_ids[:, cache_position[0]:]
                else:
                    input_ids = input_ids[:, -1:]
                inputs_embeds = None
        elif inputs_embeds is not None:
            input_ids = None
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "cache_position": cache_position,
            **kwargs,
        }
    
    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is not None and hasattr(past_key_values, "reorder_cache"):
            past_key_values.reorder_cache(beam_idx)
        return past_key_values


class ArlowProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "images_kwargs": {},
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }


class ArlowProcessor(ProcessorMixin):
    r"""
    Constructs an Arlow processor which wraps an image processor, a tokenizer, and a video processor into a single
    processor. Follows Qwen3VL's strategy for placeholder expansion and timestamp prompts.

    Args:
        image_processor: Required image processor.
        tokenizer: Required tokenizer (`ArlowTokenizer` or `ArlowTokenizerFast`).
        video_processor: Required video processor for video support.
        chat_template: Optional chat template string.
    """

    attributes = ["image_processor", "tokenizer", "video_processor"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = ("ArlowTokenizer", "ArlowTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
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

    def __call__(
        self,
        images: ImageInput | None = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] | None = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[ArlowProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            ArlowProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = {}
        video_inputs = {}
        image_grid_thw = None
        video_grid_thw = None

        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs.get("image_grid_thw")

        if videos is not None:
            video_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = video_inputs.get("video_grid_thw")
            # preserve metadata for timestamp prompts if provided
            if "return_metadata" not in kwargs and "video_metadata" in video_inputs:
                video_metadata = video_inputs.pop("video_metadata")
            else:
                video_metadata = video_inputs.get("video_metadata")

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # will edit in-place

        # Expand image placeholders by computed token budget
        if image_grid_thw is not None:
            merge_len = self.image_processor.merge_size**2 if hasattr(self.image_processor, "merge_size") else 1
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_len
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        # Expand video placeholders into per-frame segments with optional timestamps
        if video_grid_thw is not None:
            merge_len = self.video_processor.merge_size**2 if hasattr(self.video_processor, "merge_size") else 1
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    # build per-frame blocks
                    video_placeholder = ""
                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_len

                    # compute timestamps if metadata exists, otherwise just omit
                    curr_timestamps = None
                    if video_metadata is not None:
                        metadata = video_metadata[index]
                        if getattr(metadata, "fps", None) is None:
                            logger.warning_once(
                                "Arlow requires video fps to build timestamp prompts; defaulting to fps=24."
                            )
                            metadata.fps = 24 if metadata.fps is None else metadata.fps
                        curr_timestamps = self._calculate_timestamps(
                            metadata.frames_indices, metadata.fps, self.video_processor.merge_size
                        )

                    for frame_idx in range(video_grid_thw[index][0]):
                        if curr_timestamps is not None:
                            curr_time = curr_timestamps[frame_idx]
                            video_placeholder += f"<{curr_time:.1f} seconds>"
                        video_placeholder += (
                            self.vision_start_token + "<|placeholder|>" * frame_seqlen + self.vision_end_token
                        )

                    compound = f"{self.vision_start_token}{self.video_token}{self.vision_end_token}"
                    if compound in text[i]:
                        text[i] = text[i].replace(compound, video_placeholder, 1)
                    else:
                        text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    index += 1

                text[i] = text[i].replace("<|placeholder|>", self.video_token)

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
            images_kwargs = ArlowProcessorKwargs._defaults.get("images_kwargs", {}).copy()
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or getattr(self.image_processor, "merge_size", 1)
            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        if video_sizes is not None and self.video_processor is not None:
            videos_kwargs = ArlowProcessorKwargs._defaults.get("videos_kwargs", {}).copy()
            videos_kwargs.update(kwargs)
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

    def _calculate_timestamps(self, indices: Union[list[int], np.ndarray], video_fps: float, merge_size: int = 2):
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        timestamps = [idx / video_fps for idx in indices]
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps


class ArlowForSequenceClassification(GenericForSequenceClassification, ArlowPreTrainedModel): ...


class ArlowForQuestionAnswering(GenericForQuestionAnswering, ArlowPreTrainedModel): ...


class ArlowForTokenClassification(GenericForTokenClassification, ArlowPreTrainedModel): ...


__all__ = [
    "ArlowConfig",
    "ArlowVisionConfig",
    "ArlowPreTrainedModel",
    "ArlowModel",
    "ArlowForCausalLM",
    "ArlowMultimodalModel",
    "ArlowForConditionalGeneration",
    "ArlowVisionTransformer",
    "ArlowProcessor",
    "ArlowForSequenceClassification",
    "ArlowForQuestionAnswering",
    "ArlowForTokenClassification",
]
