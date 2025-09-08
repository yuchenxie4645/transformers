from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import is_flash_attn_available
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

if is_flash_attn_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward

from ...processing_utils import Unpack
from ...utils import (TransformersKwargs, auto_docstring, can_return_tuple,
                      logging)
from ...utils.generic import check_model_inputs
from .configuration_arlow import ArlowConfig

logger = logging.get_logger(__name__)


class ArlowRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class ArlowRotaryEmbedding(nn.Module):
    def __init__(self, config: ArlowConfig, device=None):
        super().__init__()
        # Validate rope_scaling
        if getattr(config, "rope_scaling", None) is not None and not isinstance(
            config.rope_scaling, dict
        ):
            raise ValueError("rope_scaling must be a dict if provided")
        if isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
            if self.rope_type is None:
                logger.warning(
                    "rope_scaling provided without 'rope_type'/'type'; defaulting to 'default'."
                )
                self.rope_type = "default"
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            # Even/odd interleaving: expand frequencies for interleaved dims
            cos = freqs.cos().repeat_interleave(2, dim=-1) * self.attention_scaling
            sin = freqs.sin().repeat_interleave(2, dim=-1) * self.attention_scaling

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


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class ArlowAttention(nn.Module):
    def __init__(self, config: ArlowConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or (
            self.hidden_size // self.num_heads
        )
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

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
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],  # additive mask [B, 1, S, S]
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        padding_mask_2d: Optional[torch.Tensor] = None,  # 2D bool mask for varlen path
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        q_shape = (*input_shape, self.num_heads, self.head_dim)
        kv_shape = (*input_shape, self.num_kv_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(q_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(kv_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        use_flash = (
            is_flash_attn_available()
            and self.config.use_varlen_flash
            and padding_mask_2d is not None
            and past_key_values is None
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Keep explicit batch and sequence length for safe reshapes
        bsz, q_len = hidden_states.shape[:2]

        if use_flash:
            # Use HF unified FA2 wrapper
            attn_unpad = _flash_attention_forward(
                query_states.transpose(1, 2),  # [B,H,S,D] -> [B,S,H,D]
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                attention_mask=padding_mask_2d,  # [B,S]
                query_length=q_len,
                is_causal=True,
                dropout=self.attention_dropout if self.training else 0.0,
                position_ids=None,
            )
            # attn_unpad is [B, S, H, D]; fold heads using dynamic dims to avoid any mismatch
            attn_output = attn_unpad.contiguous().view(bsz, q_len, -1)
            attn_weights = None
        else:
            # Fallback: SDPA with causal=True and padding via additive mask
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float32)

            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                # When providing an additive mask, do not enable causal path.
                is_causal=attention_mask is None,
            )
            # Ensure we restore [B, S, H*D] using dynamic dims; avoid flattening [S] into channels
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_weights = None

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
        self.self_attn = ArlowAttention(config=config, layer_idx=layer_idx)
        self.mlp = ArlowMLP(config)
        self.input_layernorm = ArlowRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ArlowRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
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
        padding_mask_2d: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            cache_position=cache_position,
            padding_mask_2d=padding_mask_2d,
            **kwargs,
        )
        hidden_states = residual + self.resid_dropout(attn_out)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = residual + self.resid_dropout(mlp_out)
        return hidden_states


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

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                ArlowDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
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

    @check_model_inputs
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
        # Raise when both are provided or both are missing
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Create additive causal mask for SDPA fallback
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Preserve original 2D padding mask for varlen path
        padding_mask_2d = attention_mask

        # Gradient checkpointing support
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting use_cache=False."
                )
            use_cache = False

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if (
                self.gradient_checkpointing
                and self.training
                and past_key_values is None
            ):

                def create_custom_forward(module):
                    def custom_forward(hidden_states):
                        return module(
                            hidden_states,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_values=past_key_values,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            padding_mask_2d=padding_mask_2d,
                            **kwargs,
                        )

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    padding_mask_2d=padding_mask_2d,
                    **kwargs,
                )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
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
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Align labels to sliced logits window before shifting
            labels_window = labels[:, slice_indices]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_window[..., 1:].contiguous()
            ignore_index = (
                self.config.pad_token_id
                if self.config.pad_token_id is not None
                else -100
            )
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
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            **kwargs,
        }

    # reorder cache (beam)
    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is not None and hasattr(past_key_values, "reorder_cache"):
            past_key_values.reorder_cache(beam_idx)
        return past_key_values


__all__ = ["ArlowPreTrainedModel", "ArlowModel", "ArlowForCausalLM"]
