import math
from typing import Any, Optional

import torch
import torch.nn as nn
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func

from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_arlow import ArlowConfig


logger = logging.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# RoPE helpers
# ──────────────────────────────────────────────────────────────────────────
def build_rope_cache(
    max_seq_len: int,
    head_dim: int,
    base: float = 10_000.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
):
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=dtype, device=device) / head_dim))
    seq = torch.arange(max_seq_len, dtype=dtype, device=device)[:, None]
    freqs = seq * theta[None, :]  # [S, D/2]
    sin, cos = freqs.sin(), freqs.cos()  # [S, D/2]
    sin = sin[None, :, None, :]  # [1, S, 1, D/2]
    cos = cos[None, :, None, :]
    return sin, cos


def apply_rotary(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    # x : [B, S, H, D]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    x[..., 0::2] = x1 * cos - x2 * sin
    x[..., 1::2] = x1 * sin + x2 * cos
    return x


# ──────────────────────────────────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────────────────────────────────
class ArlowRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * scale) * self.weight


# ──────────────────────────────────────────────────────────────────────────
# Grouped‑Query Flash‑Attention block
# ──────────────────────────────────────────────────────────────────────────
class ArlowGroupedQueryAttention(nn.Module):
    def __init__(self, config: ArlowConfig, is_cross_attn: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads if not is_cross_attn else config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout_p = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        # RoPE cache
        self.register_buffer("rope_sin", None, persistent=False)
        self.register_buffer("rope_cos", None, persistent=False)
        self.rope_theta = config.rope_theta
        self.max_pos = config.max_position_embeddings

        w = self.out_proj.weight
        if not getattr(w, "is_meta", False):
            with torch.no_grad():
                w.mul_(1 / math.sqrt(2.0))

    # internal ----------------------------------------------------------------
    def _maybe_build_rope(self, seq_len: int, dtype: torch.dtype, device):
        if (
            self.rope_sin is None
            or self.rope_sin.size(1) < seq_len
            or self.rope_sin.dtype != dtype
            or self.rope_sin.device != device
        ):
            sin, cos = build_rope_cache(self.max_pos, self.head_dim, self.rope_theta, dtype, device)
            self.rope_sin, self.rope_cos = sin, cos

    # forward -----------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, S_q, D]
        attention_mask: Optional[torch.Tensor] = None,  # [B, S_q]
        encoder_hidden_states: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,  # ignored in varlen path
    ) -> torch.Tensor:
        use_flash = hidden_states.dtype in [torch.bfloat16, torch.float16]

        bsz, seqlen_q, _ = hidden_states.size()
        # pick KV source
        kv_input = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        seqlen_kv = kv_input.size(1)

        # 1) Q/K/V projections
        q = self.q_proj(hidden_states).view(bsz, seqlen_q, self.num_heads, self.head_dim)
        k = self.k_proj(kv_input).view(bsz, seqlen_kv, self.num_kv_heads, self.head_dim)
        v = self.v_proj(kv_input).view(bsz, seqlen_kv, self.num_kv_heads, self.head_dim)

        # 2) RoPE
        self._maybe_build_rope(max(seqlen_q, seqlen_kv), q.dtype, q.device)
        q = apply_rotary(q, self.rope_sin[:, :seqlen_q], self.rope_cos[:, :seqlen_q])
        k = apply_rotary(k, self.rope_sin[:, :seqlen_kv], self.rope_cos[:, :seqlen_kv])

        # 3) Build masks for unpadding
        if encoder_hidden_states is None:
            mask_q = attention_mask
            mask_kv = attention_mask
        else:
            mask_q = torch.ones(bsz, seqlen_q, dtype=torch.bool, device=q.device)
            mask_kv = attention_mask

        # 4) Unpad Q, K, V
        if use_flash:
            # 🚀 FlashAttention (varlen packed path)
            q_unpad, q_idx, q_cu, q_max, _ = unpad_input(q, mask_q)
            k_unpad, _, k_cu, k_max, _ = unpad_input(k, mask_kv)
            v_unpad, _, _, _, _ = unpad_input(v, mask_kv)

            kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)

            attn_unpad = flash_attn_varlen_kvpacked_func(
                q_unpad,
                kv_unpad,
                q_cu,
                k_cu,
                q_max,
                k_max,
                self.dropout_p if self.training else 0.0,
                1.0 / math.sqrt(self.head_dim),
                causal=(encoder_hidden_states is None),
            )

            attn_padded = pad_input(attn_unpad, q_idx, bsz, seqlen_q)

        else:
            # 🧠 PyTorch standard scaled dot-product attention (fallback)
            q_ = q.permute(0, 2, 1, 3)  # [B, H, S_q, D]
            k_ = k.permute(0, 2, 1, 3)  # [B, H_kv, S_kv, D]
            v_ = v.permute(0, 2, 1, 3)  # [B, H_kv, S_kv, D]

            # If grouped-query attention, repeat kv to match num_heads
            if self.num_kv_heads != self.num_heads:
                repeat_factor = self.num_heads // self.num_kv_heads
                k_ = k_.repeat_interleave(repeat_factor, dim=1)
                v_ = v_.repeat_interleave(repeat_factor, dim=1)

            attn_scores = torch.matmul(q_, k_.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if mask_q is not None:
                # Turn [B, S] -> [B, 1, 1, S] and apply additive mask
                attn_mask = (1.0 - mask_kv.unsqueeze(1).unsqueeze(2)) * -1e9
                attn_scores = attn_scores + attn_mask

            attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
            attn_probs = torch.nn.functional.dropout(attn_probs, p=self.dropout_p, training=self.training)

            attn_padded = torch.matmul(attn_probs, v_)  # [B, H, S, D]
            attn_padded = attn_padded.permute(0, 2, 1, 3).contiguous()  # [B, S, H, D]

        # shared reshape & projection
        attn_out = attn_padded.view(bsz, seqlen_q, self.num_heads, self.head_dim)
        attn_out = attn_out.reshape(bsz, seqlen_q, self.hidden_size)

        #       if not use_flash:
        #           print(f"[Fallback] Using regular attention – dtype: {hidden_states.dtype}")
        #       else:
        #           print(f"[Flash] Using FlashAttention – dtype: {hidden_states.dtype}")

        return self.out_proj(attn_out)


# ──────────────────────────────────────────────────────────────────────────
# Transformer layer (with checkpoint hooks)
# ──────────────────────────────────────────────────────────────────────────
class ArlowFlashTransformerLayer(nn.Module):
    def __init__(self, config: ArlowConfig):
        super().__init__()
        self.gradient_checkpointing = False

        self.self_attn = ArlowGroupedQueryAttention(config)
        self.cross_attn = (
            ArlowGroupedQueryAttention(config, is_cross_attn=True) if config.use_cross_attention else None
        )

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.SiLU() if config.hidden_act == "silu" else nn.ReLU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.norm1 = ArlowRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = ArlowRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_cross = ArlowRMSNorm(config.hidden_size, eps=config.rms_norm_eps) if self.cross_attn else None
        self.dropout = nn.Dropout(config.attention_dropout)

    # ---------------------------------------------------------------------
    def _sa_block(self, hidden_states, attn_mask):
        return self.self_attn(self.norm1(hidden_states), attn_mask)

    def _ca_block(self, hidden_states, enc_hidden, enc_mask):
        return self.cross_attn(self.norm_cross(hidden_states), enc_mask, enc_hidden)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # self‑attention
        sa_out = (
            torch.utils.checkpoint.checkpoint(self._sa_block, hidden_states, attention_mask)
            if self.training and self.gradient_checkpointing
            else self._sa_block(hidden_states, attention_mask)
        )
        hidden_states = hidden_states + self.dropout(sa_out)

        # cross‑attention
        if self.cross_attn is not None and encoder_hidden_states is not None:
            ca_out = (
                torch.utils.checkpoint.checkpoint(
                    self._ca_block, hidden_states, encoder_hidden_states, encoder_attention_mask
                )
                if self.training and self.gradient_checkpointing
                else self._ca_block(hidden_states, encoder_hidden_states, encoder_attention_mask)
            )
            hidden_states = hidden_states + self.dropout(ca_out)

        # MLP
        mlp_out = self.mlp(self.norm2(hidden_states))
        hidden_states = hidden_states + self.dropout(mlp_out)
        return hidden_states


# ──────────────────────────────────────────────────────────────────────────
# Pre‑trained base
# ──────────────────────────────────────────────────────────────────────────
class ArlowPreTrainedModel(PreTrainedModel):
    config_class = ArlowConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ArlowFlashTransformerLayer"]
    _skip_keys_device_placement = ["rope_sin", "rope_cos"]  # Skip RoPE buffers in device placement
    _supports_flash_attn_2 = True
    _supports_sdpa = True

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

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ArlowFlashTransformerLayer):
            module.gradient_checkpointing = value

    is_loaded_in_8bit = False
    is_loaded_in_4bit = False


# ──────────────────────────────────────────────────────────────────────────
# Encoder stack
# ──────────────────────────────────────────────────────────────────────────
class ArlowModel(ArlowPreTrainedModel):
    def __init__(self, config: ArlowConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([ArlowFlashTransformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = ArlowRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
        return self.norm(hidden_states)


# ──────────────────────────────────────────────────────────────────────────
# Causal‑LM wrapper
# ──────────────────────────────────────────────────────────────────────────
class ArlowForCausalLM(ArlowPreTrainedModel, GenerationMixin):
    def __init__(self, config: ArlowConfig):
        super().__init__(config)
        self.model = ArlowModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.lm_head, self.model.embed_tokens)

    # manual toggle remains for user convenience
    def gradient_checkpointing_enable(self, **_):
        self.gradient_checkpointing = True
        for layer in self.model.layers:
            layer.gradient_checkpointing = True

    # forward ----------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    # tied‑weight helpers ----------------------------------------------------
    def state_dict(self, *a, **kw):
        sd = super().state_dict(*a, **kw)
        if self.config.tie_word_embeddings:
            sd.pop("lm_head.weight", None)
        return sd

    def _load_from_state_dict(self, sd, prefix, local_md, strict, missing, unexp, errs):
        if self.config.tie_word_embeddings and f"{prefix}lm_head.weight" not in sd:
            sd[f"{prefix}lm_head.weight"] = sd[f"{prefix}model.embed_tokens.weight"]
        super()._load_from_state_dict(sd, prefix, local_md, strict, missing, unexp, errs)


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────
__all__ = [
    "ArlowRMSNorm",
    "ArlowGroupedQueryAttention",
    "ArlowFlashTransformerLayer",
    "ArlowPreTrainedModel",
    "ArlowModel",
    "ArlowForCausalLM",
]
