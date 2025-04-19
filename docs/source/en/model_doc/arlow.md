# ArlowGPT — A Future-Ready Decoder for CLM + Vision Fusion

**ArlowGPT** is a decoder-only Transformer architecture purpose-built for **large-scale causal language modeling (CLM)**. Designed for performance, extensibility, and vision integration, Arlow features a lightweight core, frozen cross-attention pathways, and seamless compatibility with FlashAttention and grouped-query attention, as well as standard pytorch attention for easy quants + fine tuning with LoRA or QLoRA.

While currently optimized for text generation tasks, **Arlow is architected with future multimodal backbones in mind** — allowing for straightforward upgrades like image-text fusion due to the inclusion of cross attention weights.

**Important: FlashAttention VarLen does NOT support any other DATA TYPE other than **BF16** and **FP16**. Therefore my implemenation includes conditionally applying flash attention when detected DTYPE to be FP16 or BF16.**

---

## What’s Included

This repo contains the full modeling stack and configuration setup required to register and train Arlow models using the Hugging Face ecosystem.

### `configuration_arlow.py`
Defines the `ArlowConfig` class — the blueprint for model architecture and hyperparameters. It includes:
- Support for grouped-query attention
- FlashAttention-ready toggles
- Cross-attention control (`use_cross_attention`)
- Hugging Face `model_type = "arlow"` registration
- Tied embeddings and rotary position embedding (RoPE) parameters

### `modeling_arlow.py`
Implements the full modeling logic, including:
- `ArlowPreTrainedModel`: Base class with weight init and checkpoint support
- `ArlowModel`: Backbone-only decoder stack (no LM head)
- `ArlowForCausalLM`: Decoder stack + LM head + loss + generation logic

Key features:
- FlashAttention integration (with **varlen** QKV path)
- RoPE rotary embedding
- Grouped Query Attention
- Cross-attention blocks for encoder-decoder pipelines

### Unit Tests (`test_modeling_arlow.py`)
- Covers `ArlowForCausalLM` and `ArlowModel` forward passes
- Tests `generate()` output shape
- Save/load weight consistency
- Full Hugging Face `check_repo.py` compliance via `all_model_classes`

### 🔁 Pretraining Script (WIP)
Includes pretraining logic compatible with:
- `🤗 Trainer`
- FlashAttention 2.0 (if installed)
- Gradient checkpointing
- Streaming/large-scale datasets
- Mixed precision (`bfloat16` or `fp16`) 

---

## Naming Convention & Integration

- All modeling classes use the prefix **`Arlow`** (e.g. `ArlowModel`, `ArlowConfig`)
- However, for HF compatibility, `ArlowConfig.model_type = "arlow"`  
  (so checkpoint naming and auto model loading works via `AutoModelForCausalLM.from_pretrained()`)

---

## Designed For:

- **Decoder-only causal LM pretraining**
- **Large-scale inference** with FlashAttention and GQA
- **Easy opt-in to encoder-decoder setups** (via frozen cross-attention)
- **Future multimodal support**, such as vision transformer backbones or retrieval components
- **Lightweight adaptation** (LoRA, adapters, etc.)
- **Conditionally apply Flash attention methods for BF16 and FP16 DTypes**

## ArlowTokenizer
[[autodoc]] ArlowTokenizer

## ArlowTokenizerFast
[[autodoc]] ArlowTokenizerFast

## ArlowConfig
[[autodoc]] ArlowConfig

## ArlowForCausalLM
[[autodoc]] ArlowForCausalLM

## ArlowModel
[[autodoc]] ArlowModel

## ArlowRMSNorm
[[autodoc]] ArlowRMSNorm

## ArlowPreTrainedModel
[[autodoc]] ArlowPreTrainedModel

## ArlowGroupedQueryAttention
[[autodoc]] ArlowGroupedQueryAttention

## ArlowFlashTransformerLayer
[[autodoc]] ArlowFlashTransformerLayer
