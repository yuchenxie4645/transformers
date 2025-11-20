
# Arlow

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Arlow is a multimodal generative model that blends a Qwen2/3-VL style vision stack with a Gemma-like decoder and tooling
adapted from the Qwen processors. The source code highlights these direct inspirations with `Inspired by transformers.models.*`
comments on every copied or closely adapted block so it stays obvious which upstream implementation informed Arlow.

At a high level Arlow offers:

- **Vision encoder (Qwen2-VL inspired)** that patchifies images & videos with rotary-aware attention, deformable biasing, DeepStack skips, and token budgets.
- **Text decoder (Gemma + Qwen2 blend)** with grouped-query attention, RoPE / M-ROPE, sliding-window layers, and optional gated visual fusion.
- **Multimodal bridge** that projects vision tokens into the text space, tracks rope deltas, and can inject DeepStack or timestamp-aligned hints.
- **Processor & tokenizer tooling (Qwen3-VL inspired)** that expands `<image>` / `<video>` placeholders, supports pan-and-scan crops, adaptive video sampling, and multimodal chat templates.

The model natively handles:

- Single or batched conversations that mix pure text, images, and/or long-form videos.
- Fine-grained visual prompts such as pan-and-scan crops or timestamp-aligned descriptions.
- Text-only workloads by instantiating `ArlowForCausalLM` with the same checkpoints.

### Architecture Walkthrough

#### Text decoder
- Uses the Gemma-style RMSNorm/MLP stack with grouped-query attention borrowed from Qwen2.
- Supports mixed full/sliding attention via `layer_types`, configurable RoPE scaling, and gating hooks that can blend DeepStack features back into selected layers.
- Keeps cache- and FlashAttention-friendly APIs (rope-aware cache positions, causal mask factory, SDPA/Flash2 dispatch).

#### Vision encoder
- Reuses the Qwen2-VL patch embed → rotary-attention → MLP block pipeline with optional deformable biasing and progressive patching.
- Tracks DeepStack layers so intermediate vision features can be re-injected into the decoder (either always or via learned gates).
- Provides helpers such as `get_image_features`/`get_video_features` that map placeholder metadata back to token slices.

#### Multimodal bridge
- `ArlowModel` aligns the modalities by projecting vision tokens to the text hidden size, concatenating them with prompt embeddings, then computing joint positional ids (M-ROPE for vision, text RoPE for language).
- Rope deltas are cached so assisted decoding or multi-image prompts reuse the expensive indexing work.

#### Pre/Post-processing
- `ArlowProcessor` mirrors the Qwen3-VL processor: it expands `<image>` / `<video>` placeholders into the exact number of required tokens, injects timestamp hints, and supports batch mixes of media types.
- `ArlowImageProcessor(Fast)` performs dynamic resizing, optional pan-and-scan crops, patch merging, and emits `image_grid_thw` metadata that the model needs.
- `ArlowVideoProcessor` adds several sampling strategies (`uniform`, `fps_based`, `motion_adaptive`) plus safeguards for volumetric token budgets.

### Input Preparation & Special Tokens

- Text prompts should use `<image>` / `<video>` markers (or the tokenizer’s equivalent special ids). The processor expands each marker into `<|vision_start|> ... <|vision_end|>` spans sized to match the actual grid metadata.
- Videos can optionally receive timestamp supervision: when `timestamp_alignment=True`, each frame placeholder is preceded by `<{time} seconds>` tokens so the decoder can ground outputs.
- When pan-and-scan is enabled, additional `<image>` markers get injected automatically so croppings share the original context sentence.

### Processor knobs you might care about

- **Dynamic resolution**: `images_kwargs={"size": {...}, "disable_grouping": False}` allows heterogeneous aspect ratios without wasting tokens.
- **Pan-and-Scan**: set `do_pan_and_scan=True` plus the `pan_and_scan_*` thresholds to capture tall/ultra-wide content while respecting the mm token budget.
- **Video sampling**: choose between uniform sampling (`sample_strategy="uniform"`), deterministic FPS-based sampling (`"fps_based"`), or motion-adaptive sampling (provide raw frames to favor segments with action).
- **Token budgeting**: `mm_tokens_per_image` / `mm_tokens_per_video` in `ArlowConfig` and `max_tokens_per_video` in the processor guard against prompt explosions.

These knobs pair tightly with the `image_grid_thw` / `video_grid_thw` metadata that the processor returns—always forward them to the model alongside `pixel_values`/`pixel_values_videos`.

## Usage Examples

### Text-only Generation

For text-only tasks, use `ArlowForCausalLM`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "your-arlow-model",
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
tokenizer = AutoTokenizer.from_pretrained("your-arlow-model")

prompt = "Explain the concept of large language models."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(response)
```

### Single Image Inference

For vision-language tasks, use `ArlowForConditionalGeneration`:

```python
import torch
from transformers import ArlowForConditionalGeneration, AutoProcessor

model = ArlowForConditionalGeneration.from_pretrained(
    "your-arlow-vlm-model",
    dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("your-arlow-vlm-model")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "path/to/image.jpg"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)
```

### Video Understanding

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/path/to/video.mp4"},
            {"type": "text", "text": "What happens in this video?"}
        ]
    }
]

inputs = processor.apply_chat_template(
    conversation,
    fps=1,  # Sample 1 frame per second
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)
```

### Batch Mixed Media Inference

The model can process batches with mixed media types:

```python
# Multiple images
conversation1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image1.jpg"},
            {"type": "image", "path": "/path/to/image2.jpg"},
            {"type": "text", "text": "Compare these two images."}
        ]
    }
]

# Pure text
conversation2 = [
    {
        "role": "user",
        "content": "What is machine learning?"
    }
]

# Mixed media
conversation3 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image.jpg"},
            {"type": "video", "path": "/path/to/video.mp4"},
            {"type": "text", "text": "What are the common themes?"}
        ]
    }
]

conversations = [conversation1, conversation2, conversation3]

inputs = processor.apply_chat_template(
    conversations,
    fps=1,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    padding=True
).to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)
```

## Usage Tips

### Flash Attention 2

To enable Flash Attention 2 for faster inference:

```bash
pip install -U flash-attn --no-build-isolation
```

Then load the model with:

```python
model = ArlowForConditionalGeneration.from_pretrained(
    "your-arlow-model",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
```

Note: Flash Attention 2 requires `torch.float16` or `torch.bfloat16` dtype.

### Quantization

For reduced memory usage, quantize the model with bitsandbytes:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = ArlowForConditionalGeneration.from_pretrained(
    "your-arlow-model",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### Position IDs, Cache & FA2 Packing

- Always pass `cache_position` from `generate` into manual forward calls if you override `prepare_inputs_for_generation`. The text model differentiates between prefill vs. decode steps based on that tensor.
- When using FlashAttention-2 the model expects packed `position_ids` of shape `[4, batch, seq]` (text row + 3 M-ROPE rows). The processor handles this automatically; if you craft inputs manually ensure you concatenate `[text_positions; vision_positions]`.
- Mixed-modal batches can skip re-feeding `pixel_values` during decoding. `ArlowForConditionalGeneration.prepare_inputs_for_generation` already clears them when `cache_position[0] != 0`, so copy that behavior in custom loops.

### Processor Debugging Tips

- Call `processor._get_num_multimodal_tokens(...)` to sanity-check that the token budget matches your prompt before invoking the heavy image/video preprocessing.
- Set `return_mm_token_type_ids=True` to obtain a mask of multimodal placeholder positions. This is handy when computing loss masks or when you want to inject DeepStack features selectively.
- Enable `timestamp_alignment` only when your video metadata includes fps/frame indices; otherwise the processor will warn and fall back to a default FPS of 24.

## ArlowConfig

[[autodoc]] ArlowConfig

## ArlowTextConfig

[[autodoc]] ArlowTextConfig

## ArlowVisionConfig

[[autodoc]] ArlowVisionConfig

## ArlowProcessor

[[autodoc]] ArlowProcessor

## ArlowImageProcessor

[[autodoc]] ArlowImageProcessor
    - preprocess

## ArlowRMSNorm

[[autodoc]] ArlowRMSNorm
    - forward

## ArlowTextModel

[[autodoc]] ArlowTextModel
    - forward

## ArlowVLVisionModel

[[autodoc]] ArlowVLVisionModel
    - forward

## ArlowModel

[[autodoc]] ArlowModel
    - forward

## ArlowForCausalLM

[[autodoc]] ArlowForCausalLM
    - forward

## ArlowForConditionalGeneration

[[autodoc]] ArlowForConditionalGeneration
    - forward

## ArlowForSequenceClassification

[[autodoc]] ArlowForSequenceClassification
    - forward

## ArlowForTokenClassification

[[autodoc]] ArlowForTokenClassification
    - forward

## ArlowForQuestionAnswering

[[autodoc]] ArlowForQuestionAnswering
    - forward
