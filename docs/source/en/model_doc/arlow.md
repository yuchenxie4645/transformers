
# Arlow

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Arlow is a multimodal vision-language model that combines a Qwen2-VL style vision encoder with a Gemma-like text decoder. The model supports images, videos, and text in a unified architecture with advanced features like:

- **Dynamic resolution** for handling arbitrary image sizes efficiently
- **M-ROPE (Multimodal Rotary Position Embedding)** for 3D positional encoding of visual content
- **DeepStack visual injection** for multi-layer visual feature fusion
- **Pan-and-scan** for preserving detail in high-resolution images
- **Multiple video sampling strategies** (uniform, fps-based, motion-adaptive)

The model natively handles single or batched conversations mixing text, images, and videos.

## Usage example

### Single Image Inference

```python
import torch
from transformers import ArlowForConditionalGeneration, AutoProcessor

model = ArlowForConditionalGeneration.from_pretrained(
    "yuchenxie/arlow-vlm",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("yuchenxie/arlow-vlm")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://example.com/image.jpg"},
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

### Text-only Generation

For text-only tasks, use `ArlowForCausalLM`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "yuchenxie/arlow-text",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("yuchenxie/arlow-text")

prompt = "Explain the concept of large language models."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
```

## Usage Tips

### Image Resolution

The model supports a wide range of resolutions. Configure the minimum and maximum pixels to balance quality and computation:

```python
processor = AutoProcessor.from_pretrained(
    "yuchenxie/arlow-vlm",
    min_pixels=256 * 28 * 28,
    max_pixels=1024 * 28 * 28
)
```

This ensures each image uses 256-1024 tokens. The factor of 28 comes from the patch size (14) times the temporal patch size (2).

### Flash Attention 2

For faster inference, install Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

Then load the model with:

```python
model = ArlowForConditionalGeneration.from_pretrained(
    "yuchenxie/arlow-vlm",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
```

Flash Attention 2 requires `torch.float16` or `torch.bfloat16` dtype.

### Quantization

For reduced memory usage with bitsandbytes:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = ArlowForConditionalGeneration.from_pretrained(
    "yuchenxie/arlow-vlm",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### Video Sampling Strategies

Configure video frame sampling in the processor:

```python
# Uniform sampling (default)
inputs = processor.apply_chat_template(conversation, fps=1)

# Motion-adaptive sampling (for action-heavy videos)
inputs = processor.apply_chat_template(
    conversation,
    sample_strategy="motion_adaptive"
)
```

### Timestamp Alignment

Enable timestamp supervision for video grounding tasks:

```python
inputs = processor.apply_chat_template(
    conversation,
    timestamp_alignment=True
)
```

This injects `<{time} seconds>` tokens before each frame for temporal grounding.

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

## ArlowVideoProcessor

[[autodoc]] ArlowVideoProcessor
    - preprocess

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
