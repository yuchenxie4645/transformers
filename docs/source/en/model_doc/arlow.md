
# Arlow

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Arlow is a vision-language model that combines a visual encoder with a text decoder for multimodal understanding and generation. The model is built on the Transformer architecture with several key features:

- **Vision Encoder**: Processes images and videos through a vision transformer with rotary position embeddings (RoPE)
- **Text Decoder**: Decoder-only architecture with grouped query attention (GQA), RoPE, and sliding window attention support
- **Multimodal Integration**: Seamlessly fuses vision and text modalities using M-ROPE (Multimodal Rotary Position Embedding)
- **Flexible Usage**: Supports both text-only (ArlowForCausalLM) and multimodal (ArlowForConditionalGeneration) tasks

The model can handle:
- Single and multiple images
- Video inputs with temporal understanding
- Pure text generation
- Mixed multimodal conversations

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
