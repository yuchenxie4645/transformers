# tests/models/arlow/test_modeling_arlow.py

import sys
import tempfile
import unittest

import torch

from transformers import ArlowConfig, ArlowForCausalLM, ArlowModel
from transformers.testing_utils import require_torch, torch_device


# Optional: gracefully skip if FlashAttention isn't installed
try:
    import flash_attn  # noqa: F401
except ImportError:
    print("⚠️ flash-attn not found — skipping Arlow model tests.")
    sys.exit(0)

torch.manual_seed(1337)

all_model_classes = (ArlowForCausalLM, ArlowModel)


class ArlowModelTester(unittest.TestCase):
    def setUp(self):
        self.config = ArlowConfig(
            vocab_size=2048,
            hidden_size=256,
            intermediate_size=1024,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_hidden_layers=4,
            max_position_embeddings=128,
            attention_dropout=0.0,
            pad_token_id=0,
        )

    @require_torch
    def test_forward_pass(self):
        model = ArlowForCausalLM(self.config).to(torch_device).eval()
        input_ids = torch.randint(0, self.config.vocab_size, (2, 32), device=torch_device)
        with torch.no_grad():
            outputs = model(input_ids)
        self.assertEqual(outputs.logits.shape, (2, 32, self.config.vocab_size))

    @require_torch
    def test_save_and_load(self):
        model = ArlowForCausalLM(self.config).to(torch_device).eval()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            loaded_model = ArlowForCausalLM.from_pretrained(tmp_dir).to(torch_device)
        self.assertTrue(torch.allclose(model.lm_head.weight, loaded_model.lm_head.weight, atol=1e-5))

    @require_torch
    def test_generate_output(self):
        model = ArlowForCausalLM(self.config).to(torch_device).eval()
        input_ids = torch.randint(0, self.config.vocab_size, (1, 16), device=torch_device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model.generate(input_ids=input_ids, max_new_tokens=8)
        self.assertEqual(output.shape[1], 16 + 8)

    @require_torch
    def test_arlow_model_forward(self):
        model = ArlowModel(self.config).to(torch_device).eval()
        input_ids = torch.randint(0, self.config.vocab_size, (2, 16), device=torch_device)
        with torch.no_grad():
            output = model(input_ids)
        self.assertEqual(output.shape, (2, 16, self.config.hidden_size))


if __name__ == "__main__":
    unittest.main()
