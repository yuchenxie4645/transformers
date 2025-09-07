import tempfile
import unittest

import torch

from transformers import ArlowConfig, ArlowForCausalLM, ArlowModel
from transformers.testing_utils import require_torch, torch_device


class _DummyCache:
    def __init__(self):
        self.called_with = None

    def reorder_cache(self, beam_idx):
        self.called_with = beam_idx


class ArlowModelingTests(unittest.TestCase):
    def _tiny_config(self) -> ArlowConfig:
        return ArlowConfig(
            vocab_size=257,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            max_position_embeddings=64,
            attention_dropout=0.0,
            resid_dropout=0.0,
            mlp_dropout=0.0,
            attention_bias=False,
            use_varlen_flash=False,  # Standard SDPA path
            pad_token_id=0,
        )

    @require_torch
    def test_model_forward(self):
        cfg = self._tiny_config()
        model = ArlowModel(cfg).to(torch_device).eval()
        input_ids = torch.randint(0, cfg.vocab_size, (2, 16), device=torch_device)
        with torch.no_grad():
            out = model(input_ids=input_ids)
        self.assertEqual(out.last_hidden_state.shape, (2, 16, cfg.hidden_size))

    @require_torch
    def test_causallm_forward_and_loss(self):
        cfg = self._tiny_config()
        model = ArlowForCausalLM(cfg).to(torch_device).train(False)
        bs, seqlen = 2, 20
        input_ids = torch.randint(0, cfg.vocab_size, (bs, seqlen), device=torch_device)
        labels = input_ids.clone()
        # set some labels to ignore index via pad_token_id or -100
        labels[:, :2] = cfg.pad_token_id if cfg.pad_token_id is not None else -100
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=labels, logits_to_keep=8)
        self.assertIsNotNone(out.loss)
        self.assertEqual(out.logits.shape, (bs, 8, cfg.vocab_size))

    @require_torch
    def test_generate_greedy(self):
        cfg = self._tiny_config()
        model = ArlowForCausalLM(cfg).to(torch_device).eval()
        inp = torch.randint(0, cfg.vocab_size, (1, 10), device=torch_device)
        with torch.no_grad():
            generated = model.generate(input_ids=inp, max_new_tokens=6)
        self.assertEqual(generated.shape[1], 16)

    @require_torch
    def test_save_and_load_and_tying(self):
        cfg = self._tiny_config()
        model = ArlowForCausalLM(cfg).to(torch_device).eval()
        # weight tying check (values close)
        wte = model.model.get_input_embeddings().weight
        lm = model.lm_head.weight
        self.assertTrue(torch.allclose(wte[:10], lm[:10], atol=1e-5))
        # roundtrip
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d)
            reloaded = ArlowForCausalLM.from_pretrained(d).to(torch_device).eval()
        self.assertTrue(torch.allclose(model.lm_head.weight, reloaded.lm_head.weight, atol=1e-5))

    @require_torch
    def test_reorder_cache_helper(self):
        cfg = self._tiny_config()
        model = ArlowForCausalLM(cfg).to(torch_device).eval()
        dummy_cache = _DummyCache()
        beam_idx = torch.tensor([1, 0], device=torch_device)
        out = model._reorder_cache(dummy_cache, beam_idx)
        self.assertIs(out, dummy_cache)
        self.assertIsNotNone(dummy_cache.called_with)

    @require_torch
    def test_standard_sdpa_path(self):
        """Test standard SDPA attention path with padding mask"""
        cfg = self._tiny_config()
        cfg.use_varlen_flash = False  # Force standard path
        model = ArlowModel(cfg).to(torch_device).eval()
        
        # Create input with padding
        bs, seqlen = 2, 16
        input_ids = torch.randint(1, cfg.vocab_size, (bs, seqlen), device=torch_device)
        # Add padding tokens at the end
        input_ids[:, -4:] = cfg.pad_token_id
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != cfg.pad_token_id).long()
        
        with torch.no_grad():
            # This should use standard F.scaled_dot_product_attention
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        
        self.assertEqual(out.last_hidden_state.shape, (bs, seqlen, cfg.hidden_size))


if __name__ == "__main__":
    unittest.main()


