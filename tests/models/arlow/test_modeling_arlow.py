import sys, unittest, tempfile, torch
from transformers.testing_utils import require_torch, torch_device

# ---- abort early if flash‑attn is missing -------------------------------------------------
try:
    import flash_attn                     # noqa: F401
except ImportError:                       # pragma: no cover
    print("⚠️  flash‑attn not found — skipping Arlow model tests.")
    sys.exit(0)

# ---- pull exported symbols straight from transformers -------------------------------------
from transformers import ArlowConfig, ArlowForCausalLM   # ✅ new import path

torch.manual_seed(1234)

class ArlowBasicTest(unittest.TestCase):
    def _cfg(self):
        return ArlowConfig(
            vocab_size=2048,
            hidden_size=320,
            intermediate_size=1280,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_hidden_layers=4,
            max_position_embeddings=256,
            attention_dropout=0.0,
            pad_token_id=0,
        )

    @require_torch
    def test_forward(self):
        model = ArlowForCausalLM(self._cfg()).to(torch_device).bfloat16()
        ids   = torch.randint(5, 2048, (2, 32), device=torch_device)
        out   = model(ids)
        self.assertEqual(out.logits.shape, (2, 32, 2048))

    @require_torch
    def test_save_load(self):
        model = ArlowForCausalLM(self._cfg()).to(torch_device).bfloat16()
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp, safe_serialization=False)
            reloaded = ArlowForCausalLM.from_pretrained(
                tmp,
                low_cpu_mem_usage=False,   # <= keeps real tensors, not meta
                torch_dtype=torch.bfloat16,
                device_map=None,
            )
        self.assertTrue(torch.equal(model.lm_head.weight, reloaded.lm_head.weight))

if __name__ == "__main__":
    unittest.main()
