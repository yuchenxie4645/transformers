"""Testing suite for the PyTorch Arlow model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        ArlowConfig,
        ArlowForCausalLM,
        ArlowForQuestionAnswering,
        ArlowForSequenceClassification,
        ArlowForTokenClassification,
        ArlowModel,
    )
    from transformers.models.arlow.modeling_arlow import ArlowRotaryEmbedding


class ArlowModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = ArlowConfig
        base_model_class = ArlowModel
        causal_lm_class = ArlowForCausalLM


@require_torch
class ArlowModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            ArlowModel,
            ArlowForCausalLM,
            ArlowForSequenceClassification,
            ArlowForTokenClassification,
            ArlowForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": ArlowModel,
            "text-generation": ArlowForCausalLM,
            "text-classification": ArlowForSequenceClassification,
            "token-classification": ArlowForTokenClassification,
            "question-answering": ArlowForQuestionAnswering,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    model_tester_class = ArlowModelTester
    rotary_embedding_layer = ArlowRotaryEmbedding  # Enables RoPE tests if set

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = ArlowForCausalLM if is_torch_available() else None


@require_torch
class ArlowIntegrationTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_model_basic_functionality(self):
        """Test basic model functionality with small inputs."""
        config = ArlowConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,  # Same as num_attention_heads for simplicity
            max_position_embeddings=128,
        )
        model = ArlowForCausalLM(config).to(torch_device).eval()

        # Test forward pass
        input_ids = torch.randint(0, config.vocab_size, (1, 10), device=torch_device)
        with torch.no_grad():
            outputs = model(input_ids)

        self.assertEqual(outputs.logits.shape, (1, 10, config.vocab_size))

        # Test generation
        generated = model.generate(input_ids, max_new_tokens=5, do_sample=False)
        self.assertEqual(generated.shape[1], 15)  # original 10 + 5 new tokens

    @slow
    def test_model_with_attention_mask(self):
        """Test model with attention mask for padding."""
        config = ArlowConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            pad_token_id=0,
        )
        model = ArlowForCausalLM(config).to(torch_device).eval()

        # Create input with padding
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]], device=torch_device)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], device=torch_device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        self.assertEqual(outputs.logits.shape, (2, 5, config.vocab_size))

    @slow
    def test_model_rope_scaling(self):
        """Test RoPE scaling functionality."""
        config = ArlowConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            rope_scaling={"type": "linear", "factor": 2.0},
        )
        model = ArlowModel(config).to(torch_device).eval()

        # Test with sequence longer than original max_position_embeddings
        input_ids = torch.randint(0, config.vocab_size, (1, 80), device=torch_device)

        with torch.no_grad():
            outputs = model(input_ids)

        self.assertEqual(outputs.last_hidden_state.shape, (1, 80, config.hidden_size))

    @slow
    def test_model_training_mode(self):
        """Test model in training mode with loss computation."""
        config = ArlowConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
        )
        model = ArlowForCausalLM(config).to(torch_device).train()

        input_ids = torch.randint(0, config.vocab_size, (2, 10), device=torch_device)
        labels = input_ids.clone()

        outputs = model(input_ids, labels=labels)

        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.logits.shape, (2, 10, config.vocab_size))

        # Test backward pass
        outputs.loss.backward()

        # Check that gradients are computed
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                break

    @slow
    def test_model_gradient_checkpointing(self):
        """Test gradient checkpointing functionality."""
        config = ArlowConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,  # More layers for gradient checkpointing
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
        )
        model = ArlowForCausalLM(config).to(torch_device).train()
        model.gradient_checkpointing_enable()

        input_ids = torch.randint(0, config.vocab_size, (1, 20), device=torch_device)
        labels = input_ids.clone()

        outputs = model(input_ids, labels=labels)
        outputs.loss.backward()

        # Check that gradients are computed despite gradient checkpointing
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                break

    @slow
    def test_model_sliding_window_attention(self):
        """Test sliding window attention with long sequences."""
        config = ArlowConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=512,
            use_sliding_window=True,
            sliding_window=128,
            max_window_layers=2,  # First 2 layers use full attention, last 2 use sliding window
        )
        model = ArlowForCausalLM(config).to(torch_device).eval()

        # Test with a sequence longer than sliding window
        input_ids = torch.randint(0, config.vocab_size, (1, 200), device=torch_device)
        with torch.no_grad():
            outputs = model(input_ids)

        self.assertEqual(outputs.logits.shape, (1, 200, config.vocab_size))

        # Verify layer types were set correctly
        self.assertEqual(config.layer_types[0], "full_attention")
        self.assertEqual(config.layer_types[1], "full_attention")
        self.assertEqual(config.layer_types[2], "sliding_attention")
        self.assertEqual(config.layer_types[3], "sliding_attention")

        # Test generation with sliding window
        generated = model.generate(input_ids[:, :10], max_new_tokens=20, do_sample=False)
        self.assertEqual(generated.shape[1], 30)  # original 10 + 20 new tokens

    @slow
    def test_model_sliding_window_vs_full_attention(self):
        """Test that sliding window and full attention produce similar results on short sequences."""
        # For short sequences (shorter than sliding window), results should be identical
        seq_len = 50
        
        # Config with sliding window
        config_sliding = ArlowConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=512,
            use_sliding_window=True,
            sliding_window=128,
            max_window_layers=0,  # All layers use sliding window
        )
        
        # Config without sliding window (all full attention)
        config_full = ArlowConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=512,
            use_sliding_window=False,
        )

        model_sliding = ArlowForCausalLM(config_sliding).to(torch_device).eval()
        model_full = ArlowForCausalLM(config_full).to(torch_device).eval()

        # Copy weights to make them identical
        model_full.load_state_dict(model_sliding.state_dict(), strict=False)

        input_ids = torch.randint(0, 1000, (1, seq_len), device=torch_device)
        
        with torch.no_grad():
            outputs_sliding = model_sliding(input_ids)
            outputs_full = model_full(input_ids)

        # For sequences shorter than sliding window, outputs should be very similar
        torch.testing.assert_close(outputs_sliding.logits, outputs_full.logits, rtol=1e-3, atol=1e-3)
