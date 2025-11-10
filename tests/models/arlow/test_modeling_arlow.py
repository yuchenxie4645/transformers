"""Testing suite for the PyTorch Arlow model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import numpy as np
    import torch

    from transformers import (
        ArlowConfig,
        ArlowForCausalLM,
        ArlowForQuestionAnswering,
        ArlowForSequenceClassification,
        ArlowForTokenClassification,
        ArlowModel,
    )
    from transformers.models.arlow.image_processing_arlow import ArlowImageProcessor
    from transformers.models.arlow.video_processing_arlow import ArlowVideoProcessor
    from transformers.models.arlow.modeling_arlow import ArlowTextRotaryEmbedding


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
    rotary_embedding_layer = ArlowTextRotaryEmbedding  # Enables RoPE tests if set
    test_can_load_with_device_context_manager = False
    test_can_load_with_global_device_set = False
    test_disk_offload_safetensors = False

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = ArlowForCausalLM if is_torch_available() else None

    def test_config_multimodal_defaults(self):
        config = ArlowConfig()
        self.assertEqual(config.max_position_embeddings, 32768)
        self.assertEqual(config.mm_tokens_per_image, 512)
        self.assertEqual(config.mm_tokens_per_video, 1024)
        self.assertEqual(config.video_max_frames, 768)
        vision_config = config.vision_config
        self.assertTrue(hasattr(vision_config, "deepstack_visual_indexes"))
        self.assertIsInstance(vision_config.deepstack_visual_indexes, list)


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

    def test_video_processor_temporal_budget(self):
        video_processor = ArlowVideoProcessor(
            patch_size=14,
            temporal_patch_size=2,
            merge_size=2,
            max_tokens_per_video=16,
        )
        video = torch.randn(24, 3, 112, 112)
        outputs = video_processor(videos=[video], return_metadata=True)
        grid = outputs["video_grid_thw"][0]
        grid = grid.tolist() if hasattr(grid, "tolist") else grid
        tokens = int(grid[0]) * int(grid[1]) * int(grid[2])
        self.assertLessEqual(tokens, video_processor.max_tokens_per_video)
        metadata = outputs["video_metadata"][0]
        self.assertEqual(
            len(metadata.frames_indices),
            int(grid[0]) * video_processor.temporal_patch_size,
        )

    def test_video_processor_motion_adaptive_sampling(self):
        np.random.seed(0)
        max_frames = 12
        video_processor = ArlowVideoProcessor(
            sample_strategy="motion_adaptive",
            max_frames=max_frames,
        )
        # Build a video with abrupt change to emphasize motion-heavy frames
        video = torch.zeros(max_frames, 3, 56, 56)
        video[max_frames // 2 :] = 5.0
        outputs = video_processor(
            videos=[video],
            num_frames=6,
            return_metadata=True,
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
        )
        metadata = outputs["video_metadata"][0]
        indices = np.asarray(metadata.frames_indices)
        self.assertGreaterEqual(len(indices), 6)
        self.assertEqual(len(indices) % video_processor.temporal_patch_size, 0)
        self.assertTrue(np.all(indices[:-1] <= indices[1:]))
        self.assertTrue(np.all((indices >= 0) & (indices < max_frames)))

    def test_image_pan_and_scan_generates_crops(self):
        image_processor = ArlowImageProcessor(
            do_pan_and_scan=True,
            pan_and_scan_min_crop_size=64,
            pan_and_scan_min_ratio_to_activate=1.0,
            pan_and_scan_max_num_crops=4,
        )
        image = torch.randn(3, 128, 512)
        outputs = image_processor(images=[image])
        num_crops = outputs["num_crops"][0]
        self.assertGreaterEqual(num_crops, 1)
        grid = outputs["image_grid_thw"][0]
        self.assertEqual(int(grid[0]), 1)

    def test_mrope_section_alignment(self):
        config = ArlowConfig(
            hidden_size=128,
            num_attention_heads=4,
            mrope_sections=[12, 10, 10],
            vision_config={"embed_dim": 96, "num_heads": 3},
        )
        text_sections = config.mrope_sections
        vision_sections = config.vision_config.mrope_sections
        self.assertEqual(
            sum(vision_sections),
            config.vision_config.embed_dim // config.vision_config.num_heads,
        )
        text_total = sum(text_sections)
        vision_total = sum(vision_sections)
        for text_ratio, vision_ratio in zip(
            [sec / text_total for sec in text_sections],
            [sec / vision_total for sec in vision_sections],
        ):
            self.assertAlmostEqual(text_ratio, vision_ratio, delta=0.1)

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

    def test_model_visual_deepstack_outputs(self):
        from transformers import ArlowVisionConfig

        config = ArlowConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            image_token_id=10,
            vision_start_token_id=11,
            vision_end_token_id=12,
            head_dim=16,
            mrope_sections=[4, 6, 6],
            vision_config=ArlowVisionConfig(
                depth=2,
                embed_dim=32,
                hidden_size=64,
                num_heads=4,
                patch_size=14,
                spatial_merge_size=2,
                temporal_patch_size=2,
                deepstack_visual_indexes=[0],
            ),
        )

        model = ArlowModel(config).to(torch_device).eval()
        self.assertTrue(model.visual.enable_deepstack)

        pixel_values = torch.randn(1, 3, 2, 28, 28, device=torch_device)
        image_grid_thw = torch.tensor([[2, 2, 2]], device=torch_device)

        pooled, deepstack = model.get_image_features(
            pixel_values,
            image_grid_thw=image_grid_thw,
            return_deepstack=True,
        )

        self.assertEqual(len(pooled), 1)
        self.assertEqual(pooled[0].shape, (1, config.hidden_size))
        self.assertIsNotNone(deepstack)
        self.assertEqual(len(deepstack), len(config.vision_config.deepstack_visual_indexes))
        self.assertEqual(len(deepstack[0]), 1)
        self.assertEqual(deepstack[0][0].shape, (1, config.hidden_size))

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


@require_torch
class ArlowMultimodalIntegrationTest(unittest.TestCase):
    """Integration tests for Arlow multimodal (vision + text) functionality."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_model_multimodal_forward_image(self):
        """Test multimodal model with image inputs."""
        from transformers import ArlowForConditionalGeneration, ArlowVisionConfig

        # Create a small config for testing
        config = ArlowConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            image_token_id=10,
            vision_start_token_id=11,
            vision_end_token_id=12,
            head_dim=16,  # Explicitly set head_dim
            mrope_sections=[4, 6, 6],  # Sum = 16 = head_dim
        )

        # Add vision config
        config.vision_config = ArlowVisionConfig(
            depth=2,
            embed_dim=32,
            hidden_size=64,
            num_heads=4,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )

        model = ArlowForConditionalGeneration(config).to(torch_device).eval()

        # Create dummy inputs
        # Text with image placeholder
        input_ids = torch.tensor([[1, 2, 11, 10, 12, 3, 4, 5]], device=torch_device)

        # Create dummy pixel values (batch, channels, temporal, height, width)
        # For images, temporal = 1
        pixel_values = torch.randn(1, 3, 2, 28, 28, device=torch_device)  # temporal_patch_size=2

        # Grid dimensions: (num_images, 3) as [temporal=1, height, width]
        # After patching: 28/14 = 2 patches per dimension
        image_grid_thw = torch.tensor([[2, 2, 2]], device=torch_device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        # Check output shapes
        self.assertIsNotNone(outputs.logits)
        self.assertEqual(outputs.logits.shape[0], 1)  # batch size
        self.assertEqual(outputs.logits.shape[2], config.vocab_size)

    def test_model_multimodal_forward_video(self):
        """Test multimodal model with video inputs."""
        from transformers import ArlowForConditionalGeneration, ArlowVisionConfig

        # Create a small config for testing
        config = ArlowConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            video_token_id=13,
            vision_start_token_id=11,
            vision_end_token_id=12,
            head_dim=16,
            mrope_sections=[4, 6, 6],
        )

        config.vision_config = ArlowVisionConfig(
            depth=2,
            embed_dim=32,
            hidden_size=64,
            num_heads=4,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )

        model = ArlowForConditionalGeneration(config).to(torch_device).eval()

        # Text with video placeholder
        input_ids = torch.tensor([[1, 2, 11, 13, 12, 3, 4, 5]], device=torch_device)

        # Video: (batch, channels, temporal, height, width)
        pixel_values_videos = torch.randn(1, 3, 4, 28, 28, device=torch_device)

        # Grid: (num_videos, 3) as [temporal, height, width] after patching
        video_grid_thw = torch.tensor([[2, 2, 2]], device=torch_device)  # 4/2=2, 28/14=2

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

        self.assertIsNotNone(outputs.logits)
        self.assertEqual(outputs.logits.shape[0], 1)
        self.assertEqual(outputs.logits.shape[2], config.vocab_size)

    def test_model_multimodal_forward_image_and_video(self):
        """Test multimodal model with both image and video inputs."""
        from transformers import ArlowForConditionalGeneration, ArlowVisionConfig

        config = ArlowConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=256,
            image_token_id=10,
            video_token_id=13,
            vision_start_token_id=11,
            vision_end_token_id=12,
            head_dim=16,
            mrope_sections=[4, 6, 6],
        )

        config.vision_config = ArlowVisionConfig(
            depth=2,
            embed_dim=32,
            hidden_size=64,
            num_heads=4,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )

        model = ArlowForConditionalGeneration(config).to(torch_device).eval()

        # Text with both image and video placeholders
        input_ids = torch.tensor([[1, 11, 10, 12, 2, 3, 11, 13, 12, 4, 5]], device=torch_device)

        # Image and video inputs
        pixel_values = torch.randn(1, 3, 2, 28, 28, device=torch_device)
        pixel_values_videos = torch.randn(1, 3, 4, 28, 28, device=torch_device)

        image_grid_thw = torch.tensor([[2, 2, 2]], device=torch_device)
        video_grid_thw = torch.tensor([[2, 2, 2]], device=torch_device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )

        self.assertIsNotNone(outputs.logits)
        self.assertEqual(outputs.logits.shape[0], 1)

    def test_model_multimodal_with_labels(self):
        """Test multimodal model training with labels."""
        from transformers import ArlowForConditionalGeneration, ArlowVisionConfig

        config = ArlowConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            image_token_id=10,
            vision_start_token_id=11,
            pad_token_id=0,
            head_dim=16,
            mrope_sections=[4, 6, 6],
        )

        config.vision_config = ArlowVisionConfig(
            depth=2,
            embed_dim=32,
            hidden_size=64,
            num_heads=4,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )

        model = ArlowForConditionalGeneration(config).to(torch_device).train()

        input_ids = torch.tensor([[1, 2, 11, 10, 3, 4, 5]], device=torch_device)
        pixel_values = torch.randn(1, 3, 2, 28, 28, device=torch_device)
        image_grid_thw = torch.tensor([[2, 2, 2]], device=torch_device)

        # Labels for training
        labels = input_ids.clone()
        labels[:, :3] = -100  # Ignore loss on prompt

        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )

        # Check that loss is computed
        self.assertIsNotNone(outputs.loss)
        self.assertGreater(outputs.loss.item(), 0)

        # Test backward pass
        outputs.loss.backward()

        # Check that gradients are computed
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                break

    def test_model_multimodal_generation(self):
        """Test generation with multimodal model."""
        from transformers import ArlowForConditionalGeneration, ArlowVisionConfig

        config = ArlowConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            image_token_id=10,
            vision_start_token_id=11,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            head_dim=16,
            mrope_sections=[4, 6, 6],
        )

        config.vision_config = ArlowVisionConfig(
            depth=2,
            embed_dim=32,
            hidden_size=64,
            num_heads=4,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )

        model = ArlowForConditionalGeneration(config).to(torch_device).eval()

        input_ids = torch.tensor([[1, 11, 10, 3]], device=torch_device)
        pixel_values = torch.randn(1, 3, 2, 28, 28, device=torch_device)
        image_grid_thw = torch.tensor([[2, 2, 2]], device=torch_device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=10,
                do_sample=False,
            )

        # Check that tokens were generated
        self.assertEqual(generated.shape[0], 1)
        self.assertGreater(generated.shape[1], input_ids.shape[1])

    def test_model_mrope_position_ids(self):
        """Test M-ROPE position IDs computation for multimodal inputs."""
        from transformers import ArlowModel, ArlowVisionConfig

        config = ArlowConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            image_token_id=10,
            vision_start_token_id=11,
            head_dim=16,
            mrope_sections=[4, 6, 6],
        )

        config.vision_config = ArlowVisionConfig(
            depth=2,
            embed_dim=32,
            hidden_size=64,
            num_heads=4,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )

        model = ArlowModel(config).to(torch_device).eval()

        # Test get_rope_index method
        input_ids = torch.tensor([[1, 2, 11, 10, 3, 4, 5]], device=torch_device)
        image_grid_thw = torch.tensor([[2, 2, 2]], device=torch_device)

        position_ids, rope_deltas = model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
        )

        # Check shapes
        self.assertEqual(position_ids.shape[0], 3)  # [temporal, height, width]
        self.assertEqual(position_ids.shape[1], input_ids.shape[0])  # batch
        self.assertEqual(rope_deltas.shape[0], input_ids.shape[0])  # batch

    def test_text_model_accepts_4d_position_ids(self):
        """Ensure text model supports packed 4D position_ids [text; 3D mrope]."""
        from transformers.models.arlow.modeling_arlow import ArlowTextConfig, ArlowTextModel

        text_config = ArlowTextConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
        )
        model = ArlowTextModel(text_config).to(torch_device).eval()

        batch, seq = 1, 8
        input_ids = torch.randint(0, text_config.vocab_size, (batch, seq), device=torch_device)
        text_pos = torch.arange(seq, device=torch_device).unsqueeze(0).expand(batch, -1)
        # dummy 3D mrope positions
        vision_pos = torch.stack([text_pos, text_pos, text_pos], dim=0)
        position_ids = torch.cat([text_pos.unsqueeze(0), vision_pos], dim=0)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, position_ids=position_ids)

        self.assertEqual(outputs.last_hidden_state.shape[:2], (batch, seq))

    def test_vision_fa2_dispatch_runs_when_available(self):
        """If FA2 is available, ensure vision path can dispatch without errors."""
        try:
            from transformers import is_flash_attn_available
        except Exception:
            return  # environment may not expose the helper; bail out silently

        if not is_flash_attn_available():
            self.skipTest("Flash Attention 2 not available in environment")

        from transformers import ArlowForConditionalGeneration, ArlowVisionConfig

        config = ArlowConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            image_token_id=10,
            vision_start_token_id=11,
            vision_end_token_id=12,
            head_dim=16,
            mrope_sections=[4, 6, 6],
        )
        config.vision_config = ArlowVisionConfig(
            depth=2,
            embed_dim=32,
            hidden_size=64,
            num_heads=4,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )
        model = ArlowForConditionalGeneration(config).to(torch_device).eval()
        # request FA2 on the vision side
        model.model.visual.config._attn_implementation = "flash_attention_2"

        input_ids = torch.tensor([[1, 2, 11, 10, 12, 3]], device=torch_device)
        pixel_values = torch.randn(1, 3, 2, 28, 28, device=torch_device)
        image_grid_thw = torch.tensor([[2, 2, 2]], device=torch_device)

        with torch.no_grad():
            _ = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

    def test_vision_encoder_standalone(self):
        """Test vision encoder as a standalone component."""
        from transformers.models.arlow.modular_arlow import ArlowVisionConfig, ArlowVLVisionModel

        config = ArlowVisionConfig(
            depth=2,
            embed_dim=32,
            hidden_size=64,
            num_heads=4,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )

        vision_model = ArlowVLVisionModel._from_config(config)
        vision_model = vision_model.to(torch_device).eval()

        # Test image input
        pixel_values = torch.randn(1, 3, 2, 28, 28, device=torch_device)
        grid_thw = torch.tensor([[2, 2, 2]], device=torch_device)

        with torch.no_grad():
            vision_embeddings = vision_model(pixel_values, grid_thw)

        # Check output shape
        self.assertIsNotNone(vision_embeddings)
        self.assertEqual(vision_embeddings.ndim, 2)  # (tokens, hidden_size)
        self.assertEqual(vision_embeddings.shape[1], config.hidden_size)

    def test_model_gradient_checkpointing_multimodal(self):
        """Test gradient checkpointing with multimodal inputs."""
        from transformers import ArlowForConditionalGeneration, ArlowVisionConfig

        config = ArlowConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            image_token_id=10,
            vision_start_token_id=11,
            pad_token_id=0,
            head_dim=16,
            mrope_sections=[4, 6, 6],
        )

        config.vision_config = ArlowVisionConfig(
            depth=4,
            embed_dim=32,
            hidden_size=64,
            num_heads=4,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )

        model = ArlowForConditionalGeneration(config).to(torch_device).train()
        model.gradient_checkpointing_enable()

        input_ids = torch.tensor([[1, 2, 11, 10, 3, 4, 5]], device=torch_device)
        pixel_values = torch.randn(1, 3, 2, 28, 28, device=torch_device)
        image_grid_thw = torch.tensor([[2, 2, 2]], device=torch_device)
        labels = input_ids.clone()

        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )

        outputs.loss.backward()

        # Verify gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                break
