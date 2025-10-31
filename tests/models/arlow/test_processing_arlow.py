"""Comprehensive testing suite for Arlow processor."""

import json
import numpy as np
import pytest
import torch
import unittest
from pathlib import Path

from transformers.models.arlow.image_processing_arlow import ArlowImageProcessor
from transformers.models.arlow.image_processing_arlow_fast import ArlowImageProcessorFast
from transformers.models.arlow.video_processing_arlow import ArlowVideoProcessor
from transformers.models.arlow.processing_arlow import ArlowProcessor
from transformers.models.arlow.tokenization_arlow import ArlowTokenizer
from transformers.testing_utils import require_torch, require_vision


@pytest.mark.parametrize("fast", [False, True])
def test_image_processor_preprocess_and_grid(fast):
	processor = ArlowImageProcessorFast() if fast else ArlowImageProcessor()
	# make two images with different shapes
	img1 = torch.randint(0, 255, (3, 320, 480), dtype=torch.uint8)
	img2 = torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)
	out = processor.preprocess([img1, img2], return_tensors="pt")
	assert "pixel_values" in out and "image_grid_thw" in out
	assert out["pixel_values"].ndim == 3
	assert out["image_grid_thw"].shape[-1] == 3


def test_get_number_of_image_patches_matches_preprocess():
	processor = ArlowImageProcessorFast()
	img = torch.randint(0, 255, (3, 320, 640), dtype=torch.uint8)
	out = processor.preprocess([img], return_tensors="pt")
	grid = out["image_grid_thw"][0]
	# tokens per image = grid_h * grid_w
	expected = (grid[1] * grid[2]).item()
	got = processor.get_number_of_image_patches(320, 640)
	assert got == expected


def test_video_processor_grid_and_values():
	vp = ArlowVideoProcessor()
	# make a simple 8-frame video (TCHW)
	video = torch.randint(0, 255, (8, 3, 128, 128), dtype=torch.uint8)
	out = vp.preprocess([video], return_tensors="pt", patch_size=14, temporal_patch_size=2, merge_size=2)
	assert "pixel_values_videos" in out and "video_grid_thw" in out
	assert out["pixel_values_videos"].ndim == 3
	assert out["video_grid_thw"].shape[-1] == 3


def test_processor_placeholder_sizing(tmp_path):
	# build a tiny tokenizer vocab to allow conversion
	vocab = {"<|endoftext|>":0, "<image>":1, "<video>":2, "<|vision_start|>":3, "<|vision_end|>":4, "hello":5}
	merges = "#version: 0.2\na b\n"
	(vp := tmp_path/"vocab.json").write_text(str({k:v for k,v in vocab.items()}))
	(mp := tmp_path/"merges.txt").write_text(merges)
	tok = ArlowTokenizer(vocab_file=str(vp), merges_file=str(mp))
	ip = ArlowImageProcessorFast()
	vidp = ArlowVideoProcessor()
	proc = ArlowProcessor(image_processor=ip, tokenizer=tok, video_processor=vidp)
	text = ["hello <image> and <|vision_start|><video><|vision_end|>"]
	img = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
	video = torch.randint(0, 255, (8, 3, 112, 112), dtype=torch.uint8)
	out = proc(images=[img], videos=[video], text=text)
	assert "input_ids" in out and "pixel_values" in out and "pixel_values_videos" in out


@require_torch
@require_vision  
class ArlowProcessorTest(unittest.TestCase):
	"""Comprehensive test suite for ArlowProcessor."""

	@classmethod
	def setUpClass(cls):
		"""Create a temporary tokenizer for testing."""
		import tempfile
		cls.tmp_dir = tempfile.mkdtemp()
		
		# Create vocabulary
		vocab = {
			"<|endoftext|>": 0,
			"<image>": 1,
			"<video>": 2,
			"<|vision_start|>": 3,
			"<|vision_end|>": 4,
			"hello": 5,
			"world": 6,
			"test": 7,
			"image": 8,
			"video": 9,
		}
		
		# Write vocab file
		vocab_path = Path(cls.tmp_dir) / "vocab.json"
		with open(vocab_path, "w") as f:
			json.dump(vocab, f)
		
		# Write merges file
		merges_path = Path(cls.tmp_dir) / "merges.txt"
		with open(merges_path, "w") as f:
			f.write("#version: 0.2\nh e\nl l\no o\n")
		
		cls.tokenizer = ArlowTokenizer(vocab_file=str(vocab_path), merges_file=str(merges_path))
		cls.image_processor = ArlowImageProcessorFast()
		cls.video_processor = ArlowVideoProcessor()

	def test_processor_initialization(self):
		"""Test that processor initializes correctly with all components."""
		processor = ArlowProcessor(
			image_processor=self.image_processor,
			tokenizer=self.tokenizer,
			video_processor=self.video_processor,
		)
		
		self.assertIsNotNone(processor.image_processor)
		self.assertIsNotNone(processor.tokenizer)
		self.assertIsNotNone(processor.video_processor)
		self.assertEqual(processor.image_token, "<image>")
		self.assertEqual(processor.video_token, "<video>")

	def test_image_only_processing(self):
		"""Test processing with images only."""
		processor = ArlowProcessor(
			image_processor=self.image_processor,
			tokenizer=self.tokenizer,
			video_processor=self.video_processor,
		)
		
		text = ["hello <image> world"]
		images = [torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)]
		
		outputs = processor(images=images, text=text, return_tensors="pt")
		
		self.assertIn("input_ids", outputs)
		self.assertIn("pixel_values", outputs)
		self.assertIn("image_grid_thw", outputs)
		self.assertNotIn("pixel_values_videos", outputs)

	def test_video_only_processing(self):
		"""Test processing with videos only."""
		processor = ArlowProcessor(
			image_processor=self.image_processor,
			tokenizer=self.tokenizer,
			video_processor=self.video_processor,
		)
		
		text = ["hello <video> world"]
		videos = [torch.randint(0, 255, (8, 3, 112, 112), dtype=torch.uint8)]
		
		outputs = processor(videos=videos, text=text, return_tensors="pt")
		
		self.assertIn("input_ids", outputs)
		self.assertIn("pixel_values_videos", outputs)
		self.assertIn("video_grid_thw", outputs)
		self.assertNotIn("pixel_values", outputs)

	def test_mixed_image_video_processing(self):
		"""Test processing with both images and videos."""
		processor = ArlowProcessor(
			image_processor=self.image_processor,
			tokenizer=self.tokenizer,
			video_processor=self.video_processor,
		)
		
		text = ["hello <image> and <video> world"]
		images = [torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)]
		videos = [torch.randint(0, 255, (8, 3, 112, 112), dtype=torch.uint8)]
		
		outputs = processor(images=images, videos=videos, text=text, return_tensors="pt")
		
		self.assertIn("input_ids", outputs)
		self.assertIn("pixel_values", outputs)
		self.assertIn("image_grid_thw", outputs)
		self.assertIn("pixel_values_videos", outputs)
		self.assertIn("video_grid_thw", outputs)

	def test_multiple_images_in_text(self):
		"""Test processing text with multiple image placeholders."""
		processor = ArlowProcessor(
			image_processor=self.image_processor,
			tokenizer=self.tokenizer,
			video_processor=self.video_processor,
		)
		
		text = ["<image> hello <image> world"]
		images = [
			torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8),
			torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8),
		]
		
		outputs = processor(images=images, text=text, return_tensors="pt")
		
		self.assertIn("input_ids", outputs)
		self.assertIn("pixel_values", outputs)
		self.assertEqual(len(outputs["image_grid_thw"]), 2)

	def test_batch_processing(self):
		"""Test processing multiple text-image pairs."""
		processor = ArlowProcessor(
			image_processor=self.image_processor,
			tokenizer=self.tokenizer,
			video_processor=self.video_processor,
		)
		
		texts = ["hello <image> world", "test <image> image"]
		images = [
			[torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)],
			[torch.randint(0, 255, (3, 336, 336), dtype=torch.uint8)],
		]
		
		# Need padding since different image sizes result in different token counts
		outputs = processor(images=images, text=texts, padding=True, return_tensors="pt")
		
		self.assertIn("input_ids", outputs)
		self.assertIn("pixel_values", outputs)
		# Should have 2 image grids (one per text)
		self.assertEqual(len(outputs["image_grid_thw"]), 2)

	def test_placeholder_expansion(self):
		"""Test that image/video placeholders are properly expanded."""
		processor = ArlowProcessor(
			image_processor=self.image_processor,
			tokenizer=self.tokenizer,
			video_processor=self.video_processor,
		)
		
		text = ["<image>"]
		images = [torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)]
		
		# Process to get grid info
		outputs = processor(images=images, text=text, return_tensors="pt")
		
		# The text should have been expanded with image tokens
		# Verify that input_ids length > original text length
		self.assertGreater(outputs["input_ids"].shape[1], 1)

	def test_vision_tokens_wrapping(self):
		"""Test that video tokens are wrapped with vision start/end tokens."""
		processor = ArlowProcessor(
			image_processor=self.image_processor,
			tokenizer=self.tokenizer,
			video_processor=self.video_processor,
		)
		
		text = ["<|vision_start|><video><|vision_end|>"]
		videos = [torch.randint(0, 255, (8, 3, 112, 112), dtype=torch.uint8)]
		
		outputs = processor(videos=videos, text=text, return_tensors="pt")
		
		self.assertIn("input_ids", outputs)
		# Vision tokens should be present in the tokenized output
		input_ids = outputs["input_ids"][0]
		self.assertIn(processor.vision_start_token_id, input_ids.tolist())
		self.assertIn(processor.vision_end_token_id, input_ids.tolist())

	def test_mm_token_type_ids(self):
		"""Test multimodal token type ID generation."""
		processor = ArlowProcessor(
			image_processor=self.image_processor,
			tokenizer=self.tokenizer,
			video_processor=self.video_processor,
		)
		
		text = ["hello <image> world"]
		images = [torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)]
		
		outputs = processor(
			images=images,
			text=text,
			return_tensors="pt",
			return_mm_token_type_ids=True,
		)
		
		self.assertIn("mm_token_type_ids", outputs)
		# Image tokens should have type_id=1
		mm_token_type_ids = outputs["mm_token_type_ids"]
		self.assertGreater(mm_token_type_ids.sum(), 0)

	def test_get_num_multimodal_tokens(self):
		"""Test calculation of multimodal token counts."""
		processor = ArlowProcessor(
			image_processor=self.image_processor,
			tokenizer=self.tokenizer,
			video_processor=self.video_processor,
		)
		
		# Test with image sizes
		vision_data = processor._get_num_multimodal_tokens(
			image_sizes=[(224, 224), (448, 224)]
		)
		
		self.assertIn("num_image_tokens", vision_data)
		self.assertIn("num_image_patches", vision_data)
		self.assertEqual(len(vision_data["num_image_tokens"]), 2)

	def test_different_image_sizes(self):
		"""Test processing images with different sizes."""
		processor = ArlowProcessor(
			image_processor=self.image_processor,
			tokenizer=self.tokenizer,
			video_processor=self.video_processor,
		)
		
		text = ["<image> and <image>"]
		images = [
			torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8),
			torch.randint(0, 255, (3, 448, 224), dtype=torch.uint8),
		]
		
		outputs = processor(images=images, text=text, return_tensors="pt")
		
		# Should have different grid dimensions
		grids = outputs["image_grid_thw"]
		self.assertNotEqual(
			(grids[0][1].item(), grids[0][2].item()),
			(grids[1][1].item(), grids[1][2].item()),
		)

	def test_text_only_processing(self):
		"""Test processing with text only (no images/videos)."""
		processor = ArlowProcessor(
			image_processor=self.image_processor,
			tokenizer=self.tokenizer,
			video_processor=self.video_processor,
		)
		
		text = ["hello world"]
		outputs = processor(text=text, return_tensors="pt")
		
		self.assertIn("input_ids", outputs)
		self.assertNotIn("pixel_values", outputs)
		self.assertNotIn("pixel_values_videos", outputs)

	def test_post_process_generation(self):
		"""Test post-processing of generated outputs."""
		processor = ArlowProcessor(
			image_processor=self.image_processor,
			tokenizer=self.tokenizer,
			video_processor=self.video_processor,
		)
		
		# Simulate generated token IDs
		generated_ids = torch.tensor([[5, 6, 7]])  # hello world test
		
		decoded = processor.post_process_image_text_to_text(generated_ids)
		
		self.assertIsInstance(decoded, list)
		self.assertEqual(len(decoded), 1)

	def test_timestamp_calculation(self):
		"""Test video timestamp calculation for frame indices."""
		processor = ArlowProcessor(
			image_processor=self.image_processor,
			tokenizer=self.tokenizer,
			video_processor=self.video_processor,
		)
		
		indices = [0, 1, 2, 3, 4, 5, 6, 7]
		fps = 2.0
		merge_size = 2
		
		timestamps = processor._calculate_timestamps(indices, fps, merge_size)
		
		self.assertEqual(len(timestamps), len(indices) // merge_size)
		# Timestamps should be in seconds
		self.assertGreater(timestamps[-1], 0)

	def test_empty_images(self):
		"""Test handling of empty image list."""
		processor = ArlowProcessor(
			image_processor=self.image_processor,
			tokenizer=self.tokenizer,
			video_processor=self.video_processor,
		)
		
		text = ["hello world"]
		outputs = processor(images=None, text=text, return_tensors="pt")
		
		self.assertIn("input_ids", outputs)
		self.assertNotIn("pixel_values", outputs)

	def test_processor_attributes(self):
		"""Test that processor has all expected attributes."""
		processor = ArlowProcessor(
			image_processor=self.image_processor,
			tokenizer=self.tokenizer,
			video_processor=self.video_processor,
		)
		
		self.assertTrue(hasattr(processor, "image_token"))
		self.assertTrue(hasattr(processor, "video_token"))
		self.assertTrue(hasattr(processor, "image_token_id"))
		self.assertTrue(hasattr(processor, "video_token_id"))
		self.assertTrue(hasattr(processor, "vision_start_token"))
		self.assertTrue(hasattr(processor, "vision_end_token"))
		self.assertTrue(hasattr(processor, "vision_start_token_id"))
		self.assertTrue(hasattr(processor, "vision_end_token_id"))
