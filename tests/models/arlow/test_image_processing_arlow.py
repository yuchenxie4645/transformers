
import itertools
import tempfile
import unittest

import numpy as np

from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, load_image
from transformers.models.arlow.image_processing_arlow import smart_resize
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs, prepare_video_inputs
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import ArlowImageProcessor

    if is_torchvision_available():
        from transformers import ArlowImageProcessorFast


class ArlowImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        num_frames=10,
        min_resolution=56,
        max_resolution=1024,
        min_pixels=56 * 56,
        max_pixels=28 * 28 * 1280,
        do_normalize=True,
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
        do_resize=True,
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
        do_convert_rgb=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.image_mean = IMAGENET_STANDARD_MEAN
        self.image_std = IMAGENET_STANDARD_STD
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "merge_size": self.merge_size,
        }

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        images = prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )
        return [[image] for image in images]

    def prepare_video_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_video_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            num_frames=self.num_frames,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


@require_torch
@require_vision
class ArlowImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = ArlowImageProcessor if is_vision_available() else None
    fast_image_processing_class = ArlowImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = ArlowImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "min_pixels"))
            self.assertTrue(hasattr(image_processing, "max_pixels"))
            self.assertTrue(hasattr(image_processing, "do_convert_rgb"))
            self.assertTrue(hasattr(image_processing, "patch_size"))
            self.assertTrue(hasattr(image_processing, "temporal_patch_size"))
            self.assertTrue(hasattr(image_processing, "merge_size"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.min_pixels, 56 * 56)
            self.assertEqual(image_processor.max_pixels, 28 * 28 * 1280)

            image_processor = image_processing_class.from_dict(
                self.image_processor_dict, min_pixels=256 * 256, max_pixels=640 * 640
            )
            self.assertEqual(image_processor.min_pixels, 256 * 256)
            self.assertEqual(image_processor.max_pixels, 640 * 640)

    def test_select_best_resolution(self):
        # Test with a final resize resolution
        best_resolution = smart_resize(561, 278, factor=28)
        self.assertEqual(best_resolution, (560, 280))

    def test_call_pil(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            for image in image_inputs:
                self.assertIsInstance(image[0], Image.Image)

            # Test not batched input
            process_out = image_processing(image_inputs[0], return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (4900, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

            # Test batched
            process_out = image_processing(image_inputs, return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (34300, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]] * 7)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

    def test_call_numpy(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image[0], np.ndarray)

            # Test not batched input
            process_out = image_processing(image_inputs[0], return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (4900, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

            # Test batched
            process_out = image_processing(image_inputs, return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (34300, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]] * 7)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

    def test_call_pytorch(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)

            for image in image_inputs:
                self.assertIsInstance(image[0], torch.Tensor)

            # Test not batched input
            process_out = image_processing(image_inputs[0], return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (4900, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

            # Test batched
            process_out = image_processing(image_inputs, return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (34300, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]] * 7)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

    @unittest.skip(reason="ArlowImageProcessor doesn't treat 4 channel PIL and numpy consistently yet")
    def test_call_numpy_4_channels(self):
        pass

    def test_nested_input(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)

            # Test batched as a list of images
            process_out = image_processing(image_inputs, return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (34300, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]] * 7)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

            # Test batched as a nested list of images, where each sublist is one batch
            image_inputs_nested = image_inputs[:3] + image_inputs[3:]
            process_out = image_processing(image_inputs_nested, return_tensors="pt")
            encoded_images_nested = process_out.pixel_values
            image_grid_thws_nested = process_out.image_grid_thw
            expected_output_image_shape = (34300, 1176)
            expected_image_grid_thws = torch.Tensor([[1, 70, 70]] * 7)
            self.assertEqual(tuple(encoded_images_nested.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

            # Image processor should return same pixel values, independently of input format
            self.assertTrue((encoded_images_nested == encoded_images).all())
            self.assertTrue((image_grid_thws_nested == expected_image_grid_thws).all())

    def test_custom_image_size(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            with tempfile.TemporaryDirectory() as tmpdirname:
                image_processing.save_pretrained(tmpdirname)
                image_processor_loaded = image_processing_class.from_pretrained(
                    tmpdirname, max_pixels=56 * 56, min_pixels=28 * 28
                )

            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            process_out = image_processor_loaded(image_inputs, return_tensors="pt")
            expected_output_video_shape = [112, 1176]
            self.assertListEqual(list(process_out.pixel_values.shape), expected_output_video_shape)

    def test_custom_pixels(self):
        pixel_choices = frozenset(itertools.product((100, 150, 200, 20000), (100, 150, 200, 20000)))
        for image_processing_class in self.image_processor_list:
            image_processor_dict = self.image_processor_dict.copy()
            for a_pixels, b_pixels in pixel_choices:
                image_processor_dict["min_pixels"] = min(a_pixels, b_pixels)
                image_processor_dict["max_pixels"] = max(a_pixels, b_pixels)
                image_processor = image_processing_class(**image_processor_dict)
                image_inputs = self.image_processor_tester.prepare_image_inputs()
                # Just checking that it doesn't raise an error
                image_processor(image_inputs, return_tensors="pt")

    @require_vision
    @require_torch
    def test_slow_fast_equivalence(self):
        dummy_image = load_image(url_to_local_path("http://images.cocodataset.org/val2017/000000039769.jpg"))

        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_image, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_image, return_tensors="pt")

        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)
        self.assertEqual(encoding_slow.image_grid_thw.dtype, encoding_fast.image_grid_thw.dtype)
        self._assert_slow_fast_tensors_equivalence(
            encoding_slow.image_grid_thw.float(), encoding_fast.image_grid_thw.float()
        )

    @require_vision
    @require_torch
    def test_slow_fast_equivalence_batched(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        if hasattr(self.image_processor_tester, "do_center_crop") and self.image_processor_tester.do_center_crop:
            self.skipTest(
                reason="Skipping as do_center_crop is True and center_crop functions are not equivalent for fast and slow processors"
            )

        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_images, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_images, return_tensors="pt")

        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)
        self.assertEqual(encoding_slow.image_grid_thw.dtype, encoding_fast.image_grid_thw.dtype)
        self._assert_slow_fast_tensors_equivalence(
            encoding_slow.image_grid_thw.float(), encoding_fast.image_grid_thw.float()
        )

    def test_get_num_patches_without_images(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            num_patches = image_processing.get_number_of_image_patches(height=100, width=100, images_kwargs={})
            self.assertEqual(num_patches, 64)

            num_patches = image_processing.get_number_of_image_patches(height=200, width=50, images_kwargs={})
            self.assertEqual(num_patches, 56)

            num_patches = image_processing.get_number_of_image_patches(
                height=100, width=100, images_kwargs={"patch_size": 28}
            )
            self.assertEqual(num_patches, 16)

    def test_extreme_aspect_ratios(self):
        """Test handling of extreme aspect ratios."""
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            
            # Test wide image
            wide_image = Image.new("RGB", (4000, 100))
            process_out = image_processing(wide_image, return_tensors="pt")
            self.assertIsNotNone(process_out.pixel_values)
            self.assertIsNotNone(process_out.image_grid_thw)
            
            # Test tall image
            tall_image = Image.new("RGB", (100, 4000))
            process_out = image_processing(tall_image, return_tensors="pt")
            self.assertIsNotNone(process_out.pixel_values)
            self.assertIsNotNone(process_out.image_grid_thw)

    def test_min_max_pixel_constraints(self):
        """Test that min/max pixel constraints are properly enforced."""
        for image_processing_class in self.image_processor_list:
            # Test with very small min_pixels
            image_processing = image_processing_class(
                **{**self.image_processor_dict, "min_pixels": 28 * 28, "max_pixels": 56 * 56}
            )
            small_image = Image.new("RGB", (56, 56))
            process_out = image_processing(small_image, return_tensors="pt")
            total_pixels = process_out.image_grid_thw[0][1] * process_out.image_grid_thw[0][2] * 14 * 14
            self.assertGreaterEqual(total_pixels, 28 * 28)
            self.assertLessEqual(total_pixels, 56 * 56 * 2)  # Allow some margin

    def test_patch_grouping_consistency(self):
        """Test that patch grouping produces consistent results."""
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            
            # Create images with same resolution
            images = [Image.new("RGB", (224, 224)) for _ in range(4)]
            
            # Process with grouping enabled (default)
            out_grouped = image_processing(images, return_tensors="pt")
            
            # Process without grouping
            out_ungrouped = image_processing(images, return_tensors="pt", disable_grouping=True)
            
            # Results should be identical
            torch.testing.assert_close(out_grouped.pixel_values, out_ungrouped.pixel_values)
            torch.testing.assert_close(out_grouped.image_grid_thw.float(), out_ungrouped.image_grid_thw.float())

    def test_temporal_padding(self):
        """Test that temporal dimension is properly padded for image inputs."""
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            
            image = Image.new("RGB", (224, 224))
            process_out = image_processing(image, return_tensors="pt")
            
            # Temporal dimension should be 1 after processing (images have 1 "frame")
            self.assertEqual(process_out.image_grid_thw[0][0].item(), 1)

    def test_different_resolutions_batch(self):
        """Test processing a batch with different resolutions."""
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            
            # Create images with different resolutions
            images = [
                Image.new("RGB", (224, 224)),
                Image.new("RGB", (448, 224)),
                Image.new("RGB", (224, 448)),
                Image.new("RGB", (336, 336)),
            ]
            
            process_out = image_processing(images, return_tensors="pt")
            
            # All images should be processed
            self.assertEqual(len(process_out.image_grid_thw), 4)
            
            # Grid dimensions should vary based on input resolution
            grids = process_out.image_grid_thw
            # Verify that different input sizes produce different grid dimensions
            grid_set = set((g[1].item(), g[2].item()) for g in grids)
            self.assertGreater(len(grid_set), 1)  # Should have at least 2 different grid sizes

