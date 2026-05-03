"""PIL image processor class for Arlow."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring, is_torchvision_available, logging
from .image_processing_arlow import ArlowImageProcessorKwargs, _ArlowImageProcessorMixin, smart_resize


if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF


logger = logging.get_logger(__name__)


def _prepare_float_image_for_resize(image: np.ndarray) -> np.ndarray:
    """Convert out-of-range float images to uint8 so PIL resizing stays well defined."""
    if np.issubdtype(image.dtype, np.floating):
        min_value = float(np.min(image))
        max_value = float(np.max(image))
        if min_value < 0.0 or max_value > 1.0:
            logger.warning_once(
                "Received a floating-point image outside [0, 1] while resizing; clipping to [0, 255] before resize."
            )
            return np.clip(image, 0.0, 255.0).astype(np.uint8)
    return image


@auto_docstring
class ArlowImageProcessorPil(_ArlowImageProcessorMixin, PilBackend):
    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[ArlowImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        do_pan_and_scan: bool,
        pan_and_scan_min_crop_size: int,
        pan_and_scan_max_num_crops: int,
        pan_and_scan_min_ratio_to_activate: float,
        **kwargs,
    ) -> BatchFeature:
        _ = disable_grouping

        expanded_images, num_crops_per_image = self._expand_with_pan_and_scan(
            images=images,
            do_pan_and_scan=do_pan_and_scan,
            pan_and_scan_min_crop_size=pan_and_scan_min_crop_size,
            pan_and_scan_max_num_crops=pan_and_scan_max_num_crops,
            pan_and_scan_min_ratio_to_activate=pan_and_scan_min_ratio_to_activate,
        )

        processed_patches = []
        processed_grids = []
        for image in expanded_images:
            height, width = image.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                image = _prepare_float_image_for_resize(image)
                image = self.resize(
                    image=image,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            else:
                resized_height, resized_width = height, width

            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)

            patches = np.expand_dims(image, axis=0)
            if patches.shape[0] % temporal_patch_size != 0:
                repeats = np.repeat(
                    patches[-1][np.newaxis], temporal_patch_size - (patches.shape[0] % temporal_patch_size), axis=0
                )
                patches = np.concatenate([patches, repeats], axis=0)

            grid_t = patches.shape[0] // temporal_patch_size
            channel = patches.shape[1]
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
            patches = patches.reshape(
                grid_t,
                temporal_patch_size,
                channel,
                grid_h,
                patch_size,
                grid_w,
                patch_size,
            )
            # Row-major token order: T, H, W. Feature order: C, temporal_patch, patch_h, patch_w.
            patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
            flatten_patches = patches.reshape(
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )
            processed_patches.append(flatten_patches)
            processed_grids.append([grid_t, grid_h, grid_w])

        if len(processed_patches) > 0:
            pixel_values = np.concatenate(processed_patches, axis=0)
            image_grid_thw = np.asarray(processed_grids, dtype=np.int64)
        else:
            feature_size = 3 * temporal_patch_size * patch_size * patch_size
            pixel_values = np.zeros((0, feature_size), dtype=np.float32)
            image_grid_thw = np.zeros((0, 3), dtype=np.int64)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw, "num_crops": num_crops_per_image},
            tensor_type=return_tensors,
        )


__all__ = ["ArlowImageProcessorPil"]
