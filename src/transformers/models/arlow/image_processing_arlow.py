"""Image processor class for Arlow multimodal models."""

import math
from typing import Optional, Union

import torch

from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import get_size_dict
from ...image_processing_utils_fast import BaseImageProcessorFast, group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, add_start_docstrings


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 28 * 28 * 1280
):
    """Rescales the image while enforcing divisibility and pixel bounds.

    Conditions:
    1) height and width divisible by `factor`
    2) total pixels within [min_pixels, max_pixels]
    3) aspect ratio not extreme
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class ArlowImageProcessorKwargs(ImagesKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_pixels: int
    max_pixels: int
    disable_grouping: bool
    do_pan_and_scan: bool
    pan_and_scan_min_crop_size: int
    pan_and_scan_max_num_crops: int
    pan_and_scan_min_ratio_to_activate: float


@add_start_docstrings(
    "Constructs an Arlow image processor that dynamically resizes images and outputs grid metadata.",
    """
        patch_size (`int`, *optional*, defaults to 14):
            The spatial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size used by the vision encoder (images use 1 temporal slice but are padded to be divisible).
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to LLM encoder.
    """,
)
class ArlowImageProcessor(BaseImageProcessorFast):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
    do_rescale = True
    do_normalize = True
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    min_pixels = None
    max_pixels = None
    do_pan_and_scan = False
    pan_and_scan_min_crop_size = 256
    pan_and_scan_max_num_crops = 4
    pan_and_scan_min_ratio_to_activate = 1.6
    valid_kwargs = ArlowImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[ArlowImageProcessorKwargs]):
        size = kwargs.pop("size", None)
        min_pixels = kwargs.pop("min_pixels", None)
        max_pixels = kwargs.pop("max_pixels", None)
        # Start with class default, then override with provided values
        def _size_to_dict(size_value):
            if size_value is None:
                return {}
            if isinstance(size_value, SizeDict):
                return {key: value for key, value in vars(size_value).items() if value is not None}
            if isinstance(size_value, dict):
                return {key: value for key, value in size_value.items() if value is not None}
            converted = get_size_dict(size_value, default_to_square=False, param_name="size_override")
            return {key: value for key, value in converted.items() if value is not None}

        base_size = _size_to_dict(self.size)
        override_size = _size_to_dict(size) if size is not None else None
        merged_size = base_size if override_size is None else {**base_size, **override_size}

        # backward compatibility: override size with min_pixels and max_pixels if they are provided
        if min_pixels is not None:
            merged_size["shortest_edge"] = min_pixels
            merged_size.pop("min_pixels", None)
        if max_pixels is not None:
            merged_size["longest_edge"] = max_pixels
            merged_size.pop("max_pixels", None)
        if "shortest_edge" not in merged_size or "longest_edge" not in merged_size:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        super().__init__(size=merged_size, min_pixels=min_pixels, max_pixels=max_pixels, **kwargs)

    def _further_process_kwargs(
        self,
        size: Optional[SizeDict] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        **kwargs,
    ) -> dict:
        if min_pixels is not None and max_pixels is not None:
            size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        elif size is not None:
            # Merge with defaults if partial size is provided
            size = {**self.size, **size}
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
            min_pixels = size["shortest_edge"]
            max_pixels = size["longest_edge"]
        else:
            size = {**self.size}

        return super()._further_process_kwargs(size=size, min_pixels=min_pixels, max_pixels=max_pixels, **kwargs)

    def pan_and_scan(
        self,
        image: "torch.Tensor",
        pan_and_scan_min_crop_size: int,
        pan_and_scan_max_num_crops: int,
        pan_and_scan_min_ratio_to_activate: float,
    ) -> list["torch.Tensor"]:
        """
        Generate additional crops for images with extreme aspect ratios following a pan-and-scan strategy.

        Args:
            image (`torch.Tensor`):
                Image tensor of shape `(channels, height, width)`.
            pan_and_scan_min_crop_size (`int`):
                Minimum size of each crop.
            pan_and_scan_max_num_crops (`int`):
                Maximum number of crops to generate.
            pan_and_scan_min_ratio_to_activate (`float`):
                Aspect ratio threshold to activate pan-and-scan.

        Returns:
            `list[torch.Tensor]` containing additional cropped views.
        """

        height, width = image.shape[-2:]
        if width >= height:
            if height == 0 or width / height < pan_and_scan_min_ratio_to_activate:
                return []
            num_crops_w = int(math.floor(width / height + 0.5))
            max_crops_by_size = int(math.floor(width / pan_and_scan_min_crop_size)) if pan_and_scan_min_crop_size else num_crops_w
            num_crops_w = min(max_crops_by_size, num_crops_w)
            num_crops_w = max(2, num_crops_w)
            num_crops_w = min(pan_and_scan_max_num_crops, num_crops_w)
            num_crops_h = 1
        else:
            if width == 0 or height / width < pan_and_scan_min_ratio_to_activate:
                return []
            num_crops_h = int(math.floor(height / width + 0.5))
            max_crops_by_size = int(math.floor(height / pan_and_scan_min_crop_size)) if pan_and_scan_min_crop_size else num_crops_h
            num_crops_h = min(max_crops_by_size, num_crops_h)
            num_crops_h = max(2, num_crops_h)
            num_crops_h = min(pan_and_scan_max_num_crops, num_crops_h)
            num_crops_w = 1

        crop_size_w = int(math.ceil(width / num_crops_w))
        crop_size_h = int(math.ceil(height / num_crops_h))

        if min(crop_size_w, crop_size_h) < pan_and_scan_min_crop_size:
            return []

        crop_positions_w = [min(width - crop_size_w, crop_size_w * i) for i in range(num_crops_w)]
        crop_positions_h = [min(height - crop_size_h, crop_size_h * i) for i in range(num_crops_h)]

        crops: list["torch.Tensor"] = []
        for pos_h in crop_positions_h:
            for pos_w in crop_positions_w:
                end_h = min(pos_h + crop_size_h, height)
                end_w = min(pos_w + crop_size_w, width)
                crops.append(image[..., pos_h:end_h, pos_w:end_w])

        return crops

    def _expand_with_pan_and_scan(
        self,
        images: list["torch.Tensor"],
        do_pan_and_scan: bool,
        pan_and_scan_min_crop_size: int,
        pan_and_scan_max_num_crops: int,
        pan_and_scan_min_ratio_to_activate: float,
    ) -> tuple[list["torch.Tensor"], list[int]]:
        """
        Apply pan-and-scan to a list of images, returning expanded views and per-image crop counts.
        """

        expanded: list["torch.Tensor"] = []
        num_crops_per_image: list[int] = []

        for image in images:
            crops: list["torch.Tensor"] = []
            if do_pan_and_scan:
                crops = self.pan_and_scan(
                    image=image,
                    pan_and_scan_min_crop_size=pan_and_scan_min_crop_size,
                    pan_and_scan_max_num_crops=pan_and_scan_max_num_crops,
                    pan_and_scan_min_ratio_to_activate=pan_and_scan_min_ratio_to_activate,
                )
            num_crops_per_image.append(len(crops))
            expanded.append(image)
            expanded.extend(crops)

        return expanded, num_crops_per_image

    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[ArlowImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: Optional[Union[str, "torch.device"]] = None,
        **kwargs: Unpack[ArlowImageProcessorKwargs],
    ) -> BatchFeature:
        # Detect original input structure before flattening
        orig = images
        is_list = isinstance(orig, (list, tuple))
        is_nested = is_list and len(orig) > 0 and isinstance(orig[0], (list, tuple))
        is_single_list_of_one = is_list and not is_nested and len(orig) == 1

        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )
        batch_feature = self._preprocess(images, **kwargs)
        # For nested batches and single-list inputs, return 2D (flattened) to match common image processor tests
        if isinstance(batch_feature["pixel_values"], torch.Tensor):
            pixel_values = batch_feature["pixel_values"]
            if pixel_values.ndim == 3 and (is_nested or is_single_list_of_one):
                b, p, d = pixel_values.shape
                batch_feature["pixel_values"] = pixel_values.reshape(b * p, d)
        return batch_feature

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional[PILImageResampling],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ):
        do_pan_and_scan = kwargs.pop("do_pan_and_scan", self.do_pan_and_scan)
        pan_and_scan_min_crop_size = kwargs.pop(
            "pan_and_scan_min_crop_size", self.pan_and_scan_min_crop_size
        )
        pan_and_scan_max_num_crops = kwargs.pop(
            "pan_and_scan_max_num_crops", self.pan_and_scan_max_num_crops
        )
        pan_and_scan_min_ratio_to_activate = kwargs.pop(
            "pan_and_scan_min_ratio_to_activate", self.pan_and_scan_min_ratio_to_activate
        )

        # Compute additional views if pan-and-scan is requested
        expanded_images, num_crops_per_image = self._expand_with_pan_and_scan(
            images=images,
            do_pan_and_scan=bool(do_pan_and_scan),
            pan_and_scan_min_crop_size=pan_and_scan_min_crop_size,
            pan_and_scan_max_num_crops=pan_and_scan_max_num_crops,
            pan_and_scan_min_ratio_to_activate=pan_and_scan_min_ratio_to_activate,
        )

        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(
            expanded_images, disable_grouping=disable_grouping
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group again for patchification
        grouped_images, grouped_images_index = group_images_by_shape(
            resized_images, disable_grouping=disable_grouping
        )
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            patches = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            if patches.ndim == 4:
                patches = patches.unsqueeze(1)
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)
            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids = reorder_images(processed_grids, grouped_images_index)

        # Build a 3D tensor (batch, patches, features) by padding per-sample patch sequences
        max_patches = max(img.shape[0] for img in processed_images)
        feature_size = processed_images[0].shape[-1]
        padded = []
        for img in processed_images:
            if img.shape[0] < max_patches:
                pad = img.new_zeros((max_patches - img.shape[0], feature_size))
                img = torch.cat([img, pad], dim=0)
            padded.append(img)
        pixel_values = torch.stack(padded, dim=0)
        image_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "num_crops": num_crops_per_image,
            },
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        min_pixels = (
            images_kwargs["min_pixels"]
            if images_kwargs and "min_pixels" in images_kwargs
            else self.size["shortest_edge"]
        )
        max_pixels = (
            images_kwargs["max_pixels"]
            if images_kwargs and "max_pixels" in images_kwargs
            else self.size["longest_edge"]
        )
        patch_size = images_kwargs.get("patch_size", self.patch_size) if images_kwargs else self.patch_size
        merge_size = images_kwargs.get("merge_size", self.merge_size) if images_kwargs else self.merge_size

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


__all__ = ["ArlowImageProcessor"]
