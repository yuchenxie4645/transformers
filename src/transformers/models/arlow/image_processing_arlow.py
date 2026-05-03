import math
from collections.abc import Iterable

import numpy as np
import torch

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, ImageInput, PILImageResampling, SizeDict
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, is_torchvision_available


if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 28 * 28 * 1280
):
    """Rescale the image while enforcing divisibility and pixel bounds."""
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
    r"""
    min_pixels (`int`, *optional*, defaults to `56 * 56`):
        Lower bound for the resized image area.
    max_pixels (`int`, *optional*, defaults to `28 * 28 * 1280`):
        Upper bound for the resized image area.
    patch_size (`int`, *optional*, defaults to 14):
        Spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        Temporal patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        Merge size used to collapse vision patches before the language model.
    do_pan_and_scan (`bool`, *optional*, defaults to `False`):
        Whether to create extra crops for extreme aspect ratios.
    pan_and_scan_min_crop_size (`int`, *optional*, defaults to 256):
        Smallest crop edge allowed during pan-and-scan.
    pan_and_scan_max_num_crops (`int`, *optional*, defaults to 4):
        Maximum number of extra crops to add per image.
    pan_and_scan_min_ratio_to_activate (`float`, *optional*, defaults to 1.6):
        Minimum aspect ratio required to activate pan-and-scan.
    """

    min_pixels: int
    max_pixels: int
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    do_pan_and_scan: bool
    pan_and_scan_min_crop_size: int
    pan_and_scan_max_num_crops: int
    pan_and_scan_min_ratio_to_activate: float


class _ArlowImageProcessorMixin:
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
    default_to_square = False
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

        size = dict(self.size) if size is None else size
        if isinstance(size, SizeDict):
            size = dict(size)
        elif not isinstance(size, dict):
            size = get_size_dict(size=size, default_to_square=False, param_name="size")
        else:
            size = {key: value for key, value in size.items() if value is not None}

        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
            size.pop("min_pixels", None)
        if max_pixels is not None:
            size["longest_edge"] = max_pixels
            size.pop("max_pixels", None)
        if "shortest_edge" not in size or "longest_edge" not in size:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        super().__init__(
            size=size,
            min_pixels=size["shortest_edge"],
            max_pixels=size["longest_edge"],
            **kwargs,
        )

    def _standardize_kwargs(
        self,
        size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        **kwargs,
    ) -> dict:
        if min_pixels is not None and max_pixels is not None:
            size = SizeDict(shortest_edge=min_pixels, longest_edge=max_pixels)
        kwargs = super()._standardize_kwargs(size=size, **kwargs)
        size = kwargs.get("size", self.size)
        if not size.shortest_edge or not size.longest_edge:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        kwargs["min_pixels"] = size.shortest_edge
        kwargs["max_pixels"] = size.longest_edge
        return kwargs

    @staticmethod
    def _flatten_unpadded_pixel_values(pixel_values, image_grid_thw):
        if (
            pixel_values is None
            or image_grid_thw is None
            or getattr(pixel_values, "ndim", None) != 3
            or getattr(image_grid_thw, "ndim", None) != 2
            or image_grid_thw.shape[-1] != 3
        ):
            return pixel_values

        if isinstance(image_grid_thw, torch.Tensor):
            patch_counts = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()
        else:
            patch_counts = [int(grid_t * grid_h * grid_w) for grid_t, grid_h, grid_w in image_grid_thw.tolist()]

        total_patches = sum(int(count) for count in patch_counts)
        feature_size = pixel_values.shape[-1]

        if isinstance(pixel_values, np.ndarray):
            flattened = np.zeros((total_patches, feature_size), dtype=pixel_values.dtype)
        else:
            flattened = pixel_values.new_zeros((total_patches, feature_size))

        offset = 0
        for image_idx, patch_count in enumerate(patch_counts):
            count = int(patch_count)
            if count > 0:
                flattened[offset : offset + count] = pixel_values[image_idx, :count]
            offset += count

        return flattened

    def __call__(self, images: ImageInput, *args, **kwargs: Unpack[ArlowImageProcessorKwargs]) -> BatchFeature:
        batch_feature = self.preprocess(images, *args, **kwargs)
        return batch_feature

    def preprocess(self, images: ImageInput, *args, **kwargs: Unpack[ArlowImageProcessorKwargs]) -> BatchFeature:
        batch_feature = super().preprocess(images, *args, **kwargs)
        if "pixel_values" in batch_feature and "image_grid_thw" in batch_feature:
            batch_feature["pixel_values"] = self._flatten_unpadded_pixel_values(
                batch_feature["pixel_values"], batch_feature["image_grid_thw"]
            )
        return batch_feature

    def pan_and_scan(
        self,
        image,
        pan_and_scan_min_crop_size: int,
        pan_and_scan_max_num_crops: int,
        pan_and_scan_min_ratio_to_activate: float,
    ) -> list:
        height, width = image.shape[-2:]
        if width >= height:
            if height == 0 or width / height < pan_and_scan_min_ratio_to_activate:
                return []
            num_crops_w = int(math.floor(width / height + 0.5))
            max_crops_by_size = (
                int(math.floor(width / pan_and_scan_min_crop_size)) if pan_and_scan_min_crop_size else num_crops_w
            )
            num_crops_w = min(max_crops_by_size, num_crops_w)
            num_crops_w = max(2, num_crops_w)
            num_crops_w = min(pan_and_scan_max_num_crops, num_crops_w)
            num_crops_h = 1
        else:
            if width == 0 or height / width < pan_and_scan_min_ratio_to_activate:
                return []
            num_crops_h = int(math.floor(height / width + 0.5))
            max_crops_by_size = (
                int(math.floor(height / pan_and_scan_min_crop_size)) if pan_and_scan_min_crop_size else num_crops_h
            )
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

        crops = []
        for pos_h in crop_positions_h:
            for pos_w in crop_positions_w:
                end_h = min(pos_h + crop_size_h, height)
                end_w = min(pos_w + crop_size_w, width)
                crops.append(image[..., pos_h:end_h, pos_w:end_w])
        return crops

    def _expand_with_pan_and_scan(
        self,
        images: list,
        do_pan_and_scan: bool,
        pan_and_scan_min_crop_size: int,
        pan_and_scan_max_num_crops: int,
        pan_and_scan_min_ratio_to_activate: float,
    ) -> tuple[list, list[int]]:
        expanded = []
        num_crops_per_image = []

        for image in images:
            crops = []
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

        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


@auto_docstring
class ArlowImageProcessor(_ArlowImageProcessorMixin, TorchvisionBackend):
    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[ArlowImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
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
        expanded_images, num_crops_per_image = self._expand_with_pan_and_scan(
            images=images,
            do_pan_and_scan=do_pan_and_scan,
            pan_and_scan_min_crop_size=pan_and_scan_min_crop_size,
            pan_and_scan_max_num_crops=pan_and_scan_max_num_crops,
            pan_and_scan_min_ratio_to_activate=pan_and_scan_min_ratio_to_activate,
        )

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
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
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
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - (patches.shape[1] % temporal_patch_size), 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)
            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h,
                patch_size,
                grid_w,
                patch_size,
            )
            # Row-major token order: T, H, W. Feature order: C, temporal_patch, patch_h, patch_w.
            patches = patches.permute(0, 1, 4, 6, 3, 2, 5, 7).contiguous()
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids = reorder_images(processed_grids, grouped_images_index)

        if len(processed_images) > 0:
            max_patches = max(image.shape[0] for image in processed_images)
            feature_size = processed_images[0].shape[-1]
            padded = []
            for image in processed_images:
                if image.shape[0] < max_patches:
                    padding = image.new_zeros((max_patches - image.shape[0], feature_size))
                    image = torch.cat([image, padding], dim=0)
                padded.append(image)
            pixel_values = torch.stack(padded, dim=0)
            image_grid_thw = torch.tensor(processed_grids, dtype=torch.long)
        else:
            feature_size = 3 * temporal_patch_size * patch_size * patch_size
            pixel_values = torch.zeros((0, 0, feature_size), dtype=torch.float32)
            image_grid_thw = torch.zeros((0, 3), dtype=torch.long)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw, "num_crops": num_crops_per_image},
            tensor_type=return_tensors,
        )


__all__ = ["ArlowImageProcessor"]
