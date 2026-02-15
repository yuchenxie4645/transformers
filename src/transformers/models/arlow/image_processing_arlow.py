import math

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, resize, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, add_start_docstrings, logging


logger = logging.get_logger(__name__)


# Inspired by transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 28 * 28 * 1280
):
    """Rescales the image while enforcing divisibility and pixel bounds."""
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


def _prepare_float_image_for_resize(image: np.ndarray) -> np.ndarray:
    """Convert out-of-range float images to uint8 so PIL resize is well-defined."""
    if np.issubdtype(image.dtype, np.floating):
        min_value = float(np.min(image))
        max_value = float(np.max(image))
        if min_value < 0.0 or max_value > 1.0:
            logger.warning_once(
                "Received a floating-point image outside [0, 1] while resizing; clipping to [0, 255] before resize."
            )
            return np.clip(image, 0.0, 255.0).astype(np.uint8)
    return image


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
class ArlowImageProcessor(BaseImageProcessor):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
    do_rescale = True
    rescale_factor = 1 / 255
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

        self.do_resize = kwargs.pop("do_resize", self.do_resize)
        self.resample = kwargs.pop("resample", self.resample)
        self.do_rescale = kwargs.pop("do_rescale", self.do_rescale)
        self.rescale_factor = kwargs.pop("rescale_factor", self.rescale_factor)
        self.do_normalize = kwargs.pop("do_normalize", self.do_normalize)
        self.image_mean = kwargs.pop("image_mean", self.image_mean)
        self.image_std = kwargs.pop("image_std", self.image_std)
        self.do_convert_rgb = kwargs.pop("do_convert_rgb", self.do_convert_rgb)
        self.patch_size = kwargs.pop("patch_size", self.patch_size)
        self.temporal_patch_size = kwargs.pop("temporal_patch_size", self.temporal_patch_size)
        self.merge_size = kwargs.pop("merge_size", self.merge_size)
        self.do_pan_and_scan = kwargs.pop("do_pan_and_scan", self.do_pan_and_scan)
        self.pan_and_scan_min_crop_size = kwargs.pop("pan_and_scan_min_crop_size", self.pan_and_scan_min_crop_size)
        self.pan_and_scan_max_num_crops = kwargs.pop("pan_and_scan_max_num_crops", self.pan_and_scan_max_num_crops)
        self.pan_and_scan_min_ratio_to_activate = kwargs.pop(
            "pan_and_scan_min_ratio_to_activate", self.pan_and_scan_min_ratio_to_activate
        )

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

        if min_pixels is not None:
            merged_size["shortest_edge"] = min_pixels
            merged_size.pop("min_pixels", None)
        if max_pixels is not None:
            merged_size["longest_edge"] = max_pixels
            merged_size.pop("max_pixels", None)
        if "shortest_edge" not in merged_size or "longest_edge" not in merged_size:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        self.size = merged_size
        self.min_pixels = merged_size["shortest_edge"]
        self.max_pixels = merged_size["longest_edge"]
        super().__init__(**kwargs)

    @staticmethod
    def _flatten_unpadded_pixel_values(pixel_values, image_grid_thw):
        if (
            pixel_values is None
            or image_grid_thw is None
            or getattr(pixel_values, "ndim", None) != 3
            or getattr(image_grid_thw, "ndim", None) != 2
        ):
            return pixel_values

        grid_values = (
            image_grid_thw.tolist() if hasattr(image_grid_thw, "tolist") else np.asarray(image_grid_thw).tolist()
        )
        patch_counts = [int(grid_t * grid_h * grid_w) for grid_t, grid_h, grid_w in grid_values]
        total_patches = sum(patch_counts)
        feature_size = pixel_values.shape[-1]

        if isinstance(pixel_values, np.ndarray):
            flattened = np.zeros((total_patches, feature_size), dtype=pixel_values.dtype)
        else:
            flattened = pixel_values.new_zeros((total_patches, feature_size))

        offset = 0
        for image_idx, patch_count in enumerate(patch_counts):
            if patch_count > 0:
                flattened[offset : offset + patch_count] = pixel_values[image_idx, :patch_count]
            offset += patch_count

        return flattened

    def __call__(self, images: ImageInput, *args, **kwargs: Unpack[ArlowImageProcessorKwargs]) -> BatchFeature:
        batch_feature = self.preprocess(images, *args, **kwargs)
        if "pixel_values" in batch_feature and "image_grid_thw" in batch_feature:
            batch_feature["pixel_values"] = self._flatten_unpadded_pixel_values(
                batch_feature["pixel_values"], batch_feature["image_grid_thw"]
            )
        return batch_feature

    def pan_and_scan(
        self,
        image: np.ndarray,
        pan_and_scan_min_crop_size: int,
        pan_and_scan_max_num_crops: int,
        pan_and_scan_min_ratio_to_activate: float,
    ) -> list[np.ndarray]:
        """Generate additional crops for images with extreme aspect ratios."""
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

        crops: list[np.ndarray] = []
        for pos_h in crop_positions_h:
            for pos_w in crop_positions_w:
                end_h = min(pos_h + crop_size_h, height)
                end_w = min(pos_w + crop_size_w, width)
                crops.append(image[..., pos_h:end_h, pos_w:end_w])
        return crops

    def _expand_with_pan_and_scan(
        self,
        images: list[np.ndarray],
        do_pan_and_scan: bool,
        pan_and_scan_min_crop_size: int,
        pan_and_scan_max_num_crops: int,
        pan_and_scan_min_ratio_to_activate: float,
    ) -> tuple[list[np.ndarray], list[int]]:
        expanded: list[np.ndarray] = []
        num_crops_per_image: list[int] = []

        for image in images:
            crops: list[np.ndarray] = []
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

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: dict[str, int],
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        return_tensors: str | TensorType | None,
        do_convert_rgb: bool,
        input_data_format: str | ChannelDimension | None,
        do_pan_and_scan: bool,
        pan_and_scan_min_crop_size: int,
        pan_and_scan_max_num_crops: int,
        pan_and_scan_min_ratio_to_activate: float,
    ) -> BatchFeature:
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        images = [to_numpy_array(image) for image in images]
        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input "
                "images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])

        processed_images = []
        for image in images:
            height, width = get_image_size(image, channel_dim=input_data_format)
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                image = _prepare_float_image_for_resize(image)
                image = resize(
                    image=image,
                    size=(resized_height, resized_width),
                    resample=resample,
                    input_data_format=input_data_format,
                )

            if do_rescale:
                image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )
            image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_channel_dim=input_data_format)
            processed_images.append(image)

        expanded_images, num_crops_per_image = self._expand_with_pan_and_scan(
            images=processed_images,
            do_pan_and_scan=do_pan_and_scan,
            pan_and_scan_min_crop_size=pan_and_scan_min_crop_size,
            pan_and_scan_max_num_crops=pan_and_scan_max_num_crops,
            pan_and_scan_min_ratio_to_activate=pan_and_scan_min_ratio_to_activate,
        )

        processed_patches = []
        processed_grids = []
        for image in expanded_images:
            channel, resized_height, resized_width = image.shape
            alignment = patch_size * merge_size
            if resized_height % alignment != 0 or resized_width % alignment != 0:
                # Pan-and-scan crops can end up off-grid; realign to patch/merge multiples.
                aligned_height = max(alignment, round(resized_height / alignment) * alignment)
                aligned_width = max(alignment, round(resized_width / alignment) * alignment)
                image = _prepare_float_image_for_resize(image)
                image = resize(
                    image=image,
                    size=(aligned_height, aligned_width),
                    resample=resample,
                    input_data_format=ChannelDimension.FIRST,
                )
                channel, resized_height, resized_width = image.shape
            patches = image[np.newaxis]
            if patches.shape[0] % temporal_patch_size != 0:
                repeats = np.repeat(
                    patches[-1][np.newaxis], temporal_patch_size - (patches.shape[0] % temporal_patch_size), axis=0
                )
                patches = np.concatenate([patches, repeats], axis=0)

            grid_t = patches.shape[0] // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
            patches = patches.reshape(
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
            patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
            flatten_patches = patches.reshape(
                grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
            )
            processed_patches.append(flatten_patches)
            processed_grids.append([grid_t, grid_h, grid_w])

        if len(processed_patches) > 0:
            max_patches = max(patches.shape[0] for patches in processed_patches)
            feature_size = processed_patches[0].shape[-1]
            padded = []
            for patches in processed_patches:
                if patches.shape[0] < max_patches:
                    padding = np.zeros((max_patches - patches.shape[0], feature_size), dtype=patches.dtype)
                    patches = np.concatenate([patches, padding], axis=0)
                padded.append(patches)
            pixel_values = np.stack(padded, axis=0)
            image_grid_thw = np.asarray(processed_grids, dtype=np.int64)
        else:
            feature_size = 3 * temporal_patch_size * patch_size * patch_size
            pixel_values = np.zeros((0, 0, feature_size), dtype=np.float32)
            image_grid_thw = np.zeros((0, 3), dtype=np.int64)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw, "num_crops": num_crops_per_image},
            tensor_type=return_tensors,
        )

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool | None = None,
        size: dict[str, int] | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        resample: PILImageResampling | None = None,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        do_normalize: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        merge_size: int | None = None,
        disable_grouping: bool | None = None,
        do_convert_rgb: bool | None = None,
        do_pan_and_scan: bool | None = None,
        pan_and_scan_min_crop_size: int | None = None,
        pan_and_scan_max_num_crops: int | None = None,
        pan_and_scan_min_ratio_to_activate: float | None = None,
        return_tensors: str | TensorType | None = None,
        input_data_format: str | ChannelDimension | None = None,
    ) -> BatchFeature:
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        if size is not None:
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
            min_pixels = size["shortest_edge"]
            max_pixels = size["longest_edge"]
        elif min_pixels is not None and max_pixels is not None:
            size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        else:
            size = {**self.size}

        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        do_pan_and_scan = do_pan_and_scan if do_pan_and_scan is not None else self.do_pan_and_scan
        pan_and_scan_min_crop_size = (
            pan_and_scan_min_crop_size if pan_and_scan_min_crop_size is not None else self.pan_and_scan_min_crop_size
        )
        pan_and_scan_max_num_crops = (
            pan_and_scan_max_num_crops if pan_and_scan_max_num_crops is not None else self.pan_and_scan_max_num_crops
        )
        pan_and_scan_min_ratio_to_activate = (
            pan_and_scan_min_ratio_to_activate
            if pan_and_scan_min_ratio_to_activate is not None
            else self.pan_and_scan_min_ratio_to_activate
        )
        _ = disable_grouping

        images = make_flat_list_of_images(images)
        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be PIL.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
        )

        return self._preprocess(
            images=images,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            return_tensors=return_tensors,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            do_pan_and_scan=do_pan_and_scan,
            pan_and_scan_min_crop_size=pan_and_scan_min_crop_size,
            pan_and_scan_max_num_crops=pan_and_scan_max_num_crops,
            pan_and_scan_min_ratio_to_activate=pan_and_scan_min_ratio_to_activate,
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
