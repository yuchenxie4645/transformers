# Copyright 2026 Arlow Constellations. All rights reserved.
# Written and edited by IRIS.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring, is_torchvision_available
from .image_processing_pil_arlow import ArlowImageProcessorKwargs, _ArlowImageProcessorMixin, smart_resize


if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF


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
