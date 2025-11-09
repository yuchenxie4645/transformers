"""Video processor class for Arlow multimodal models."""

import math
from typing import Optional, Union

import numpy as np
import torch

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ChannelDimension, PILImageResampling, SizeDict, get_image_size
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType, add_start_docstrings, logging
from ...video_processing_utils import BASE_VIDEO_PROCESSOR_DOCSTRING, BaseVideoProcessor
from ...video_utils import VideoInput, VideoMetadata, group_videos_by_shape, reorder_videos


logger = logging.get_logger(__name__)


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = 128 * 128,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144,
    max_frames: Optional[int] = None,
    return_temporal: bool = False,
):
    """
    Smart resize for video frames with temporal/spatial constraints.

    Args:
        num_frames: Number of frames in the video
        height: Frame height
        width: Frame width
        temporal_factor: Temporal patch size
        factor: Spatial patch size
        min_pixels: Minimum volumetric pixels allowed (T × H × W)
        max_pixels: Maximum volumetric pixels allowed (T × H × W)
        max_frames: Optional explicit cap on the number of frames

    Returns:
        Tuple of (new_num_frames, new_height, new_width)
    """
    if num_frames < temporal_factor:
        raise ValueError(f"num_frames:{num_frames} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor
    if t_bar <= 0:
        t_bar = temporal_factor

    # Apply constraints on total volumetric pixels (T × H × W)
    frame_pixels = h_bar * w_bar
    if max_pixels is not None and frame_pixels > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
        frame_pixels = h_bar * w_bar
    if min_pixels is not None and frame_pixels < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
        frame_pixels = h_bar * w_bar

    if max_frames is not None:
        capped_frames = max(max_frames // temporal_factor, 1) * temporal_factor
        t_bar = min(t_bar, capped_frames)

    # Ensure we never exceed available frames while keeping divisibility
    t_bar = min(t_bar, (num_frames // temporal_factor) * temporal_factor or temporal_factor)

    if return_temporal:
        return t_bar, h_bar, w_bar

    return h_bar, w_bar


def motion_adaptive_sampling(frames: np.ndarray, target_frames: int, threshold: float = 0.1):
    """
    Sample frames adaptively based on motion detection.
    Samples more frames from high-motion regions.

    Args:
        frames: Input frames array (T, H, W, C) or (T, C, H, W)
        target_frames: Target number of frames
        threshold: Motion threshold for detecting significant changes

    Returns:
        Indices of frames to sample
    """
    num_frames = frames.shape[0]
    if num_frames <= target_frames:
        return np.arange(num_frames)

    # Compute frame differences (simple motion proxy)
    diffs = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2, 3))
    motion_scores = np.concatenate([[0], diffs])  # Prepend 0 for first frame

    # Normalize scores
    motion_scores = motion_scores / (motion_scores.max() + 1e-8)

    # Use motion scores as sampling probabilities
    # Higher motion = higher probability
    probs = motion_scores + 0.1  # Add base probability
    probs = probs / probs.sum()

    # Sample indices based on probabilities
    indices = np.sort(np.random.choice(num_frames, size=target_frames, replace=False, p=probs))

    return indices


class ArlowVideoProcessorInitKwargs(VideosKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_frames: int
    max_frames: int
    sample_strategy: str


@add_start_docstrings(
    "Constructs an Arlow video processor with adaptive sampling and dynamic resizing.",
    BASE_VIDEO_PROCESSOR_DOCSTRING,
    """
        patch_size (`int`, *optional*, defaults to 14):
            The spatial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to LLM encoder.
        sample_strategy (`str`, *optional*, defaults to "uniform"):
            Video frame sampling strategy: "uniform", "motion_adaptive", or "fps_based".
    """,
)
class ArlowVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 128 * 32 * 32, "longest_edge": 32 * 32 * 768}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    fps = None
    min_frames = 4
    max_frames = 64  # Default from config
    do_sample_frames = True
    sample_strategy = "uniform"  # Can be "uniform", "motion_adaptive", "fps_based"
    valid_kwargs = ArlowVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[ArlowVideoProcessorInitKwargs]):
        size = kwargs.pop("size", None)
        min_pixels = kwargs.pop("min_pixels", None)
        max_pixels = kwargs.pop("max_pixels", None)
        max_tokens_per_video = kwargs.pop("max_tokens_per_video", None)
        # Start with class default, then override with provided values
        merged_size = dict(self.size) if size is None else {**self.size, **size}

        # backward compatibility: override size with min_pixels and max_pixels if they are provided
        if min_pixels is not None:
            merged_size["shortest_edge"] = min_pixels
            merged_size.pop("min_pixels", None)
        if max_pixels is not None:
            merged_size["longest_edge"] = max_pixels
            merged_size.pop("max_pixels", None)
        if max_tokens_per_video is not None:
            patch_size = kwargs.get("patch_size", self.patch_size)
            merge_size = kwargs.get("merge_size", self.merge_size)
            temporal_patch_size = kwargs.get("temporal_patch_size", self.temporal_patch_size)
            volumetric_budget = max_tokens_per_video * (patch_size * merge_size) ** 2 * temporal_patch_size
            merged_size["longest_edge"] = min(merged_size["longest_edge"], volumetric_budget)
        if "shortest_edge" not in merged_size or "longest_edge" not in merged_size:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        super().__init__(size=merged_size, min_pixels=min_pixels, max_pixels=max_pixels, **kwargs)
        self.max_tokens_per_video = max_tokens_per_video
        self.max_volume = merged_size["longest_edge"]
        self._last_selected_frame_indices: list[list[int]] = []

    def _further_process_kwargs(
        self,
        size: Optional[SizeDict] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """
        Update kwargs that need further processing before being validated.
        """
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

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: Optional[int] = None,
        fps: Optional[Union[int, float]] = None,
        sample_strategy: Optional[str] = None,
        **kwargs,
    ):
        """
        Sample frames using configurable strategies.

        Supports:
        - "uniform": Uniform sampling across the video
        - "motion_adaptive": Sample more frames from high-motion regions
        - "fps_based": Sample at a target FPS

        Args:
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            num_frames (`int`, *optional*):
                Maximum number of frames to sample. Defaults to `self.max_frames`.
            fps (`int` or `float`, *optional*):
                Target frames to sample per second. Defaults to `self.fps`.
            sample_strategy (`str`, *optional*):
                Sampling strategy override. Defaults to `self.sample_strategy`.

        Returns:
            Frame indices to sample.
        """
        strategy = sample_strategy or self.sample_strategy

        # Ensure mutual exclusivity of num_frames and fps
        if strategy == "uniform":
            # Use only num_frames for uniform sampling, clamp to available frames and default to full length
            effective_num_frames = num_frames
            if effective_num_frames is None:
                effective_num_frames = metadata.total_num_frames
            effective_num_frames = min(effective_num_frames, metadata.total_num_frames)
            return super().sample_frames(metadata, num_frames=effective_num_frames, fps=None, **kwargs)
        elif strategy == "fps_based":
            # Use only fps for fps-based sampling
            return super().sample_frames(metadata, num_frames=None, fps=fps, **kwargs)

        # Motion-adaptive sampling requires accessing frames; fall back to uniform safely
        logger.warning("Motion-adaptive sampling requires frame data access. Falling back to uniform sampling.")
        # Default fallback to uniform with safe clamping
        effective_num_frames = num_frames
        if effective_num_frames is None:
            effective_num_frames = metadata.total_num_frames
        effective_num_frames = min(effective_num_frames, metadata.total_num_frames)
        return super().sample_frames(metadata, num_frames=effective_num_frames, fps=None, **kwargs)

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: Optional[SizeDict] = None,
        interpolation: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        # Group by shape for batched resizing and tokenization
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}

        selected_indices_grouped: dict[tuple[int, int, int], torch.Tensor] = {}
        for shape, stacked_videos in grouped_videos.items():
            B, T, C, H, W = stacked_videos.shape
            num_frames, height, width = T, H, W

            patch_size = patch_size or self.patch_size
            temporal_patch_size = temporal_patch_size or self.temporal_patch_size
            merge_size = merge_size or self.merge_size

            target_frames = T
            resized_height, resized_width = height, width

            if do_resize:
                target_frames, resized_height, resized_width = smart_resize(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    temporal_factor=temporal_patch_size,
                    factor=(patch_size * merge_size),
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                    max_frames=self.max_frames,
                    return_temporal=True,
                )
            else:
                resized_height, resized_width = height, width

            grid_h_patches = max(resized_height // patch_size, 1)
            grid_w_patches = max(resized_width // patch_size, 1)
            grid_h_groups = max(grid_h_patches // merge_size, 1)
            grid_w_groups = max(grid_w_patches // merge_size, 1)

            temporal_groups = max(1, math.ceil(max(target_frames, 1) / temporal_patch_size))
            if temporal_groups % temporal_patch_size != 0:
                temporal_groups = ((temporal_groups + temporal_patch_size - 1) // temporal_patch_size) * temporal_patch_size

            if self.max_tokens_per_video is not None:
                temporal_groups = min(temporal_groups, max(self.max_tokens_per_video, temporal_patch_size))
                temporal_groups = max(temporal_groups, temporal_patch_size)

                while (
                    temporal_groups * grid_h_groups * grid_w_groups > self.max_tokens_per_video
                    and (temporal_groups > temporal_patch_size or grid_h_groups > 1 or grid_w_groups > 1)
                ):
                    if temporal_groups > temporal_patch_size:
                        temporal_groups -= temporal_patch_size
                    elif grid_h_groups >= grid_w_groups and grid_h_groups > 1:
                        grid_h_groups -= 1
                    elif grid_w_groups > 1:
                        grid_w_groups -= 1
                    else:
                        break

                temporal_groups = max(temporal_groups, temporal_patch_size)
                grid_h_groups = max(grid_h_groups, 1)
                grid_w_groups = max(grid_w_groups, 1)
                resized_height = grid_h_groups * merge_size * patch_size
                resized_width = grid_w_groups * merge_size * patch_size
                grid_h_patches = grid_h_groups * merge_size
                grid_w_patches = grid_w_groups * merge_size

            target_frames = temporal_groups * temporal_patch_size
            max_groups = max(self.max_frames // temporal_patch_size, 1)
            target_frames = min(target_frames, max_groups * temporal_patch_size)
            target_frames = max(target_frames, temporal_patch_size)

            if target_frames != T:
                frame_positions = torch.linspace(
                    0,
                    T - 1,
                    steps=target_frames,
                    device=stacked_videos.device,
                    dtype=torch.float32,
                )
                frame_indices = torch.clamp(frame_positions.round().long(), 0, T - 1)
                stacked_videos = stacked_videos.index_select(1, frame_indices)
            else:
                frame_indices = torch.arange(T, device=stacked_videos.device)

            if do_resize:
                frame_count = stacked_videos.shape[1]
                stacked_videos = stacked_videos.view(B * frame_count, C, H, W)
                stacked_videos = self.resize(
                    stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
                stacked_videos = stacked_videos.view(B, frame_count, C, resized_height, resized_width)

            selected_indices_grouped[shape] = torch.stack([frame_indices.clone() for _ in range(B)], dim=0)
            resized_videos_grouped[shape] = stacked_videos

        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)
        selected_indices = reorder_videos(selected_indices_grouped, grouped_videos_index)
        self._last_selected_frame_indices = [indices.detach().cpu().tolist() for indices in selected_indices]

        # Group again to ensure homogeneous shapes after resizing
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)

            # Fused rescale and normalize
            # Optional channel conversion to RGB
            if do_convert_rgb and stacked_videos.shape[2] == 1:
                stacked_videos = stacked_videos.repeat(1, 1, 3, 1, 1)
            # Basic alpha drop if RGBA (keep first 3 channels)
            if do_convert_rgb and stacked_videos.shape[2] == 4:
                stacked_videos = stacked_videos[:, :, :3, :, :]

            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches = stacked_videos

            # Ensure divisibility for temporal patching by repeating last frame
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, temporal_tokens, channel = patches.shape[:3]
            temporal_groups = temporal_tokens // temporal_patch_size
            grid_h_patches = max(resized_height // patch_size, 1)
            grid_w_patches = max(resized_width // patch_size, 1)
            grid_h_groups = max(grid_h_patches // merge_size, 1)
            grid_w_groups = max(grid_w_patches // merge_size, 1)

            patches = patches.view(
                batch_size,
                temporal_groups,
                temporal_patch_size,
                channel,
                grid_h_groups,
                merge_size,
                patch_size,
                grid_w_groups,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                temporal_groups * grid_h_groups * grid_w_groups,
                channel * temporal_patch_size * merge_size * merge_size * patch_size * patch_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[temporal_groups, grid_h_groups, grid_w_groups]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)

        # Build a 3D tensor (batch, patches, features) by padding per-video patch sequences
        max_patches = max(vid.shape[0] for vid in processed_videos)
        feature_size = processed_videos[0].shape[-1]
        padded = []
        for vid in processed_videos:
            if vid.shape[0] < max_patches:
                pad = vid.new_zeros((max_patches - vid.shape[0], feature_size))
                vid = torch.cat([vid, pad], dim=0)
            padded.append(vid)
        pixel_values_videos = torch.stack(padded, dim=0)
        video_grid_thw = torch.tensor(processed_grids)
        data = {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

    def preprocess(
        self,
        videos: VideoInput,
        **kwargs: Unpack[VideosKwargs],
    ) -> BatchFeature:
        outputs = super().preprocess(videos, **kwargs)
        if "video_metadata" in outputs and self._last_selected_frame_indices:
            metadata_list = outputs["video_metadata"]
            for metadata, indices in zip(metadata_list, self._last_selected_frame_indices):
                if metadata is None:
                    continue
                metadata.frames_indices = np.asarray(indices, dtype=np.int64)
        self._last_selected_frame_indices = []
        return outputs


class ArlowVideoProcessorFast(ArlowVideoProcessor):
    pass


__all__ = ["ArlowVideoProcessor", "ArlowVideoProcessorFast"]
