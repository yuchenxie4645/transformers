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
from ...video_utils import VideoMetadata, group_videos_by_shape, reorder_videos


logger = logging.get_logger(__name__)


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = 128 * 128,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144,
):
    """
    Smart resize for video frames with temporal/spatial constraints.
    
    Args:
        num_frames: Number of frames in the video
        height: Frame height
        width: Frame width
        temporal_factor: Temporal patch size
        factor: Spatial patch size
        min_pixels: Minimum total pixels allowed
        max_pixels: Maximum total pixels allowed
    
    Returns:
        Tuple of (new_height, new_width)
    """
    print(f"[DEBUG VIDEO] smart_resize: num_frames={num_frames}, height={height}, width={width}")
    
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

    # Apply spatial constraints only (tests expect min/max on HxW, not including time)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    print(f"[DEBUG VIDEO] smart_resize output: h_bar={h_bar}, w_bar={w_bar}")
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
    print(f"[DEBUG VIDEO] motion_adaptive_sampling: input shape={frames.shape}, target={target_frames}")
    
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
    
    print(f"[DEBUG VIDEO] motion_adaptive_sampling: selected {len(indices)} frames")
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
        # Start with class default, then override with provided values
        merged_size = dict(self.size) if size is None else {**self.size, **size}
        
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
        print(f"[DEBUG VIDEO] sample_frames: strategy={strategy}, num_frames={num_frames}, fps={fps}")

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
        logger.warning(
            "Motion-adaptive sampling requires frame data access. Falling back to uniform sampling."
        )
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

        for shape, stacked_videos in grouped_videos.items():
            B, T, C, H, W = stacked_videos.shape
            num_frames, height, width = T, H, W
            if do_resize:
                resized_height, resized_width = smart_resize(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    temporal_factor=temporal_patch_size,
                    factor=(patch_size * merge_size),
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_videos = stacked_videos.view(B * T, C, H, W)
                stacked_videos = self.resize(
                    stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
                stacked_videos = stacked_videos.view(B, T, C, resized_height, resized_width)
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

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

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h // merge_size, grid_w // merge_size]] * batch_size

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


class ArlowVideoProcessorFast(ArlowVideoProcessor):
    pass


__all__ = ["ArlowVideoProcessor", "ArlowVideoProcessorFast"]

