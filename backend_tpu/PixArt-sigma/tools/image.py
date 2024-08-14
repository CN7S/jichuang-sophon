import numpy as np
from PIL import Image
import cv2

def make_grid_np(
    tensor: np.ndarray,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: tuple = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
) -> np.ndarray:
    # If input is a list of ndarrays, stack them along the batch dimension
    if isinstance(tensor, list):
        tensor = np.stack(tensor, axis=0)

    # Ensure tensor has shape (B, C, H, W)
    if tensor.ndim == 2:  # single image H x W
        tensor = tensor[np.newaxis, np.newaxis, ...]
    elif tensor.ndim == 3:  # single image with channels
        tensor = tensor[np.newaxis, ...]
    elif tensor.ndim == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = np.concatenate((tensor, tensor, tensor), axis=1)

    # Normalize if specified
    if normalize:
        tensor = tensor.copy()  # avoid modifying tensor in-place

        def norm_ip(img, low, high):
            np.clip(img, low, high, out=img)
            img -= low
            img /= max(high - low, 1e-5)

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(np.min(t)), float(np.max(t)))

        if scale_each:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = tensor.shape[2] + padding, tensor.shape[3] + padding
    num_channels = tensor.shape[1]
    grid_shape = (num_channels, height * ymaps + padding, width * xmaps + padding)
    grid = np.full(grid_shape, pad_value, dtype=tensor.dtype)

    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[
                :,
                y * height + padding : (y + 1) * height,
                x * width + padding : (x + 1) * width,
            ] = tensor[k]
            k += 1

    return grid


def save_image(
    tensor,
    fp,
    format = None,
    **kwargs,
) -> None:
    grid = make_grid_np(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = np.clip((grid * 255 + 0.5), a_min = 0, a_max = 255)
    ndarr = np.transpose(ndarr, (1, 2, 0))
    ndarr = ndarr.astype(np.uint8)
    
    # cv2.imwrite(fp, ndarr)
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)