"""
Shared utility functions for DINOv3 EM embedding notebooks (Task 2 & 3).
"""

import numpy as np
import torch
from PIL import Image
from skimage.transform import resize as sk_resize


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def normalize_uint8(arr):
    """Contrast-stretch a 2D array to uint8 [0, 255] via 1/99th percentile."""
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    a = np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)
    return (a * 255).astype(np.uint8)


def center_crop(pil_img, size):
    """Crop a square of `size` × `size` pixels from the centre of pil_img.

    Raises ValueError if size > min(width, height).
    """
    w, h = pil_img.size
    if size > min(w, h):
        raise ValueError(
            f"crop size {size} is larger than the shortest image dimension "
            f"(w={w}, h={h})."
        )
    left = (w - size) // 2
    top  = (h - size) // 2
    return pil_img.crop((left, top, left + size, top + size))


def em_to_rgb(slice_2d, scale=1.0, center_crop_size=None):
    """Convert a 2D EM slice to a uint8 RGB PIL Image.

    Args:
        slice_2d        : 2D numpy array (H, W)
        scale           : resize factor (>1 upsample, <1 downsample, 1.0 = no-op)
        center_crop_size: if not None, crop a square of this size after scaling
    """
    gray = normalize_uint8(slice_2d)
    rgb  = np.stack([gray] * 3, axis=-1)
    img  = Image.fromarray(rgb)
    if scale != 1.0:
        new_w = max(1, round(img.width  * scale))
        new_h = max(1, round(img.height * scale))
        resample = Image.BICUBIC if scale > 1.0 else Image.LANCZOS
        img = img.resize((new_w, new_h), resample=resample)
    if center_crop_size is not None:
        img = center_crop(img, center_crop_size)
    return img


def pad_to_patch_multiple(pil_img, patch_size=16):
    """Pad image so that width and height are multiples of patch_size."""
    w, h = pil_img.size
    new_w = ((w + patch_size - 1) // patch_size) * patch_size
    new_h = ((h + patch_size - 1) // patch_size) * patch_size
    if new_w == w and new_h == h:
        return pil_img
    padded = Image.new(pil_img.mode, (new_w, new_h), 0)
    padded.paste(pil_img, (0, 0))
    return padded


def prepare_mito_for_input(mito_slice, em_slice, scale=1.0, center_crop_size=None):
    """Align a mito mask to the model input space (same transforms as em_to_rgb).

    Steps:
      1. Resize mito to full EM spatial dimensions
      2. Apply scale
      3. Apply center crop (if requested)

    Returns a boolean (H', W') mask matching the model input.
    """
    em_h, em_w = em_slice.shape
    mask = (mito_slice > 0).astype(np.float32)
    if mask.shape != (em_h, em_w):
        mask = sk_resize(mask, (em_h, em_w), order=0, anti_aliasing=False)
    if scale != 1.0:
        new_h = max(1, round(em_h * scale))
        new_w = max(1, round(em_w * scale))
        mask = sk_resize(mask, (new_h, new_w), order=0, anti_aliasing=False)
    if center_crop_size is not None:
        h, w = mask.shape
        top  = (h - center_crop_size) // 2
        left = (w - center_crop_size) // 2
        mask = mask[top:top + center_crop_size, left:left + center_crop_size]
    return mask > 0.5


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def minmax(x):
    """Normalise array to [0, 1]."""
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def make_overlay(em_img, mito_mask, alpha=0.4, color=None):
    """Blend a binary mito mask onto an RGB uint8 image.

    Args:
        em_img    : (H, W, 3) uint8 numpy array
        mito_mask : (H, W) bool array
        alpha     : blend weight for the overlay colour
        color     : RGB colour as array [r, g, b] in [0, 1]; default red
    """
    if color is None:
        color = np.array([1.0, 0.2, 0.2])
    rgb = em_img.astype(np.float32) / 255.0
    rgb[mito_mask] = (1 - alpha) * rgb[mito_mask] + alpha * np.array(color)
    return rgb


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def zscore(tokens):
    """Z-score each token across its embedding dimension (approximates LayerNorm)."""
    mean = tokens.mean(axis=-1, keepdims=True)
    std  = tokens.std(axis=-1, keepdims=True)
    return (tokens - mean) / (std + 1e-8)


@torch.no_grad()
def get_embeddings(pil_img, model, processor, device,
                   patch_size, layer_idx, num_register_tokens):
    """Run a PIL image through DINOv3 and return CLS + patch embeddings.

    Args:
        pil_img             : PIL Image (will be padded to patch multiple)
        model               : HuggingFace DINOv3 model
        processor           : HuggingFace AutoImageProcessor
        device              : torch device string
        patch_size          : model patch size in pixels
        layer_idx           : hidden_states index to extract (0=patch embed, 1–12=blocks)
        num_register_tokens : number of register tokens (from model.config)

    Returns:
        cls_token   : (D,)         z-scored CLS embedding
        patch_tokens: (nh, nw, D)  z-scored patch embeddings
    """
    pil_img = pad_to_patch_multiple(pil_img, patch_size)
    w, h = pil_img.size
    nh, nw = h // patch_size, w // patch_size

    inputs = processor(
        images=pil_img,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).to(device)

    outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer_idx].cpu().numpy()   # (1, 1 + R + N, D)

    cls_token    = hidden[0, 0]                                    # (D,)
    patch_tokens = hidden[0, 1 + num_register_tokens:]             # (N, D)
    patch_tokens = patch_tokens.reshape(nh, nw, -1)                # (nh, nw, D)

    return cls_token, patch_tokens


def upsample_patch_embeddings(patch_grid, target_h, target_w):
    """Bilinearly upsample patch embeddings to pixel resolution.

    Args:
        patch_grid : (nh, nw, D)
        target_h   : target height in pixels
        target_w   : target width  in pixels

    Returns:
        (target_h, target_w, D) dense pixel embeddings
    """
    nh, nw, D = patch_grid.shape
    t = torch.from_numpy(patch_grid).permute(2, 0, 1).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(
        t, size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    return t.squeeze(0).permute(1, 2, 0).numpy()   # (target_h, target_w, D)
