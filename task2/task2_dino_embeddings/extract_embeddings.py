"""
Extract patch-level DINOv3 embeddings from an EM .npy file.

For each z-slice in the input file, the script:
  1. Converts the raw EM slice to RGB (with optional rescaling)
  2. Runs DINOv3 and extracts patch embeddings from the requested layer
  3. Saves the raw patch grid — NOT upsampled — together with the target
     image size needed to upsample later

Upsampling to pixel resolution is intentionally deferred so the saved file
stays small.  To get dense (H, W, D) embeddings after loading:

    import numpy as np
    from utils import upsample_patch_embeddings

    data      = np.load("..._patches.npz")
    patches   = data["patches"]    # (N, nh, nw, D)
    target_hw = data["target_hw"]  # (N, 2)  — original [H, W] before padding
    padded_hw = data["padded_hw"]  # (N, 2)  — [H, W] after patch-multiple padding

    # Upsample to padded size first, then crop to original size to avoid
    # edge artefacts from the zero-padding region.
    dense_grids = [
        upsample_patch_embeddings(patches[i], *padded_hw[i])[:target_hw[i,0], :target_hw[i,1]]
        for i in range(len(patches))
    ]

Output filename:
    <out_dir>/<stem>_imgscale<IMG_SCALE>_layer<LAYER_IDX>_patches.npz

Usage:
    python extract_embeddings.py jrc_mus-kidney_em_s3_10slices.npy
    python extract_embeddings.py jrc_mus-kidney_em_s3_10slices.npy --img-scale 2.0 --layer-idx 9
    python extract_embeddings.py jrc_mus-kidney_em_s3_10slices.npy --img-scale 1.0 --layer-idx 6 --model facebook/dinov3-vitb16-pretrain-lvd1689m
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

# ── shared utilities (task2/shared/utils.py) ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from utils import em_to_rgb, get_embeddings, pad_to_patch_multiple


# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL    = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_SCALE    = 1.0
DEFAULT_LAYER    = 9
DEFAULT_PATCH_SZ = 16
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract patch-level DINOv3 embeddings from an EM .npy file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "em_file",
        metavar="EM_FILE",
        help="Filename (or full path) of the input EM .npy file. "
             "If no directory is given, --data-dir is prepended.",
    )
    p.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        metavar="DIR",
        help=f"Directory containing EM .npy files (default: {DEFAULT_DATA_DIR}). "
             "Ignored when EM_FILE already contains a directory component.",
    )
    p.add_argument(
        "--img-scale",
        type=float,
        default=DEFAULT_SCALE,
        metavar="S",
        help=f"Resize factor applied before the model (default: {DEFAULT_SCALE}). "
             ">1 upsamples (more patches), <1 downsamples.",
    )
    p.add_argument(
        "--layer-idx",
        type=int,
        default=DEFAULT_LAYER,
        metavar="L",
        help=f"Hidden-state index to extract (0=patch embed, 1–12=blocks; default: {DEFAULT_LAYER}).",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="MODEL_ID",
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SZ,
        metavar="P",
        help=f"Model patch size in pixels (default: {DEFAULT_PATCH_SZ}).",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        metavar="DIR",
        help="Output directory (default: data/embeddings/).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    em_file = Path(args.em_file)
    if em_file.parent == Path("."):          # no directory given — prepend data-dir
        em_file = Path(args.data_dir) / em_file
    em_path = em_file.resolve()
    if not em_path.exists():
        sys.exit(f"ERROR: file not found — {em_path}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path(__file__).parent.parent / "data" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build output filename
    scale_str = f"{args.img_scale:g}".replace(".", "p")   # e.g. 2.0 → "2", 1.5 → "1p5"
    out_name  = f"{em_path.stem}_imgscale{scale_str}_layer{args.layer_idx}_patches.npz"
    out_path  = out_dir / out_name

    print(f"Input  : {em_path}")
    print(f"Output : {out_path}")
    print(f"Model  : {args.model}")
    print(f"Layer  : {args.layer_idx}  (0=patch embed, 1-12=transformer blocks)")
    print(f"Scale  : {args.img_scale}x")
    print()

    # ── load EM data ──────────────────────────────────────────────────────────
    em = np.load(em_path)
    if em.ndim != 3:
        sys.exit(f"ERROR: expected a 3-D (N, H, W) array, got shape {em.shape}")
    print(f"Loaded EM  : {em.shape}  dtype={em.dtype}")

    # ── load model ────────────────────────────────────────────────────────────
    hf_token  = os.getenv("HF_TOKEN")
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Loading model …")

    processor = AutoImageProcessor.from_pretrained(args.model, token=hf_token)
    model     = AutoModel.from_pretrained(args.model, token=hf_token)
    model.eval().to(device)

    num_register_tokens = model.config.num_register_tokens
    print(f"Register tokens : {num_register_tokens}")
    print()

    # ── extract patch embeddings (no upsampling) ──────────────────────────────
    patch_list = []
    target_hw  = []
    padded_hw  = []

    for i, sl in enumerate(em):
        pil_img = em_to_rgb(sl, scale=args.img_scale)
        target_h, target_w = pil_img.height, pil_img.width

        # Compute padded size (same padding applied inside get_embeddings)
        padded      = pad_to_patch_multiple(pil_img, args.patch_size)
        padded_h, padded_w = padded.height, padded.width

        _, patch_tokens = get_embeddings(
            pil_img, model, processor, device,
            args.patch_size, args.layer_idx, num_register_tokens,
        )

        patch_list.append(patch_tokens)
        target_hw.append([target_h, target_w])
        padded_hw.append([padded_h, padded_w])
        print(f"  slice {i:2d}: patches={patch_tokens.shape}  "
              f"target=({target_h}, {target_w})  padded=({padded_h}, {padded_w})")

    patches   = np.stack(patch_list, axis=0).astype(np.float32)   # (N, nh, nw, D)
    target_hw = np.array(target_hw, dtype=np.int32)                # (N, 2)
    padded_hw = np.array(padded_hw, dtype=np.int32)                # (N, 2)

    print(f"\nPatches shape : {patches.shape}  (N, nh, nw, D)")
    print(f"Target HW     : {target_hw.shape}  (N, 2)")
    print(f"Padded HW     : {padded_hw.shape}  (N, 2)")

    np.savez(out_path, patches=patches, target_hw=target_hw, padded_hw=padded_hw)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
