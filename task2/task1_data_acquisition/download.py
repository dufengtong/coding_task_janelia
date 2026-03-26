"""
Download EM image data and mitochondria segmentations from OpenOrganelle (Janelia COSEM).

Data is stored publicly on AWS S3 in N5 format (no credentials required):
    s3://janelia-cosem-datasets/{dataset_id}/{dataset_id}.n5

Usage:
    python download.py --dataset jrc_hela-3
    python download.py --dataset jrc_hela-2 --scale s2 --slices 10
    python download.py --all-mito --scale s3 --slices 5
    python download.py --list-datasets

Reference:
    https://openorganelle.janelia.org/faq
"""

import argparse
import os
import sys
import numpy as np
import zarr
import dask.array as da

MITO_DATASETS = [
    "jrc_hela-3",
    "jrc_cos7-11",
    "jrc_mus-kidney",
    "jrc_mus-liver"
]

S3_BUCKET = "s3://janelia-cosem-datasets"

# Resolution level: s0=full res, s1=half, s2=quarter, s3=eighth ...
DEFAULT_SCALE = "s2"

# Number of evenly-spaced 2D z-slices to download
DEFAULT_NUM_SLICES = 5

# Candidate sub-paths for EM volumes inside the N5 container
EM_CANDIDATES = [
    "em/fibsem-uint16",
    "em/fibsem-uint8",
    "em",
]

# Candidate sub-paths for mito segmentations inside the N5 container.
# Actual path varies by dataset.
MITO_CANDIDATES = [
    "labels/mito_seg",        # segmentation 
    "labels/empanada-mito_seg",
]

def open_n5_group(dataset_id: str) -> zarr.Group:
    """Open the root N5 group for a dataset (anonymous S3 access)."""
    url = f"{S3_BUCKET}/{dataset_id}/{dataset_id}.n5"
    store = zarr.N5FSStore(url, anon=True)
    return zarr.open(store, mode="r")


def find_array(group: zarr.Group, candidates: list[str], scale: str) -> tuple[str, zarr.Array] | tuple[None, None]:
    """
    Try each candidate sub-path + scale level in the group.
    Returns (path_used, zarr_array) or (None, None) if nothing found.
    """
    for base in candidates:
        path = f"{base}/{scale}"
        try:
            node = group[path]
            if isinstance(node, zarr.Array):
                return path, node
        except KeyError:
            pass
    return None, None


def download_slices(arr: zarr.Array, num_slices: int) -> np.ndarray:
    """
    Download `num_slices` evenly-spaced 2D slices along the z-axis using dask
    for efficient chunked parallel reads.
    """
    dask_arr = da.from_array(arr, chunks=arr.chunks)
    z_size = arr.shape[0]
    indices = np.linspace(0, z_size - 1, num=min(num_slices, z_size), dtype=int)
    # Compute slice-by-slice to keep memory usage bounded
    slices = np.stack([dask_arr[int(i)].compute() for i in indices], axis=0)
    return slices


def download_dataset(
    dataset_id: str,
    scale: str = DEFAULT_SCALE,
    num_slices: int = DEFAULT_NUM_SLICES,
    output_dir: str = "data",
) -> None:
    """Download EM + mito segmentation slices for a single dataset."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Dataset : {dataset_id}")
    print(f"Scale   : {scale}  (s0=full res, higher number = lower res)")
    print(f"Slices  : {num_slices}  (evenly spaced along z-axis)")
    print(f"Output  : {os.path.abspath(output_dir)}/")
    print("=" * 60)

    # Open the root N5 group
    print(f"\nOpening N5 container: {S3_BUCKET}/{dataset_id}/{dataset_id}.n5")
    try:
        group = open_n5_group(dataset_id)
    except Exception as e:
        print(f"  ERROR: Could not open dataset — {e}")
        return

    # ---- EM raw image -------------------------------------------------------
    path, arr = find_array(group, EM_CANDIDATES, scale)
    print(f"\n[EM]")
    if arr is None:
        print(f"  WARNING: EM array not found at scale '{scale}' "
              f"(tried: {', '.join(EM_CANDIDATES)}). Skipping.")
    else:
        print(f"  Path   : {path}")
        print(f"  Shape  : {arr.shape}  dtype: {arr.dtype}  chunks: {arr.chunks}")
        slices = download_slices(arr, num_slices)
        out_path = os.path.join(output_dir, f"{dataset_id}_em_{scale}_{len(slices)}slices.npy")
        np.save(out_path, slices)
        print(f"  Saved  : {slices.shape} -> {out_path}")

    # ---- Mitochondria segmentation ------------------------------------------
    path, arr = find_array(group, MITO_CANDIDATES, scale)
    print(f"\n[Mito segmentation]")
    if arr is None:
        print(f"  INFO: Mito segmentation not found at scale '{scale}' "
              f"(tried: {', '.join(MITO_CANDIDATES)}).")
        print(f"  This dataset may not have mito segmentations, or try a different --scale.")

    else:
        print(f"  Path   : {path}")
        print(f"  Shape  : {arr.shape}  dtype: {arr.dtype}  chunks: {arr.chunks}")
        slices = download_slices(arr, num_slices)
        out_path = os.path.join(output_dir, f"{dataset_id}_mito_{scale}_{len(slices)}slices.npy")
        np.save(out_path, slices)
        print(f"  Saved  : {slices.shape} -> {out_path}")


def list_datasets() -> None:
    print("Known mitochondria datasets on OpenOrganelle:")
    print()
    for ds in MITO_DATASETS:
        print(f"  {ds:<22}  https://openorganelle.janelia.org/datasets/{ds}")
    print()
    print("N5 root for each dataset:")
    print(f"  {S3_BUCKET}/<dataset_id>/<dataset_id>.n5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download EM + mito segmentation slices from OpenOrganelle (Janelia COSEM).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        metavar="NAME",
        help="Dataset name, e.g. jrc_hela-3. Use --list-datasets to see all options.",
    )
    parser.add_argument(
        "--all-mito",
        action="store_true",
        help="Download all known mitochondria datasets.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print available mitochondria datasets and exit.",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default=DEFAULT_SCALE,
        metavar="LEVEL",
        help=f"Resolution level: s0 (full), s1, s2, s3 ... (default: {DEFAULT_SCALE})",
    )
    parser.add_argument(
        "--slices",
        type=int,
        default=DEFAULT_NUM_SLICES,
        metavar="N",
        help=f"Number of 2D z-slices to download per dataset (default: {DEFAULT_NUM_SLICES})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"),
        metavar="DIR",
        help="Directory to save .npy files (default: ../data/raw/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_datasets:
        list_datasets()
        sys.exit(0)

    if not args.dataset and not args.all_mito:
        print("ERROR: Provide --dataset <name>, --all-mito, or --list-datasets.")
        sys.exit(1)

    datasets = MITO_DATASETS if args.all_mito else [args.dataset]

    for ds in datasets:
        download_dataset(
            dataset_id=ds,
            scale=args.scale,
            num_slices=args.slices,
            output_dir=args.output_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
