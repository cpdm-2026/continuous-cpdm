# scripts/prepare_data.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Allow running from repo root:
# python scripts/prepare_data.py ...
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from cpdm.data import prepare_two_domain_folders


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare two endpoint image folders for CPDM training. "
            "This script does not download datasets. "
            "Please download the raw datasets manually and pass their local folders."
        )
    )

    parser.add_argument(
        "--cond1_src",
        required=True,
        help=(
            "Source image directory for endpoint/domain 1. "
            "For scalar CPDM, this corresponds to s_z=+1."
        ),
    )

    parser.add_argument(
        "--cond2_src",
        required=True,
        help=(
            "Source image directory for endpoint/domain 2. "
            "For scalar CPDM, this corresponds to s_z=-1."
        ),
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help=(
            "Output directory. The script will create out_dir/cond1, "
            "out_dir/cond2, and out_dir/manifest.json."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="copy",
        help=(
            "How to place files into the prepared folders. "
            "Default is copy. Symlink may fail on some Windows/Drive setups."
        ),
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help=(
            "Maximum number of images for both cond1 and cond2. "
            "Use per-domain options below to override separately."
        ),
    )

    parser.add_argument(
        "--max_images_cond1",
        type=int,
        default=None,
        help="Maximum number of images for cond1 only.",
    )

    parser.add_argument(
        "--max_images_cond2",
        type=int,
        default=None,
        help="Maximum number of images for cond2 only.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic image selection.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    manifest = prepare_two_domain_folders(
        cond1_src=args.cond1_src,
        cond2_src=args.cond2_src,
        out_dir=args.out_dir,
        max_images=args.max_images,
        max_images_cond1=args.max_images_cond1,
        max_images_cond2=args.max_images_cond2,
        seed=args.seed,
        mode=args.mode,
        overwrite=args.overwrite,
    )

    print("[done] prepared two-domain dataset")
    print(f"[done] out_dir: {manifest['out_dir']}")
    print(f"[done] cond1 images: {manifest['cond1']['num_images']}")
    print(f"[done] cond2 images: {manifest['cond2']['num_images']}")
    print(f"[done] manifest: {Path(manifest['out_dir']) / 'manifest.json'}")


if __name__ == "__main__":
    main()