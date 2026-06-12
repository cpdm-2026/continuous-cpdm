# scripts/fid_stats_build.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


# Allow running from repo root:
#   python scripts/fid_stats_build.py ...
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from cpdm.evals.fid_core import (  # noqa: E402
    build_inception_avgpool_extractor,
    inception_feats_from_paths,
    list_images,
    mu_sigma_from_feats,
)


DEFAULT_PREPROCESS_NOTE = (
    "TrainResize128x128 THEN "
    "InceptionResize299+CenterCrop+ImageNetNormalize"
)
DEFAULT_EXTRACTOR_NOTE = "torchvision InceptionV3 DEFAULT avgpool"
DEFAULT_SAMPLING_NOTE = "all_prepared_images_recursive_sorted"


# Metadata / sanity helpers
def _as_np_str(value: str) -> np.ndarray:
    return np.array([str(value)])


def _as_np_int(value: int) -> np.ndarray:
    return np.array([int(value)], dtype=np.int64)


def _default_out_npz(out_dir: str, domain_name: str, cache_suffix: str) -> str:
    filename = f"inception_cache__{domain_name}_{cache_suffix}.npz"
    return str(Path(out_dir) / filename)


def _discover_immediate_classes(real_root: str) -> List[str]:
    """Return immediate subdirectory names that contain at least one image."""
    root = Path(real_root)
    classes: List[str] = []

    if not root.exists() or not root.is_dir():
        return classes

    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        try:
            has_images = len(list_images(str(child), recursive=True)) > 0
        except Exception:
            has_images = False

        if has_images:
            classes.append(child.name)

    return classes


def _raw_sanity(paths: List[str], n: int = 8) -> Dict[str, object]:
    """Lightweight raw RGB sanity check for cache-building logs.

    This does not affect cache values. It only verifies that input files can be
    opened as RGB uint8 images before the fixed Inception preprocessing defined
    in cpdm.evals.fid_core.
    """
    picked = paths[: max(0, min(int(n), len(paths)))]

    out: Dict[str, object] = {
        "sample_n": int(len(picked)),
        "bad": 0,
        "modes": {},
        "sizes": {},
        "min": None,
        "max": None,
        "mean_avg": None,
        "std_avg": None,
    }

    if not picked:
        return out

    mins: List[float] = []
    maxs: List[float] = []
    means: List[float] = []
    stds: List[float] = []
    modes: Dict[str, int] = {}
    sizes: Dict[str, int] = {}
    bad = 0

    for path in picked:
        try:
            img = Image.open(path).convert("RGB")
            arr = np.asarray(img)
            modes[img.mode] = modes.get(img.mode, 0) + 1
            sizes[str(img.size)] = sizes.get(str(img.size), 0) + 1
            mins.append(float(arr.min()))
            maxs.append(float(arr.max()))
            means.append(float(arr.mean()))
            stds.append(float(arr.std()))
        except Exception:
            bad += 1

    out.update(
        {
            "bad": int(bad),
            "modes": modes,
            "sizes": sizes,
            "min": float(min(mins)) if mins else None,
            "max": float(max(maxs)) if maxs else None,
            "mean_avg": float(np.mean(means)) if means else None,
            "std_avg": float(np.mean(stds)) if stds else None,
        }
    )
    return out


def _load_extra_metadata(json_text: str) -> Dict[str, str]:
    if not json_text:
        return {}

    parsed = json.loads(json_text)
    if not isinstance(parsed, dict):
        raise ValueError("--extra_metadata_json must be a JSON object.")

    return {str(k): str(v) for k, v in parsed.items()}


def _save_real_cache(
    out_npz: str,
    feats: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    domain_name: str,
    preprocess_note: str,
    extractor_note: str,
    real_root: str,
    n_listed: int,
    dataset_name: str = "",
    sampling_note: str = DEFAULT_SAMPLING_NOTE,
    class_names: Optional[List[str]] = None,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Save real-domain cache in the format expected by cpdm.evals.fid_core.

    Official reproduction path:
      1. scripts/prepare_data.py decides filtering, deterministic selection,
         resizing, and the prepared folder contents.
      2. this script uses all images in that prepared folder and computes
         fixed InceptionV3 avgpool statistics.
    """
    out_path = Path(out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "feats": feats.astype(np.float32),
        "mu": mu.astype(np.float64),
        "sigma": sigma.astype(np.float64),
        "n": _as_np_int(feats.shape[0]),
        "domain": _as_np_str(domain_name),
        "preprocess": _as_np_str(preprocess_note),
        "extractor": _as_np_str(extractor_note),
        "real_root": _as_np_str(real_root),
        "sampling": _as_np_str(sampling_note),
        "n_listed": _as_np_int(n_listed),
        "n_used": _as_np_int(feats.shape[0]),
    }

    if dataset_name:
        payload["dataset"] = _as_np_str(dataset_name)

    # Backward-friendly alias for the old PlantVillage cache metadata.
    if domain_name.lower() == "leaf":
        payload["plant_root"] = _as_np_str(real_root)

    if class_names:
        payload["classes"] = np.array(class_names, dtype=object)

    if extra_metadata:
        for key, value in extra_metadata.items():
            if key in payload:
                raise ValueError(f"Extra metadata key conflicts with cache key: {key}")
            payload[key] = _as_np_str(value)

    np.savez_compressed(out_path, **payload)


# CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixed real-domain InceptionV3 avgpool feature cache for "
            "FID/KID evaluation. This script always uses all images in the "
            "prepared real_root folder; data selection belongs to prepare_data.py."
        )
    )

    parser.add_argument(
        "--real_root",
        required=True,
        help=(
            "Prepared real-image folder for one endpoint domain. "
            "All images under this folder are used."
        ),
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help=(
            "Output directory for the cache. Unless --out_npz is provided, the "
            "filename is inception_cache__<domain_name>_<cache_suffix>.npz."
        ),
    )
    parser.add_argument(
        "--out_npz",
        default=None,
        help=(
            "Optional explicit output .npz path. If provided, this overrides "
            "--out_dir/--cache_suffix filename construction."
        ),
    )
    parser.add_argument(
        "--domain_name",
        required=True,
        help="Domain name stored in the cache, e.g. flower, leaf, male, female.",
    )
    parser.add_argument(
        "--cache_suffix",
        default="train128",
        help="Filename suffix used when --out_npz is not provided.",
    )
    parser.add_argument(
        "--dataset_name",
        default="",
        help="Optional dataset name stored in cache metadata.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively list images. Enabled by default.",
    )
    parser.add_argument(
        "--non_recursive",
        action="store_false",
        dest="recursive",
        help="Only list images directly under real_root.",
    )
    parser.add_argument(
        "--preprocess_note",
        default=DEFAULT_PREPROCESS_NOTE,
        help="Metadata note for preprocessing protocol.",
    )
    parser.add_argument(
        "--extractor_note",
        default=DEFAULT_EXTRACTOR_NOTE,
        help="Metadata note for feature extractor protocol.",
    )
    parser.add_argument(
        "--sampling_note",
        default=DEFAULT_SAMPLING_NOTE,
        help=(
            "Metadata note describing how the prepared folder is used. "
            "This script itself does not sample."
        ),
    )
    parser.add_argument(
        "--store_classes",
        action="store_true",
        help=(
            "Store immediate subdirectory names containing images as `classes` "
            "metadata."
        ),
    )
    parser.add_argument(
        "--extra_metadata_json",
        default="",
        help='Optional JSON object of extra string metadata, e.g. \'{"split":"train"}\'.',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help=(
            "Batch size for Inception feature extraction. This affects memory/speed, "
            "not the intended statistics."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="PyTorch DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda, cuda:0, or cpu. Default: auto.",
    )
    parser.add_argument(
        "--raw_sanity_n",
        type=int,
        default=8,
        help="Number of raw RGB images to inspect for sanity logs.",
    )
    parser.add_argument(
        "--verify_cache",
        action="store_true",
        help="Reload saved npz and verify cached mu/sigma against features.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    out_npz = (
        str(args.out_npz)
        if args.out_npz
        else _default_out_npz(
            out_dir=args.out_dir,
            domain_name=args.domain_name,
            cache_suffix=args.cache_suffix,
        )
    )

    paths = list_images(args.real_root, recursive=bool(args.recursive))

    if len(paths) == 0:
        raise RuntimeError(f"No real images found under real_root: {args.real_root}")

    extra_metadata = _load_extra_metadata(args.extra_metadata_json)
    class_names = _discover_immediate_classes(args.real_root) if args.store_classes else []
    sanity = _raw_sanity(paths, n=int(args.raw_sanity_n))

    print("\n========== [REAL CACHE BUILD] ==========")
    print(f"[REAL ROOT] {args.real_root}")
    print(f"[DOMAIN] {args.domain_name}")
    print(f"[LISTED] {len(paths)}")
    print(f"[USING] {len(paths)}")
    print("[SAMPLING] all prepared images; no sampling in fid_stats_build.py")
    print(f"[OUT NPZ] {out_npz}")
    print(f"[PREPROCESS] {args.preprocess_note}")
    print(f"[EXTRACTOR] {args.extractor_note}")
    if args.dataset_name:
        print(f"[DATASET] {args.dataset_name}")
    if class_names:
        print(f"[CLASSES] {len(class_names)} classes")
    print(f"[RAW SANITY] {json.dumps(sanity, ensure_ascii=False)}")

    bundle = build_inception_avgpool_extractor(device=args.device)

    feats, meta = inception_feats_from_paths(
        paths,
        bundle=bundle,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )

    mu, sigma = mu_sigma_from_feats(feats)

    _save_real_cache(
        out_npz=out_npz,
        feats=feats,
        mu=mu,
        sigma=sigma,
        domain_name=args.domain_name,
        preprocess_note=args.preprocess_note,
        extractor_note=args.extractor_note,
        real_root=args.real_root,
        n_listed=len(paths),
        dataset_name=args.dataset_name,
        sampling_note=args.sampling_note,
        class_names=class_names,
        extra_metadata=extra_metadata,
    )

    print("\n========== [SAVED] ==========")
    print(f"[NPZ] {out_npz}")
    print(f"[FEATS] {feats.shape} {feats.dtype}")
    print(f"[MU] {mu.shape} {mu.dtype}")
    print(f"[SIGMA] {sigma.shape} {sigma.dtype}")
    print(f"[N_OK] {meta['n_ok']}")
    print(f"[N_SKIP_BATCHES] {meta['n_skip_batches']}")
    print(f"[SEC] {meta['sec']:.2f}")
    for key, value in meta.items():
        if key.startswith("gpu_"):
            print(f"[{key}] {value:.2f}")

    if args.verify_cache:
        z = np.load(out_npz, allow_pickle=True)
        mu_re, sigma_re = mu_sigma_from_feats(z["feats"].astype(np.float32))
        mu_diff = float(np.max(np.abs(mu_re - z["mu"].astype(np.float64))))
        sigma_diff = float(np.max(np.abs(sigma_re - z["sigma"].astype(np.float64))))

        print("\n========== [VERIFY] ==========")
        print(f"[max|mu_recomputed-mu_cache|] {mu_diff:.6g}")
        print(f"[max|sigma_recomputed-sigma_cache|] {sigma_diff:.6g}")


if __name__ == "__main__":
    main()
