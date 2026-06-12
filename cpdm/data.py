# data.py

from __future__ import annotations
import numpy as np
import os
import json
import shutil
import random
from pathlib import Path
from typing import Optional, Dict, Any, List

import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

from .config import IMG_SIZE, ALLOW_EXT


# TensorFlow dataset loader
def make_dataset(
    root_dir: str,
    batch_size: int,
    seed: int = 42,
    shuffle_buf: int = 8192,
):
    """Create an image-only tf.data.Dataset from a prepared image folder.

    Expected folder:
        root_dir/
          000000.jpg
          000001.jpg
          ...

    The returned dataset yields image batches only, not condition tensors.
    In training, cond1_ds and cond2_ds are endpoint image datasets.
    The actual condition tensor is generated inside train.py.
    """
    ds = image_dataset_from_directory(
        root_dir,
        labels=None,
        label_mode=None,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=False,
        interpolation="bilinear",
        seed=seed,
    )

    opt = tf.data.Options()
    opt.experimental_deterministic = True
    ds = ds.with_options(opt)

    ds = ds.map(
        lambda x: tf.cast(x, tf.float32) / 127.5 - 1.0,
        num_parallel_calls=1,
    )

    ds = ds.unbatch()
    ds = ds.shuffle(
        shuffle_buf,
        seed=seed,
        reshuffle_each_iteration=False,
    )
    ds = ds.batch(batch_size, drop_remainder=True)

    return ds.prefetch(1)


# Data preparation utilities
def collect_image_files(
    root_dir: str,
    allowed_ext: Optional[set[str]] = None,
) -> List[Path]:
    """Recursively collect image files under root_dir.

    This function does not assume any dataset-specific folder structure.
    It simply finds valid image files under the given directory.
    """
    root = Path(root_dir)

    if not root.exists():
        raise FileNotFoundError(f"source directory does not exist: {root_dir}")

    if not root.is_dir():
        raise NotADirectoryError(f"source path is not a directory: {root_dir}")

    if allowed_ext is None:
        allowed_ext = ALLOW_EXT

    allowed_ext = {str(e).lower() for e in allowed_ext}

    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in allowed_ext
    ]

    files = sorted(files, key=lambda p: str(p.relative_to(root)).lower())

    if not files:
        raise RuntimeError(
            f"No image files found under {root_dir}. "
            f"Allowed extensions: {sorted(allowed_ext)}"
        )

    return files


def _prepare_empty_dir(path: str, overwrite: bool = False):
    path = Path(path)

    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"target path exists but is not a directory: {path}")

        existing = list(path.iterdir())
        if existing and not overwrite:
            raise RuntimeError(
                f"target directory is not empty: {path}\n"
                f"Use overwrite=True to replace it."
            )

        if overwrite:
            shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def _copy_or_symlink(src: Path, dst: Path, mode: str):
    mode = str(mode).lower()

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "symlink":
        os.symlink(src.resolve(), dst)
        return

    raise ValueError(f"Unknown mode={mode!r}. Expected 'copy' or 'symlink'.")


def prepare_domain_folder(
    src_dir: str,
    dst_dir: str,
    max_images: Optional[int] = None,
    seed: int = 42,
    mode: str = "copy",
    overwrite: bool = False,
    prefix: str = "",
) -> Dict[str, Any]:
    """Prepare one endpoint image folder.

    The source directory may contain nested folders.
    Selected files are copied or symlinked into a flat target folder.

    Output:
        dst_dir/
          000000.jpg
          000001.jpg
          ...

    Returns a manifest dictionary.
    """
    files = collect_image_files(src_dir)

    rng = random.Random(int(seed))
    files = list(files)
    rng.shuffle(files)

    if max_images is not None:
        files = files[:max_images]

    _prepare_empty_dir(dst_dir, overwrite=overwrite)

    dst_root = Path(dst_dir)
    src_root = Path(src_dir)

    records = []

    for i, src in enumerate(files):
        ext = src.suffix.lower()
        name = f"{prefix}{i:06d}{ext}"
        dst = dst_root / name

        _copy_or_symlink(src, dst, mode=mode)

        records.append(
            {
                "index": i,
                "source": str(src),
                "source_relative": str(src.relative_to(src_root)),
                "target": str(dst),
            }
        )

    manifest = {
        "src_dir": str(src_root),
        "dst_dir": str(dst_root),
        "mode": str(mode).lower(),
        "seed": int(seed),
        "max_images": max_images,
        "num_images": len(files),
        "records": records,
    }

    return manifest


def prepare_two_domain_folders(
    cond1_src: str,
    cond2_src: str,
    out_dir: str,
    max_images: Optional[int] = None,
    max_images_cond1: Optional[int] = None,
    max_images_cond2: Optional[int] = None,
    seed: int = 42,
    mode: str = "copy",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Prepare two endpoint domains for CPDM training.

    Output:
        out_dir/
          cond1/
            000000.jpg
            ...
          cond2/
            000000.jpg
            ...
          manifest.json

    Notes:
        cond1 and cond2 are image endpoint domains.
        They are not condition tensors.

        For scalar CPDM:
            cond1 corresponds to the positive endpoint, s_z = +1.
            cond2 corresponds to the negative endpoint, s_z = -1.
    """
    out_root = Path(out_dir)

    if overwrite and out_root.exists():
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    if max_images_cond1 is None:
        max_images_cond1 = max_images

    if max_images_cond2 is None:
        max_images_cond2 = max_images

    cond1_manifest = prepare_domain_folder(
        src_dir=cond1_src,
        dst_dir=str(out_root / "cond1"),
        max_images=max_images_cond1,
        seed=seed,
        mode=mode,
        overwrite=overwrite,
        prefix="",
    )

    cond2_manifest = prepare_domain_folder(
        src_dir=cond2_src,
        dst_dir=str(out_root / "cond2"),
        max_images=max_images_cond2,
        seed=seed + 1,
        mode=mode,
        overwrite=overwrite,
        prefix="",
    )

    manifest = {
        "out_dir": str(out_root),
        "mode": str(mode).lower(),
        "seed": int(seed),
        "cond1": cond1_manifest,
        "cond2": cond2_manifest,
    }

    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[prepare] cond1 images: {cond1_manifest['num_images']}")
    print(f"[prepare] cond2 images: {cond2_manifest['num_images']}")
    print(f"[prepare] manifest saved to {manifest_path}")

    return manifest

def make_clip_dataset(
    bank_npz: str,
    batch_size: int,
    seed: int = 42,
    shuffle_buf: int = 8192,
    expected_dim: int = 512,
    root_dir: Optional[str] = None,
):
    """Create an embedding-only tf.data.Dataset from a CLIP bank.

    Expected bank format from scripts/build_clip_bank.py:
        paths       : object/string array, shape [N]
        embeddings  : float32 array, shape [N, D]

    This dataset must use the same shuffle protocol as make_dataset():
        shuffle(seed, shuffle_buf, reshuffle_each_iteration=False)
        batch(drop_remainder=True)

    Therefore, when used with cond1_ds / cond2_ds, pass the same
    batch_size, seed, and shuffle_buf.
    """
    bank_path = Path(bank_npz)

    if not bank_path.exists():
        raise FileNotFoundError(f"CLIP bank not found: {bank_npz}")

    data = np.load(str(bank_path), allow_pickle=True)

    if "embeddings" not in data:
        raise KeyError(
            f"'embeddings' key not found in CLIP bank: {bank_npz}. "
            f"Available keys={list(data.keys())}"
        )

    embeddings = np.asarray(data["embeddings"], dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(
            f"CLIP embeddings must be rank-2 [N,D], got shape={embeddings.shape}"
        )

    if embeddings.shape[-1] != int(expected_dim):
        raise ValueError(
            f"CLIP embedding dim mismatch: expected {expected_dim}, "
            f"got {embeddings.shape[-1]} from {bank_npz}"
        )

    # Optional safety check:
    # compare bank path order with the current prepared image folder order.
    if root_dir is not None:
        if "paths" not in data:
            raise KeyError(
                f"'paths' key is required for root_dir order check, "
                f"but not found in {bank_npz}."
            )

        bank_paths = []
        for p in data["paths"].tolist():
            if isinstance(p, bytes):
                p = p.decode("utf-8")
            bank_paths.append(str(Path(str(p)).resolve()))

        current_paths = [
            str(p.resolve())
            for p in collect_image_files(root_dir)
        ]

        if bank_paths != current_paths:
            raise ValueError(
                "CLIP bank path order does not match current image folder order.\n"
                f"bank_npz={bank_npz}\n"
                f"root_dir={root_dir}\n"
                f"bank_count={len(bank_paths)} | current_count={len(current_paths)}\n"
                "Rebuild the CLIP bank from the same prepared folder, or disable "
                "root_dir checking if this mismatch is expected."
            )

    ds = tf.data.Dataset.from_tensor_slices(embeddings)

    opt = tf.data.Options()
    opt.experimental_deterministic = True
    ds = ds.with_options(opt)

    ds = ds.shuffle(
        shuffle_buf,
        seed=seed,
        reshuffle_each_iteration=False,
    )

    ds = ds.batch(batch_size, drop_remainder=True)

    return ds.prefetch(1)