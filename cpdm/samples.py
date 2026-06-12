# samples.py

from __future__ import annotations

import os
import shutil
import copy
import math
from typing import Optional, Tuple

import numpy as np

from .config import IMG_SIZE, CHANNELS, TrainConfig
from .build_and_load import build_and_load_latest
from .sample_core import (
    sample_from_context,
    sample_cpdm_sweep_from_context,
    make_sz_sweep_values,
    CPDM_MODELS,
)


# Image conversion helper
def to_uint8_safe(x: np.ndarray, debug_prefix: str = ""):
    """Convert generated images to uint8 safely.

    The sampler normally returns images in [-1, 1].
    This helper also handles uint8 and 0-255-like arrays for debugging.
    """
    x = np.asarray(x)

    if x.ndim != 4:
        raise ValueError(f"{debug_prefix} expected 4D (B,H,W,C), got {x.shape}")

    if x.dtype == np.uint8:
        info = {
            "mode": "already_uint8",
            "in_min": int(x.min()),
            "in_max": int(x.max()),
            "oob_ratio": 0.0,
            "out_min": int(x.min()),
            "out_max": int(x.max()),
        }
        return x, info

    x_f = x.astype(np.float32)
    vmin = float(np.min(x_f))
    vmax = float(np.max(x_f))

    if vmin >= -2.0 and vmax <= 2.0:
        y = (x_f + 1.0) * 127.5
        mode = "assume_-1_1"
        oob = np.mean((x_f < -1.0) | (x_f > 1.0))
    elif vmin >= -0.1 and vmax <= 1.5:
        y = x_f * 255.0
        mode = "assume_0_1"
        oob = np.mean((x_f < 0.0) | (x_f > 1.0))
    else:
        y = x_f
        mode = "assume_0_255"
        oob = np.mean((x_f < 0.0) | (x_f > 255.0))

    y = np.clip(y, 0.0, 255.0)
    y = np.rint(y).astype(np.uint8)

    info = {
        "mode": mode,
        "in_min": vmin,
        "in_max": vmax,
        "oob_ratio": float(oob),
        "out_min": int(y.min()),
        "out_max": int(y.max()),
    }
    return y, info


# Directory helper

def prepare_out_dir(out_dir: str, overwrite: bool = False):
    """Prepare output directory for png image saving.

    If png files already exist and overwrite=False, this raises an error.
    This prevents accidental overwriting of generated samples.
    """
    if os.path.isdir(out_dir):
        existing_png = [
            f for f in os.listdir(out_dir)
            if f.lower().endswith(".png")
        ]

        if existing_png and not overwrite:
            raise RuntimeError(
                f"output dir already has {len(existing_png)} png files: {out_dir}\n"
                f"Set overwrite=True if you really want to replace them."
            )

        if overwrite:
            shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)


# Normal preview mode

def preview_samples_by_model_name(
    model_name: str,
    model_dir: Optional[str] = None,
    step: Optional[int] = None,
    cfg: Optional[TrainConfig] = None,
    n: int = 8,
    domain: str = "cond1",
    s_z=None,
    cond=None,
    z=None,
    shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
    start_idx: int = 0,
    base_seed=(1234, 5678),
    same_noise: bool = False,
    title: Optional[str] = None,
    cols: Optional[int] = None,
    dpi: int = 120,
):
    """Preview generated samples from any supported model.

    This function only displays images with matplotlib.
    It does not save files.

    Condition handling is delegated to sample_from_context:
        - baseline / shift models: domain or cond
        - CPDM models: s_z
        - CLIP-like models: cond if provided
    """
    import matplotlib.pyplot as plt

    model_name = str(model_name).lower()

    if cfg is None:
        cfg = TrainConfig()

    cfg = copy.copy(cfg)
    cfg.train_model = model_name

    ctx = build_and_load_latest(
        model_dir=model_dir,
        cfg=cfg,
        step=step,
    )

    imgs = sample_from_context(
        ctx,
        cfg=cfg,
        n=n,
        domain=domain,
        s_z=s_z,
        cond=cond,
        z=z,
        shape=shape,
        start_idx=start_idx,
        base_seed=base_seed,
        same_noise=same_noise,
    )

    if hasattr(imgs, "numpy"):
        imgs = imgs.numpy()

    def to_vis(x):
        x = np.asarray(x)
        if x.min() < 0:
            x = (x + 1.0) / 2.0
        return np.clip(x, 0.0, 1.0)

    n = int(imgs.shape[0])

    if cols is None:
        cols = min(n, 8)

    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(2.0 * cols, 2.0 * rows),
        dpi=dpi,
    )

    axes = np.asarray(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(to_vis(imgs[i]))
            ax.set_title(f"{i}", fontsize=8)

    if title is None:
        title = f"{model_name} preview"

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return fig, imgs


# Normal save mode
def save_samples_by_model_name(
    model_name: str,
    model_dir: Optional[str],
    out_dir: str,
    step: Optional[int] = None,
    cfg: Optional[TrainConfig] = None,
    n_samples: int = 1000,
    batch_size: int = 256,
    domain: str = "cond1",
    s_z=None,
    cond=None,
    shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
    base_seed: Tuple[int, int] = (2026, 1),
    overwrite: bool = False,
    print_every: int = 1,
):
    """Save generated samples as 00000.png, 00001.png, ...

    Normal mode:
        - fixed domain / condition for baselines
        - fixed s_z for CPDM
        - fixed cond for CLIP-like models if provided

    RNG protocol:
        start_idx = saved
        base_seed is fixed across models or conditions when desired.
    """
    from PIL import Image
    from tqdm import tqdm

    model_name = str(model_name).lower()

    if cfg is None:
        cfg = TrainConfig()

    cfg = copy.copy(cfg)
    cfg.train_model = model_name

    ctx = build_and_load_latest(
        model_dir=model_dir,
        cfg=cfg,
        step=step,
    )

    prepare_out_dir(out_dir, overwrite=overwrite)

    saved = 0
    batch_idx = 0

    pbar = tqdm(total=n_samples, desc=f"Sampling ({model_name}/{domain})")

    while saved < n_samples:
        cur_bs = min(int(batch_size), int(n_samples) - saved)

        imgs_tf = sample_from_context(
            ctx,
            cfg=cfg,
            n=cur_bs,
            domain=domain,
            s_z=s_z,
            cond=cond,
            z=None,
            shape=shape,
            start_idx=saved,
            base_seed=base_seed,
            same_noise=False,
        )

        imgs_np = imgs_tf.numpy()
        imgs_u8, info = to_uint8_safe(
            imgs_np,
            debug_prefix=f"[batch {batch_idx}] ",
        )

        if (batch_idx % print_every) == 0:
            print(
                f"[batch {batch_idx}] mode={info['mode']} "
                f"in[{info['in_min']:.3f},{info['in_max']:.3f}] "
                f"oob={info['oob_ratio'] * 100:.2f}% "
                f"out[{info['out_min']},{info['out_max']}] "
                f"shape={imgs_u8.shape} dtype={imgs_u8.dtype}"
            )

        for i in range(cur_bs):
            global_idx = saved + i
            Image.fromarray(imgs_u8[i], mode="RGB").save(
                os.path.join(out_dir, f"{global_idx:05d}.png")
            )

        saved += cur_bs
        batch_idx += 1
        pbar.update(cur_bs)

    pbar.close()
    print(f"[done] {n_samples} images saved to {out_dir}")


# CPDM s_z sweep preview mode
def preview_cpdm_sweep_by_model_name(
    model_name: str,
    model_dir: Optional[str] = None,
    cfg: Optional[TrainConfig] = None,
    step: Optional[int] = None,
    shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
    sweep_step: float = 0.2,
    start_idx: int = 0,
    base_seed=(1234, 5678),
    seed_per_image: int = 1,
    title: str = "CPDM $s_z$ Sweep Preview",
    dpi: int = 150,
):
    """Preview CPDM s_z sweep.

    Preview mode is intentionally fixed to one image per s_z.

    sample_cpdm_sweep_from_context returns:
        imgs: [n_sz, seed_per_image, H, W, C]

    Here:
        seed_per_image must be 1
        displayed images are imgs[:, 0]
    """
    import matplotlib.pyplot as plt

    model_name = str(model_name).lower()

    if model_name not in CPDM_MODELS:
        raise ValueError(
            f"preview_cpdm_sweep_by_model_name only supports "
            f"{sorted(CPDM_MODELS)}, but got {model_name!r}."
        )

    seed_per_image = int(seed_per_image)
    if seed_per_image != 1:
        raise ValueError(
            "preview_cpdm_sweep_by_model_name requires seed_per_image == 1. "
            "Use save_cpdm_sweep_by_model_name for multiple samples per s_z."
        )   

    if cfg is None:
        cfg = TrainConfig()

    cfg = copy.copy(cfg)
    cfg.train_model = model_name

    ctx = build_and_load_latest(
        model_dir=model_dir,
        cfg=cfg,
        step=step,
    )

    imgs, top_sz, bottom_sz, sz_values = sample_cpdm_sweep_from_context(
        ctx,
        sweep_step=sweep_step,
        shape=shape,
        start_idx=start_idx,
        base_seed=base_seed,
        seed_per_image=seed_per_image,
    )

    if hasattr(imgs, "numpy"):
        imgs = imgs.numpy()

    # [n_sz, 1, H, W, C] -> [n_sz, H, W, C]
    imgs = imgs[:, 0]

    def to_vis(x):
        x = np.asarray(x)
        if x.min() < 0:
            x = (x + 1.0) / 2.0
        return np.clip(x, 0.0, 1.0)

    sz_lookup = {
        round(float(sz), 4): idx
        for idx, sz in enumerate(sz_values)
    }

    n_cols = max(len(top_sz), len(bottom_sz))

    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(2.0 * n_cols, 4.2),
        dpi=dpi,
    )

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    for row, sz_row in enumerate([top_sz, bottom_sz]):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis("off")

            if col >= len(sz_row):
                continue

            sz = round(float(sz_row[col]), 4)
            if sz not in sz_lookup:
                raise ValueError(
                    f"s_z={sz} exists in layout values but not in unique sz_values. "
                    f"Check sweep_step={sweep_step}."
                )

            img_idx = sz_lookup[sz]

            ax.imshow(to_vis(imgs[img_idx]))
            ax.set_title(f"$s_z={sz_row[col]:.1f}$", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

    return fig, imgs, top_sz, bottom_sz, sz_values



# CPDM s_z sweep save mode
def save_cpdm_sweep_by_model_name(
    model_name: str,
    model_dir: Optional[str],
    out_dir: str,
    cfg: Optional[TrainConfig] = None,
    step: Optional[int] = None,
    n_samples_per_sz: int = 1000,
    batch_size: int = 100,
    shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
    sweep_step: float = 0.2,
    base_seed: Tuple[int, int] = (2026, 1),
    overwrite: bool = False,
    print_every: int = 1,
):
    """Save CPDM sweep samples into one folder per unique s_z value.

    sample_cpdm_sweep_from_context returns:
        imgs: [n_sz, cur_bs, H, W, C]

    RNG semantics:
        - fixed sample index across different s_z values uses the same noise identity
        - fixed s_z with different sample index uses different noise
        - chunk start_idx increments by saved
    """
    from PIL import Image
    from tqdm import tqdm

    model_name = str(model_name).lower()

    if model_name not in CPDM_MODELS:
        raise ValueError(
            f"save_cpdm_sweep_by_model_name only supports "
            f"{sorted(CPDM_MODELS)}, but got {model_name!r}."
        )

    if cfg is None:
        cfg = TrainConfig()

    cfg = copy.copy(cfg)
    cfg.train_model = model_name

    ctx = build_and_load_latest(
        model_dir=model_dir,
        cfg=cfg,
        step=step,
    )

    def format_sz_dirname(sz: float) -> str:
        sz = float(sz)

        if abs(sz) < 1e-8:
            return "sz_0_0"

        sign = "p" if sz > 0 else "m"
        val = f"{abs(sz):.1f}".replace(".", "_")
        return f"sz_{sign}{val}"

    if overwrite and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    _, _, sz_values = make_sz_sweep_values(sweep_step)

    sz_dirs = []
    for sz in sz_values:
        sz_dir = os.path.join(out_dir, format_sz_dirname(float(sz)))
        prepare_out_dir(sz_dir, overwrite=False)
        sz_dirs.append(sz_dir)

    saved = 0
    batch_idx = 0

    pbar = tqdm(
        total=n_samples_per_sz,
        desc=f"CPDM sweep sampling ({model_name})",
    )

    while saved < n_samples_per_sz:
        cur_bs = min(int(batch_size), int(n_samples_per_sz) - saved)

        imgs_tf, top_sz, bottom_sz, sz_values = sample_cpdm_sweep_from_context(
            ctx,
            sweep_step=sweep_step,
            shape=shape,
            start_idx=saved,
            base_seed=base_seed,
            seed_per_image=cur_bs,
        )

        imgs_np = imgs_tf.numpy()

        # imgs_np: [n_sz, cur_bs, H, W, C]
        for sz_idx, sz in enumerate(sz_values):
            imgs_u8, info = to_uint8_safe(
                imgs_np[sz_idx],
                debug_prefix=f"[batch {batch_idx} / sz={float(sz):.1f}] ",
            )

            if (batch_idx % print_every) == 0 and sz_idx == 0:
                print(
                    f"[batch {batch_idx}] mode={info['mode']} "
                    f"in[{info['in_min']:.3f},{info['in_max']:.3f}] "
                    f"oob={info['oob_ratio'] * 100:.2f}% "
                    f"out[{info['out_min']},{info['out_max']}] "
                    f"chunk_shape={imgs_np.shape}"
                )

            sz_dir = sz_dirs[sz_idx]

            for i in range(cur_bs):
                global_idx = saved + i
                Image.fromarray(imgs_u8[i], mode="RGB").save(
                    os.path.join(sz_dir, f"{global_idx:05d}.png")
                )

        saved += cur_bs
        batch_idx += 1
        pbar.update(cur_bs)

    pbar.close()
    print(f"[done] {n_samples_per_sz} images per s_z saved to {out_dir}")