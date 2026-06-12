# build_and_load.py
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from .config import (
    SEED_T_EPS,
    SAVE_DIR,
    TrainConfig,
    DriftCfg,
)
from .schedules import cosine_beta_schedule, alpha_tables
from .drift import DriftA_NoGain
from .model import build_model, build_shift_fn


LOSS_BASELINE = "baseline"    # Standard DDPM epsilon-prediction loss.
LOSS_BASE_CPDM = "base_cpdm"   # CPDM eta/n-target loss: eta = eps + r_t.
LOSS_CONTINUOUS_CPDM = "continuous_cpdm"            # Base CPDM plus local n-target pair consistency.
LOSS_COND_QUAD_SHIFT_DDPM = "cond_quad_shift_ddpm"  # Conditional quadratic shift baseline loss.


@dataclass
class BuildLoadContext:
    """
    Container returned by build/load helpers.
    It stores the loaded model objects together with resolved artifact paths,
    diffusion schedules, optional drift modules, optional shift predictors, and
    optional condition banks.
    """
    train_model: str
    loss_mode: str
    condition_dim: int

    model: tf.keras.Model
    tables: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]

    drift: Optional[DriftA_NoGain] = None      # Present only for CPDM variants.
    shift_fn: Optional[tf.keras.Model] = None  # Present only for Shift-DDPM variants.
    condition_bank_paths: Optional[Tuple[str, str]] = None
    model_spec: Optional[Dict[str, Any]] = None

    output_dir: Optional[str] = None
    model_dir: Optional[str] = None
    weights_dir: Optional[str] = None
    tf_ckpt_dir: Optional[str] = None
    clip_bank_dir: Optional[str] = None
    fid_stats_dir: Optional[str] = None

    denoise_weight_path: Optional[str] = None
    shift_weight_path: Optional[str] = None
    proto_path: Optional[str] = None
    condition_bank: Optional[Dict[str, tf.Tensor]] = None


# train_model registry
def resolve_train_model(cfg: TrainConfig) -> Dict[str, Any]:
    """Resolve the public train_model switch into build/load behavior."""
    train_model = str(getattr(cfg, "train_model", "continuous_cpdm")).lower()

    if train_model == "onehot":
        return {
            "train_model": train_model,
            "loss_mode": LOSS_BASELINE,
            "condition_dim": 2,
            "uses_drift": False,
            "uses_shift": False,
            "expects_batch_cond": False,
            "uses_condition_bank": False,
        }

    if train_model == "joint256":
        return {
            "train_model": train_model,
            "loss_mode": LOSS_BASELINE,
            "condition_dim": 256,
            "uses_drift": False,
            "uses_shift": False,
            "expects_batch_cond": False,
            "uses_condition_bank": False,
        }

    if train_model == "clip_img":
        return {
            "train_model": train_model,
            "loss_mode": LOSS_BASELINE,
            "condition_dim": 512,
            "uses_drift": False,
            "uses_shift": False,
            "expects_batch_cond": False,
            "uses_condition_bank": True,
        }

    if train_model == "clip_text":
        return {
            "train_model": train_model,
            "loss_mode": LOSS_BASELINE,
            "condition_dim": 512,
            "uses_drift": False,
            "uses_shift": False,
            "expects_batch_cond": False,
            "uses_condition_bank": True,
        }

    if train_model == "base_cpdm":
        return {
            "train_model": train_model,
            "loss_mode": LOSS_BASE_CPDM,
            "condition_dim": 1,
            "uses_drift": True,
            "uses_shift": False,
            "expects_batch_cond": False,
            "uses_condition_bank": False,
        }

    if train_model == "continuous_cpdm":
        return {
            "train_model": train_model,
            "loss_mode": LOSS_CONTINUOUS_CPDM,
            "condition_dim": 1,
            "uses_drift": True,
            "uses_shift": False,
            "expects_batch_cond": False,
            "uses_condition_bank": False,
        }

    if train_model in {"shift_ddpm", "cond_quad_shift_ddpm"}:
        return {
            "train_model": "cond_quad_shift_ddpm",
            "loss_mode": LOSS_COND_QUAD_SHIFT_DDPM,
            "condition_dim": 1,
            "uses_drift": False,
            "uses_shift": True,
            "shift_type": "original",
            "expects_batch_cond": False,
            "uses_condition_bank": False,
        }

    if train_model in {
        "shift_ddpm_larger",
        "shift_ddpm_pp",
        "cond_quad_shift_ddpm_larger",
    }:
        return {
            "train_model": "cond_quad_shift_ddpm_larger",
            "loss_mode": LOSS_COND_QUAD_SHIFT_DDPM,
            "condition_dim": 1,
            "uses_drift": False,
            "uses_shift": True,
            "shift_type": "larger",
            "expects_batch_cond": False,
            "uses_condition_bank": False,
        }

    raise ValueError(
            f"Unknown train_model={train_model!r}. Expected one of "
            "{'onehot', 'joint256', 'clip_img', 'clip_text', "
            "'base_cpdm', 'continuous_cpdm', 'cond_quad_shift_ddpm', 'cond_quad_shift_ddpm_larger'}."
        )

# Path / weight helpers

def _model_dir_name(train_model: str) -> str:
    """Return the public model artifact directory name."""
    return str(train_model).lower()


def _base_dir_or_default(path: Optional[str], cfg: TrainConfig) -> str:
    """Resolve the user-provided artifact root or model-specific path."""
    if path is not None:
        return str(path)

    if hasattr(cfg, "output_dir"):
        return str(getattr(cfg, "output_dir"))

    if hasattr(cfg, "save_dir"):
        return str(getattr(cfg, "save_dir"))

    return SAVE_DIR


def _resolve_artifact_paths(
    model_dir: Optional[str],
    cfg: TrainConfig,
    train_model: str,
) -> Dict[str, str]:
    """Resolve the public B-layout artifact paths.

    Supported inputs:
        model_dir = outputs/leaf_flower
        model_dir = outputs/leaf_flower/continuous_cpdm

    Public layout:
        outputs/<run>/
          prototypes/
          clip_bank/
          fid_stats/
          <model>/
            weights/
            tf_ckpt/
    """
    base = os.path.normpath(_base_dir_or_default(model_dir, cfg))
    model_name = _model_dir_name(train_model)

    known_models = {
        "onehot",
        "joint256",
        "clip_img",
        "clip_text",
        "base_cpdm",
        "continuous_cpdm",
        "cond_quad_shift_ddpm",
        "cond_quad_shift_ddpm_larger",
    }

    # If the provided path already points to a model-specific directory,
    # use its parent as output_dir. Otherwise, treat it as output_dir.
    if os.path.basename(base) in known_models:
        resolved_model_dir = base
        output_dir = os.path.dirname(base)
    else:
        output_dir = base
        resolved_model_dir = os.path.join(output_dir, model_name)

    return {
        "output_dir": output_dir,
        "model_dir": resolved_model_dir,
        "weights_dir": os.path.join(resolved_model_dir, "weights"),
        "tf_ckpt_dir": os.path.join(resolved_model_dir, "tf_ckpt"),
        "proto_path": os.path.join(output_dir, "prototypes"),
        "clip_bank_dir": os.path.join(output_dir, "clip_bank"),
        "fid_stats_dir": os.path.join(output_dir, "fid_stats"),
    }


def _find_weight(weights_dir: str, prefix: str, step: Optional[int] = None) -> str:
    if step is None:
        pattern = os.path.join(weights_dir, f"{prefix}_step*.weights.h5")
    else:
        pattern = os.path.join(
            weights_dir,
            f"{prefix}_step{int(step):07d}.weights.h5",
        )

    paths = sorted(glob.glob(pattern))
    paths = [p for p in paths if "_ema" not in os.path.basename(p)]

    if not paths:
        raise FileNotFoundError(
            f"No weight file found: prefix={prefix!r}, step={step}, dir={weights_dir}"
        )

    return paths[-1]


def _make_tables(cfg: TrainConfig) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    betas = cosine_beta_schedule(cfg.K)
    alphas, alphabars, _ = alpha_tables(betas)
    return betas, alphas, alphabars


def _get_proto_path(paths: Dict[str, str], cfg: TrainConfig) -> str:
    """
    Resolve CPDM prototype path.
    If cfg.proto_path is None, use the run-level prototype directory resolved
    from output_dir/model_dir. This is useful when sampling from different runs
    such as Leaf/Flower and CelebA.
    """
    explicit = getattr(cfg, "proto_path", None)
    if explicit is not None:
        return str(explicit)

    return paths["proto_path"]


def _make_drift_cfg(cfg: TrainConfig) -> DriftCfg:
    return DriftCfg(
        K=cfg.K,
        A=cfg.A,
        tau0=float(getattr(cfg, "tau0", 1e-4)),
        lp_sigma=float(getattr(cfg, "lp_sigma", 3.0)),
        kappa=float(getattr(cfg, "kappa", 1.0)),
        freeze_prototypes=True,
        time_schedule=str(getattr(cfg, "time_schedule", "linear")),
        uhat_mode=str(getattr(cfg, "uhat_mode", "dataset_diff")),
        uhat_seed=int(getattr(cfg, "uhat_seed", getattr(cfg, "RNG_SEED", SEED_T_EPS))),
        uhat_norm_target=float(getattr(cfg, "uhat_norm_target", 0.6)),
        proto_target_count=int(getattr(cfg, "proto_target_count", 512)),
    )


# Model / condition-bank loaders

def _load_denoise_model(
    cfg: TrainConfig,
    model_spec: Dict[str, Any],
    weights_dir: str,
    step: Optional[int] = None,
) -> Tuple[tf.keras.Model, str]:
    del cfg

    train_model = model_spec["train_model"]
    condition_dim = int(model_spec["condition_dim"])

    model = build_model(
        condition_dim=condition_dim,
        train_model=train_model,
    )

    weight_path = _find_weight(weights_dir, prefix="denoise_fn", step=step)
    print("[weights] loading denoise_fn:", weight_path)
    model.load_weights(weight_path)

    return model, weight_path


def _get_condition_bank_paths(
    paths: Dict[str, str],
    cfg: TrainConfig,
    train_model: str,
) -> Tuple[str, str]:
    """Resolve separate cond1/cond2 CLIP bank paths.

    Public convention under <output_dir>/clip_bank:

        clip_img:
            cond1_clip_img_bank.npz
            cond2_clip_img_bank.npz

        clip_text:
            cond1_clip_text_bank.npz
            cond2_clip_text_bank.npz

    Each file must contain:
        embeddings: [N, D]
    """
    train_model = str(train_model).lower()

    if train_model == "clip_img":
        cond1_name = "cond1_clip_img_bank.npz"
        cond2_name = "cond2_clip_img_bank.npz"
    elif train_model == "clip_text":
        cond1_name = "cond1_clip_text_bank.npz"
        cond2_name = "cond2_clip_text_bank.npz"
    else:
        raise ValueError(
            "condition banks are only used for clip_img/clip_text, "
            f"got train_model={train_model!r}."
        )

    # Explicit domain-wise overrides first. Custom filenames are allowed here;
    # the path assigned to cond1 is used as the cond1 condition stream.
    cond1_path = getattr(cfg, "cond1_condition_bank_path", None)
    cond2_path = getattr(cfg, "cond2_condition_bank_path", None)

    if cond1_path is not None or cond2_path is not None:
        if cond1_path is None or cond2_path is None:
            raise ValueError(
                "Both cond1_condition_bank_path and cond2_condition_bank_path "
                "must be provided together."
            )
        return str(cond1_path), str(cond2_path)

    # Directory convention.
    candidate_dirs = []

    clip_bank_dir = getattr(cfg, "clip_bank_dir", None)
    if clip_bank_dir is not None:
        candidate_dirs.append(str(clip_bank_dir))

    candidate_dirs.append(paths["clip_bank_dir"])

    # Backward-compatible fallbacks for older local layouts.
    candidate_dirs.append(os.path.join(paths["model_dir"], "clip_bank"))
    candidate_dirs.append(os.path.join(paths["model_dir"], "metadata"))

    # Remove duplicates while preserving order.
    seen = set()
    unique_candidate_dirs = []
    for root in candidate_dirs:
        root = os.path.normpath(str(root))
        if root not in seen:
            seen.add(root)
            unique_candidate_dirs.append(root)

    for root in unique_candidate_dirs:
        cond1 = os.path.join(root, cond1_name)
        cond2 = os.path.join(root, cond2_name)

        if os.path.exists(cond1) and os.path.exists(cond2):
            return cond1, cond2

    # Return expected paths from the first candidate so the loader raises a clear error.
    root = unique_candidate_dirs[0]
    return (
        os.path.join(root, cond1_name),
        os.path.join(root, cond2_name),
    )


def _load_single_condition_bank(
    path: str,
    expected_dim: int,
    domain_key: str,
) -> np.ndarray:
    """Load one domain-wise CLIP condition bank.

    Expected npz key:
        embeddings: [N, D]
    """
    if path is None:
        raise ValueError(f"{domain_key} condition bank path is required.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"{domain_key} condition bank not found: {path}")

    data = np.load(path, allow_pickle=True)

    if "embeddings" not in data:
        raise KeyError(
            f"'embeddings' key not found in {domain_key} condition bank: {path}. "
            f"Available keys={list(data.keys())}"
        )

    arr = np.asarray(data["embeddings"], dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(
            f"{domain_key} condition bank must be rank-2 [N,D]. "
            f"Got shape={arr.shape} from {path}"
        )

    if arr.shape[-1] != expected_dim:
        raise ValueError(
            f"{domain_key} condition dim mismatch. "
            f"expected_dim={expected_dim}, got {arr.shape[-1]} from {path}"
        )

    return arr


def _load_condition_bank(
    paths: Tuple[str, str],
    expected_dim: int,
) -> Dict[str, tf.Tensor]:
    """Load separate cond1/cond2 CLIP condition banks."""
    cond1_path, cond2_path = paths

    cond1 = _load_single_condition_bank(
        cond1_path,
        expected_dim=expected_dim,
        domain_key="cond1",
    )
    cond2 = _load_single_condition_bank(
        cond2_path,
        expected_dim=expected_dim,
        domain_key="cond2",
    )

    print(
        "[cond_bank] loaded separate banks | "
        f"cond1={cond1.shape} <- {cond1_path} | "
        f"cond2={cond2.shape} <- {cond2_path}"
    )

    return {
        "cond1": tf.constant(cond1, tf.float32),
        "cond2": tf.constant(cond2, tf.float32),
    }


def _add_context_paths(
    kwargs: Dict[str, Any],
    paths: Dict[str, str],
) -> Dict[str, Any]:
    """Attach resolved path metadata to BuildLoadContext kwargs."""
    kwargs.update(
        output_dir=paths["output_dir"],
        model_dir=paths["model_dir"],
        weights_dir=paths["weights_dir"],
        tf_ckpt_dir=paths["tf_ckpt_dir"],
        clip_bank_dir=paths["clip_bank_dir"],
        fid_stats_dir=paths["fid_stats_dir"],
    )
    return kwargs


# Build/load families
def build_and_load_extra_base(
    model_dir: Optional[str],
    cfg: TrainConfig,
    step: Optional[int] = None,
) -> BuildLoadContext:
    """Build/load non-metadata baseline.

    Supported:
        onehot:
            external cond is generated as [B,2] one-hot.

        joint256:
            external cond is generated as [B] integer label id.
            Joint256DenoiseFn maps label id -> Embedding(2,256) internally.
    """
    model_spec = resolve_train_model(cfg)
    train_model = model_spec["train_model"]

    if train_model not in {"onehot", "joint256"}:
        raise ValueError(f"build_and_load_extra_base got train_model={train_model!r}")

    paths = _resolve_artifact_paths(
        model_dir,
        cfg,
        train_model=train_model,
    )

    tables = _make_tables(cfg)
    model, denoise_path = _load_denoise_model(
        cfg,
        model_spec,
        paths["weights_dir"],
        step=step,
    )

    return BuildLoadContext(
        **_add_context_paths(
            {
                "train_model": train_model,
                "loss_mode": model_spec["loss_mode"],
                "condition_dim": int(model_spec["condition_dim"]),
                "model": model,
                "tables": tables,
                "model_spec": model_spec,
                "denoise_weight_path": denoise_path,
            },
            paths,
        )
    )


def build_and_load_clip(
    model_dir: Optional[str],
    cfg: TrainConfig,
    step: Optional[int] = None,
) -> BuildLoadContext:
    """Build/load CLIP-image or CLIP-text baseline.

    Supported:
        clip_img
        clip_text

    These require domain-wise condition banks:
        cond1: [N,512]
        cond2: [N,512]
    """
    model_spec = resolve_train_model(cfg)
    train_model = model_spec["train_model"]

    if train_model not in {"clip_img", "clip_text"}:
        raise ValueError(f"build_and_load_clip got train_model={train_model!r}")

    paths = _resolve_artifact_paths(
        model_dir,
        cfg,
        train_model=train_model,
    )

    tables = _make_tables(cfg)
    model, denoise_path = _load_denoise_model(
        cfg,
        model_spec,
        paths["weights_dir"],
        step=step,
    )

    bank_paths = _get_condition_bank_paths(paths, cfg, train_model)
    condition_bank = _load_condition_bank(
        bank_paths,
        expected_dim=int(model_spec["condition_dim"]),
    )

    return BuildLoadContext(
        **_add_context_paths(
            {
                "train_model": train_model,
                "loss_mode": model_spec["loss_mode"],
                "condition_dim": int(model_spec["condition_dim"]),
                "model": model,
                "tables": tables,
                "condition_bank": condition_bank,
                "model_spec": model_spec,
                "denoise_weight_path": denoise_path,
                "condition_bank_paths": bank_paths,
            },
            paths,
        )
    )


def build_and_load_cpdm(
    model_dir: Optional[str],
    cfg: TrainConfig,
    step: Optional[int] = None,
) -> BuildLoadContext:
    """Build/load Base CPDM or Continuous CPDM.

    Supported:
        base_cpdm
        continuous_cpdm

    Sampling reuses the prototype generated during training.
    For dataset_diff mode, a missing prototype is treated as an error.
    """
    model_spec = resolve_train_model(cfg)
    train_model = model_spec["train_model"]

    if train_model not in {"base_cpdm", "continuous_cpdm"}:
        raise ValueError(f"build_and_load_cpdm got train_model={train_model!r}")

    paths = _resolve_artifact_paths(
        model_dir,
        cfg,
        train_model=train_model,
    )

    tables = _make_tables(cfg)
    betas, _, _ = tables

    model, denoise_path = _load_denoise_model(
        cfg,
        model_spec,
        paths["weights_dir"],
        step=step,
    )

    proto_path = _get_proto_path(paths, cfg)
    uhat_mode = str(getattr(cfg, "uhat_mode", "dataset_diff")).lower()

    def _expected_uhat_file(proto_path_: str, uhat_mode_: str) -> str:
        if proto_path_.endswith(".npz"):
            return proto_path_

        name_map = {
            "dataset_diff": "uhat16_diff.npz",
            "const": "uhat16_const.npz",
            "random": "uhat16_random.npz",
        }

        if uhat_mode_ not in name_map:
            raise ValueError(
                f"Unknown uhat_mode={uhat_mode_!r}. "
                f"Expected one of {sorted(name_map)}."
            )

        return os.path.join(proto_path_, name_map[uhat_mode_])

    expected_proto_file = _expected_uhat_file(proto_path, uhat_mode)

    if uhat_mode == "dataset_diff" and not os.path.exists(expected_proto_file):
        raise FileNotFoundError(
            "CPDM sampling with uhat_mode='dataset_diff' requires an existing "
            f"prototype file. Expected={expected_proto_file}, proto_path={proto_path}"
        )

    drift = DriftA_NoGain(
        betas,
        _make_drift_cfg(cfg),
    )

    # drift.py resolves mode-specific npz names when proto_path is a directory.
    # For dataset_diff this should load an existing file.
    # For const/random this may create deterministic u_hat if not present.
    drift.warmup_and_save_if_needed(
        None,
        None,
        proto_path,
        target_count=int(getattr(cfg, "proto_target_count", 512)),
    )

    return BuildLoadContext(
        **_add_context_paths(
            {
                "train_model": train_model,
                "loss_mode": model_spec["loss_mode"],
                "condition_dim": int(model_spec["condition_dim"]),
                "model": model,
                "drift": drift,
                "tables": tables,
                "model_spec": model_spec,
                "denoise_weight_path": denoise_path,
                "proto_path": proto_path,
            },
            paths,
        )
    )


def build_and_load_shift_ddpm(
    model_dir: Optional[str],
    cfg: TrainConfig,
    step: Optional[int] = None,
) -> BuildLoadContext:
    """Build/load conditional Quadratic-Shift-DDPM.

    This path is separated because Shift-DDPM requires:
        denoise_fn weights
        shift_fn weights
    """
    model_spec = resolve_train_model(cfg)
    train_model = model_spec["train_model"]

    if train_model not in {"cond_quad_shift_ddpm", "cond_quad_shift_ddpm_larger"}:
        raise ValueError(f"build_and_load_shift_ddpm got train_model={train_model!r}")

    paths = _resolve_artifact_paths(
        model_dir,
        cfg,
        train_model=train_model,
    )

    tables = _make_tables(cfg)

    model, denoise_path = _load_denoise_model(
        cfg,
        model_spec,
        paths["weights_dir"],
        step=step,
    )

    shift_fn = build_shift_fn(
        shift_type=str(model_spec.get("shift_type", getattr(cfg, "shift_type", "original"))),
        num_cond=int(getattr(cfg, "shift_num_cond", 2)),
        out_ch=int(getattr(cfg, "shift_out_ch", 3)),
    )

    shift_path = _find_weight(paths["weights_dir"], prefix="shift_fn", step=step)
    print("[weights] loading shift_fn:", shift_path)
    shift_fn.load_weights(shift_path)

    return BuildLoadContext(
        **_add_context_paths(
            {
                "train_model": train_model,
                "loss_mode": model_spec["loss_mode"],
                "condition_dim": int(model_spec["condition_dim"]),
                "model": model,
                "shift_fn": shift_fn,
                "tables": tables,
                "model_spec": model_spec,
                "denoise_weight_path": denoise_path,
                "shift_weight_path": shift_path,
            },
            paths,
        )
    )


def build_and_load_latest(
    model_dir: Optional[str],
    cfg: TrainConfig,
    step: Optional[int] = None,
) -> BuildLoadContext:
    """Top-level dispatcher for sampling/build scripts.

    This function only builds and loads objects.
    It does not run reverse sampling.

    The model_dir argument accepts either:
        - dataset/run root, e.g. outputs/leaf_flower
        - model-specific directory, e.g. outputs/leaf_flower/continuous_cpdm
    """
    model_spec = resolve_train_model(cfg)
    train_model = model_spec["train_model"]

    if train_model in {"onehot", "joint256"}:
        return build_and_load_extra_base(model_dir, cfg, step=step)

    if train_model in {"clip_img", "clip_text"}:
        return build_and_load_clip(model_dir, cfg, step=step)

    if train_model in {"base_cpdm", "continuous_cpdm"}:
        return build_and_load_cpdm(model_dir, cfg, step=step)

    if train_model in {"cond_quad_shift_ddpm", "cond_quad_shift_ddpm_larger"}:
        return build_and_load_shift_ddpm(model_dir, cfg, step=step)

    raise RuntimeError(f"Unexpected train_model={train_model!r}")
