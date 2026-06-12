# scripts/sample.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path
# Allow running from repo root:
# python scripts/sample.py ...

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cpdm.config import SAVE_DIR, TrainConfig

from cpdm.samples import (
    preview_samples_by_model_name,
    save_samples_by_model_name,
    preview_cpdm_sweep_by_model_name,
    save_cpdm_sweep_by_model_name,
)


ALL_MODELS = [
    "onehot",
    "joint256",
    "clip_img",
    "clip_text",
    "base_cpdm",
    "continuous_cpdm",
    "cond_quad_shift_ddpm",
    "cond_quad_shift_ddpm_larger",
]

CPDM_MODELS = [
    "base_cpdm",
    "continuous_cpdm",
]


def _base_seed(xs):
    if xs is None:
        return None
    if len(xs) != 2:
        raise ValueError("--base_seed requires exactly two integers.")
    return (int(xs[0]), int(xs[1]))


# Build a sampling config with artifact root and optional CLIP bank overrides.
def _make_sample_cfg(args, artifact_dir: str | None = None):
    cfg = TrainConfig()
    cfg.train_model = args.model
    cfg.proto_path = None
    if artifact_dir is not None:
        cfg.output_dir = str(artifact_dir)

    if hasattr(args, "clip_bank_dir") and args.clip_bank_dir is not None:
        cfg.clip_bank_dir = str(args.clip_bank_dir)
    else:
        cfg.clip_bank_dir = None

    if hasattr(args, "cond1_clip_bank"):
        cfg.cond1_condition_bank_path = args.cond1_clip_bank

    if hasattr(args, "cond2_clip_bank"):
        cfg.cond2_condition_bank_path = args.cond2_clip_bank

    return cfg


def build_parser():
    parser = argparse.ArgumentParser(
        description="Sampling CLI for CPDM / baseline diffusion models."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # Normal preview
    p = subparsers.add_parser(
        "preview",
        help="Preview generated samples with matplotlib.",
    )

    p.add_argument("--model", required=True, choices=ALL_MODELS)
    p.add_argument(
        "--output_dir",
        default=SAVE_DIR,
        help="Run-level artifact root, e.g. ./outputs/leaf_flower.",
    )
    p.add_argument(
        "--model_dir",
        default=None,
        help=(
            "Optional artifact path. Can be either run root "
            "./outputs/leaf_flower or model-specific dir "
            "./outputs/leaf_flower/continuous_cpdm."
        ),
    )
    p.add_argument("--step", type=int, default=None)

    p.add_argument("--n", type=int, default=8)
    p.add_argument("--domain", default="cond1", choices=["cond1", "cond2"])
    p.add_argument("--s_z", type=float, default=None)

    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--base_seed", type=int, nargs=2, default=[1234, 5678])
    p.add_argument("--same_noise", action="store_true")

    p.add_argument("--title", default=None)
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument(
        "--cond1_clip_bank",
        default=None,
        help="Explicit cond1 CLIP bank path. Allows custom filenames such as flower_clip_img_bank.npz.",
    )
    p.add_argument(
        "--cond2_clip_bank",
        default=None,
        help="Explicit cond2 CLIP bank path. Must be provided together with --cond1_clip_bank.",
    )
    p.add_argument(
        "--clip_bank_dir",
        default=None,
        help=(
            "Directory containing default CLIP banks, e.g. "
            "cond1_clip_img_bank.npz and cond2_clip_img_bank.npz."
        ),
    )
    # Normal save
    p = subparsers.add_parser(
        "save",
        help="Save generated samples under a fixed condition.",
    )

    p.add_argument("--model", required=True, choices=ALL_MODELS)
    p.add_argument("--output_dir", default=SAVE_DIR)
    p.add_argument("--model_dir", default=None)
    p.add_argument("--step", type=int, default=None)

    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=256)

    p.add_argument("--domain", default="cond1", choices=["cond1", "cond2"])
    p.add_argument("--s_z", type=float, default=None)

    p.add_argument("--base_seed", type=int, nargs=2, default=[2026, 1])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--print_every", type=int, default=1)
    p.add_argument(
        "--cond1_clip_bank",
        default=None,
        help="Explicit cond1 CLIP bank path. Allows custom filenames such as flower_clip_img_bank.npz.",
    )
    p.add_argument(
        "--cond2_clip_bank",
        default=None,
        help="Explicit cond2 CLIP bank path. Must be provided together with --cond1_clip_bank.",
    )
    p.add_argument(
        "--clip_bank_dir",
        default=None,
        help=(
            "Directory containing default CLIP banks, e.g. "
            "cond1_clip_img_bank.npz and cond2_clip_img_bank.npz."
        ),
    )

    # CPDM sweep preview
    p = subparsers.add_parser(
        "sweep-preview",
        help="Preview CPDM s_z sweep. One image per s_z.",
    )

    p.add_argument("--model", required=True, choices=CPDM_MODELS)
    p.add_argument("--output_dir", default=SAVE_DIR)
    p.add_argument("--model_dir", default=None)
    p.add_argument("--step", type=int, default=None)

    p.add_argument("--sweep_step", type=float, default=0.2)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--base_seed", type=int, nargs=2, default=[1234, 5678])

    p.add_argument("--title", default="CPDM $s_z$ Sweep Preview")
    p.add_argument("--dpi", type=int, default=150)

    # CPDM sweep save
    p = subparsers.add_parser(
        "sweep-save",
        help="Save CPDM sweep samples into one folder per s_z value.",
    )

    p.add_argument("--model", required=True, choices=CPDM_MODELS)
    p.add_argument("--output_dir", default=SAVE_DIR)
    p.add_argument("--model_dir", default=None)
    p.add_argument("--step", type=int, default=None)

    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_samples_per_sz", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=100)

    p.add_argument("--sweep_step", type=float, default=0.2)
    p.add_argument("--base_seed", type=int, nargs=2, default=[2026, 1])

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--print_every", type=int, default=1)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    artifact_dir = args.model_dir if args.model_dir is not None else args.output_dir
    if args.command == "preview":
        cfg = _make_sample_cfg(args, artifact_dir=artifact_dir)
        preview_samples_by_model_name(
            model_name=args.model,
            model_dir=artifact_dir,
            step=args.step,
            cfg=cfg,
            n=args.n,
            domain=args.domain,
            s_z=args.s_z,
            start_idx=args.start_idx,
            base_seed=_base_seed(args.base_seed),
            same_noise=args.same_noise,
            title=args.title,
            cols=args.cols,
            dpi=args.dpi,
        )

    elif args.command == "save":
        cfg = _make_sample_cfg(args, artifact_dir=artifact_dir)
        save_samples_by_model_name(
            model_name=args.model,
            model_dir=artifact_dir,
            out_dir=args.out_dir,
            step=args.step,
            cfg=cfg,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            domain=args.domain,
            s_z=args.s_z,
            base_seed=_base_seed(args.base_seed),
            overwrite=args.overwrite,
            print_every=args.print_every,
        )

    elif args.command == "sweep-preview":
        cfg = _make_sample_cfg(args, artifact_dir=artifact_dir)
        preview_cpdm_sweep_by_model_name(
            model_name=args.model,
            model_dir=artifact_dir,
            step=args.step,
            cfg=cfg,
            sweep_step=args.sweep_step,
            start_idx=args.start_idx,
            base_seed=_base_seed(args.base_seed),
            seed_per_image=1,
            title=args.title,
            dpi=args.dpi,
        )

    elif args.command == "sweep-save":
        cfg = _make_sample_cfg(args, artifact_dir=artifact_dir)
        save_cpdm_sweep_by_model_name(
            model_name=args.model,
            model_dir=artifact_dir,
            out_dir=args.out_dir,
            step=args.step,
            cfg=cfg,
            n_samples_per_sz=args.n_samples_per_sz,
            batch_size=args.batch_size,
            sweep_step=args.sweep_step,
            base_seed=_base_seed(args.base_seed),
            overwrite=args.overwrite,
            print_every=args.print_every,
        )

    else:
        raise RuntimeError(f"Unknown command: {args.command!r}")


if __name__ == "__main__":
    main()