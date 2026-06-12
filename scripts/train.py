# scripts/train.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

# Allow running from repo root:
# python scripts/train.py ...

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpdm.config import BATCH_SIZE, SAVE_DIR, TrainConfig
from cpdm.data import make_dataset, make_clip_dataset
from cpdm.train import train_alt

SUPPORTED_SIMPLE_MODELS = [
    "onehot",
    "joint256",
    "clip_img",
    "clip_text",
    "base_cpdm",
    "continuous_cpdm",
    "cond_quad_shift_ddpm",
    "cond_quad_shift_ddpm_larger",
]

CLIP_BANK_MODELS = {"clip_img", "clip_text"}


def _resolve_default_clip_bank_paths(
    clip_bank_dir: str,
    model: str,
    cond1_clip_bank: Optional[str] = None,
    cond2_clip_bank: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve default domain-wise CLIP bank paths.

    Default public convention:
        clip_img:
            cond1_clip_img_bank.npz
            cond2_clip_img_bank.npz

        clip_text:
            cond1_clip_text_bank.npz
            cond2_clip_text_bank.npz

    Explicit --cond1_clip_bank / --cond2_clip_bank override the default names.
    """
    model = str(model).lower()

    if model not in CLIP_BANK_MODELS:
        return None, None

    bank_root = Path(clip_bank_dir)

    if model == "clip_img":
        cond1_name = "cond1_clip_img_bank.npz"
        cond2_name = "cond2_clip_img_bank.npz"
    elif model == "clip_text":
        cond1_name = "cond1_clip_text_bank.npz"
        cond2_name = "cond2_clip_text_bank.npz"
    else:
        raise ValueError(f"Unsupported CLIP-bank model: {model}")

    cond1_path = Path(cond1_clip_bank) if cond1_clip_bank else bank_root / cond1_name
    cond2_path = Path(cond2_clip_bank) if cond2_clip_bank else bank_root / cond2_name

    return str(cond1_path), str(cond2_path)


def _validate_clip_banks_if_needed(args) -> Tuple[Optional[str], Optional[str]]:
    """Fail fast when a CLIP baseline is requested but bank files are missing."""
    cond1_bank, cond2_bank = _resolve_default_clip_bank_paths(
        clip_bank_dir=args.clip_bank_dir,
        model=args.model,
        cond1_clip_bank=args.cond1_clip_bank,
        cond2_clip_bank=args.cond2_clip_bank,
    )

    if args.model not in CLIP_BANK_MODELS:
        return None, None

    missing = []
    if cond1_bank is None or not Path(cond1_bank).exists():
        missing.append(f"cond1: {cond1_bank}")
    if cond2_bank is None or not Path(cond2_bank).exists():
        missing.append(f"cond2: {cond2_bank}")

    if missing:
        raise FileNotFoundError(
            "CLIP condition bank file(s) not found.\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nExpected default naming under <output_dir>/clip_bank:\n"
            + "  cond1_clip_img_bank.npz / cond2_clip_img_bank.npz for clip_img\n"
            + "  cond1_clip_text_bank.npz / cond2_clip_text_bank.npz for clip_text\n"
            + "Or pass explicit --cond1_clip_bank and --cond2_clip_bank."
        )

    return cond1_bank, cond2_bank


def build_parser():
    parser = argparse.ArgumentParser(
        description="Training CLI for CPDM and baseline diffusion models."
    )

    # Dataset / model
    parser.add_argument("--model", required=True, choices=SUPPORTED_SIMPLE_MODELS)
    parser.add_argument(
        "--cond1_dir",
        required=True,
        help=("Prepared cond1 image folder. "
              "For scalar CPDM, cond1 corresponds to s_z=+1."
        ),
    )
    parser.add_argument(
        "--cond2_dir",
        required=True,
        help=("Prepared cond2 image folder. "
              "For scalar CPDM, cond2 corresponds to s_z=-1."
        ),
    )

    # Output layout
    parser.add_argument(
        "--output_dir",
        default=SAVE_DIR,
        help=(
            "Dataset/run-level artifact root. Model-specific artifacts are saved "
            "under <output_dir>/<model>/."
        ),
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        help=(
            "Optional model-specific output directory override. "
            "If omitted, uses <output_dir>/<model>."
        ),
    )

    # CLIP condition banks
    parser.add_argument(
        "--clip_bank_dir",
        default=None,
        help=(
            "Directory containing domain-wise CLIP condition banks. "
            "If omitted, uses <output_dir>/clip_bank."
        ),
    )
    parser.add_argument(
        "--cond1_clip_bank",
        default=None,
        help=(
            "Optional explicit condition-bank .npz for cond1. "
            "Overrides --clip_bank_dir naming convention."
        ),
    )
    parser.add_argument(
        "--cond2_clip_bank",
        default=None,
        help=(
            "Optional explicit condition-bank .npz for cond2. "
            "Overrides --clip_bank_dir naming convention."
        ),
    )

    # Dataset
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--shuffle_buf", type=int, default=8192)

    # Training

    # Number of diffusion timesteps.
    parser.add_argument("--K", type=int, default=1000) 
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=3.0)
    parser.add_argument("--rng_seed", type=int, default=777)

    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from latest checkpoint if available. Default: True.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_false",
        dest="resume",
        help="Disable checkpoint resume.",
    )

    # CPDM drift

    # CPDM drift strength.
    parser.add_argument("--A", type=float, default=2.0)
    parser.add_argument("--kappa", type=float, default=1.0)
    # CPDM drift time schedule.
    parser.add_argument("--time_schedule", type=str, default="linear")
    parser.add_argument(
        "--uhat_mode",
        type=str,
        default="dataset_diff",
        choices=["dataset_diff", "const", "random"],
    )
    parser.add_argument("--uhat_seed", type=int, default=777)
    parser.add_argument("--uhat_norm_target", type=float, default=0.6)
    parser.add_argument("--proto_target_count", type=int, default=512)

    # Continuous CPDM
    parser.add_argument("--pair_diff_lambda", type=float, default=0.1)
    parser.add_argument("--pair_delta_min", type=float, default=0.01)
    parser.add_argument("--pair_delta_max", type=float, default=0.05)
    parser.add_argument("--pair_eps", type=float, default=1e-3)
    parser.add_argument("--boundary_band_min", type=float, default=0.95)
    parser.add_argument("--boundary_band_max", type=float, default=1.0)

    # Shift-DDPM
    parser.add_argument("--shift_type", type=str, default="original")
    parser.add_argument("--shift_weight", type=float, default=0.5)
    parser.add_argument("--shift_num_cond", type=int, default=2)
    parser.add_argument("--shift_out_ch", type=int, default=3)

    return parser


def make_config(
    args,
    cond1_bank: Optional[str] = None,
    cond2_bank: Optional[str] = None,
) -> TrainConfig:
    cfg = TrainConfig()

    # Output layout
    output_dir = Path(args.output_dir)

    if args.save_dir is not None:
        model_dir = Path(args.save_dir)
    else:
        model_dir = output_dir / str(args.model)

    clip_bank_dir = Path(args.clip_bank_dir)

    cfg.output_dir = str(output_dir)
    cfg.save_dir = str(model_dir)
    cfg.model_dir = str(model_dir)

    cfg.weights_dir = str(model_dir / "weights")
    cfg.ckpt_dir = str(model_dir / "tf_ckpt")
    cfg.proto_path = str(output_dir / "prototypes")
    cfg.clip_bank_dir = str(clip_bank_dir)
    cfg.fid_stats_dir = str(output_dir / "fid_stats")

    # Core training
    cfg.train_model = str(args.model)
    cfg.K = int(args.K)
    cfg.total_steps = int(args.total_steps)
    cfg.save_every = int(args.save_every)
    cfg.lr = float(args.lr)
    cfg.grad_clip = float(args.grad_clip)
    cfg.resume = bool(args.resume)
    cfg.RNG_SEED = int(args.rng_seed)
    cfg.extra_save_steps = (int(args.total_steps),)

    # Dataset protocol
    cfg.batch_size = int(args.batch_size)
    cfg.data_seed = int(args.data_seed)
    cfg.shuffle_buf = int(args.shuffle_buf)

    # CLIP condition bank protocol
    cfg.cond1_condition_bank_path = cond1_bank
    cfg.cond2_condition_bank_path = cond2_bank
    cfg.uses_condition_bank = bool(args.model in CLIP_BANK_MODELS)

    # CPDM / drift
    cfg.A = float(args.A)
    cfg.kappa = float(args.kappa)
    cfg.time_schedule = str(args.time_schedule)
    cfg.uhat_mode = str(args.uhat_mode)
    cfg.uhat_seed = int(args.uhat_seed)
    cfg.uhat_norm_target = float(args.uhat_norm_target)
    cfg.proto_target_count = int(args.proto_target_count)

    # Continuous CPDM
    cfg.pair_diff_lambda = float(args.pair_diff_lambda)
    cfg.pair_delta_min = float(args.pair_delta_min)
    cfg.pair_delta_max = float(args.pair_delta_max)
    cfg.pair_eps = float(args.pair_eps)
    cfg.boundary_band_min = float(args.boundary_band_min)
    cfg.boundary_band_max = float(args.boundary_band_max)

    # Shift-DDPM
    cfg.shift_type = str(args.shift_type)
    cfg.shift_weight = float(args.shift_weight)
    cfg.shift_num_cond = int(args.shift_num_cond)
    cfg.shift_out_ch = int(args.shift_out_ch)

    return cfg


def main():
    parser = build_parser()
    args = parser.parse_args()
    # Dynamic default: <output_dir>/clip_bank.
    # This keeps the same code path for leaf_flower, celeb_gender, and future runs.
    if args.clip_bank_dir is None:
        args.clip_bank_dir = str(Path(args.output_dir) / "clip_bank")

    cond1_bank, cond2_bank = _validate_clip_banks_if_needed(args)
    cfg = make_config(args, cond1_bank=cond1_bank, cond2_bank=cond2_bank)

    print("[paths] output_dir:", cfg.output_dir)
    print("[paths] model_dir:", cfg.model_dir)
    print("[paths] weights_dir:", cfg.weights_dir)
    print("[paths] ckpt_dir:", cfg.ckpt_dir)
    print("[paths] proto_path:", cfg.proto_path)

    if args.model in CLIP_BANK_MODELS:
        print("[paths] clip_bank_dir:", cfg.clip_bank_dir)
        print("[paths] cond1_clip_bank:", cfg.cond1_condition_bank_path)
        print("[paths] cond2_clip_bank:", cfg.cond2_condition_bank_path)

    print("[data] loading cond1 image dataset:", args.cond1_dir)
    cond1_ds = make_dataset(
        root_dir=args.cond1_dir,
        batch_size=args.batch_size,
        seed=args.data_seed,
        shuffle_buf=args.shuffle_buf,
    )

    print("[data] loading cond2 image dataset:", args.cond2_dir)
    cond2_ds = make_dataset(
        root_dir=args.cond2_dir,
        batch_size=args.batch_size,
        seed=args.data_seed + 1,
        shuffle_buf=args.shuffle_buf,
    )

    if args.model in CLIP_BANK_MODELS:
        cond1_emb_ds = make_clip_dataset(
            bank_npz=cond1_bank,
            batch_size=args.batch_size,
            seed=args.data_seed,
            shuffle_buf=args.shuffle_buf,
            expected_dim=512,
            root_dir=args.cond1_dir,
        )

        cond2_emb_ds = make_clip_dataset(
            bank_npz=cond2_bank,
            batch_size=args.batch_size,
            seed=args.data_seed + 1,
            shuffle_buf=args.shuffle_buf,
            expected_dim=512,
            root_dir=args.cond2_dir,
        )
    else:
        cond1_emb_ds = None
        cond2_emb_ds = None

    train_alt(
        cond1_ds,
        cond2_ds,
        cfg,
        cond1_emb_ds=cond1_emb_ds,
        cond2_emb_ds=cond2_emb_ds,
    )

if __name__ == "__main__":
    main()
