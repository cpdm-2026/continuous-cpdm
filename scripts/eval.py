# scripts/eval.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple


# Allow running from repo root:
# python scripts/eval.py ...
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from cpdm.evals.fid import (
    evaluate_endpoint_fid,
    evaluate_sweep_fid,
)

from cpdm.evals.probabilistic import (
    plot_signed_logit_response,
    save_image_level_probabilistic_response,
    save_signed_logit_response,
)

# Evaluation CLI for endpoint FID/KID, CPDM sweep FID/KID,
# and probabilistic signed-logit response analysis.
def _add_fid_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--gen_root",
        required=True,
        help="Generated sample root.",
    )
    p.add_argument(
        "--real_cache_cond1",
        required=True,
        help="Real Inception cache for cond1 endpoint.",
    )
    p.add_argument(
        "--real_cache_cond2",
        required=True,
        help="Real Inception cache for cond2 endpoint.",
    )
    p.add_argument(
        "--out_csv",
        required=True,
        help="Output CSV path.",
    )
    p.add_argument(
        "--out_json",
        default=None,
        help="Optional output JSON path.",
    )
    p.add_argument(
        "--cond1_name",
        default="flower",
        help="Human-readable name for cond1 real domain.",
    )
    p.add_argument(
        "--cond2_name",
        default="leaf",
        help="Human-readable name for cond2 real domain.",
    )
    p.add_argument(
        "--model_tag",
        default=None,
        help="Optional model tag written to result rows.",
    )
    p.add_argument(
        "--max_gen_images",
        type=int,
        default=0,
        help="Use at most this many generated images per folder. 0 means all.",
    )
    p.add_argument(
        "--gen_sample_seed",
        type=int,
        default=42,
        help="Seed for deterministic generated-image subset sampling.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for Inception feature extraction.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="PyTorch DataLoader workers.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda, cuda:0, or cpu. Default: auto.",
    )
    p.add_argument(
        "--no_kid",
        action="store_true",
        help="Disable KID computation and compute FID only.",
    )
    p.add_argument(
        "--kid_subset_size",
        type=int,
        default=1000,
        help="KID subset size.",
    )
    p.add_argument(
        "--kid_n_subsets",
        type=int,
        default=50,
        help="Number of KID subsets.",
    )
    p.add_argument(
        "--kid_seed",
        type=int,
        default=123,
        help="KID random seed.",
    )
    p.add_argument(
        "--export_subset",
        action="store_true",
        help="Copy the selected generated subset for debugging.",
    )
    p.add_argument(
        "--export_root",
        default=None,
        help="Optional subset export root.",
    )


def _add_generated_model_dir_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--gen_root",
        "--generated_model_dir",
        dest="generated_model_dir",
        required=True,
        help=(
            "Generated s_z sweep root containing local sz_* folders, e.g. "
            "sz_p1_0, sz_p0_8, sz_0_0, sz_m1_0."
        ),
    )


def _add_prob_image_common_args(p: argparse.ArgumentParser) -> None:
    _add_generated_model_dir_arg(p)
    p.add_argument(
        "--positive_cache",
        "--real_cache_positive",
        dest="positive_cache_path",
        required=True,
        help="Real Inception feature cache for the positive endpoint domain.",
    )
    p.add_argument(
        "--negative_cache",
        "--real_cache_negative",
        dest="negative_cache_path",
        required=True,
        help="Real Inception feature cache for the negative endpoint domain.",
    )
    p.add_argument(
        "--positive_name",
        default="flower",
        help="Positive endpoint name. Positive logit points toward this domain.",
    )
    p.add_argument(
        "--negative_name",
        default="leaf",
        help="Negative endpoint name.",
    )
    p.add_argument(
        "--image_level_csv",
        "--out_image_level_csv",
        dest="image_level_csv",
        default=None,
        help=(
            "Output image-level probabilistic response CSV. "
            "Default: generated_model_dir/<model_tag>__image_level_probabilistic_response.csv"
        ),
    )
    p.add_argument(
        "--model_tag",
        default=None,
        help="Optional model tag written to result rows.",
    )
    p.add_argument(
        "--max_gen_images",
        type=int,
        default=0,
        help="Use at most this many generated images per s_z folder. 0 means all.",
    )
    p.add_argument(
        "--gen_sample_seed",
        type=int,
        default=42,
        help="Seed for deterministic generated-image subset sampling.",
    )
    p.add_argument(
        "--classifier_seed",
        type=int,
        default=42,
        help="Seed for real-feature classifier train/test split.",
    )
    p.add_argument(
        "--classifier_test_size",
        type=float,
        default=0.2,
        help="Held-out split ratio for the real-feature classifier sanity check.",
    )
    p.add_argument(
        "--classifier_max_iter",
        type=int,
        default=3000,
        help="Maximum iterations for logistic regression classifier.",
    )
    p.add_argument(
        "--classifier_C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for logistic regression classifier.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for Inception feature extraction.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="PyTorch DataLoader workers.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda, cuda:0, or cpu. Default: auto.",
    )


def _add_signed_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--image_level_csv",
        required=True,
        help="Input image-level probabilistic response CSV.",
    )
    p.add_argument(
        "--signed_logit_csv",
        "--out_signed_logit_csv",
        dest="signed_logit_csv",
        default=None,
        help=(
            "Output signed logit response CSV. Default: same folder as "
            "image_level_csv or generated_model_dir default convention."
        ),
    )
    p.add_argument(
        "--gen_root",
        "--generated_model_dir",
        dest="generated_model_dir",
        default=None,
        help="Optional generated model dir used for default output naming.",
    )
    p.add_argument(
        "--model_tag",
        default=None,
        help="Optional model tag used for default output naming.",
    )


def _add_plot_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--signed_logit_csv",
        required=True,
        help="Input signed logit response CSV.",
    )
    p.add_argument(
        "--gen_root",
        "--generated_model_dir",
        dest="generated_model_dir",
        default=None,
        help="Optional generated model dir used for default output naming.",
    )
    p.add_argument(
        "--model_tag",
        default=None,
        help="Optional model tag for plot title/output naming.",
    )
    p.add_argument(
        "--out_png",
        default=None,
        help="Output PNG path. Default uses generated_model_dir/model_tag convention.",
    )
    p.add_argument(
        "--out_pdf",
        default=None,
        help="Output PDF path. Default uses generated_model_dir/model_tag convention.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Plot DPI.",
    )
    p.add_argument(
        "--show_plot",
        action="store_true",
        help="Show the matplotlib window after saving. Default is save-only.",
    )


def _print_saved(label: str, path: Optional[str]) -> None:
    if path:
        print(f"[{label}] {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluation CLI for CPDM experiments."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # FID endpoint mode
    p = subparsers.add_parser(
        "fid-endpoint",
        help=(
            "Evaluate endpoint/domain samples. "
            "Expected gen_root/cond1 and gen_root/cond2."
        ),
    )
    _add_fid_common_args(p)

    # FID sweep mode
    p = subparsers.add_parser(
        "fid-sweep",
        help=(
            "Evaluate CPDM s_z sweep samples. "
            "Expected gen_root/sz_p1_0, sz_p0_8, ..., sz_m1_0."
        ),
    )
    _add_fid_common_args(p)

    # Probabilistic image-level response
    p = subparsers.add_parser(
        "prob-image-level",
        help=(
            "Save full image-level probabilistic response CSV. "
            "This computes p_positive, p_negative, signed_prob, logit, "
            "and pred_domain for each generated image."
        ),
    )
    _add_prob_image_common_args(p)

    # Probabilistic signed-logit aggregation
    p = subparsers.add_parser(
        "prob-signed-logit",
        help=(
            "Aggregate image_level.csv into paper-style signed logit response CSV."
        ),
    )
    _add_signed_common_args(p)

    # Probabilistic signed-logit plot
    p = subparsers.add_parser(
        "prob-plot-logit",
        help="Plot paper-style signed classifier logit response across s_z.",
    )
    _add_plot_common_args(p)

     
    # Probabilistic full pipeline
    p = subparsers.add_parser(
        "prob-all",
        help=(
            "Run image-level response -> signed-logit CSV -> signed-logit plot."
        ),
    )
    _add_prob_image_common_args(p)
    p.add_argument(
        "--signed_logit_csv",
        "--out_signed_logit_csv",
        dest="signed_logit_csv",
        default=None,
        help="Output signed logit response CSV.",
    )
    p.add_argument(
        "--out_png",
        default=None,
        help="Output signed-logit response PNG path.",
    )
    p.add_argument(
        "--out_pdf",
        default=None,
        help="Output signed-logit response PDF path.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Plot DPI.",
    )
    p.add_argument(
        "--no_plot",
        action="store_true",
        help="Only save CSV files; do not create the signed-logit response plot.",
    )
    p.add_argument(
        "--show_plot",
        action="store_true",
        help="Show the matplotlib window after saving. Default is save-only.",
    )

    return parser


# Command handlers
def _run_fid_endpoint(args: argparse.Namespace) -> None:
    evaluate_endpoint_fid(
        gen_root=args.gen_root,
        real_cache_cond1=args.real_cache_cond1,
        real_cache_cond2=args.real_cache_cond2,
        out_csv=args.out_csv,
        out_json=args.out_json,
        cond1_name=args.cond1_name,
        cond2_name=args.cond2_name,
        model_tag=args.model_tag,
        max_gen_images=args.max_gen_images,
        gen_sample_seed=args.gen_sample_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        compute_kid=not bool(args.no_kid),
        kid_subset_size=args.kid_subset_size,
        kid_n_subsets=args.kid_n_subsets,
        kid_seed=args.kid_seed,
        export_subset=args.export_subset,
        export_root=args.export_root,
    )
    _print_saved("FID CSV", args.out_csv)
    _print_saved("FID JSON", args.out_json)


def _run_fid_sweep(args: argparse.Namespace) -> None:
    evaluate_sweep_fid(
        gen_root=args.gen_root,
        real_cache_cond1=args.real_cache_cond1,
        real_cache_cond2=args.real_cache_cond2,
        out_csv=args.out_csv,
        out_json=args.out_json,
        cond1_name=args.cond1_name,
        cond2_name=args.cond2_name,
        model_tag=args.model_tag,
        max_gen_images=args.max_gen_images,
        gen_sample_seed=args.gen_sample_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        compute_kid=not bool(args.no_kid),
        kid_subset_size=args.kid_subset_size,
        kid_n_subsets=args.kid_n_subsets,
        kid_seed=args.kid_seed,
        export_subset=args.export_subset,
        export_root=args.export_root,
    )
    _print_saved("FID CSV", args.out_csv)
    _print_saved("FID JSON", args.out_json)


def _run_prob_image_level(args: argparse.Namespace):
    df = save_image_level_probabilistic_response(
        generated_model_dir=args.generated_model_dir,
        positive_cache_path=args.positive_cache_path,
        negative_cache_path=args.negative_cache_path,
        positive_name=args.positive_name,
        negative_name=args.negative_name,
        out_csv_path=args.image_level_csv,
        model_tag=args.model_tag,
        max_gen_images=args.max_gen_images,
        gen_sample_seed=args.gen_sample_seed,
        classifier_seed=args.classifier_seed,
        classifier_test_size=args.classifier_test_size,
        classifier_max_iter=args.classifier_max_iter,
        classifier_C=args.classifier_C,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    print(f"[IMAGE LEVEL ROWS] {len(df)}")
    _print_saved("IMAGE LEVEL CSV", df.attrs.get("image_level_csv_path"))
    return df


def _run_prob_signed_logit(args: argparse.Namespace):
    df = save_signed_logit_response(
        image_level=args.image_level_csv,
        out_csv_path=args.signed_logit_csv,
        generated_model_dir=args.generated_model_dir,
        model_tag=args.model_tag,
    )
    print(f"[SIGNED LOGIT ROWS] {len(df)}")
    _print_saved("SIGNED LOGIT CSV", df.attrs.get("signed_logit_csv_path"))
    return df


def _run_prob_plot_logit(args: argparse.Namespace) -> Tuple[object, str, str]:
    fig, png_path, pdf_path = plot_signed_logit_response(
        signed_logit_response=args.signed_logit_csv,
        generated_model_dir=args.generated_model_dir,
        model_tag=args.model_tag,
        out_png_path=args.out_png,
        out_pdf_path=args.out_pdf,
        dpi=args.dpi,
        show=bool(args.show_plot),
    )
    _print_saved("SIGNED LOGIT PNG", png_path)
    _print_saved("SIGNED LOGIT PDF", pdf_path)
    return fig, png_path, pdf_path


def _run_prob_all(args: argparse.Namespace) -> None:
    image_df = save_image_level_probabilistic_response(
        generated_model_dir=args.generated_model_dir,
        positive_cache_path=args.positive_cache_path,
        negative_cache_path=args.negative_cache_path,
        positive_name=args.positive_name,
        negative_name=args.negative_name,
        out_csv_path=args.image_level_csv,
        model_tag=args.model_tag,
        max_gen_images=args.max_gen_images,
        gen_sample_seed=args.gen_sample_seed,
        classifier_seed=args.classifier_seed,
        classifier_test_size=args.classifier_test_size,
        classifier_max_iter=args.classifier_max_iter,
        classifier_C=args.classifier_C,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    _print_saved("IMAGE LEVEL CSV", image_df.attrs.get("image_level_csv_path"))

    # Use the DataFrame directly so prob-all does not depend on knowing the
    # default image-level CSV path returned by probabilistic.py.
    signed_df = save_signed_logit_response(
        image_level=image_df,
        out_csv_path=args.signed_logit_csv,
        generated_model_dir=args.generated_model_dir,
        model_tag=args.model_tag,
    )
    _print_saved("SIGNED LOGIT CSV", signed_df.attrs.get("signed_logit_csv_path"))

    if not bool(args.no_plot):
        _, png_path, pdf_path = plot_signed_logit_response(
            signed_logit_response=signed_df,
            generated_model_dir=args.generated_model_dir,
            model_tag=args.model_tag,
            out_png_path=args.out_png,
            out_pdf_path=args.out_pdf,
            dpi=args.dpi,
            show=bool(args.show_plot),
        )
        _print_saved("SIGNED LOGIT PNG", png_path)
        _print_saved("SIGNED LOGIT PDF", pdf_path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "fid-endpoint":
        _run_fid_endpoint(args)
    elif args.command == "fid-sweep":
        _run_fid_sweep(args)
    elif args.command == "prob-image-level":
        _run_prob_image_level(args)
    elif args.command == "prob-signed-logit":
        _run_prob_signed_logit(args)
    elif args.command == "prob-plot-logit":
        _run_prob_plot_logit(args)
    elif args.command == "prob-all":
        _run_prob_all(args)
    else:
        raise RuntimeError(f"Unknown command: {args.command!r}")


if __name__ == "__main__":
    main()
