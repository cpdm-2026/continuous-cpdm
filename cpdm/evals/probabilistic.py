# cpdm/evals/probabilistic.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fid_core import build_inception_avgpool_extractor
from .probabilistic_core import (
    aggregate_signed_logit_response,
    compute_image_level_probabilistic_response,
    train_real_feature_classifier_from_paths,
)


# Path helpers
def _infer_model_tag(generated_model_dir: str, model_tag: Optional[str] = None) -> str:
    if model_tag is not None:
        return str(model_tag)
    return os.path.basename(str(generated_model_dir).rstrip("/"))


def _default_image_level_csv_path(
    generated_model_dir: str,
    model_tag: Optional[str] = None,
) -> str:
    model_tag = _infer_model_tag(generated_model_dir, model_tag=model_tag)
    return os.path.join(
        generated_model_dir,
        f"{model_tag}__image_level_probabilistic_response.csv",
    )


def _default_signed_logit_csv_path(
    generated_model_dir: str,
    model_tag: Optional[str] = None,
) -> str:
    model_tag = _infer_model_tag(generated_model_dir, model_tag=model_tag)
    return os.path.join(
        generated_model_dir,
        f"{model_tag}__signed_logit_response.csv",
    )


def _default_signed_logit_plot_paths(
    generated_model_dir: str,
    model_tag: Optional[str] = None,
) -> Tuple[str, str]:
    model_tag = _infer_model_tag(generated_model_dir, model_tag=model_tag)
    png_path = os.path.join(
        generated_model_dir,
        f"{model_tag}__signed_logit_response.png",
    )
    pdf_path = os.path.join(
        generated_model_dir,
        f"{model_tag}__signed_logit_response.pdf",
    )
    return png_path, pdf_path


# Public wrappers
def save_image_level_probabilistic_response(
    generated_model_dir: str,
    positive_cache_path: str,
    negative_cache_path: str,
    positive_name: str = "positive",
    negative_name: str = "negative",
    out_csv_path: Optional[str] = None,
    model_tag: Optional[str] = None,
    max_gen_images: int = 0,
    gen_sample_seed: int = 42,
    classifier_seed: int = 42,
    classifier_test_size: float = 0.2,
    classifier_max_iter: int = 3000,
    classifier_C: float = 1.0,
    classifier_class_weight: str = "balanced",
    batch_size: int = 128,
    num_workers: int = 0,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Save full image-level probabilistic response CSV.

    This is the full evidence table. Each row corresponds to one generated
    image and stores probability/logit/prediction metadata.

    Returns:
        DataFrame that was written to out_csv_path.
    """
    model_tag = _infer_model_tag(generated_model_dir, model_tag=model_tag)

    classifier_bundle = train_real_feature_classifier_from_paths(
        positive_cache_path=positive_cache_path,
        negative_cache_path=negative_cache_path,
        positive_name=positive_name,
        negative_name=negative_name,
        seed=classifier_seed,
        test_size=classifier_test_size,
        max_iter=classifier_max_iter,
        C=classifier_C,
        class_weight=classifier_class_weight,
    )

    extractor_bundle = build_inception_avgpool_extractor(device=device)

    rows = compute_image_level_probabilistic_response(
        generated_model_dir=generated_model_dir,
        classifier_bundle=classifier_bundle,
        bundle=extractor_bundle,
        model_tag=model_tag,
        max_gen_images=max_gen_images,
        gen_sample_seed=gen_sample_seed,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    df = pd.DataFrame(rows)

    if not df.empty:
        sort_cols = [col for col in ["s_z", "image_path"] if col in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)

    if out_csv_path is None:
        out_csv_path = _default_image_level_csv_path(
            generated_model_dir=generated_model_dir,
            model_tag=model_tag,
        )

    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    return df



def save_signed_logit_response(
    image_level: Any,
    out_csv_path: Optional[str] = None,
    generated_model_dir: Optional[str] = None,
    model_tag: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate image-level evidence into paper-style signed logit response.

    The output intentionally stays minimal so that it directly matches the
    paper-style plot:
        x-axis -> s_z
        y-axis -> logit_mean
        error  -> logit_sem

    Args:
        image_level:
            One of:
            - image-level DataFrame
            - path to image-level CSV
            - row iterable
        out_csv_path:
            If None, a default path is inferred when generated_model_dir is
            provided or when image_level is a CSV path.
    """
    df = aggregate_signed_logit_response(image_level)
    df = pd.DataFrame(df)

    if not df.empty:
        sort_cols = [col for col in ["s_z"] if col in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)

    if out_csv_path is None:
        if generated_model_dir is None:
            if isinstance(image_level, (str, os.PathLike)):
                generated_model_dir = str(Path(image_level).resolve().parent)
            else:
                raise ValueError(
                    "generated_model_dir must be provided when out_csv_path is "
                    "None and image_level is not a CSV path."
                )

        out_csv_path = _default_signed_logit_csv_path(
            generated_model_dir=generated_model_dir,
            model_tag=model_tag,
        )

    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    return df



def plot_signed_logit_response(
    signed_logit_response: Any,
    generated_model_dir: Optional[str] = None,
    model_tag: Optional[str] = None,
    out_png_path: Optional[str] = None,
    out_pdf_path: Optional[str] = None,
    dpi: int = 180,
    show: bool = True,
):
    """Plot the signed classifier logit response across s_z.

    This function is intentionally styled to match the original Colab plot as
    closely as possible.
    """
    if isinstance(signed_logit_response, pd.DataFrame):
        df = signed_logit_response.copy()
    elif isinstance(signed_logit_response, (str, os.PathLike)):
        df = pd.read_csv(signed_logit_response)
        if generated_model_dir is None:
            generated_model_dir = str(Path(signed_logit_response).resolve().parent)
    else:
        df = pd.DataFrame(list(signed_logit_response))

    required = {"s_z", "logit_mean", "logit_sem"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"signed logit response is missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("signed logit response is empty; nothing to plot.")

    df = df.sort_values("s_z").reset_index(drop=True)

    if model_tag is None:
        if "model_tag" in df.columns and len(df["model_tag"].dropna()) > 0:
            model_tag = str(df["model_tag"].dropna().iloc[0])
        elif generated_model_dir is not None:
            model_tag = _infer_model_tag(generated_model_dir)
        else:
            model_tag = "model"

    if out_png_path is None or out_pdf_path is None:
        if generated_model_dir is None:
            raise ValueError(
                "generated_model_dir must be provided when out_png_path or "
                "out_pdf_path is None."
            )
        default_png, default_pdf = _default_signed_logit_plot_paths(
            generated_model_dir=generated_model_dir,
            model_tag=model_tag,
        )
        if out_png_path is None:
            out_png_path = default_png
        if out_pdf_path is None:
            out_pdf_path = default_pdf

    positive_domain = None
    if "positive_domain" in df.columns and len(df["positive_domain"].dropna()) > 0:
        positive_domain = str(df["positive_domain"].dropna().iloc[0])

    ylabel = (
        f"Classifier logit: {positive_domain} positive"
        if positive_domain
        else "Classifier logit: positive endpoint"
    )

    fig = plt.figure(figsize=(8.5, 4.5), dpi=dpi)
    plt.plot(
        df["s_z"],
        df["logit_mean"],
        marker="o",
        linewidth=2,
        label="Mean classifier logit",
    )
    plt.fill_between(
        df["s_z"],
        df["logit_mean"] - df["logit_sem"],
        df["logit_mean"] + df["logit_sem"],
        alpha=0.2,
        label="± SEM",
    )
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel(r"$s_z$")
    plt.ylabel(ylabel)
    plt.title(f"Signed classifier response across $s_z$ ({model_tag})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    Path(out_png_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png_path, bbox_inches="tight")
    plt.savefig(out_pdf_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, str(out_png_path), str(out_pdf_path)
