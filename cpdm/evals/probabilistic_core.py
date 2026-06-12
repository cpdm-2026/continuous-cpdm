# cpdm/evals/probabilistic_core.py

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch

from .fid_core import (
    FeatureExtractorBundle,
    RealCache,
    build_inception_avgpool_extractor,
    load_real_cache,
    list_images,
    sample_paths,
)


@dataclass
class ClassifierBundle:
    """Real-feature classifier for probabilistic response evaluation.

    Convention:
        positive domain -> label 1
        negative domain -> label 0

    Therefore, a positive classifier logit means that the generated image is
    closer to the positive endpoint domain under the trained real-feature
    classifier. In the Leaf/Flower setting, this is typically:

        positive_domain = flower
        negative_domain = leaf
    """

    classifier: Any
    positive_name: str
    negative_name: str
    positive_cache_path: str
    negative_cache_path: str
    seed: int
    test_size: float
    heldout_acc: float
    heldout_auc: float
    n_positive: int
    n_negative: int


@dataclass(frozen=True)
class SzSweepDir:
    """One generated s_z sweep directory in the local repository format."""

    s_z: float
    folder_name: str
    folder_path: str


# Local s_z sweep folder parsing
def parse_sz_dirname(name: str) -> float:
    """Parse local CPDM sweep directory names.

    This matches the local repository naming used by sample.py sweep-save:

        sz_p1_0 -> +1.0
        sz_p0_8 -> +0.8
        sz_p0_6 -> +0.6
        sz_0_0  ->  0.0
        sz_m0_2 -> -0.2
        sz_m1_0 -> -1.0

    The old Colab-style folders such as `50K_1.0` are intentionally not
    supported here. Those were one-off notebook output names, while the public
    repository uses the `sz_*` convention.
    """
    base = os.path.basename(str(name).rstrip("/"))

    if base == "sz_0_0":
        return 0.0

    match = re.match(r"^sz_([pm])(\d+)_(\d+)$", base)
    if match is None:
        raise ValueError(f"Invalid local s_z directory name: {name!r}")

    sign, integer_part, decimal_part = match.groups()
    value = float(f"{integer_part}.{decimal_part}")

    if sign == "m":
        value = -value

    return float(value)


def find_sz_sweep_dirs(generated_model_dir: str) -> List[SzSweepDir]:
    """Find local generated s_z sweep folders under generated_model_dir.

    Expected structure:
        generated_model_dir/
          sz_p1_0/
          sz_p0_8/
          sz_p0_6/
          ...
          sz_0_0/
          ...
          sz_m1_0/

    Returns:
        List of SzSweepDir sorted by s_z in ascending order. The ascending
        order is convenient for plotting from -1 to +1.
    """
    root = Path(generated_model_dir)

    if not root.exists():
        raise FileNotFoundError(
            f"generated_model_dir does not exist: {generated_model_dir}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"generated_model_dir is not a directory: {generated_model_dir}"
        )

    out: List[SzSweepDir] = []

    for path in root.iterdir():
        if not path.is_dir():
            continue

        try:
            s_z = parse_sz_dirname(path.name)
        except ValueError:
            continue

        out.append(
            SzSweepDir(
                s_z=float(s_z),
                folder_name=path.name,
                folder_path=str(path),
            )
        )

    if not out:
        raise RuntimeError(
            f"No local s_z sweep folders found under {generated_model_dir}. "
            "Expected folders like sz_p1_0, sz_p0_8, sz_0_0, sz_m1_0."
        )

    out.sort(key=lambda item: float(item.s_z))
    return out


# Real-feature classifier
def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, score))
    except Exception:
        return float("nan")


def train_real_feature_classifier(
    positive_cache: RealCache,
    negative_cache: RealCache,
    seed: int = 42,
    test_size: float = 0.2,
    max_iter: int = 3000,
    C: float = 1.0,
    class_weight: str = "balanced",
) -> ClassifierBundle:
    """Train a logistic classifier on real endpoint Inception features.

    The classifier is trained only on real caches. It is then applied to
    generated samples to estimate image-level endpoint probability and signed
    classifier logit.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_pos = positive_cache.feats.astype(np.float32)
    x_neg = negative_cache.feats.astype(np.float32)

    y_pos = np.ones(len(x_pos), dtype=np.int64)
    y_neg = np.zeros(len(x_neg), dtype=np.int64)

    x = np.concatenate([x_pos, x_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=y,
    )

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=int(max_iter),
            class_weight=class_weight,
            C=float(C),
            solver="lbfgs",
            random_state=int(seed),
        ),
    )

    classifier.fit(x_train, y_train)

    prob_test = classifier.predict_proba(x_test)[:, 1]
    pred_test = (prob_test >= 0.5).astype(np.int64)

    heldout_acc = float(accuracy_score(y_test, pred_test))
    heldout_auc = _safe_auc(y_test, prob_test)

    return ClassifierBundle(
        classifier=classifier,
        positive_name=str(positive_cache.name),
        negative_name=str(negative_cache.name),
        positive_cache_path=str(positive_cache.path),
        negative_cache_path=str(negative_cache.path),
        seed=int(seed),
        test_size=float(test_size),
        heldout_acc=heldout_acc,
        heldout_auc=heldout_auc,
        n_positive=int(len(x_pos)),
        n_negative=int(len(x_neg)),
    )


def train_real_feature_classifier_from_paths(
    positive_cache_path: str,
    negative_cache_path: str,
    positive_name: str = "positive",
    negative_name: str = "negative",
    seed: int = 42,
    test_size: float = 0.2,
    max_iter: int = 3000,
    C: float = 1.0,
    class_weight: str = "balanced",
) -> ClassifierBundle:
    """Load two real feature caches and train the real-feature classifier."""
    positive_cache = load_real_cache(positive_cache_path, name=positive_name)
    negative_cache = load_real_cache(negative_cache_path, name=negative_name)

    return train_real_feature_classifier(
        positive_cache=positive_cache,
        negative_cache=negative_cache,
        seed=seed,
        test_size=test_size,
        max_iter=max_iter,
        C=C,
        class_weight=class_weight,
    )


# Feature extraction with path preservation
class SafeImagePathDataset(torch.utils.data.Dataset):
    """Image dataset that returns transformed image and source path."""

    def __init__(self, paths: Sequence[str], transform):
        self.paths = [str(p) for p in paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), path
        except Exception:
            return None


def safe_path_collate(batch):
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    images, paths = zip(*batch)
    return torch.stack(images, dim=0), list(paths)


@torch.no_grad()
def inception_feats_from_paths_with_paths(
    paths: Sequence[str],
    bundle: FeatureExtractorBundle,
    batch_size: int = 128,
    num_workers: int = 0,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """Extract Inception features while preserving valid image paths.

    fid_core.inception_feats_from_paths returns features only. The image-level
    probabilistic CSV needs path-level alignment, so this local helper keeps
    the valid path list in the same order as extracted features.
    """
    paths = [str(p) for p in paths]

    if len(paths) == 0:
        raise RuntimeError("No image paths were provided for feature extraction.")

    dataset = SafeImagePathDataset(paths=paths, transform=bundle.transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(bundle.device.type == "cuda"),
        collate_fn=safe_path_collate,
    )

    feats_list: List[np.ndarray] = []
    valid_paths: List[str] = []
    n_ok = 0
    n_skip_batches = 0

    if bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=bundle.device)

    t0 = time.time()

    for batch in loader:
        if batch is None:
            n_skip_batches += 1
            continue

        images, batch_paths = batch
        images = images.to(bundle.device, non_blocking=True)

        feat = bundle.extractor(images)["feat"]
        feat = feat.squeeze(-1).squeeze(-1)
        feat = feat.detach().cpu().numpy().astype(np.float32)

        feats_list.append(feat)
        valid_paths.extend(batch_paths)
        n_ok += int(feat.shape[0])

    if n_ok < 2:
        raise RuntimeError(
            f"Too few valid images for probabilistic response: n_ok={n_ok}"
        )

    feats = np.concatenate(feats_list, axis=0)
    elapsed = time.time() - t0

    meta: Dict[str, Any] = {
        "n_ok": int(n_ok),
        "n_listed": int(len(paths)),
        "n_skip_batches": int(n_skip_batches),
        "sec": float(elapsed),
    }

    if bundle.device.type == "cuda":
        meta.update(
            {
                "gpu_mem_alloc_MB": float(
                    torch.cuda.memory_allocated(bundle.device) / (1024**2)
                ),
                "gpu_mem_reserved_MB": float(
                    torch.cuda.memory_reserved(bundle.device) / (1024**2)
                ),
                "gpu_peak_alloc_MB": float(
                    torch.cuda.max_memory_allocated(bundle.device) / (1024**2)
                ),
                "gpu_peak_reserved_MB": float(
                    torch.cuda.max_memory_reserved(bundle.device) / (1024**2)
                ),
            }
        )

    return feats, valid_paths, meta


# Image-level probabilistic response
def compute_image_level_rows_from_features(
    feats: np.ndarray,
    paths: Sequence[str],
    classifier_bundle: ClassifierBundle,
    model_tag: str,
    s_z: float,
    folder_name: str,
    generated_dir: str,
    generated_model_dir: str,
    gen_n_listed: Optional[int] = None,
    gen_sample_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Convert generated features into full image-level response rows."""
    if len(paths) != int(feats.shape[0]):
        raise ValueError(
            f"Number of paths and features differ: len(paths)={len(paths)}, "
            f"feats.shape[0]={feats.shape[0]}"
        )

    classifier = classifier_bundle.classifier

    p_positive = classifier.predict_proba(feats)[:, 1].astype(np.float64)
    p_negative = 1.0 - p_positive
    logit = classifier.decision_function(feats).astype(np.float64)

    rows: List[Dict[str, Any]] = []

    for img_path, p_pos, p_neg, logit_value in zip(
        paths,
        p_positive,
        p_negative,
        logit,
    ):
        pred_domain = (
            classifier_bundle.positive_name
            if float(p_pos) >= 0.5
            else classifier_bundle.negative_name
        )

        rows.append(
            {
                "model_tag": str(model_tag),
                "s_z": float(s_z),
                "folder_name": str(folder_name),
                "generated_model_dir": str(generated_model_dir),
                "generated_dir": str(generated_dir),
                "image_path": str(img_path),
                "positive_domain": str(classifier_bundle.positive_name),
                "negative_domain": str(classifier_bundle.negative_name),
                "p_positive": float(p_pos),
                "p_negative": float(p_neg),
                "signed_prob": float(p_pos - p_neg),
                "logit": float(logit_value),
                "pred_domain": str(pred_domain),
                "gen_n_listed": "" if gen_n_listed is None else int(gen_n_listed),
                "gen_sample_seed": int(gen_sample_seed),
                "classifier_seed": int(classifier_bundle.seed),
                "classifier_test_size": float(classifier_bundle.test_size),
                "classifier_heldout_acc": float(classifier_bundle.heldout_acc),
                "classifier_heldout_auc": float(classifier_bundle.heldout_auc),
            }
        )

    return rows


def compute_image_level_probabilistic_response(
    generated_model_dir: str,
    classifier_bundle: ClassifierBundle,
    bundle: Optional[FeatureExtractorBundle] = None,
    model_tag: Optional[str] = None,
    max_gen_images: int = 0,
    gen_sample_seed: int = 42,
    batch_size: int = 128,
    num_workers: int = 0,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Compute full image-level probabilistic response for an s_z sweep.

    This is the core engine for:
        save_image_level_probabilistic_response(...)

    It returns one row per valid generated image. The row contains probability,
    signed probability, classifier logit, predicted domain, and path metadata.
    """
    if model_tag is None:
        model_tag = os.path.basename(str(generated_model_dir).rstrip("/"))

    if bundle is None:
        bundle = build_inception_avgpool_extractor(device=device)

    sweep_dirs = find_sz_sweep_dirs(generated_model_dir)
    rows: List[Dict[str, Any]] = []

    for sweep_dir in sweep_dirs:
        all_paths = list_images(sweep_dir.folder_path, recursive=True)
        use_paths = sample_paths(
            all_paths,
            k=max_gen_images,
            seed=gen_sample_seed,
        )

        if len(use_paths) == 0:
            raise RuntimeError(
                f"No images found under generated directory: {sweep_dir.folder_path}"
            )

        feats, valid_paths, meta = inception_feats_from_paths_with_paths(
            use_paths,
            bundle=bundle,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        part_rows = compute_image_level_rows_from_features(
            feats=feats,
            paths=valid_paths,
            classifier_bundle=classifier_bundle,
            model_tag=str(model_tag),
            s_z=float(sweep_dir.s_z),
            folder_name=sweep_dir.folder_name,
            generated_dir=sweep_dir.folder_path,
            generated_model_dir=generated_model_dir,
            gen_n_listed=len(all_paths),
            gen_sample_seed=gen_sample_seed,
        )

        for row in part_rows:
            row["gen_n_used"] = int(feats.shape[0])
            row["extract_sec"] = float(meta["sec"])
            row["gen_skip_batches"] = int(meta["n_skip_batches"])
            for key, value in meta.items():
                if key.startswith("gpu_"):
                    row[key] = value

        rows.extend(part_rows)

    rows.sort(key=lambda r: (float(r["s_z"]), str(r["image_path"])))
    return rows


# Signed logit aggregation for the paper-style response plot
def aggregate_signed_logit_response(image_level: Any) -> List[Dict[str, Any]]:
    """Aggregate image-level rows into signed logit response rows.

    This intentionally stays minimal because the output is designed to reproduce
    the paper figure:

        x-axis: s_z
        y-axis: mean classifier logit toward the positive endpoint
        error:   SEM over generated images

    Required columns:
        model_tag, s_z, logit

    Optional columns propagated when present:
        positive_domain, negative_domain
    """
    if isinstance(image_level, pd.DataFrame):
        df = image_level.copy()
    elif isinstance(image_level, (str, os.PathLike)):
        df = pd.read_csv(image_level)
    else:
        df = pd.DataFrame(list(image_level))

    required = {"model_tag", "s_z", "logit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"image-level data is missing required columns: {sorted(missing)}"
        )

    group_cols = ["model_tag", "s_z"]
    for optional in ["positive_domain", "negative_domain"]:
        if optional in df.columns:
            group_cols.append(optional)

    rows: List[Dict[str, Any]] = []

    for keys, group in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = {col: value for col, value in zip(group_cols, keys)}
        logits = group["logit"].astype(float).to_numpy()
        n = int(len(logits))

        logit_std = float(np.std(logits, ddof=0))
        logit_sem = float(logit_std / np.sqrt(n)) if n > 0 else float("nan")

        rows.append(
            {
                **base,
                "n": n,
                "logit_mean": float(np.mean(logits)),
                "logit_std": logit_std,
                "logit_sem": logit_sem,
            }
        )

    rows.sort(key=lambda row: float(row["s_z"]))
    return rows
