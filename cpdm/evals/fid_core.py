# cpdm/evals/fid_core.py

from __future__ import annotations

import os
import csv
import json
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image
from scipy import linalg

import torch
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor


ALLOW_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class RealCache:
    """Precomputed real-data Inception feature cache."""

    name: str
    path: str
    feats: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    n: int


@dataclass
class FeatureExtractorBundle:
    """Inception feature extractor and preprocessing bundle."""

    device: torch.device
    extractor: torch.nn.Module
    transform: transforms.Compose


# Inception feature extractor
def build_inception_avgpool_extractor(
    device: Optional[str] = None,
) -> FeatureExtractorBundle:
    """Build torchvision Inception-v3 avgpool feature extractor.

    Output feature shape:
        [B, 2048]

    Important:
        This preprocessing must match the preprocessing used when building
        the real-data FID caches.
    """
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    from torchvision.models import inception_v3, Inception_V3_Weights

    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights)
    model.eval().to(device_obj)

    extractor = create_feature_extractor(
        model,
        return_nodes={"avgpool": "feat"},
    )
    extractor.eval().to(device_obj)

    transform = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return FeatureExtractorBundle(
        device=device_obj,
        extractor=extractor,
        transform=transform,
    )


# Image path utilities
def list_images(root: str, recursive: bool = True) -> List[str]:
    """List image files under a directory."""
    root_path = Path(root)

    if not root_path.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")

    if not root_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    if recursive:
        paths = [
            p
            for p in root_path.rglob("*")
            if p.is_file() and p.suffix.lower() in ALLOW_EXT
        ]
    else:
        paths = [
            p
            for p in root_path.iterdir()
            if p.is_file() and p.suffix.lower() in ALLOW_EXT
        ]

    paths = sorted(paths, key=lambda p: str(p).lower())
    return [str(p) for p in paths]


def sample_paths(
    paths: List[str],
    k: int = 0,
    seed: int = 42,
) -> List[str]:
    """Return all paths if k<=0, otherwise deterministic random subset."""
    if k is None or int(k) <= 0 or int(k) >= len(paths):
        return list(paths)

    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(paths), size=int(k), replace=False)
    idx = np.sort(idx)

    return [paths[i] for i in idx]


def export_subset_files(
    paths: List[str],
    out_dir: str,
    clear_first: bool = True,
    save_list_txt: bool = True,
) -> str:
    """Copy selected evaluation images for debugging/reproducibility."""
    out_path = Path(out_dir)

    if clear_first and out_path.exists():
        shutil.rmtree(out_path)

    out_path.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(paths):
        src_path = Path(src)
        dst = out_path / f"{i:05d}{src_path.suffix.lower()}"
        shutil.copy2(src_path, dst)

    if save_list_txt:
        with open(out_path / "selected_paths.txt", "w", encoding="utf-8") as f:
            for p in paths:
                f.write(str(p) + "\n")

    return str(out_path)


# Dataset / feature extraction
class SafeImageDataset(torch.utils.data.Dataset):
    """Image dataset that skips unreadable files through safe_collate."""

    def __init__(self, paths: List[str], transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img)
        except Exception:
            return None


def safe_collate(batch):
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None

    return torch.stack(batch, dim=0)


@torch.no_grad()
def inception_feats_from_paths(
    paths: List[str],
    bundle: FeatureExtractorBundle,
    batch_size: int = 128,
    num_workers: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extract Inception avgpool features from image paths."""
    if len(paths) == 0:
        raise RuntimeError("No image paths were provided for feature extraction.")

    dataset = SafeImageDataset(paths, transform=bundle.transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(bundle.device.type == "cuda"),
        collate_fn=safe_collate,
    )

    feats_list = []
    n_ok = 0
    n_skip_batches = 0

    if bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=bundle.device)

    t0 = time.time()

    for x in loader:
        if x is None:
            n_skip_batches += 1
            continue

        x = x.to(bundle.device, non_blocking=True)

        feat = bundle.extractor(x)["feat"]
        feat = feat.squeeze(-1).squeeze(-1)
        feat = feat.detach().cpu().numpy().astype(np.float32)

        feats_list.append(feat)
        n_ok += feat.shape[0]

    if n_ok < 2:
        raise RuntimeError(f"Too few valid images for FID/KID: n_ok={n_ok}")

    feats = np.concatenate(feats_list, axis=0)
    elapsed = time.time() - t0

    meta = {
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

    return feats, meta


# FID / KID metrics
def mu_sigma_from_feats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute feature mean and covariance."""
    x = feats.astype(np.float64)
    mu = x.mean(axis=0)
    sigma = np.cov(x, rowvar=False)
    return mu, sigma


def frechet_distance(
    mu1,
    sigma1,
    mu2,
    sigma2,
    eps: float = 1e-6,
) -> float:
    """Compute Fréchet distance between two Gaussian feature distributions."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(
        (sigma1 + np.eye(sigma1.shape[0]) * eps)
        @ (sigma2 + np.eye(sigma2.shape[0]) * eps),
        disp=False,
    )

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    value = (
        diff.dot(diff)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2.0 * np.trace(covmean)
    )

    return float(value)


def split_fid_from_feats(feats: np.ndarray, seed: int = 0) -> float:
    """Compute split-FID inside one generated set as a sanity diagnostic."""
    n = feats.shape[0]

    if n < 20:
        return float("nan")

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    half = n // 2

    f1 = feats[perm[:half]]
    f2 = feats[perm[half : 2 * half]]

    mu1, sigma1 = mu_sigma_from_feats(f1)
    mu2, sigma2 = mu_sigma_from_feats(f2)

    return frechet_distance(mu1, sigma1, mu2, sigma2)


def _poly_mmd2_torch(
    x,
    y,
    degree: int = 3,
    gamma=None,
    coef0: float = 1.0,
):
    m = x.shape[0]

    if gamma is None:
        gamma = 1.0 / x.shape[1]

    kxx = (gamma * (x @ x.t()) + coef0).pow(degree)
    kyy = (gamma * (y @ y.t()) + coef0).pow(degree)
    kxy = (gamma * (x @ y.t()) + coef0).pow(degree)

    sum_kxx = (kxx.sum() - kxx.diag().sum()) / (m * (m - 1))
    sum_kyy = (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
    sum_kxy = kxy.mean()

    return sum_kxx + sum_kyy - 2.0 * sum_kxy


def kid_from_feats(
    feats_gen: np.ndarray,
    feats_real: np.ndarray,
    bundle: FeatureExtractorBundle,
    subset_size: int = 1000,
    n_subsets: int = 50,
    seed: int = 0,
) -> Tuple[float, float]:
    """Compute polynomial-kernel KID from generated and real features."""
    n_gen = feats_gen.shape[0]
    n_real = feats_real.shape[0]
    m = min(int(subset_size), int(n_gen), int(n_real))

    if m < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(int(seed))
    vals = []

    x_all = torch.from_numpy(feats_gen.astype(np.float64)).to(bundle.device)
    y_all = torch.from_numpy(feats_real.astype(np.float64)).to(bundle.device)

    for _ in range(int(n_subsets)):
        i_gen = rng.choice(n_gen, size=m, replace=False)
        i_real = rng.choice(n_real, size=m, replace=False)

        value = _poly_mmd2_torch(x_all[i_gen], y_all[i_real])
        vals.append(float(value.detach().cpu()))

    vals = np.array(vals, dtype=np.float64)

    mean = float(vals.mean())
    std = float(vals.std(ddof=1) if len(vals) > 1 else 0.0)

    return mean, std


# Real cache loading
def load_real_cache(path: str, name: str) -> RealCache:
    """Load a real-data feature cache.

    Expected npz keys:
        feats
        mu
        sigma

    Optional:
        n
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Real cache missing for {name}: {path}")

    data = np.load(path, allow_pickle=True)

    required = {"feats", "mu", "sigma"}
    missing = required - set(data.files)

    if missing:
        raise ValueError(
            f"Real cache {path} is missing keys: {sorted(missing)}. "
            f"Available keys: {data.files}"
        )

    feats = data["feats"].astype(np.float32)
    mu = data["mu"].astype(np.float64)
    sigma = data["sigma"].astype(np.float64)

    if "n" in data.files:
        n = int(np.asarray(data["n"]).reshape(-1)[0])
    else:
        n = int(feats.shape[0])

    return RealCache(
        name=str(name),
        path=str(path),
        feats=feats,
        mu=mu,
        sigma=sigma,
        n=n,
    )


def load_two_real_caches(
    cond1_cache: str,
    cond2_cache: str,
    cond1_name: str = "cond1",
    cond2_name: str = "cond2",
) -> Dict[str, RealCache]:
    """Load real caches for two endpoint domains."""
    return {
        "cond1": load_real_cache(cond1_cache, name=cond1_name),
        "cond2": load_real_cache(cond2_cache, name=cond2_name),
    }


# Single generated-folder evaluation
def compute_against_real(
    feats_g: np.ndarray,
    mu_g: np.ndarray,
    sigma_g: np.ndarray,
    real_cache: RealCache,
    bundle: FeatureExtractorBundle,
    compute_kid: bool = True,
    kid_subset_size: int = 1000,
    kid_n_subsets: int = 50,
    kid_seed: int = 123,
) -> Dict[str, float]:
    """Compute FID and optional KID against one real cache."""
    fid = frechet_distance(
        mu_g,
        sigma_g,
        real_cache.mu,
        real_cache.sigma,
    )

    out = {
        "FID": float(fid),
    }

    if compute_kid:
        kid_mean, kid_std = kid_from_feats(
            feats_g,
            real_cache.feats,
            bundle=bundle,
            subset_size=kid_subset_size,
            n_subsets=kid_n_subsets,
            seed=kid_seed,
        )

        out["KID_mean"] = float(kid_mean)
        out["KID_std"] = float(kid_std)

    return out


def evaluate_single_gen_dir(
    gen_dir: str,
    real: Dict[str, RealCache],
    bundle: FeatureExtractorBundle,
    max_gen_images: int = 0,
    gen_sample_seed: int = 42,
    batch_size: int = 128,
    num_workers: int = 0,
    compute_kid: bool = True,
    kid_subset_size: int = 1000,
    kid_n_subsets: int = 50,
    kid_seed: int = 123,
    export_subset: bool = False,
    export_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate one generated image directory against cond1/cond2 real caches.

    This function does not know endpoint/sweep naming.
    It only receives a single generated image directory and compares it
    against both real endpoint caches.
    """
    all_paths = list_images(gen_dir, recursive=True)
    use_paths = sample_paths(
        all_paths,
        k=max_gen_images,
        seed=gen_sample_seed,
    )

    if len(use_paths) == 0:
        raise RuntimeError(f"No images found under generated directory: {gen_dir}")

    if export_subset:
        if export_dir is None:
            export_dir = os.path.join(gen_dir, "_eval_subset")
        export_dir = export_subset_files(use_paths, export_dir)
    else:
        export_dir = ""

    feats_g, meta_g = inception_feats_from_paths(
        use_paths,
        bundle=bundle,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    mu_g, sigma_g = mu_sigma_from_feats(feats_g)
    gen_split_fid = split_fid_from_feats(feats_g, seed=gen_sample_seed)

    metrics_cond1 = compute_against_real(
        feats_g,
        mu_g,
        sigma_g,
        real["cond1"],
        bundle=bundle,
        compute_kid=compute_kid,
        kid_subset_size=kid_subset_size,
        kid_n_subsets=kid_n_subsets,
        kid_seed=kid_seed,
    )

    metrics_cond2 = compute_against_real(
        feats_g,
        mu_g,
        sigma_g,
        real["cond2"],
        bundle=bundle,
        compute_kid=compute_kid,
        kid_subset_size=kid_subset_size,
        kid_n_subsets=kid_n_subsets,
        kid_seed=kid_seed + 9999,
    )

    out = {
        "gen_dir": str(gen_dir),
        "gen_n_listed": int(len(all_paths)),
        "gen_n_used": int(feats_g.shape[0]),
        "max_gen_images": int(max_gen_images or 0),
        "gen_sample_seed": int(gen_sample_seed),
        "export_subset": bool(export_subset),
        "export_dir": str(export_dir),
        "gen_split_fid": float(gen_split_fid),
        "gen_extract_sec": float(meta_g["sec"]),
        "gen_skip_batches": int(meta_g["n_skip_batches"]),
        "FID_cond1": float(metrics_cond1["FID"]),
        "FID_cond2": float(metrics_cond2["FID"]),
    }

    if compute_kid:
        out.update(
            {
                "KID_cond1_mean": float(metrics_cond1["KID_mean"]),
                "KID_cond1_std": float(metrics_cond1["KID_std"]),
                "KID_cond2_mean": float(metrics_cond2["KID_mean"]),
                "KID_cond2_std": float(metrics_cond2["KID_std"]),
            }
        )

    for key, value in meta_g.items():
        if key.startswith("gpu_"):
            out[key] = value

    return out


# Row saving utilities
def save_rows_csv(rows: List[Dict[str, Any]], out_csv: str) -> None:
    """Save list-of-dict rows to CSV."""
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise RuntimeError("No rows to save.")

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[saved] {out_csv}")


def save_rows_json(rows: List[Dict[str, Any]], out_json: str) -> None:
    """Save list-of-dict rows to JSON."""
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"[saved] {out_json}")