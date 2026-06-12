# drift.py
import os
import numpy as np
import tensorflow as tf

from .schedules import alpha_tables, make_drift_schedule, psi_to_drift_scalar

# Default CPDM basis used in the main comparison table.
UHAT_DATASET_DIFF = "dataset_diff" 
UHAT_CONST = "const"
UHAT_RANDOM = "random"
VALID_UHAT_MODES = {UHAT_DATASET_DIFF, UHAT_CONST, UHAT_RANDOM}


# Normalize global mean pixel L2 norm so different u_hat modes are compared
# mainly by direction rather than by scale.
def _normalize_mean_hwk(x, target=0.6, eps=1e-8): 
    """Scale an HxWxC field so the mean pixel L2 norm is target."""
    x = tf.convert_to_tensor(x, tf.float32)
    norms = tf.norm(x, axis=-1, keepdims=True)
    mean_norm = tf.reduce_mean(norms)
    scale = float(target) / (mean_norm + eps)
    return tf.stop_gradient(x * scale)


def _normalize_pixel_l2_np(x: np.ndarray, target=0.6, eps=1e-8) -> np.ndarray:
    """Scale every spatial vector so each pixel channel-vector has L2 norm target."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return (float(target) * x / norms).astype(np.float32)


class DriftA_NoGain:
    """Fixed-basis CPDM drift module.

    This module builds or loads a fixed image-space basis u_hat and returns
    the noise/eta-space drift

        r_t(c) = A * C_t * psi_to_drift_scalar(s_z) * u_hat.

    The corresponding x-space mean shift is not returned here; sampling code
    applies sqrt(1 - alpha_bar_t) separately.

    The saved .npz contains only the final u-hat field under key "uhat16".
    The basis type is determined by the mode-specific filename:
        dataset_diff -> uhat16_diff.npz
        const        -> uhat16_const.npz
        random       -> uhat16_random.npz
    """

    def __init__(self, betas: tf.Tensor, cfg):
        self.cfg = cfg
        self.betas = tf.cast(betas, tf.float32)
        self.alphas, self.alphabars, self.sigma_star = alpha_tables(self.betas)
        self.K = int(cfg.K)

        self.time_schedule = getattr(cfg, "time_schedule", "linear")
        tau, Ccum = make_drift_schedule(
            self.K,
            time_schedule=self.time_schedule,
            tau0=getattr(cfg, "tau0", 1e-4),
        )

        self.tau = tf.constant(tau, tf.float32)
        # Drift time schedule C_t. This is separate from the DDPM alpha_bar schedule.
        self.C_table = tf.constant(Ccum, tf.float32) 

        # Global drift strength. A is fixed during training in this implementation.
        self.A = tf.Variable(float(cfg.A), dtype=tf.float32, trainable=False) 
        # Scalar gain for psi_to_drift_scalar; kept for compatibility with spin-coordinate variants.
        self.kappa = float(getattr(cfg, "kappa", 1.0))

        g = float(self.A.numpy()) * Ccum
        self.gamma_table = tf.constant(g.astype(np.float32), tf.float32)

        self.uhat16 = None
        self.uhat_path = None
        self._uhat_cache = {}
        self._lp_sigma = float(getattr(cfg, "lp_sigma", 3.0))
        self._lp_kernel = None

        self.uhat_mode = str(getattr(cfg, "uhat_mode", UHAT_DATASET_DIFF)).lower()
        if self.uhat_mode not in VALID_UHAT_MODES:
            raise ValueError(
                f"Unknown uhat_mode={self.uhat_mode!r}. "
                f"Expected one of {sorted(VALID_UHAT_MODES)}."
            )

        # Seed used only for random u_hat construction.
        self.uhat_seed = int(getattr(cfg, "uhat_seed", getattr(cfg, "RNG_SEED", 777)))
        self.uhat_norm_target = float(getattr(cfg, "uhat_norm_target", 0.6))

        sigma_T = float(self.sigma_star[-1].numpy())
        g_T = float(self.gamma_table[-1].numpy())
        c_T = float(Ccum[-1]) if len(Ccum) else 0.0
        # Diagnostic endpoint separation scale used only for logging.
        self.r_T = 2.0 * g_T / (sigma_T + 1e-12)

        print(
            f"[info] schedule={self.time_schedule} | "
            f"uhat_mode={self.uhat_mode} | "
            f"A={float(self.A.numpy()):.3f}, C_T={c_T:.3f} "
            f"-> gamma_T={g_T:.4f}, sigma(T)={sigma_T:.4f}, r_T={self.r_T:.3f}"
        )


    # Optional low-pass utility kept for compatibility / future ablations.
    def _ensure_lp_kernel(self, C: int, ksize: int = 11, sigma: float = 3.0):
        if self._lp_kernel is not None:
            return

        ax = np.arange(ksize) - (ksize - 1) / 2.0
        xx, yy = np.meshgrid(ax, ax)
        g = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma * sigma))
        g /= np.sum(g)

        g4 = np.zeros((ksize, ksize, C, 1), dtype=np.float32)
        for c in range(C):
            g4[:, :, c, 0] = g

        self._lp_kernel = tf.constant(g4, tf.float32)

    def _lowpass(self, x):
        _, _, _, C = x.shape
        self._ensure_lp_kernel(int(C), ksize=11, sigma=self._lp_sigma)
        return tf.nn.depthwise_conv2d(
            x,
            self._lp_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )


    # u-hat path / key handling
    def _uhat_key(self) -> str:
        return "uhat16"

    def _resolve_uhat_path(self, path: str, target_count: int) -> str:
        """Resolve either a directory or an explicit .npz file path.

        If a directory is supplied, use a simple mode-specific filename.
        If an explicit .npz path is supplied, use it as-is.

        Expected files:
            dataset_diff -> uhat16_diff.npz
            const        -> uhat16_const.npz
            random       -> uhat16_random.npz

        Note:
            target_count is still used when building dataset_diff uhat,
            but it is intentionally not encoded in the filename.
            For different prototype settings, use a different save_dir/prototypes
            folder or delete the existing prototype file before rebuilding.
        """
        path = os.fspath(path)

        if path.endswith(".npz"):
            return path

        if self.uhat_mode == UHAT_DATASET_DIFF:
            filename = "uhat16_diff.npz"

        elif self.uhat_mode == UHAT_CONST:
            filename = "uhat16_const.npz"

        elif self.uhat_mode == UHAT_RANDOM:
            filename = "uhat16_random.npz"

        else:
            raise RuntimeError(f"Unhandled uhat_mode={self.uhat_mode!r}.")

        return os.path.join(path, filename)


    # u-hat creation / loading
    def warmup_and_save_if_needed(self, flower_ds, leaf_ds, path: str, target_count=512):
        """The saved .npz contains only the final u-hat field under key "uhat16".
        The basis type is determined by the mode-specific filename:
            dataset_diff -> uhat16_diff.npz
            const        -> uhat16_const.npz
            random       -> uhat16_random.npz

        A, C_t, gamma_t, kappa, and time_schedule are runtime drift configuration
        values and are not stored in the prototype file.
        """
        resolved_path = self._resolve_uhat_path(path, target_count)
        self.uhat_path = resolved_path

        if os.path.exists(resolved_path):
            self._load_uhat(resolved_path)
            return

        out_dir = os.path.dirname(resolved_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if self.uhat_mode == UHAT_DATASET_DIFF:
            self._build_dataset_diff_uhat(flower_ds, leaf_ds, resolved_path, target_count)

        elif self.uhat_mode == UHAT_CONST:
            self._build_const_uhat(resolved_path)

        elif self.uhat_mode == UHAT_RANDOM:
            self._build_random_uhat(resolved_path)

        else:
            raise RuntimeError(f"Unhandled uhat_mode={self.uhat_mode!r}.")

    def _load_uhat(self, path: str):
        data = np.load(path, allow_pickle=True)
        key = self._uhat_key()

        if key not in data.files:
            raise ValueError(
                f"Prototype file {path} does not contain expected key {key!r}. "
                f"Available keys: {data.files}. "
                "Use a mode-specific uhat file or rebuild the prototype."
            )

        self.uhat16 = tf.Variable(data[key], dtype=tf.float32, trainable=False)

        print(
            f"[proto] loaded {self.uhat_mode} uhat16 <- {path} "
            f"key={key} shape={self.uhat16.shape}"
        )

    def _save_uhat_only(self, path: str):
        key = self._uhat_key()
        np.savez(path, **{key: self.uhat16.numpy()})
        print(f"[proto] saved {self.uhat_mode} uhat16 -> {path} key={key}")

    def _build_dataset_diff_uhat(self, flower_ds, leaf_ds, path: str, target_count: int):
        if flower_ds is None or leaf_ds is None:
            raise ValueError("dataset_diff uhat_mode requires both flower_ds and leaf_ds.")

        print(
            f"[proto] building dataset-diff uhat16 "
            f"(flowers {target_count}, leaf {target_count})..."
        )

        def _mean_16x16(ds, n_imgs: int):
            sum_16 = np.zeros((16, 16, 3), dtype=np.float32)
            count = 0

            for batch in ds:
                # Supports image-only batches and (image, condition) batches.
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]

                batch_np = batch.numpy()
                B = batch_np.shape[0]

                for k in range(B):
                    img = batch_np[k:k + 1]
                    img16 = tf.image.resize(img, (16, 16), method="area").numpy()[0]
                    sum_16 += img16
                    count += 1

                    if count >= n_imgs:
                        break

                if count >= n_imgs:
                    break

            if count == 0:
                raise ValueError("[proto] no images collected in _mean_16x16")

            if count < n_imgs:
                print(f"[proto][warn] only {count} images collected (requested {n_imgs})")

            return sum_16 / float(count)

        u_up16 = _mean_16x16(flower_ds, int(target_count))
        u_dn16 = _mean_16x16(leaf_ds, int(target_count))

        uhat16 = (u_up16 - u_dn16).astype(np.float32)

        self.uhat16 = tf.Variable(uhat16, dtype=tf.float32, trainable=False)
        self._save_uhat_only(path)

    def _build_const_uhat(self, path: str):
        print("[proto] creating CONST uhat16 and saving...")

        v = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32)
        v = v * (self.uhat_norm_target / (tf.norm(v) + 1e-12))
        uhat16 = tf.tile(v[None, None, :], [16, 16, 1])

        self.uhat16 = tf.Variable(uhat16, dtype=tf.float32, trainable=False)
        self._save_uhat_only(path)

    def _build_random_uhat(self, path: str):
        print(
            f"[proto] creating RANDOM uhat16 with seed={self.uhat_seed} "
            f"and saving..."
        )

        rng = np.random.default_rng(self.uhat_seed)
        uhat16 = rng.standard_normal((16, 16, 3), dtype=np.float32)
        uhat16 = _normalize_pixel_l2_np(
            uhat16,
            target=self.uhat_norm_target,
        )

        self.uhat16 = tf.Variable(uhat16, dtype=tf.float32, trainable=False)
        self._save_uhat_only(path)

    # u-hat expansion and drift computation
    def _uhat_full(self, H: int, W: int):
        key = (int(H), int(W))

        if key in self._uhat_cache:
            return self._uhat_cache[key]

        if self.uhat16 is None:
            raise RuntimeError(
                "uhat16 not initialized; call warmup_and_save_if_needed first."
            )

        if self.uhat_mode == UHAT_CONST:
            # Preserve the exact constant vector over the full spatial field.
            v = tf.cast(self.uhat16[0, 0, :], tf.float32)
            u = tf.tile(v[None, None, :], [int(H), int(W), 1])

        else:
            # dataset_diff/random keep spatial structure through bilinear resize.
            u = tf.image.resize(
                self.uhat16[None, ...],
                (int(H), int(W)),
                method="bilinear",
            )[0]
            u = _normalize_mean_hwk(
                u,
                target=self.uhat_norm_target,
            )

        u = tf.stop_gradient(tf.cast(u, tf.float32))
        self._uhat_cache[key] = u
        return u

    def direction(self, x: tf.Tensor):
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]

        u = self._uhat_full(int(H), int(W))
        u_batch = tf.tile(u[None, ...], [B, 1, 1, 1])

        return tf.stop_gradient(u_batch)

    def c_t_batch(self, x: tf.Tensor, t_vec: tf.Tensor, s_z_batch: tf.Tensor):
        """Return CPDM noise-space drift r_t(c).

        This function returns r_t(c), not the x-space shift m_t(c).
        The sampling code forms m_t(c) = sqrt(1 - alpha_bar_t) * r_t(c).
        """
        uhat = self.direction(x)

        g = tf.gather(self.gamma_table, tf.cast(t_vec, tf.int32))
        coeff0 = psi_to_drift_scalar(s_z_batch, kappa=self.kappa)
        coeff = tf.reshape(coeff0 * g[:, None], [-1, 1, 1, 1])

        return coeff * uhat