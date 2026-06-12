# schedules.py
import math
import numpy as np
import tensorflow as tf

# DDPM cosine beta schedule with clipping for numerical stability.
def cosine_beta_schedule(K: int, s: float = 0.008):
    steps = K + 1
    t = tf.linspace(0.0, float(K), steps)
    f = tf.math.cos(((t / float(K)) + s) / (1 + s) * math.pi / 2.0) ** 2
    alphabar = f / f[0]
    betas = 1.0 - (alphabar[1:] / alphabar[:-1])
    return tf.clip_by_value(betas, 1e-4, 1e-2)

# Precompute DDPM alpha tables from beta_t.
def alpha_tables(betas: tf.Tensor):
    """Return alpha_t, alpha_bar_t, and sigma_t = sqrt(1 - alpha_bar_t)."""
    alphas = 1.0 - betas
    alphabars = tf.math.cumprod(alphas, axis=0)
    sigma_star = tf.sqrt(1.0 - alphabars)
    return alphas, alphabars, sigma_star


def make_tau_linear(K: int):
    """Linear cumulative drift schedule.

    C[t] = t / (K - 1), for t = 0, ..., K - 1.
    """
    if K <= 1:
        Ccum = np.zeros((K,), dtype=np.float32)
        tau = np.zeros((K,), dtype=np.float32)
        return tau, Ccum

    Ccum = np.linspace(0.0, 1.0, K, endpoint=True, dtype=np.float32)
    tau = np.diff(np.concatenate([[0.0], Ccum])).astype(np.float32)
    tau[0] = 0.0
    return tau, Ccum


def make_tau_step01(K: int):
    """Step-constant cumulative drift schedule.

    C[0] = 0 and C[t] = 1 for t > 0.
    """
    if K <= 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    Ccum = np.ones((K,), dtype=np.float32)
    Ccum[0] = 0.0

    tau = np.diff(np.concatenate([[0.0], Ccum])).astype(np.float32)
    return tau, Ccum


def make_tau_cosine(K: int, tau0: float = 1e-4):
    """Cosine cumulative drift schedule with C[0]=0.

    tau0 is applied only to positive-time increments tau[1:].
    This keeps the CPDM drift zero at t=0 while avoiding nearly-zero
    drift increments at later diffusion steps.
    """
    if K <= 1:
        Ccum = np.zeros((K,), dtype=np.float32)
        tau = np.zeros((K,), dtype=np.float32)
        return tau, Ccum

    u = np.linspace(0.0, 1.0, K, endpoint=True)
    C = 0.5 * (1.0 - np.cos(np.pi * u))
    C = (C - C[0]) / (C[-1] - C[0] + 1e-12)

    tau = np.diff(np.concatenate([[0.0], C])).astype(np.float32)
    tau[0] = 0.0
    tau[1:] = np.maximum(tau[1:], tau0)

    tau_sum = np.sum(tau)
    if tau_sum <= 0:
        Ccum = np.zeros((K,), dtype=np.float32)
        tau = np.zeros((K,), dtype=np.float32)
        return tau, Ccum

    tau = tau / tau_sum
    Ccum = np.cumsum(tau)
    return tau.astype(np.float32), Ccum.astype(np.float32)


def make_drift_schedule(K: int, time_schedule: str = "linear", tau0: float = 1e-4): 
    """Return tau and cumulative C_t table for the requested CPDM drift schedule.

    This schedule controls the deterministic drift strength over diffusion time.
    It is separate from the DDPM beta / alpha_bar noise schedule.

    Supported schedules:
        - "linear"
        - "cosine"
        - "step01"
    """ 
    if time_schedule == "linear":
        return make_tau_linear(K)

    if time_schedule == "cosine":
        return make_tau_cosine(K, tau0=tau0)

    if time_schedule == "step01":
        return make_tau_step01(K)

    raise ValueError(
        f"Unknown time_schedule={time_schedule!r}. "
        "Expected one of {'linear', 'cosine', 'step01'}."
    )

# Map scalar spin coordinate s_z to a drift coefficient.
# kappa is kept for compatibility with multi-coordinate spin variants.
def psi_to_drift_scalar(s_z, kappa=1.0):
    return kappa * s_z


def prepare_sz_batch(s_z, n):
    s_z = tf.convert_to_tensor(s_z, tf.float32)

    if s_z.shape.rank == 0:
        s_z = tf.reshape(s_z, [1, 1])
    elif s_z.shape.rank == 1:
        s_z = tf.reshape(s_z, [-1, 1])

    if int(s_z.shape[0]) == 1:
        s_z = tf.tile(s_z, [n, 1])

    return s_z