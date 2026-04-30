"""
S1.2 – Sensitivity of the critical exponent ν̂ to the low-rank connectivity
        parameter α
=============================================================================
Rossi & Smecca (2026), J. Neuroscience (in press)
Supplementary Section S1.2 | Figure S1.2

Rationale
---------
The proof-of-concept simulation (S1.1) uses a single value alpha = 0.70.
This script reports ν̂ as a function of alpha across [0.10, 0.95], using the
same Lyapunov protocol (N = 60, n = 20 realisations, exact Lyapunov solution).

Key findings (Section S1.2)
----------------------------
- ν̂ decreases monotonically with alpha, remaining < 0.50 across the entire range.
- For the biologically plausible range alpha ∈ [0.5, 0.9], ν̂ ≈ 0.19–0.30.
- The diamond marker at alpha = 0.70 (ν̂ = 0.19) matches the main-text value.

Output
------
- Figure S1.2: ν̂ vs α with ±s.d. band and mean-field reference line
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov
from scipy.stats import linregress

# ── Reproducibility ───────────────────────────────────────────────────────────
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ── Parameters ────────────────────────────────────────────────────────────────
N = 60
N_REAL = 20
G_C = 0.97
G_RANGE = np.linspace(0.10, 0.96, 35)
ALPHA_RANGE = np.linspace(0.10, 0.95, 18)
BIOPLAUSIBLE = (0.50, 0.90)   # biologically plausible range (HCP estimates)
ALPHA_MAIN = 0.70             # value used in S1.1


# ── Functions (identical to S1.1) ─────────────────────────────────────────────

def participation_ratio(eigenvalues: np.ndarray, n: int) -> float:
    lam = eigenvalues[eigenvalues > 0]
    if lam.size == 0:
        return 0.0
    return float(np.sum(lam) ** 2 / (n * np.sum(lam ** 2)))


def build_connectivity(g: float, alpha: float) -> np.ndarray:
    v = rng.standard_normal(N)
    v /= np.linalg.norm(v)
    G = rng.standard_normal((N, N)) / np.sqrt(N)
    return g * (alpha * np.outer(v, v) + np.sqrt(1.0 - alpha ** 2) * G)


def lyapunov_covariance(W: np.ndarray) -> np.ndarray | None:
    A = W - np.eye(N)
    try:
        C = solve_continuous_lyapunov(A, -np.eye(N))
        if np.any(np.linalg.eigvalsh(C) < -1e-8):
            return None
        return C
    except np.linalg.LinAlgError:
        return None


def fit_nu(alpha: float, n_boot: int = 5) -> tuple[float, float]:
    """Estimate ν̂ and its s.d. across n_boot sub-samples for a given alpha."""
    nu_vals = []
    for _ in range(n_boot):
        d_star = []
        for g in G_RANGE:
            vals = []
            for _ in range(N_REAL // n_boot + 1):
                W = build_connectivity(g, alpha)
                C = lyapunov_covariance(W)
                if C is None:
                    continue
                lam = np.linalg.eigvalsh(C)
                vals.append(participation_ratio(lam, N))
            d_star.append(np.mean(vals) if vals else np.nan)
        d_star = np.array(d_star)
        mask = (G_RANGE < G_C) & (~np.isnan(d_star)) & (d_star > 0)
        if mask.sum() < 4:
            continue
        delta_g = np.log10(G_C - G_RANGE[mask])
        log_d = np.log10(d_star[mask])
        slope, *_ = linregress(delta_g, log_d)
        nu_vals.append(slope)
    if not nu_vals:
        return np.nan, np.nan
    return float(np.mean(nu_vals)), float(np.std(nu_vals))


# ── Sweep over alpha ──────────────────────────────────────────────────────────
print("Sweeping alpha values (S1.2) — this may take a few minutes ...")
nu_mean = np.zeros(len(ALPHA_RANGE))
nu_std = np.zeros(len(ALPHA_RANGE))

for i, alpha in enumerate(ALPHA_RANGE):
    nu_mean[i], nu_std[i] = fit_nu(alpha)
    print(f"  alpha = {alpha:.2f}  ->  nu_hat = {nu_mean[i]:.3f} ± {nu_std[i]:.3f}")

# ── Figure S1.2 ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ALPHA_RANGE, nu_mean, "k-", linewidth=1.5, label=r"Mean $\hat{\nu}$")
ax.fill_between(ALPHA_RANGE, nu_mean - nu_std, nu_mean + nu_std,
                color="grey", alpha=0.25, label=r"$\pm$ s.d.")
ax.axhline(0.50, color="k", linestyle="--", linewidth=1,
           label=r"Mean-field: $\nu = 0.50$")

# Biologically plausible range shading
ax.axvspan(*BIOPLAUSIBLE, color="silver", alpha=0.35,
           label=r"Biologically plausible $\alpha$ (HCP, $N>1000$)")

# Diamond marker for main-text value
idx_main = np.argmin(np.abs(ALPHA_RANGE - ALPHA_MAIN))
ax.scatter([ALPHA_RANGE[idx_main]], [nu_mean[idx_main]],
           marker="D", s=60, color="k", zorder=5,
           label=rf"Main text: $\alpha={ALPHA_MAIN}$, $\hat{{\nu}}={nu_mean[idx_main]:.2f}$")

ax.set_xlabel(r"Low-rank connectivity parameter $\alpha$", fontsize=11)
ax.set_ylabel(r"Critical exponent $\hat{\nu}$", fontsize=11)
ax.set_title(
    r"Fig. S1.2 – Sensitivity of $\hat{\nu}$ to $\alpha$"
    "\n"
    r"$N=60$, $n=20$ realisations, exact Lyapunov solution",
    fontsize=10,
)
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(0, 0.65)

plt.tight_layout()
plt.savefig("fig_S1_2_alpha_sensitivity.pdf", dpi=300, bbox_inches="tight")
plt.savefig("fig_S1_2_alpha_sensitivity.png", dpi=200, bbox_inches="tight")
print("\nFigure saved: fig_S1_2_alpha_sensitivity.pdf / .png")
plt.show()
