"""
S1.1 – Proof-of-concept Lyapunov simulation
============================================
Rossi & Smecca (2026), J. Neuroscience (in press)
Supplementary Section S1.1 | Figures S1 (panels A and B)

Model
-----
N = 60 cortical populations with connectivity matrix:
    W = g * (alpha * v @ v.T + sqrt(1 - alpha**2) * G)
where v is a unit-norm random vector (low-rank synchrony component),
G is a Gaussian random matrix with entries ~ N(0, 1/N) (heterogeneous component),
alpha = 0.70 controls the synchrony/diversity balance, and g is the coupling
strength (control parameter, proxy for Phi).

Stationary covariance C is obtained from the continuous Lyapunov equation:
    (W - I) C + C (W - I).T + I = 0
solved exactly via scipy.linalg.solve_continuous_lyapunov.

The Participation Ratio D* is then computed from the eigenspectrum of C (Eq. 3).
Results are averaged over n = 20 independent connectivity realisations per g value.

Outputs
-------
- Figure S1, Panel A: D* vs g
- Figure S1, Panel B: log-log plot D* vs |g - g_c|, with power-law fit (nu_hat)
- Console output: fitted exponent nu_hat and R^2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov
from scipy.stats import linregress

# ── Reproducibility ───────────────────────────────────────────────────────────
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ── Model parameters (Section S1.1) ──────────────────────────────────────────
N = 60          # number of cortical populations
ALPHA = 0.70    # low-rank weight (synchrony component)
N_REAL = 20     # realisations per g value
G_RANGE = np.linspace(0.10, 0.97, 40)
G_C = 0.97      # critical coupling

# ── Core functions ────────────────────────────────────────────────────────────

def participation_ratio(eigenvalues: np.ndarray) -> float:
    """Normalised Participation Ratio of an eigenvalue spectrum (Eq. 3).

    D = (sum lambda_k)^2 / (N * sum lambda_k^2)

    Parameters
    ----------
    eigenvalues : 1-D array of non-negative eigenvalues
    """
    lam = eigenvalues[eigenvalues > 0]
    if lam.size == 0:
        return 0.0
    return float(np.sum(lam) ** 2 / (N * np.sum(lam ** 2)))


def build_connectivity(g: float, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Build the connectivity matrix W for one realisation.

    W = g * (alpha * v @ v.T + sqrt(1 - alpha^2) * G)
    """
    v = rng.standard_normal(N)
    v /= np.linalg.norm(v)
    G = rng.standard_normal((N, N)) / np.sqrt(N)
    return g * (alpha * np.outer(v, v) + np.sqrt(1.0 - alpha ** 2) * G)


def lyapunov_covariance(W: np.ndarray) -> np.ndarray | None:
    """Solve (W-I)C + C(W-I).T + I = 0 for the stationary covariance C.

    Returns None if the Lyapunov equation has no positive-definite solution
    (i.e., the system is unstable).
    """
    A = W - np.eye(N)
    try:
        C = solve_continuous_lyapunov(A, -np.eye(N))
        if np.any(np.linalg.eigvalsh(C) < -1e-8):
            return None
        return C
    except np.linalg.LinAlgError:
        return None


def simulate_d_star(g_values: np.ndarray, n_real: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean D* and s.d. across realisations for each coupling value g."""
    d_mean = np.zeros(len(g_values))
    d_std = np.zeros(len(g_values))
    for i, g in enumerate(g_values):
        d_vals = []
        attempts = 0
        while len(d_vals) < n_real and attempts < n_real * 5:
            W = build_connectivity(g, ALPHA, rng)
            C = lyapunov_covariance(W)
            attempts += 1
            if C is None:
                continue
            lam = np.linalg.eigvalsh(C)
            d_vals.append(participation_ratio(lam))
        if d_vals:
            d_mean[i] = np.mean(d_vals)
            d_std[i] = np.std(d_vals)
        else:
            d_mean[i] = np.nan
            d_std[i] = np.nan
    return d_mean, d_std


# ── Run simulation ────────────────────────────────────────────────────────────
print("Running Lyapunov simulation (S1.1) ...")
print(f"  N = {N}, alpha = {ALPHA}, n_realisations = {N_REAL}, g_c = {G_C}")

d_mean, d_std = simulate_d_star(G_RANGE, N_REAL)

# ── Power-law fit: D* ~ |g - g_c|^nu in the sub-critical regime ──────────────
mask_fit = (G_RANGE < G_C) & (~np.isnan(d_mean)) & (d_mean > 0)
delta_g = G_C - G_RANGE[mask_fit]
log_dg = np.log10(delta_g)
log_d = np.log10(d_mean[mask_fit])

slope, intercept, r_value, p_value, _ = linregress(log_dg, log_d)
nu_hat = slope
r_squared = r_value ** 2

print(f"\n  Fitted exponent nu_hat = {nu_hat:.3f}  (R² = {r_squared:.3f})")
print(f"  Mean-field prediction: nu = 0.50")

# ── Figure S1 ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle(
    "Fig. S1 – Proof-of-concept Lyapunov simulation\n"
    r"$N=60$, $\alpha=0.70$, $n=20$ realisations per $g$",
    fontsize=11,
)

# Panel A: D* vs g
ax = axes[0]
ax.errorbar(G_RANGE, d_mean, yerr=d_std, fmt="o", ms=4, color="k",
            ecolor="grey", elinewidth=0.8, capsize=2, label=r"$D^*$ (mean ± s.d.)")
ax.axvline(G_C, color="k", linestyle="--", linewidth=0.8, label=r"$g_c = 0.97$")
ax.set_xlabel("Coupling strength $g$", fontsize=11)
ax.set_ylabel(r"$D^*$ (Participation Ratio)", fontsize=11)
ax.set_title("Panel A", fontsize=10)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)

# Panel B: log-log D* vs |g - g_c|
ax = axes[1]
delta_g_all = G_C - G_RANGE[mask_fit]
ax.scatter(np.log10(delta_g_all), np.log10(d_mean[mask_fit]),
           color="k", s=20, zorder=3, label="Simulated $D^*$")
log_dg_fit = np.linspace(log_dg.min(), log_dg.max(), 200)
ax.plot(log_dg_fit, intercept + slope * log_dg_fit,
        color="k", linewidth=1.5, label=rf"Fit: $\hat{{\nu}} = {nu_hat:.2f}$ ($R^2={r_squared:.3f}$)")
ax.axline((0, 0), slope=0.50, color="grey", linestyle="--", linewidth=1,
          label=r"Mean-field: $\nu = 0.50$")
ax.set_xlabel(r"$\log_{10}|g - g_c|$", fontsize=11)
ax.set_ylabel(r"$\log_{10} D^*$", fontsize=11)
ax.set_title("Panel B", fontsize=10)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("fig_S1_lyapunov_simulation.pdf", dpi=300, bbox_inches="tight")
plt.savefig("fig_S1_lyapunov_simulation.png", dpi=200, bbox_inches="tight")
print("\nFigure saved: fig_S1_lyapunov_simulation.pdf / .png")
plt.show()
