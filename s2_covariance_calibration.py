"""
S2.2 – Covariance matrix construction and eigenspectrum calibration
===================================================================
Rossi & Smecca (2026), J. Neuroscience (in press)
Supplementary Section S2.2

Description
-----------
For each simulated subject, a structured 90×90 covariance matrix C is generated
whose Participation Ratio matches a per-subject target drawn from calibrated
group distributions derived from Luppi et al. (2019), Figs. 2–4.

Construction: C = I + s * v @ v.T + eps * I
where v is a unit-norm random vector and the spike amplitude s is solved
numerically (Brent's method) from PR(C) = D_target.

Group calibration targets (Section S2.2)
-----------------------------------------
    Awake      (n=15): mu_D=0.78, sd=0.06 | mu_Phi=0.82, sd=0.06
    Recovery   (n=15): mu_D=0.66, sd=0.06 | mu_Phi=0.78, sd=0.05
    MCS        (n=10): mu_D=0.44, sd=0.06 | mu_Phi=0.63, sd=0.07
    Propofol   (n=15): mu_D=0.35, sd=0.06 | mu_Phi=0.60, sd=0.06
    VS/UWS     (n=12): mu_D=0.18, sd=0.05 | mu_Phi=0.54, sd=0.05

Returns
-------
A dictionary {group_label: {"C": list[np.ndarray], "Phi": np.ndarray, "D": np.ndarray}}
ready for import by the other S2 scripts.
"""

import numpy as np
from scipy.optimize import brentq

# ── Reproducibility ───────────────────────────────────────────────────────────
RNG_SEED = 0
rng = np.random.default_rng(RNG_SEED)

# ── Constants ─────────────────────────────────────────────────────────────────
N_ROI = 90      # AAL parcellation regions
EPS = 0.03      # regularisation to ensure positive definiteness

# ── Calibration targets (Luppi et al. 2019, Figs. 2–4; S2.2) ─────────────────
GROUP_PARAMS = {
    "Awake":    {"n": 15, "mu_D": 0.78, "sd_D": 0.06, "mu_Phi": 0.82, "sd_Phi": 0.06},
    "Recovery": {"n": 15, "mu_D": 0.66, "sd_D": 0.06, "mu_Phi": 0.78, "sd_Phi": 0.05},
    "MCS":      {"n": 10, "mu_D": 0.44, "sd_D": 0.06, "mu_Phi": 0.63, "sd_Phi": 0.07},
    "Propofol": {"n": 15, "mu_D": 0.35, "sd_D": 0.06, "mu_Phi": 0.60, "sd_Phi": 0.06},
    "VS_UWS":   {"n": 12, "mu_D": 0.18, "sd_D": 0.05, "mu_Phi": 0.54, "sd_Phi": 0.05},
}

# Threshold constants (Section S2.3)
PHI_C = 0.55    # normalised PCI threshold
D_C = 0.408     # analytic spinodal: sqrt(|alpha| / 6*beta) = sqrt(1/6)


# ── Core functions ────────────────────────────────────────────────────────────

def participation_ratio(C: np.ndarray) -> float:
    """Normalised Participation Ratio of covariance matrix C (Eq. 3)."""
    lam = np.linalg.eigvalsh(C)
    lam = lam[lam > 0]
    if lam.size == 0:
        return 0.0
    return float(np.sum(lam) ** 2 / (N_ROI * np.sum(lam ** 2)))


def spike_covariance(s: float, v: np.ndarray) -> np.ndarray:
    """Build C = I + s * v @ v.T + eps * I for a given spike amplitude s."""
    return np.eye(N_ROI) * (1.0 + EPS) + s * np.outer(v, v)


def solve_spike_amplitude(d_target: float, v: np.ndarray,
                           s_lo: float = 0.0, s_hi: float = 1e5) -> float:
    """Find s such that PR(spike_covariance(s, v)) = d_target via Brent's method."""
    def objective(s):
        return participation_ratio(spike_covariance(s, v)) - d_target

    # PR decreases monotonically as s increases; check bounds
    if objective(s_lo) < 0:
        return s_lo
    if objective(s_hi) > 0:
        return s_hi
    return brentq(objective, s_lo, s_hi, xtol=1e-6)


def generate_group(params: dict) -> dict:
    """Generate synthetic subjects for one clinical/experimental group."""
    n = params["n"]
    d_targets = rng.normal(params["mu_D"], params["sd_D"], n).clip(1.0 / N_ROI, 1.0)
    phi_vals = rng.normal(params["mu_Phi"], params["sd_Phi"], n).clip(0.0, 1.0)

    cov_matrices = []
    d_vals = []
    for d_t in d_targets:
        v = rng.standard_normal(N_ROI)
        v /= np.linalg.norm(v)
        s = solve_spike_amplitude(d_t, v)
        C = spike_covariance(s, v)
        cov_matrices.append(C)
        d_vals.append(participation_ratio(C))

    return {
        "C": cov_matrices,
        "Phi": phi_vals,
        "D": np.array(d_vals),
        "D_target": d_targets,
    }


def build_dataset() -> dict:
    """Build the full calibrated dataset for all groups."""
    print("Building calibrated synthetic dataset (S2.2) ...")
    dataset = {}
    for label, params in GROUP_PARAMS.items():
        dataset[label] = generate_group(params)
        mu_D = dataset[label]["D"].mean()
        mu_Phi = dataset[label]["Phi"].mean()
        print(f"  {label:10s}  n={params['n']:2d}  "
              f"mu_D={mu_D:.3f} (target {params['mu_D']:.2f})  "
              f"mu_Phi={mu_Phi:.3f}")
    return dataset


# ── Run when executed directly ────────────────────────────────────────────────
if __name__ == "__main__":
    dataset = build_dataset()
    print("\nDataset ready. Import build_dataset() from this module in other scripts.")
