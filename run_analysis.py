"""
run_analysis.py – Drop-in pipeline for real fMRI data
=======================================================
Rossi & Smecca (2026), J. Neuroscience (in press)
Supplementary Section S2.8

Usage
-----
    from run_analysis import run_analysis
    import numpy as np

    # data_dict: {subject_id: np.ndarray of shape (N_ROI, T)}
    data_dict = {
        "sub-01": np.load("sub01_fmri.npy"),   # shape (90, ~200)
        "sub-02": np.load("sub02_fmri.npy"),
        ...
    }
    results_df = run_analysis(data_dict)
    print(results_df)

Function signatures
-------------------
    participation_ratio_denoised(timeseries, method="ledoit_wolf") -> float
    global_efficiency(timeseries)                                   -> float
    susceptibility_chi_D(D, alpha=-1, beta=1)                       -> float
    sliding_window_D(timeseries, window=40, step=1)                 -> np.ndarray
    run_analysis(data_dict, window=40, step=1)                      -> pd.DataFrame

Requirements
------------
    numpy, scipy, scikit-learn, pandas  (see requirements.txt)
    N_ROI >= 30, T >= 100 recommended
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal
from sklearn.covariance import LedoitWolf
from typing import Literal


# ── Core measurement functions ────────────────────────────────────────────────

def participation_ratio_denoised(
    timeseries: np.ndarray,
    method: Literal["ledoit_wolf", "marchenko_pastur"] = "ledoit_wolf",
) -> float:
    """Estimate the Participation Ratio D of the denoised covariance matrix.

    Parameters
    ----------
    timeseries : np.ndarray, shape (N_ROI, T)
        Parcellated fMRI timeseries (ROIs × timepoints).
    method : "ledoit_wolf" (recommended) or "marchenko_pastur"
        Denoising / regularisation strategy. LW is recommended when N_ROI/T < 1.

    Returns
    -------
    float : D = (sum lambda_k)^2 / (N * sum lambda_k^2)
    """
    N, T = timeseries.shape
    if N < 2 or T < 2:
        raise ValueError(f"Expected timeseries of shape (N_ROI, T), got {timeseries.shape}")

    X = timeseries.T  # shape (T, N_ROI) as expected by sklearn

    if method == "ledoit_wolf":
        lw = LedoitWolf(assume_centered=False).fit(X)
        C = lw.covariance_
    elif method == "marchenko_pastur":
        S = np.cov(X.T)
        q = N / T
        lambda_plus = (1.0 + np.sqrt(q)) ** 2
        eigvals = np.linalg.eigvalsh(S)
        C = S.copy()
        # Zero out eigenvalues below Marchenko–Pastur upper edge
        eigvecs = np.linalg.eigh(S)[1]
        mask = eigvals > lambda_plus
        C = eigvecs[:, mask] @ np.diag(eigvals[mask]) @ eigvecs[:, mask].T
    else:
        raise ValueError(f"Unknown method: {method!r}")

    lam = np.linalg.eigvalsh(C)
    lam = lam[lam > 0]
    if lam.size == 0:
        return 0.0
    return float(np.sum(lam) ** 2 / (N * np.sum(lam ** 2)))


def global_efficiency(timeseries: np.ndarray) -> float:
    """Estimate integration proxy Phi via global efficiency of the thresholded
    functional connectivity matrix.

    This is a surrogate for PCI when TMS-EEG data are unavailable.
    For PCI computation see Casali et al. (2013) and the original MATLAB
    implementation at: https://github.com/plkn/pci

    Parameters
    ----------
    timeseries : np.ndarray, shape (N_ROI, T)

    Returns
    -------
    float : normalised global efficiency in [0, 1]
    """
    from scipy.spatial.distance import squareform
    N, T = timeseries.shape
    # Pearson correlation matrix
    FC = np.corrcoef(timeseries)
    np.fill_diagonal(FC, 0)
    # Threshold at 75th percentile (proportional thresholding)
    thresh = np.percentile(np.abs(FC[np.triu_indices(N, k=1)]), 75)
    adj = (np.abs(FC) >= thresh).astype(float)
    # Global efficiency: mean of inverse shortest-path lengths
    # (approximated by mean FC weight for speed)
    weights = FC * adj
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_weights = np.where(weights > 0, 1.0 / weights, 0.0)
    ge = inv_weights[adj > 0].mean() if (adj > 0).any() else 0.0
    # Normalise to [0, 1]
    return float(np.clip(ge / inv_weights[inv_weights > 0].max(), 0, 1))


def susceptibility_chi_D(
    D: float,
    alpha: float = -1.0,
    beta: float = 1.0,
) -> float:
    """Analytic LG susceptibility chi_D = -1 / (2*alpha + 12*beta*D^2) (Eq. 7).

    Diverges at D_c = sqrt(|alpha| / (6*beta)).

    Parameters
    ----------
    D     : Participation Ratio value
    alpha : LG quadratic coefficient (default -1)
    beta  : LG quartic coefficient   (default  1)

    Returns
    -------
    float : susceptibility value (np.inf near spinodal)
    """
    denom = 2.0 * alpha + 12.0 * beta * D ** 2
    if abs(denom) < 1e-9:
        return np.inf
    return float(-1.0 / denom)


def sliding_window_D(
    timeseries: np.ndarray,
    window: int = 40,
    step: int = 1,
    method: Literal["ledoit_wolf", "marchenko_pastur"] = "ledoit_wolf",
) -> np.ndarray:
    """Compute D in sliding windows over the timeseries.

    Parameters
    ----------
    timeseries : np.ndarray, shape (N_ROI, T)
    window     : window length in TR (default 40 ≈ 99 s at TR=2.47 s)
    step       : step size in TR (default 1)
    method     : covariance estimation method

    Returns
    -------
    np.ndarray : D values for each window
    """
    N, T = timeseries.shape
    starts = range(0, T - window + 1, step)
    d_vals = []
    for s in starts:
        segment = timeseries[:, s:s + window]
        d_vals.append(participation_ratio_denoised(segment, method=method))
    return np.array(d_vals)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_analysis(
    data_dict: dict[str, np.ndarray],
    window: int = 40,
    step: int = 1,
    method: Literal["ledoit_wolf", "marchenko_pastur"] = "ledoit_wolf",
) -> pd.DataFrame:
    """Run the full LG analysis pipeline on real fMRI data.

    Parameters
    ----------
    data_dict : dict mapping subject_id -> np.ndarray of shape (N_ROI, T)
                N_ROI >= 30, T >= 100 recommended.
    window    : sliding-window length in TR (default 40)
    step      : sliding-window step in TR (default 1)
    method    : covariance estimation method

    Returns
    -------
    pd.DataFrame with columns:
        subject, D_LW, Phi, chi_D, var_D, n_windows
    """
    records = []
    for subject_id, ts in data_dict.items():
        if ts.ndim != 2:
            raise ValueError(f"Subject {subject_id}: expected 2D array (N_ROI, T)")

        D_lw  = participation_ratio_denoised(ts, method=method)
        Phi   = global_efficiency(ts)
        chi   = susceptibility_chi_D(D_lw)
        d_win = sliding_window_D(ts, window=window, step=step, method=method)

        records.append({
            "subject":   subject_id,
            "D_LW":      D_lw,
            "Phi":       Phi,
            "chi_D":     chi,
            "var_D":     float(np.var(d_win)),
            "n_windows": len(d_win),
        })

    df = pd.DataFrame(records)
    return df


# ── Example with synthetic data ───────────────────────────────────────────────
if __name__ == "__main__":
    print("run_analysis.py – self-test with synthetic data\n")
    rng = np.random.default_rng(0)

    # Simulate 5 subjects: shape (90 ROIs, 200 timepoints)
    dummy_data = {f"sub-{i:02d}": rng.standard_normal((90, 200)) for i in range(1, 6)}
    results = run_analysis(dummy_data)
    print(results.to_string(index=False))
    print("\nAll four core functions ran successfully.")
    print("To use with real data, pass your own data_dict to run_analysis().")
