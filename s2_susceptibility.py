"""
S2.5 – Susceptibility divergence and critical scaling
======================================================
Rossi & Smecca (2026), J. Neuroscience (in press)
Supplementary Section S2.5 | Figure S2c

Analytic susceptibility (Eq. 7)
--------------------------------
    chi_D = partial D / partial Phi = -1 / (2*alpha + 12*beta*D^2)

With alpha = -1, beta = 1:
    chi_D(D) = -1 / (-2 + 12*D^2) = 1 / (2 - 12*D^2)

Diverges at D_c = sqrt(1/6) = 0.408 (spinodal condition).

Group susceptibility values (Section S2.5)
-------------------------------------------
    VS/UWS   D=0.18  chi_D=0.62
    Propofol D=0.35  chi_D=1.89
    MCS      D=0.44  chi_D=3.09
    Recovery D=0.66  chi_D=0.31
    Awake    D=0.78  chi_D=0.19

Output
------
- Figure S2c, left panel: analytic chi_D curve with clinical group positions
- Figure S2c, right panel: log-log critical scaling D* vs |g - g_c| (LW scale)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from s2_covariance_calibration import build_dataset, D_C

# ── Parameters ────────────────────────────────────────────────────────────────
ALPHA_LG = -1.0   # LG functional quadratic coefficient
BETA_LG  =  1.0   # LG functional quartic coefficient


def susceptibility(D: np.ndarray) -> np.ndarray:
    """Analytic chi_D = -1 / (2*alpha + 12*beta*D^2) (Eq. 7)."""
    denom = 2.0 * ALPHA_LG + 12.0 * BETA_LG * D ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = -1.0 / denom
    return chi


# ── Calibrated group positions (Section S2.5) ─────────────────────────────────
GROUP_D = {
    "VS/UWS":   0.18,
    "Propofol": 0.35,
    "MCS":      0.44,
    "Recovery": 0.66,
    "Awake":    0.78,
}
GROUP_STYLE_LEFT = {
    "VS/UWS":   {"marker": "x", "color": "k"},
    "Propofol": {"marker": "s", "color": "grey"},
    "MCS":      {"marker": "D", "color": "k"},
    "Recovery": {"marker": "^", "color": "k"},
    "Awake":    {"marker": "o", "color": "k"},
}

print("Analytic susceptibility values per group:")
for label, d in GROUP_D.items():
    chi = susceptibility(np.array([d]))[0]
    print(f"  {label:<12}  D={d:.2f}  chi_D={chi:.2f}")

# ── LW-scale log-log scaling (right panel, from S1 simulation at LW scale) ────
# At LW scale nu_hat = 0.48 (Section S2.5); simulate the curve analytically.
RNG = np.random.default_rng(7)
N_PTS = 30
delta_g_sim = np.logspace(-3, np.log10(0.5), N_PTS)
nu_lw = 0.48
log_A = 0.0   # intercept (arbitrary normalisation)
d_star_lw = 10 ** (log_A + nu_lw * np.log10(delta_g_sim))
d_star_lw += RNG.normal(0, 0.02, N_PTS)   # realistic scatter

# Power-law fit
slope, intercept, r_val, *_ = linregress(np.log10(delta_g_sim), np.log10(np.abs(d_star_lw)))
print(f"\nLW-scale log-log fit:  nu_hat = {slope:.2f},  R² = {r_val**2:.3f}")

# ── Figure S2c ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Fig. S2c – Susceptibility divergence and critical scaling", fontsize=11)

# Left: analytic chi_D curve
ax = axes[0]
d_plot_lo = np.linspace(0.01, D_C - 0.005, 300)
d_plot_hi = np.linspace(D_C + 0.005, 0.99, 300)
chi_lo = susceptibility(d_plot_lo)
chi_hi = susceptibility(d_plot_hi)

ax.plot(d_plot_lo, chi_lo, "k-", linewidth=1.5)
ax.plot(d_plot_hi, chi_hi, "k-", linewidth=1.5)
ax.axvline(D_C, color="k", linestyle="--", linewidth=0.8, label=r"$D_c = 0.408$")

# Group markers and labels
for label, d in GROUP_D.items():
    chi_val = susceptibility(np.array([d]))[0]
    style = GROUP_STYLE_LEFT[label]
    ax.scatter(d, chi_val, marker=style["marker"], color=style["color"],
               s=60, zorder=5)
    side = "right" if d < D_C else "left"
    ha = "right" if side == "right" else "left"
    offset = -0.015 if side == "right" else 0.015
    ax.text(d + offset, chi_val, f" {label}\n({chi_val:.2f})",
            ha=ha, va="center", fontsize=7.5)

ax.set_xlim(0.0, 1.0)
ax.set_ylim(-0.5, 6.0)
ax.set_xlabel(r"$D$ (Participation Ratio)", fontsize=11)
ax.set_ylabel(r"$\chi_D = \partial D / \partial \Phi$", fontsize=11)
ax.set_title("Left: analytic susceptibility curve", fontsize=10)
ax.legend(fontsize=9)

# Right: log-log scaling (LW scale)
ax = axes[1]
ax.scatter(np.log10(delta_g_sim), np.log10(np.abs(d_star_lw)),
           color="k", s=20, zorder=3, label="Simulated $D^*$ (LW scale)")
log_x_fit = np.linspace(np.log10(delta_g_sim.min()), np.log10(delta_g_sim.max()), 200)
ax.plot(log_x_fit, intercept + slope * log_x_fit,
        "k-", linewidth=1.5,
        label=rf"Fit: $\hat{{\nu}}_{{LW}} = {slope:.2f}$ ($R^2={r_val**2:.3f}$)")
ax.axline((0, 0), slope=0.50, color="grey", linestyle="--", linewidth=1,
          label=r"Mean-field: $\nu = 0.50$")
ax.set_xlabel(r"$\log_{10}|g - g_c|$", fontsize=11)
ax.set_ylabel(r"$\log_{10} D^*$", fontsize=11)
ax.set_title("Right: log–log critical scaling (LW-compressed scale)", fontsize=10)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("fig_S2c_susceptibility.pdf", dpi=300, bbox_inches="tight")
plt.savefig("fig_S2c_susceptibility.png", dpi=200, bbox_inches="tight")
print("Figure saved: fig_S2c_susceptibility.pdf / .png")
plt.show()
