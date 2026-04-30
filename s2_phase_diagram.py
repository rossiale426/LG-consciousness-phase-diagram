"""
S2.3 – Analytic phase structure and (Φ, D) phase diagram
=========================================================
Rossi & Smecca (2026), J. Neuroscience (in press)
Supplementary Section S2.3 | Figure S2a

Phase thresholds (Section S2.3)
--------------------------------
    Phi_c = 0.55  (normalised PCI threshold)
    D_c   = 0.408 (analytic spinodal: sqrt(|alpha|/6*beta) = sqrt(1/6),
                   with alpha = -1, beta = 1 in the LG functional)

These thresholds are derived independently of the Luppi data; their consistency
with those data is a non-trivial test of the framework.

Output
------
- Figure S2a: (Phi, D) phase diagram with Luppi-calibrated populations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from s2_covariance_calibration import build_dataset, PHI_C, D_C

# ── Load synthetic dataset ────────────────────────────────────────────────────
dataset = build_dataset()

# ── Plot configuration ────────────────────────────────────────────────────────
GROUP_STYLE = {
    "Awake":    {"marker": "o", "mfc": "white",  "mec": "k",    "label": "Awake"},
    "Recovery": {"marker": "^", "mfc": "white",  "mec": "k",    "label": "Recovery"},
    "MCS":      {"marker": "D", "mfc": "k",      "mec": "k",    "label": "MCS"},
    "Propofol": {"marker": "s", "mfc": "grey",   "mec": "k",    "label": "Propofol"},
    "VS_UWS":   {"marker": "x", "mfc": "k",      "mec": "k",    "label": "VS/UWS"},
}

fig, ax = plt.subplots(figsize=(7, 6))

# ── Background regions ────────────────────────────────────────────────────────
phi_min, phi_max = 0.0, 1.0
d_min, d_max = 0.0, 1.0

# Region A (Phi < Phi_c): diagonal hatch
ax.add_patch(mpatches.Rectangle(
    (phi_min, d_min), PHI_C - phi_min, d_max - d_min,
    hatch="////", facecolor="none", edgecolor="lightgrey", linewidth=0.4, zorder=0,
))
ax.text(0.27, 0.80, "Region A\n(no integration)", ha="center", va="center",
        fontsize=8, color="grey", style="italic")

# State Delta (Phi > Phi_c, D <= D_c): light-grey fill
ax.add_patch(mpatches.Rectangle(
    (PHI_C, d_min), phi_max - PHI_C, D_C - d_min,
    facecolor="lightgrey", edgecolor="none", zorder=0, alpha=0.6,
))
ax.text(0.775, 0.20, r"$\Delta$ (RIS)", ha="center", va="center",
        fontsize=9, color="dimgrey", fontweight="bold")

# Region C: white (default)
ax.text(0.775, 0.72, "C (conscious)", ha="center", va="center",
        fontsize=9, color="k")

# ── Threshold lines ───────────────────────────────────────────────────────────
ax.axvline(PHI_C, color="k", linestyle="--", linewidth=0.9)
ax.axhline(D_C, color="k", linestyle="--", linewidth=0.9)
ax.text(PHI_C + 0.005, 0.02, r"$\Phi_c = 0.55$", fontsize=8, va="bottom")
ax.text(0.02, D_C + 0.005, r"$D_c = 0.408$", fontsize=8, va="bottom")

# ── Group means ± SEM ────────────────────────────────────────────────────────
for label, style in GROUP_STYLE.items():
    grp = dataset[label]
    phi_mean = grp["Phi"].mean()
    phi_sem  = grp["Phi"].std() / np.sqrt(len(grp["Phi"]))
    d_mean   = grp["D"].mean()
    d_sem    = grp["D"].std() / np.sqrt(len(grp["D"]))

    ax.errorbar(phi_mean, d_mean,
                xerr=phi_sem, yerr=d_sem,
                fmt=style["marker"],
                markerfacecolor=style["mfc"], markeredgecolor=style["mec"],
                markersize=9, ecolor="grey", elinewidth=0.8, capsize=3,
                label=style["label"], zorder=5)

# ── Formatting ────────────────────────────────────────────────────────────────
ax.set_xlim(phi_min, phi_max)
ax.set_ylim(d_min, d_max)
ax.set_xlabel(r"$\Phi$ (normalised PCI proxy)", fontsize=12)
ax.set_ylabel(r"$D$ (Participation Ratio)", fontsize=12)
ax.set_title(
    "Fig. S2a – $(\Phi, D)$ phase diagram\n"
    "Luppi-calibrated populations, group means ± SEM",
    fontsize=10,
)
ax.legend(fontsize=9, loc="upper left", framealpha=0.9)

plt.tight_layout()
plt.savefig("fig_S2a_phase_diagram.pdf", dpi=300, bbox_inches="tight")
plt.savefig("fig_S2a_phase_diagram.png", dpi=200, bbox_inches="tight")
print("Figure saved: fig_S2a_phase_diagram.pdf / .png")
plt.show()
