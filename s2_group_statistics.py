"""
S2.4 – Group D ordering and statistical structure
==================================================
Rossi & Smecca (2026), J. Neuroscience (in press)
Supplementary Section S2.4 | Figure S2b

Tests
-----
- Kruskal–Wallis H test across all groups
- Pairwise Mann–Whitney U tests vs Awake (Bonferroni-corrected)
- Cohen's d effect sizes vs Awake

Output
------
- Console: full statistical table
- Figure S2b: left = Phi–D scatter per group; right = boxplots of D
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from s2_covariance_calibration import build_dataset, D_C, PHI_C

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset = build_dataset()
GROUP_ORDER = ["Awake", "Recovery", "MCS", "Propofol", "VS_UWS"]
GROUP_LABELS = ["Awake", "Recovery", "MCS", "Propofol", "VS/UWS"]

GROUP_STYLE = {
    "Awake":    {"marker": "o", "mfc": "white", "mec": "k",   "ls": "-"},
    "Recovery": {"marker": "^", "mfc": "white", "mec": "k",   "ls": "--"},
    "MCS":      {"marker": "D", "mfc": "k",     "mec": "k",   "ls": "-."},
    "Propofol": {"marker": "s", "mfc": "grey",  "mec": "k",   "ls": ":"},
    "VS_UWS":   {"marker": "x", "mfc": "k",     "mec": "k",   "ls": (0, (3,1,1,1))},
}

d_arrays = [dataset[g]["D"] for g in GROUP_ORDER]
phi_arrays = [dataset[g]["Phi"] for g in GROUP_ORDER]

# ── Kruskal–Wallis ────────────────────────────────────────────────────────────
H_stat, p_kw = kruskal(*d_arrays)
print(f"Kruskal–Wallis:  H = {H_stat:.1f},  p = {p_kw:.2e}")

# ── Pairwise Mann–Whitney vs Awake (Bonferroni) ───────────────────────────────
n_comparisons = len(GROUP_ORDER) - 1
print(f"\nPairwise Mann–Whitney vs Awake (Bonferroni α = {0.05/n_comparisons:.4f}):")
print(f"{'Group':<12} {'U':>8} {'p (raw)':>12} {'p (corr)':>12} {'Cohen d':>10} {'sig':>5}")

awake_d = dataset["Awake"]["D"]
awake_mean = awake_d.mean()
awake_std = awake_d.std(ddof=1)

for g, label in zip(GROUP_ORDER[1:], GROUP_LABELS[1:]):
    d_g = dataset[g]["D"]
    U, p_raw = mannwhitneyu(awake_d, d_g, alternative="two-sided")
    p_corr = min(p_raw * n_comparisons, 1.0)
    # Cohen's d (pooled std)
    pooled_std = np.sqrt(
        ((len(awake_d) - 1) * awake_std**2 + (len(d_g) - 1) * d_g.std(ddof=1)**2)
        / (len(awake_d) + len(d_g) - 2)
    )
    cohens_d = abs(awake_mean - d_g.mean()) / pooled_std
    sig = "***" if p_corr < 0.001 else ("**" if p_corr < 0.01 else ("*" if p_corr < 0.05 else "n.s."))
    print(f"{label:<12} {U:>8.0f} {p_raw:>12.3e} {p_corr:>12.3e} {cohens_d:>10.2f} {sig:>5}")

# ── Figure S2b ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Fig. S2b – Group D ordering and statistical structure", fontsize=11)

# Left: Phi–D scatter
ax = axes[0]
for g, label, style in zip(GROUP_ORDER, GROUP_LABELS, GROUP_STYLE.values()):
    phi_g = dataset[g]["Phi"]
    d_g = dataset[g]["D"]
    ax.scatter(phi_g, d_g, marker=style["marker"],
               facecolors=style["mfc"], edgecolors=style["mec"],
               s=30, alpha=0.7, label=label, zorder=3)
    # OLS regression line
    m, b = np.polyfit(phi_g, d_g, 1)
    phi_sorted = np.sort(phi_g)
    ax.plot(phi_sorted, m * phi_sorted + b,
            color="grey", linestyle=style["ls"], linewidth=0.9)

ax.axvline(PHI_C, color="k", linestyle="--", linewidth=0.7)
ax.axhline(D_C, color="k", linestyle="--", linewidth=0.7)
ax.set_xlabel(r"$\Phi$ (PCI proxy)", fontsize=11)
ax.set_ylabel(r"$D$ (Participation Ratio)", fontsize=11)
ax.legend(fontsize=8)
ax.set_title("Left: Φ–D covariation by group", fontsize=10)

# Right: boxplots of D
ax = axes[1]
box_data = [dataset[g]["D"] for g in GROUP_ORDER]
bp = ax.boxplot(box_data, labels=GROUP_LABELS, patch_artist=True,
                medianprops=dict(color="k", linewidth=1.5),
                whiskerprops=dict(color="k"),
                capprops=dict(color="k"),
                flierprops=dict(marker="o", ms=3, color="grey"))

fills = ["white", "lightgrey", "k", "grey", "white"]
hatches = ["", "", "", "", "////"]
for patch, fill, hatch in zip(bp["boxes"], fills, hatches):
    patch.set_facecolor(fill)
    patch.set_hatch(hatch)

ax.axhline(D_C, color="k", linestyle="--", linewidth=0.9,
           label=r"$D_c = 0.408$")
ax.set_ylabel(r"$D$ (Participation Ratio)", fontsize=11)
ax.set_title("Right: D distribution per group", fontsize=10)
ax.legend(fontsize=9)

# Significance brackets vs Awake
y_top = 1.05
for i, g in enumerate(GROUP_ORDER[1:], start=2):
    d_g = dataset[g]["D"]
    _, p_raw = mannwhitneyu(awake_d, d_g, alternative="two-sided")
    p_corr = min(p_raw * n_comparisons, 1.0)
    sig = "***" if p_corr < 0.001 else ("**" if p_corr < 0.01 else ("*" if p_corr < 0.05 else ""))
    if sig:
        ax.annotate("", xy=(i, y_top), xytext=(1, y_top),
                    arrowprops=dict(arrowstyle="-", color="k", lw=0.8))
        ax.text((1 + i) / 2, y_top + 0.01, sig, ha="center", fontsize=8)
        y_top += 0.05

plt.tight_layout()
plt.savefig("fig_S2b_group_statistics.pdf", dpi=300, bbox_inches="tight")
plt.savefig("fig_S2b_group_statistics.png", dpi=200, bbox_inches="tight")
print("\nFigure saved: fig_S2b_group_statistics.pdf / .png")
plt.show()
