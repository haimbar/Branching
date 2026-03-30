"""
run_pm_grid.py
Standalone script for section 9e: full P×M grid study.
Generates figures and prints LaTeX table rows.
Run from Branching/scripts/:
    python3 run_pm_grid.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR    = os.path.join(SCRIPT_DIR, "..", "figures")
sys.path.insert(0, SCRIPT_DIR)

from SimulationScenario import (simulate_cells, extract_count_snapshots,
                                 estimate_rates_trajectory_ols, predict_final_counts)

os.makedirs(FIG_DIR, exist_ok=True)

def fig_path(name):
    return os.path.join(FIG_DIR, name)

# ── Parameters (must match analyze_simulation.py) ─────────────────────────────
a, b, c, d = 0.5, 0.1, 0.05, 0.4
N    = 1000
TRUE = dict(a=a, b=b, c=c, d=d)
TRUE_vals   = [a, b, c, d]
param_names = ['a', 'b', 'c', 'd']

M_traj = 200   # Monte Carlo replicates (same as rest of script)

P_grid = [3, 5, 10, 20, 50]
M_grid = [5, 10, 20, 50]
P_max  = max(P_grid)   # 51 pools per replicate (P_max estim. + 1 holdout)

ests_grid    = np.empty((len(P_grid), len(M_grid), M_traj, 4))
count_ratios = np.empty((len(P_grid), len(M_grid), M_traj, 2))

print(f"Running P×M grid: P={P_grid}, M={M_grid}, {M_traj} replicates")
for m in range(M_traj):
    base_seed = 10000 + m * (P_max + 1)
    trajectories = []
    for p_idx in range(P_max + 1):
        t_, nx_, ny_, _ = simulate_cells(**TRUE, nx0=1, ny0=0, N=N,
                                         seed=base_seed + p_idx)
        trajectories.append((t_, nx_, ny_))
    X_true = float(trajectories[P_max][1][-1])
    Y_true = float(trajectories[P_max][2][-1])

    for pi, P in enumerate(P_grid):
        for mi, Mv in enumerate(M_grid):
            snap_list = [extract_count_snapshots(t_, nx_, ny_, M=Mv)
                         for t_, nx_, ny_ in trajectories[:P]]
            e = estimate_rates_trajectory_ols(snap_list)
            ests_grid[pi, mi, m] = [e['a'], e['b'], e['c'], e['d']]
            X_hat, Y_hat = predict_final_counts(e['a'], e['b'], e['c'], e['d'], N)
            count_ratios[pi, mi, m, 0] = X_hat / X_true if X_true > 0 else np.nan
            count_ratios[pi, mi, m, 1] = Y_hat / Y_true if Y_true > 0 else np.nan

    if (m + 1) % 50 == 0:
        print(f"  {m+1}/{M_traj} replicates done", flush=True)

# ── Summary tables ─────────────────────────────────────────────────────────────
P_labels = [str(P) for P in P_grid]
M_labels = [str(Mv) for Mv in M_grid]

print("\n=== RATE SD ===")
for j, (pname, true) in enumerate(zip(param_names, TRUE_vals)):
    print(f"\n  {pname}  (true = {true:.3f}):")
    print("        " + "  ".join(f"  P={P:2d}" for P in P_grid))
    for mi, Mv in enumerate(M_grid):
        row = f"  M={Mv:2d}  "
        for pi in range(len(P_grid)):
            row += f"  {ests_grid[pi, mi, :, j].std():6.4f}"
        print(row)

print("\n=== RATE BIAS ===")
for j, (pname, true) in enumerate(zip(param_names, TRUE_vals)):
    print(f"\n  {pname}  (true = {true:.3f}):")
    print("        " + "  ".join(f"  P={P:2d}" for P in P_grid))
    for mi, Mv in enumerate(M_grid):
        row = f"  M={Mv:2d}  "
        for pi in range(len(P_grid)):
            row += f"  {ests_grid[pi, mi, :, j].mean() - true:+.4f}"
        print(row)

print("\n=== MEAN X_hat/X_true ===")
print("        " + "  ".join(f"  P={P:2d}" for P in P_grid))
for mi, Mv in enumerate(M_grid):
    row = f"  M={Mv:2d}  "
    for pi in range(len(P_grid)):
        row += f"  {np.nanmean(count_ratios[pi, mi, :, 0]):6.4f}"
    print(row)

print("\n=== SD X_hat/X_true ===")
print("        " + "  ".join(f"  P={P:2d}" for P in P_grid))
for mi, Mv in enumerate(M_grid):
    row = f"  M={Mv:2d}  "
    for pi in range(len(P_grid)):
        row += f"  {np.nanstd(count_ratios[pi, mi, :, 0]):6.4f}"
    print(row)

print("\n=== MEAN Y_hat/Y_true ===")
print("        " + "  ".join(f"  P={P:2d}" for P in P_grid))
for mi, Mv in enumerate(M_grid):
    row = f"  M={Mv:2d}  "
    for pi in range(len(P_grid)):
        row += f"  {np.nanmean(count_ratios[pi, mi, :, 1]):6.4f}"
    print(row)

print("\n=== SD Y_hat/Y_true ===")
print("        " + "  ".join(f"  P={P:2d}" for P in P_grid))
for mi, Mv in enumerate(M_grid):
    row = f"  M={Mv:2d}  "
    for pi in range(len(P_grid)):
        row += f"  {np.nanstd(count_ratios[pi, mi, :, 1]):6.4f}"
    print(row)

# ── LaTeX rows ────────────────────────────────────────────────────────────────
print("\n=== LATEX ROWS ===")
for pi, P in enumerate(P_grid):
    first = True
    for mi, Mv in enumerate(M_grid):
        p_col = str(P) if first else ""
        first = False
        parts = [p_col, str(Mv)]
        for j in range(4):
            mn = ests_grid[pi, mi, :, j].mean()
            sd = ests_grid[pi, mi, :, j].std()
            parts += [f"{mn:.3f}", f"{sd:.3f}"]
        for k in range(2):
            mn = np.nanmean(count_ratios[pi, mi, :, k])
            sd = np.nanstd(count_ratios[pi, mi, :, k])
            parts += [f"{mn:.3f}", f"{sd:.3f}"]
        print("  " + " | ".join(parts))

# ── Figure 1: SD heatmap ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle(
    f"Trajectory WLS: rate-estimate SD across $P \\times M$ grid "
    f"($N={N}$, {M_traj} replicates)",
    fontsize=11)

for j, (pname, ax) in enumerate(zip(param_names,
                                    [axes[0,0], axes[0,1], axes[1,0], axes[1,1]])):
    mat = np.array([[ests_grid[pi, mi, :, j].std()
                     for pi in range(len(P_grid))]
                    for mi in range(len(M_grid))])
    im = ax.imshow(mat, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(P_grid))); ax.set_xticklabels(P_labels)
    ax.set_yticks(range(len(M_grid))); ax.set_yticklabels(M_labels)
    ax.set_xlabel("$P$ (pools)"); ax.set_ylabel("$M$ (checkpoints)")
    ax.set_title(f"SD of $\\hat{{{pname}}}$  (true {TRUE_vals[j]:.2f})", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85)
    for mi in range(len(M_grid)):
        for pi in range(len(P_grid)):
            ax.text(pi, mi, f"{mat[mi,pi]:.3f}", ha='center', va='center',
                    fontsize=7.5)

plt.tight_layout()
fig.savefig(fig_path("fig_pm_grid_sd.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved fig_pm_grid_sd.pdf")

# ── Figure 2: bias heatmap ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle(
    f"Trajectory WLS: rate-estimate bias across $P \\times M$ grid "
    f"($N={N}$, {M_traj} replicates)",
    fontsize=11)

for j, (pname, true, ax) in enumerate(zip(param_names, TRUE_vals,
                                           [axes[0,0], axes[0,1],
                                            axes[1,0], axes[1,1]])):
    mat = np.array([[ests_grid[pi, mi, :, j].mean() - true
                     for pi in range(len(P_grid))]
                    for mi in range(len(M_grid))])
    vmax = max(np.abs(mat).max(), 1e-6)
    im = ax.imshow(mat, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(P_grid))); ax.set_xticklabels(P_labels)
    ax.set_yticks(range(len(M_grid))); ax.set_yticklabels(M_labels)
    ax.set_xlabel("$P$ (pools)"); ax.set_ylabel("$M$ (checkpoints)")
    ax.set_title(f"Bias of $\\hat{{{pname}}}$  (true {true:.2f})", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85)
    for mi in range(len(M_grid)):
        for pi in range(len(P_grid)):
            ax.text(pi, mi, f"{mat[mi,pi]:+.3f}", ha='center', va='center',
                    fontsize=7.5)

plt.tight_layout()
fig.savefig(fig_path("fig_pm_grid_bias.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved fig_pm_grid_bias.pdf")

# ── Figure 3: count-ratio heatmaps ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle(
    f"Final-count prediction: $\\hat{{n}}_X(T)/n_X(T)$ and "
    f"$\\hat{{n}}_Y(T)/n_Y(T)$ across $P \\times M$ grid "
    f"($N={N}$, {M_traj} replicates)",
    fontsize=10)

ratio_specs = [
    (0, "mean", r"Mean $\hat{n}_X(T)/n_X(T)$",  axes[0, 0]),
    (0, "sd",   r"SD $\hat{n}_X(T)/n_X(T)$",    axes[0, 1]),
    (1, "mean", r"Mean $\hat{n}_Y(T)/n_Y(T)$",  axes[1, 0]),
    (1, "sd",   r"SD $\hat{n}_Y(T)/n_Y(T)$",    axes[1, 1]),
]

for k, stat, title, ax in ratio_specs:
    if stat == "mean":
        mat = np.array([[np.nanmean(count_ratios[pi, mi, :, k])
                         for pi in range(len(P_grid))]
                        for mi in range(len(M_grid))])
        vmax = max(abs(mat.min() - 1), abs(mat.max() - 1), 1e-6)
        im = ax.imshow(mat, aspect='auto', cmap='RdBu_r',
                       vmin=1 - vmax, vmax=1 + vmax)
    else:
        mat = np.array([[np.nanstd(count_ratios[pi, mi, :, k])
                         for pi in range(len(P_grid))]
                        for mi in range(len(M_grid))])
        im = ax.imshow(mat, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(P_grid))); ax.set_xticklabels(P_labels)
    ax.set_yticks(range(len(M_grid))); ax.set_yticklabels(M_labels)
    ax.set_xlabel("$P$ (pools)"); ax.set_ylabel("$M$ (checkpoints)")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.85)
    for mi in range(len(M_grid)):
        for pi in range(len(P_grid)):
            ax.text(pi, mi, f"{mat[mi,pi]:.3f}", ha='center', va='center',
                    fontsize=7.5)

plt.tight_layout()
fig.savefig(fig_path("fig_pm_grid_counts.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved fig_pm_grid_counts.pdf")

print("\nDone.")
