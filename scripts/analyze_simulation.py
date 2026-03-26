"""
analyze_simulation.py
Run the single-pool and split-pool simulations across a grid of (K, p) values,
save all figures to ../figures/, and print a summary table used in the report.

Run from the Branching/scripts/ directory:
    python analyze_simulation.py
Figures are written to Branching/figures/.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# locate this script's directory so imports and paths are robust
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR    = os.path.join(SCRIPT_DIR, "..", "figures")

sys.path.insert(0, SCRIPT_DIR)
from SimulationScenario import simulate_cells, simulate_with_splitting

os.makedirs(FIG_DIR, exist_ok=True)

def fig_path(name):
    return os.path.join(FIG_DIR, name)

# ── Fixed parameters ──────────────────────────────────────────────────────────
a, b, c, d = 0.5, 0.1, 0.05, 0.4
N    = 1000
SEED = 42

# Theoretical long-run composition (dominant eigenvector of rate matrix)
# A = [[a, c], [b, d]],  dX/dt = aX + cY,  dY/dt = bX + dY
tr_A  = a + d                               # 0.9
det_A = a * d - b * c                       # 0.195
lam1  = (tr_A + np.sqrt(tr_A**2 - 4*det_A)) / 2   # ≈ 0.537
# Eigenvector: (a - lam1)*v1 + c*v2 = 0  =>  v2/v1 = (lam1-a)/c
ev_ratio = (lam1 - a) / c                   # Y/X at equilibrium ≈ 0.732
eq_x_frac = 1 / (1 + ev_ratio)             # X/(X+Y) ≈ 0.578

print(f"Theory: λ₁ = {lam1:.4f},  eq X/(X+Y) = {eq_x_frac:.4f},  eq Y/X = {ev_ratio:.4f}")

# ── Moment-ODE helpers (used for both CI in single-pool plot and variance fig) ─
def moment_rhs(state, t, a, b, c, d):
    """RHS of the joint first/second central moment ODEs, single X founder."""
    mu1, mu2, s11, s12, s22 = state
    dmu1 = a*mu1 + c*mu2
    dmu2 = b*mu1 + d*mu2
    ds11 = 2*a*s11 + 2*c*s12 + (a*mu1 + c*mu2)
    ds12 =   b*s11 + (a+d)*s12 + c*s22
    ds22 = 2*b*s12 + 2*d*s22 + (b*mu1 + d*mu2)
    return [dmu1, dmu2, ds11, ds12, ds22]

def solve_moments(t_grid, a, b, c, d):
    """Solve moment ODEs on t_grid; return (mu1, mu2, s11, s12, s22, var_R1, var_R2)."""
    sol = odeint(moment_rhs, [1.0, 0.0, 0.0, 0.0, 0.0], t_grid, args=(a, b, c, d))
    mu1, mu2, s11, s12, s22 = sol.T
    mu_N    = mu1 + mu2
    alpha   = np.where(mu_N > 0, mu1 / mu_N, np.nan)
    beta    = np.where(mu_N > 0, mu2 / mu_N, np.nan)
    r       = np.where(mu1  > 0, mu2 / mu1,  np.nan)
    var_R1  = (beta**2*s11 + alpha**2*s22 - 2*alpha*beta*s12) / mu_N**2
    var_R2  = (r**2*s11 + s22 - 2*r*s12) / mu1**2
    return mu1, mu2, alpha, r, var_R1, var_R2

eps = lam1 - (tr_A - np.sqrt(tr_A**2 - 4*det_A)) / 2   # spectral gap λ₁ − λ₂

# ── Helper: final-state stats from a traj ────────────────────────────────────
def traj_stats(traj):
    nx = np.array(traj["total_X"], dtype=float)
    ny = np.array(traj["total_Y"], dtype=float)
    total = nx + ny
    x_frac  = np.where(total > 0, nx / total, np.nan)
    y_over_x = np.where(nx   > 0, ny / nx,   np.nan)
    return dict(
        final_X   = int(nx[-1]),
        final_Y   = int(ny[-1]),
        final_total = int(total[-1]),
        final_x_frac  = float(np.nanmean(x_frac[-max(1, len(x_frac)//10):])),   # last 10%
        final_y_over_x = float(np.nanmean(y_over_x[-max(1, len(y_over_x)//10):])),
    )

# ── 1. Single-pool benchmark ──────────────────────────────────────────────────
times_sp, nx_sp, ny_sp, _ = simulate_cells(a=a, b=b, c=c, d=d,
                                            nx0=1, ny0=0, N=N, seed=SEED)
nx_sp  = np.array(nx_sp, dtype=float)
ny_sp  = np.array(ny_sp, dtype=float)
tot_sp = nx_sp + ny_sp

# Solve moment ODEs on a grid covering the full simulation time range
t_th = np.linspace(0, max(times_sp) * 1.02, 3000)
mu1_th, mu2_th, alpha_th, r_th, var_R1, var_R2 = solve_moments(t_th, a, b, c, d)

z95 = 1.96
ci_R1_lo = np.clip(alpha_th - z95 * np.sqrt(np.maximum(var_R1, 0)), 0, 1)
ci_R1_hi = np.clip(alpha_th + z95 * np.sqrt(np.maximum(var_R1, 0)), 0, 1)
ci_R2_lo = np.maximum(r_th   - z95 * np.sqrt(np.maximum(var_R2, 0)), 0)
ci_R2_hi =            r_th   + z95 * np.sqrt(np.maximum(var_R2, 0))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
ax1.step(times_sp, nx_sp, where="post", color="steelblue", lw=2, label="X")
ax1.step(times_sp, ny_sp, where="post", color="tomato",    lw=2, label="Y")
ax1.set_ylabel("Cell count")
ax1.set_title("Single-pool simulation")
ax1.legend(); ax1.grid(True, alpha=0.3)

xfrac_sp  = np.where(tot_sp > 0, nx_sp / tot_sp, np.nan)
yoverx_sp = np.where(nx_sp  > 0, ny_sp / nx_sp,  np.nan)

# 95 % CI bands (shaded) + theoretical mean trajectory + equilibrium reference
ax2.fill_between(t_th, ci_R1_lo, ci_R1_hi,
                 color="steelblue", alpha=0.15, label="95% CI  $X/(X+Y)$")
ax2.fill_between(t_th, ci_R2_lo, ci_R2_hi,
                 color="tomato",    alpha=0.15, label="95% CI  $Y/X$")
ax2.plot(t_th, alpha_th, color="steelblue", lw=1.2, ls="-.",
         label=r"$E[X/(X+Y)]$")
ax2.plot(t_th, r_th,     color="tomato",    lw=1.2, ls="-.",
         label=r"$E[Y/X]$")
ax2.step(times_sp, xfrac_sp,  where="post", color="steelblue", lw=2,
         label=r"Simulated $X/(X+Y)$")
ax2.step(times_sp, yoverx_sp, where="post", color="tomato",    lw=2,
         label=r"Simulated $Y/X$")
ax2.axhline(eq_x_frac, ls="--", color="steelblue", alpha=0.5,
            label=f"Limit $X/(X+Y)={eq_x_frac:.3f}$")
ax2.axhline(ev_ratio,  ls="--", color="tomato",    alpha=0.5,
            label=f"Limit $Y/X={ev_ratio:.3f}$")
ax2.set_xlabel("Time"); ax2.set_ylabel("Ratio")
ax2.set_title("Ratios with 95% confidence intervals (delta method)")
ax2.legend(fontsize=7, ncol=2); ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(fig_path("fig_single.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig_single.pdf")

sp_stats = dict(
    final_X = int(nx_sp[-1]), final_Y = int(ny_sp[-1]),
    final_x_frac = float(np.nanmean(xfrac_sp[-len(xfrac_sp)//10:])),
    final_y_over_x = float(np.nanmean(yoverx_sp[-len(yoverx_sp)//10:])),
    n_pools = 1,
)

# ── 2. Split simulations: grid of K × p ──────────────────────────────────────
K_vals = [10, 50, 100]
p_vals = [0.0, 0.5, 1.0]

all_results = {}   # key = (K, p)
summary     = {}   # key = (K, p) -> stats dict

for K in K_vals:
    for p in p_vals:
        res = simulate_with_splitting(a=a, b=b, c=c, d=d,
                                      nx0=1, ny0=0, N=N, K=K, p=p,
                                      mode="sequential", seed=SEED)
        all_results[(K, p)] = res
        st = traj_stats(res["traj"])
        st["n_pools"] = len(res["pools"])
        summary[(K, p)] = st

# ── 3. Per-K comparison figures ───────────────────────────────────────────────
colors = {"X": "steelblue", "Y": "tomato"}

for K in K_vals:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col")
    fig.suptitle(f"Split simulation: K = {K}", fontsize=13)

    for col, p in enumerate(p_vals):
        res  = all_results[(K, p)]
        traj = res["traj"]
        nx   = np.array(traj["total_X"], dtype=float)
        ny   = np.array(traj["total_Y"], dtype=float)
        total  = nx + ny
        xfrac  = np.where(total > 0, nx / total, np.nan)
        yoverx = np.where(nx   > 0, ny / nx,    np.nan)
        x_ax   = traj["x"]
        xlabel = traj["xlabel"]

        # Row 0: counts
        ax = axes[0, col]
        ax.step(x_ax, nx, where="post", color="steelblue", lw=1.8, label="Total X")
        ax.step(x_ax, ny, where="post", color="tomato",    lw=1.8, label="Total Y")
        ax.set_title(f"p = {p}")
        ax.set_ylabel("Cell count" if col == 0 else "")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Row 1: ratios
        ax = axes[1, col]
        ax.step(x_ax, xfrac,  where="post", color="steelblue", lw=1.8, label=r"$X/(X+Y)$")
        ax.step(x_ax, yoverx, where="post", color="tomato",    lw=1.8, label=r"$Y/X$")
        ax.axhline(eq_x_frac, ls="--", color="steelblue", alpha=0.4)
        ax.axhline(ev_ratio,  ls="--", color="tomato",    alpha=0.4)
        ax.set_xlabel(xlabel); ax.set_ylabel("Ratio" if col == 0 else "")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_path(f"fig_K{K}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figures/fig_K{K}.pdf")

# ── 4. Cross-K ratio comparison for p=0.5 ────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Effect of K on ratios (p = 0.5, sequential)", fontsize=12)
cmap_k = plt.get_cmap("viridis")
k_colors = {K: cmap_k(i / (len(K_vals) - 1)) for i, K in enumerate(K_vals)}

for K in K_vals:
    traj = all_results[(K, 0.5)]["traj"]
    nx   = np.array(traj["total_X"], dtype=float)
    ny   = np.array(traj["total_Y"], dtype=float)
    tot  = nx + ny
    xfrac  = np.where(tot > 0, nx / tot, np.nan)
    yoverx = np.where(nx  > 0, ny / nx,  np.nan)
    col = k_colors[K]
    ax1.step(traj["x"], xfrac,  where="post", color=col, lw=2, label=f"K={K}")
    ax2.step(traj["x"], yoverx, where="post", color=col, lw=2, label=f"K={K}")

for ax, title, theory in [(ax1, r"$X/(X+Y)$", eq_x_frac),
                           (ax2, r"$Y/X$",     ev_ratio)]:
    ax.axhline(theory, ls="--", color="gray", alpha=0.6, label="Theory")
    ax.set_title(title); ax.set_xlabel("Time (cumulative)"); ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(fig_path("fig_K_comparison.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig_K_comparison.pdf")

# ── 5. Cross-p ratio comparison for K=50 ─────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Effect of p on ratios (K = 50, sequential)", fontsize=12)
p_colors = {0.0: "tomato", 0.5: "goldenrod", 1.0: "steelblue"}

for p in p_vals:
    traj = all_results[(50, p)]["traj"]
    nx   = np.array(traj["total_X"], dtype=float)
    ny   = np.array(traj["total_Y"], dtype=float)
    tot  = nx + ny
    xfrac  = np.where(tot > 0, nx / tot, np.nan)
    yoverx = np.where(nx  > 0, ny / nx,  np.nan)
    col = p_colors[p]
    ax1.step(traj["x"], xfrac,  where="post", color=col, lw=2, label=f"p={p}")
    ax2.step(traj["x"], yoverx, where="post", color=col, lw=2, label=f"p={p}")

for ax, title, theory in [(ax1, r"$X/(X+Y)$", eq_x_frac),
                           (ax2, r"$Y/X$",     ev_ratio)]:
    ax.axhline(theory, ls="--", color="gray", alpha=0.6, label="Theory")
    ax.set_title(title); ax.set_xlabel("Time (cumulative)"); ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(fig_path("fig_p_comparison.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig_p_comparison.pdf")

# ── 2b. Parallel simulations: same K × p grid ────────────────────────────────
all_results_par = {}
summary_par     = {}

for K in K_vals:
    for p in p_vals:
        res = simulate_with_splitting(a=a, b=b, c=c, d=d,
                                      nx0=1, ny0=0, N=N, K=K, p=p,
                                      mode="parallel", seed=SEED)
        all_results_par[(K, p)] = res
        st = traj_stats(res["traj"])
        st["n_pools"] = len(res["pools"])
        summary_par[(K, p)] = st

# ── 3b. Per-K figures for parallel mode ───────────────────────────────────────
for K in K_vals:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col")
    fig.suptitle(f"Split simulation (parallel): K = {K}", fontsize=13)

    for col, p in enumerate(p_vals):
        res  = all_results_par[(K, p)]
        traj = res["traj"]
        nx   = np.array(traj["total_X"], dtype=float)
        ny   = np.array(traj["total_Y"], dtype=float)
        total  = nx + ny
        xfrac  = np.where(total > 0, nx / total, np.nan)
        yoverx = np.where(nx   > 0, ny / nx,    np.nan)
        x_ax   = traj["x"]
        xlabel = traj["xlabel"]

        ax = axes[0, col]
        ax.step(x_ax, nx, where="post", color="steelblue", lw=1.8, label="Total X")
        ax.step(x_ax, ny, where="post", color="tomato",    lw=1.8, label="Total Y")
        ax.set_title(f"p = {p}")
        ax.set_ylabel("Cell count" if col == 0 else "")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        ax.step(x_ax, xfrac,  where="post", color="steelblue", lw=1.8, label=r"$X/(X+Y)$")
        ax.step(x_ax, yoverx, where="post", color="tomato",    lw=1.8, label=r"$Y/X$")
        ax.axhline(eq_x_frac, ls="--", color="steelblue", alpha=0.4)
        ax.axhline(ev_ratio,  ls="--", color="tomato",    alpha=0.4)
        ax.set_xlabel(xlabel); ax.set_ylabel("Ratio" if col == 0 else "")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_path(f"fig_par_K{K}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figures/fig_par_K{K}.pdf")

# ── 5b. Sequential vs parallel: overlaid ratio comparison for each K ──────────
for K in K_vals:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col")
    fig.suptitle(f"Sequential vs Parallel — K = {K}", fontsize=13)

    for col, p in enumerate(p_vals):
        traj_s = all_results[(K, p)]["traj"]
        traj_p = all_results_par[(K, p)]["traj"]

        for ax_row, (traj, mode_label, ls) in enumerate([
            (traj_s, "sequential", "-"),
            (traj_p, "parallel",   "--"),
        ]):
            nx  = np.array(traj["total_X"], dtype=float)
            ny  = np.array(traj["total_Y"], dtype=float)
            tot = nx + ny
            xfrac  = np.where(tot > 0, nx / tot, np.nan)
            yoverx = np.where(nx  > 0, ny / nx,  np.nan)
            x_ax   = traj["x"]

            ax = axes[ax_row, col]
            ax.step(x_ax, xfrac,  where="post", color="steelblue", lw=1.8,
                    ls=ls, label=r"$X/(X+Y)$")
            ax.step(x_ax, yoverx, where="post", color="tomato",    lw=1.8,
                    ls=ls, label=r"$Y/X$")
            ax.axhline(eq_x_frac, ls=":", color="steelblue", alpha=0.4)
            ax.axhline(ev_ratio,  ls=":", color="tomato",    alpha=0.4)
            xlabel = traj["xlabel"]
            ax.set_title(f"p = {p}  ({mode_label})" if ax_row == 0 else mode_label)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Ratio" if col == 0 else "")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_path(f"fig_seq_vs_par_K{K}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figures/fig_seq_vs_par_K{K}.pdf")

# ── 5c. Cross-mode, cross-K at p=0.5 ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Sequential vs Parallel — effect of K at p = 0.5", fontsize=12)
cmap_k = plt.get_cmap("viridis")
k_colors = {K: cmap_k(i / (len(K_vals) - 1)) for i, K in enumerate(K_vals)}

for K in K_vals:
    col = k_colors[K]
    for mode_key, ls, lw in [("seq", "-", 2.0), ("par", "--", 1.5)]:
        src = all_results if mode_key == "seq" else all_results_par
        traj = src[(K, 0.5)]["traj"]
        nx   = np.array(traj["total_X"], dtype=float)
        ny   = np.array(traj["total_Y"], dtype=float)
        tot  = nx + ny
        xfrac  = np.where(tot > 0, nx / tot, np.nan)
        yoverx = np.where(nx  > 0, ny / nx,  np.nan)
        lbl = f"K={K} {mode_key}"
        ax1.step(traj["x"], xfrac,  where="post", color=col, lw=lw, ls=ls, label=lbl)
        ax2.step(traj["x"], yoverx, where="post", color=col, lw=lw, ls=ls, label=lbl)

for ax, title, theory in [(ax1, r"$X/(X+Y)$", eq_x_frac),
                           (ax2, r"$Y/X$",     ev_ratio)]:
    ax.axhline(theory, ls=":", color="gray", alpha=0.6, label="Theory")
    ax.set_title(title); ax.set_xlabel("Time"); ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(fig_path("fig_seq_vs_par_comp.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig_seq_vs_par_comp.pdf")

# ── 6. Theoretical variance curves (moment ODEs + delta method) ──────────────
# t_th, var_R1, var_R2, eps already computed above for the single-pool CI

# large-t reference decay: K * exp(-2*epsilon*t), constant fitted at t=4
idx4   = np.searchsorted(t_th, 4.0)
K_R1   = var_R1[idx4] * np.exp(2*eps*t_th[idx4])
K_R2   = var_R2[idx4] * np.exp(2*eps*t_th[idx4])
ref_R1 = K_R1 * np.exp(-2*eps*t_th)
ref_R2 = K_R2 * np.exp(-2*eps*t_th)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(r"Theoretical variance of ratios (single $X$ founder, delta method)",
             fontsize=12)

ax1.semilogy(t_th, var_R1, color="steelblue", lw=2, label=r"$\mathrm{Var}(X/(X+Y))$")
ax1.semilogy(t_th, ref_R1, "k--", lw=1.2,
             label=rf"$\propto e^{{-2\varepsilon t}},\;\varepsilon={eps:.3f}$")
ax1.set_xlabel("Time"); ax1.set_ylabel("Variance (log scale)")
ax1.set_title(r"$\mathrm{Var}(X/(X+Y))$"); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.semilogy(t_th, var_R2, color="tomato", lw=2, label=r"$\mathrm{Var}(Y/X)$")
ax2.semilogy(t_th, ref_R2, "k--", lw=1.2,
             label=rf"$\propto e^{{-2\varepsilon t}},\;\varepsilon={eps:.3f}$")
ax2.set_xlabel("Time"); ax2.set_ylabel("Variance (log scale)")
ax2.set_title(r"$\mathrm{Var}(Y/X)$"); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(fig_path("fig_variance_theory.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig_variance_theory.pdf")
print(f"Spectral gap ε = λ₁ - λ₂ = {eps:.4f}")

# ── 7. Print summary tables ────────────────────────────────────────────────────
hdr = f"{'Scenario':<22}  {'Pools':>6}  {'X':>6}  {'Y':>6}  {'X/(X+Y)':>9}  {'Y/X':>7}"
sep = "-" * len(hdr)

def print_table(title, sp_row, results_dict):
    print(f"\n── {title} ──────────────────────────────────────────────────")
    print(hdr); print(sep)
    st = sp_row
    print(f"{'Single pool':<22}  {st['n_pools']:>6}  {st['final_X']:>6}  "
          f"{st['final_Y']:>6}  {st['final_x_frac']:>9.4f}  "
          f"{st['final_y_over_x']:>7.4f}")
    for K in K_vals:
        for p in p_vals:
            st  = results_dict[(K, p)]
            key = f"K={K}, p={p}"
            print(f"{key:<22}  {st['n_pools']:>6}  {st['final_X']:>6}  {st['final_Y']:>6}"
                  f"  {st['final_x_frac']:>9.4f}  {st['final_y_over_x']:>7.4f}")
    print(f"\n  Theory: X/(X+Y)={eq_x_frac:.4f}   Y/X={ev_ratio:.4f}")

print_table("Sequential splitting", sp_stats, summary)
print_table("Parallel splitting",   sp_stats, summary_par)

# ── 8. Optimal-p analysis: unbiased seeding condition ────────────────────────
# For each K, estimate f_X(K) = E[X/(X+Y) | size=K, started from 1 X cell]
# and f_Y(K) analogously, via Gillespie Monte Carlo.
# The bias-free condition ᾱ(K,p) = α_eq then gives p*(K).

print("\nComputing f_X(K), f_Y(K) via Monte Carlo ...", flush=True)

N_SIM_FXY = 2000
rng_fxy   = np.random.default_rng(7)

def pool_xfrac(nx0, ny0, K, a, b, c, d, rng):
    """Run one Gillespie pool until total == K; return X/(X+Y)."""
    nx, ny = int(nx0), int(ny0)
    while nx + ny < K:
        rxx = a * nx;  rxy = b * nx
        ryx = c * ny;  ryy = d * ny
        tot = rxx + rxy + ryx + ryy
        if tot == 0:
            return np.nan
        u = rng.uniform() * tot
        if   u < rxx:               nx += 1
        elif u < rxx + rxy:         ny += 1
        elif u < rxx + rxy + ryx:   nx += 1
        else:                       ny += 1
    return nx / (nx + ny)

K_grid    = [2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200]
K_grid_np = np.array(K_grid)
fX_arr    = []
fY_arr    = []
pstar_arr = []

for Kk in K_grid:
    sX = [pool_xfrac(1, 0, Kk, a, b, c, d, rng_fxy) for _ in range(N_SIM_FXY)]
    sY = [pool_xfrac(0, 1, Kk, a, b, c, d, rng_fxy) for _ in range(N_SIM_FXY)]
    fX = float(np.nanmean(sX))
    fY = float(np.nanmean(sY))
    ps = (eq_x_frac - fY) / (fX - fY) if abs(fX - fY) > 1e-8 else eq_x_frac
    ps = float(np.clip(ps, 0, 1))
    fX_arr.append(fX); fY_arr.append(fY); pstar_arr.append(ps)
    print(f"  K={Kk:4d}:  f_X={fX:.4f}  f_Y={fY:.4f}  p*={ps:.4f}", flush=True)

fX_arr    = np.array(fX_arr)
fY_arr    = np.array(fY_arr)
pstar_arr = np.array(pstar_arr)

# ── Fig 8a: f_X(K), f_Y(K) and p*(K) ────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Unbiased seeding condition", fontsize=12)

ax1.semilogx(K_grid_np, fX_arr, "o-", color="steelblue", lw=2,
             label=r"$f_X(K)$  (X founder)")
ax1.semilogx(K_grid_np, fY_arr, "s-", color="tomato",    lw=2,
             label=r"$f_Y(K)$  (Y founder)")
ax1.axhline(eq_x_frac, ls="--", color="gray", alpha=0.7,
            label=rf"$\alpha_{{eq}}={eq_x_frac:.3f}$")
ax1.set_xlabel("Splitting threshold $K$")
ax1.set_ylabel(r"$E\,[X/(X+Y)\mid\mathrm{size}=K]$")
ax1.set_title("Expected X fraction at split time")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.semilogx(K_grid_np, pstar_arr, "D-", color="darkgreen", lw=2,
             label=r"$p^*(K)$")
ax2.axhline(eq_x_frac, ls="--", color="gray", alpha=0.7,
            label=rf"$\alpha_{{eq}}={eq_x_frac:.3f}$")
ax2.set_xlabel("Splitting threshold $K$")
ax2.set_ylabel(r"Optimal seeding probability $p^*$")
ax2.set_title(r"$p^*(K)=(\alpha_{eq}-f_Y(K))\,/\,(f_X(K)-f_Y(K))$")
ax2.set_ylim(0, 1)
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(fig_path("fig_pstar_K.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig_pstar_K.pdf")

# ── Fig 8b: Bias surface ᾱ(K,p) − α_eq ─────────────────────────────────────
p_surf   = np.linspace(0, 1, 61)
# rows = K_grid, cols = p_surf
bias_mat = (np.outer(fX_arr, p_surf) + np.outer(fY_arr, 1 - p_surf)) - eq_x_frac

fig, ax = plt.subplots(figsize=(10, 6))
vmax = max(abs(bias_mat.min()), abs(bias_mat.max()))
im = ax.pcolormesh(p_surf, K_grid_np, bias_mat,
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
plt.colorbar(im, ax=ax, label=r"Bias $\;\bar{\alpha}(K,p)-\alpha_{eq}$")
ax.set_yscale("log")
ax.plot(pstar_arr, K_grid_np, "ko-", lw=2, ms=5,
        label=r"$p^*(K)$: zero-bias curve")
ax.axvline(eq_x_frac, ls="--", color="yellow", lw=1.5,
           label=rf"$p=\alpha_{{eq}}={eq_x_frac:.3f}$")
ax.set_xlabel("Seeding probability $p$")
ax.set_ylabel("Splitting threshold $K$ (log scale)")
ax.set_title(r"Bias $\bar{\alpha}(K,p)-\alpha_{eq}$  (red = X-excess, blue = Y-excess)")
ax.legend(loc="upper left")

plt.tight_layout()
fig.savefig(fig_path("fig_bias_surface.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig_bias_surface.pdf")

# ── Fig 8c: Simulated X/(X+Y) at p*, α_eq, and 0.5 for K=10 and K=50 ────────
pstar_10 = float(np.interp(10, K_grid_np, pstar_arr))
pstar_50 = float(np.interp(50, K_grid_np, pstar_arr))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(r"Sequential splitting: $X/(X+Y)$ with optimal vs standard $p$",
             fontsize=12)

cases_K10 = [
    (pstar_10,  "steelblue",  f"$p^*={pstar_10:.3f}$ (optimal)"),
    (eq_x_frac, "dodgerblue", rf"$p=\alpha_{{eq}}={eq_x_frac:.3f}$"),
    (0.5,       "lightblue",  r"$p=0.5$"),
]
cases_K50 = [
    (pstar_50,  "tomato",     f"$p^*={pstar_50:.3f}$ (optimal)"),
    (eq_x_frac, "orangered",  rf"$p=\alpha_{{eq}}={eq_x_frac:.3f}$"),
    (0.5,       "lightsalmon",r"$p=0.5$"),
]

for (ax, Kk, cases, pstar) in [(ax1, 10, cases_K10, pstar_10),
                                (ax2, 50, cases_K50, pstar_50)]:
    for pp, col, lbl in cases:
        res  = simulate_with_splitting(a=a, b=b, c=c, d=d, nx0=1, ny0=0,
                                       N=N, K=Kk, p=pp,
                                       mode="sequential", seed=SEED)
        traj = res["traj"]
        nx   = np.array(traj["total_X"], dtype=float)
        ny   = np.array(traj["total_Y"], dtype=float)
        tot  = nx + ny
        xfrac = np.where(tot > 0, nx / tot, np.nan)
        ax.step(traj["x"], xfrac, where="post", lw=1.8, color=col, label=lbl)
    ax.axhline(eq_x_frac, ls="--", color="gray", alpha=0.6,
               label=rf"Theory $\alpha_{{eq}}={eq_x_frac:.3f}$")
    ax.set_xlabel("Cumulative time")
    ax.set_ylabel(r"$X/(X+Y)$")
    ax.set_title(f"$K = {Kk}$  (optimal $p^* = {pstar:.3f}$)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(fig_path("fig_optimal_p_sim.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig_optimal_p_sim.pdf")

print("\nAll figures saved to ./figures/")
