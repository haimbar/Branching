"""
analyze_simulation.py
Run the single-pool and parallel-splitting (K=100) simulations,
save all figures to ../figures/, and print a summary table.

Sequential splitting code is retained in SimulationScenario.py but
is not executed here.

Run from the Branching/scripts/ directory:
    python3 analyze_simulation.py
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
from SimulationScenario import (simulate_cells, simulate_with_splitting,
                                 estimate_rates_single, estimate_rates_parallel,
                                 estimate_rates_counts_only, estimate_rates_first_event)

os.makedirs(FIG_DIR, exist_ok=True)

def fig_path(name):
    return os.path.join(FIG_DIR, name)

# ── Fixed parameters ──────────────────────────────────────────────────────────
a, b, c, d = 0.5, 0.1, 0.05, 0.4
N    = 1000
K    = 100          # splitting threshold used throughout
SEED = 42

# Theoretical long-run composition (dominant eigenvector of rate matrix)
# A = [[a, c], [b, d]],  dX/dt = aX + cY,  dY/dt = bX + dY
tr_A  = a + d                               # 0.9
det_A = a * d - b * c                       # 0.195
lam1  = (tr_A + np.sqrt(tr_A**2 - 4*det_A)) / 2   # ≈ 0.537
ev_ratio  = (lam1 - a) / c                 # Y/X at equilibrium ≈ 0.732
eq_x_frac = 1 / (1 + ev_ratio)             # X/(X+Y) ≈ 0.578
lam2  = tr_A - lam1
eps   = lam1 - lam2                        # spectral gap ≈ 0.173

print(f"Theory: λ₁={lam1:.4f},  ε={eps:.4f},  "
      f"eq X/(X+Y)={eq_x_frac:.4f},  eq Y/X={ev_ratio:.4f}")

# ── Moment-ODE helpers ────────────────────────────────────────────────────────
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
    """Solve moment ODEs on t_grid; return (mu1, mu2, alpha, r, var_R1, var_R2)."""
    sol = odeint(moment_rhs, [1.0, 0.0, 0.0, 0.0, 0.0], t_grid, args=(a, b, c, d))
    mu1, mu2, s11, s12, s22 = sol.T
    mu_N   = mu1 + mu2
    alpha  = np.where(mu_N > 0, mu1 / mu_N, np.nan)
    beta   = np.where(mu_N > 0, mu2 / mu_N, np.nan)
    r      = np.where(mu1  > 0, mu2 / mu1,  np.nan)
    var_R1 = (beta**2*s11 + alpha**2*s22 - 2*alpha*beta*s12) / mu_N**2
    var_R2 = (r**2*s11 + s22 - 2*r*s12) / mu1**2
    return mu1, mu2, alpha, r, var_R1, var_R2

def traj_stats(traj):
    """Final-state statistics from a traj dict."""
    nx = np.array(traj["total_X"], dtype=float)
    ny = np.array(traj["total_Y"], dtype=float)
    total    = nx + ny
    x_frac   = np.where(total > 0, nx / total, np.nan)
    y_over_x = np.where(nx   > 0, ny / nx,    np.nan)
    return dict(
        final_X      = int(nx[-1]),
        final_Y      = int(ny[-1]),
        final_x_frac    = float(np.nanmean(x_frac[-max(1, len(x_frac)//10):])),
        final_y_over_x  = float(np.nanmean(y_over_x[-max(1, len(y_over_x)//10):])),
    )

# ── 1. Single-pool benchmark ──────────────────────────────────────────────────
print("\n── Single pool ──────────────────────────────────────────────────────────")
times_sp, nx_sp, ny_sp, events_sp = simulate_cells(a=a, b=b, c=c, d=d,
                                                    nx0=1, ny0=0, N=N, seed=SEED)
nx_sp  = np.array(nx_sp, dtype=float)
ny_sp  = np.array(ny_sp, dtype=float)
tot_sp = nx_sp + ny_sp

# Moment ODE on a fine grid covering the simulation window
t_th = np.linspace(0, max(times_sp) * 1.02, 3000)
mu1_th, mu2_th, alpha_th, r_th, var_R1, var_R2 = solve_moments(t_th, a, b, c, d)

z95 = 1.96
ci_R1_lo = np.clip(alpha_th - z95*np.sqrt(np.maximum(var_R1, 0)), 0, 1)
ci_R1_hi = np.clip(alpha_th + z95*np.sqrt(np.maximum(var_R1, 0)), 0, 1)
ci_R2_lo = np.maximum(r_th - z95*np.sqrt(np.maximum(var_R2, 0)), 0)
ci_R2_hi =            r_th + z95*np.sqrt(np.maximum(var_R2, 0))

xfrac_sp  = np.where(tot_sp > 0, nx_sp / tot_sp, np.nan)
yoverx_sp = np.where(nx_sp  > 0, ny_sp / nx_sp,  np.nan)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
ax1.step(times_sp, nx_sp, where="post", color="steelblue", lw=2, label="X")
ax1.step(times_sp, ny_sp, where="post", color="tomato",    lw=2, label="Y")
ax1.set_ylabel("Cell count")
ax1.set_title("Single-pool simulation")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.fill_between(t_th, ci_R1_lo, ci_R1_hi,
                 color="steelblue", alpha=0.15, label=r"95% CI  $X/(X+Y)$")
ax2.fill_between(t_th, ci_R2_lo, ci_R2_hi,
                 color="tomato",    alpha=0.15, label=r"95% CI  $Y/X$")
ax2.plot(t_th, alpha_th, color="steelblue", lw=1.2, ls="-.", label=r"$E[X/(X+Y)]$")
ax2.plot(t_th, r_th,     color="tomato",    lw=1.2, ls="-.", label=r"$E[Y/X]$")
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

sp_stats = traj_stats({'total_X': nx_sp.tolist(), 'total_Y': ny_sp.tolist()})
sp_stats['n_pools'] = 1

# ── 2. Parallel splitting: K=100, all three p values ─────────────────────────
print("\n── Parallel splitting (K=100) ───────────────────────────────────────────")
p_vals = [0.0, 0.5, 1.0]
all_par = {}
for p in p_vals:
    res = simulate_with_splitting(a=a, b=b, c=c, d=d,
                                  nx0=1, ny0=0, N=N, K=K, p=p,
                                  mode="parallel", seed=SEED)
    all_par[p] = res
    st = traj_stats(res["traj"])
    n  = len(res["pools"])
    print(f"  p={p:.1f}: {n} pools,  X/(X+Y)={st['final_x_frac']:.4f},  Y/X={st['final_y_over_x']:.4f}")

# fig_par_K100.pdf — 2×3 grid (rows: counts, ratios; cols: p values)
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col")
fig.suptitle(f"Parallel splitting, $K={K}$", fontsize=13)
for col, p in enumerate(p_vals):
    traj   = all_par[p]["traj"]
    nx     = np.array(traj["total_X"], dtype=float)
    ny     = np.array(traj["total_Y"], dtype=float)
    total  = nx + ny
    xfrac  = np.where(total > 0, nx / total, np.nan)
    yoverx = np.where(nx   > 0, ny / nx,    np.nan)
    x_ax   = traj["x"]

    ax = axes[0, col]
    ax.step(x_ax, nx, where="post", color="steelblue", lw=1.8, label="Total X")
    ax.step(x_ax, ny, where="post", color="tomato",    lw=1.8, label="Total Y")
    ax.set_title(f"$p = {p}$")
    ax.set_ylabel("Cell count" if col == 0 else "")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, col]
    ax.step(x_ax, xfrac,  where="post", color="steelblue", lw=1.8, label=r"$X/(X+Y)$")
    ax.step(x_ax, yoverx, where="post", color="tomato",    lw=1.8, label=r"$Y/X$")
    ax.axhline(eq_x_frac, ls="--", color="steelblue", alpha=0.4)
    ax.axhline(ev_ratio,  ls="--", color="tomato",    alpha=0.4)
    ax.set_xlabel("Time"); ax.set_ylabel("Ratio" if col == 0 else "")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(fig_path("fig_par_K100.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig_par_K100.pdf")

# ── 2b. Parallel splitting: K=20 (for comparison with K=100) ─────────────────
print("\n── Parallel splitting (K=20) ────────────────────────────────────────────")
K20 = 20
all_par20 = {}
for p in p_vals:
    res = simulate_with_splitting(a=a, b=b, c=c, d=d,
                                  nx0=1, ny0=0, N=N, K=K20, p=p,
                                  mode="parallel", seed=SEED)
    all_par20[p] = res
    st = traj_stats(res["traj"])
    n  = len(res["pools"])
    print(f"  p={p:.1f}: {n} pools,  X/(X+Y)={st['final_x_frac']:.4f},  Y/X={st['final_y_over_x']:.4f}")

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col")
fig.suptitle(f"Parallel splitting, $K={K20}$", fontsize=13)
for col, p in enumerate(p_vals):
    traj   = all_par20[p]["traj"]
    nx     = np.array(traj["total_X"], dtype=float)
    ny     = np.array(traj["total_Y"], dtype=float)
    total  = nx + ny
    xfrac  = np.where(total > 0, nx / total, np.nan)
    yoverx = np.where(nx   > 0, ny / nx,    np.nan)
    x_ax   = traj["x"]

    ax = axes[0, col]
    ax.step(x_ax, nx, where="post", color="steelblue", lw=1.8, label="Total X")
    ax.step(x_ax, ny, where="post", color="tomato",    lw=1.8, label="Total Y")
    ax.set_title(f"$p = {p}$")
    ax.set_ylabel("Cell count" if col == 0 else "")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, col]
    ax.step(x_ax, xfrac,  where="post", color="steelblue", lw=1.8, label=r"$X/(X+Y)$")
    ax.step(x_ax, yoverx, where="post", color="tomato",    lw=1.8, label=r"$Y/X$")
    ax.axhline(eq_x_frac, ls="--", color="steelblue", alpha=0.4)
    ax.axhline(ev_ratio,  ls="--", color="tomato",    alpha=0.4)
    ax.set_xlabel("Time"); ax.set_ylabel("Ratio" if col == 0 else "")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(fig_path("fig_par_K20.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig_par_K20.pdf")

# ── 3. Theoretical variance curves ───────────────────────────────────────────
print("\n── Variance theory ──────────────────────────────────────────────────────")
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
print(f"Saved figures/fig_variance_theory.pdf  (ε = {eps:.4f})")

# ── 4. Summary table ──────────────────────────────────────────────────────────
hdr = f"{'Scenario':<28}  {'Pools':>6}  {'X':>6}  {'Y':>6}  {'X/(X+Y)':>9}  {'Y/X':>7}"
sep = "-" * len(hdr)
print(f"\n── Summary (N={N}) ──────────────────────────────────────────────────────")
print(hdr); print(sep)
st = sp_stats
print(f"{'Single pool':<28}  {st['n_pools']:>6}  {st['final_X']:>6}  "
      f"{st['final_Y']:>6}  {st['final_x_frac']:>9.4f}  {st['final_y_over_x']:>7.4f}")
for p in p_vals:
    st  = traj_stats(all_par[p]["traj"])
    key = f"Parallel K={K}, p={p}"
    n   = len(all_par[p]["pools"])
    print(f"{key:<28}  {n:>6}  {st['final_X']:>6}  {st['final_Y']:>6}"
          f"  {st['final_x_frac']:>9.4f}  {st['final_y_over_x']:>7.4f}")
print(f"\n  Theory: X/(X+Y)={eq_x_frac:.4f}   Y/X={ev_ratio:.4f}")

# ── 5. Estimation study: MLE box plots (M replicates) ────────────────────────
print("\n── Estimation study ─────────────────────────────────────────────────────")
M    = 200
TRUE = dict(a=a, b=b, c=c, d=d)
TRUE_vals = [a, b, c, d]

ests_single = np.empty((M, 4))
ests_par    = np.empty((M, 4))

for m in range(M):
    seed_m = 1000 + m

    # Single pool
    t_, nx_, ny_, ev_ = simulate_cells(**TRUE, nx0=1, ny0=0, N=N, seed=seed_m)
    e = estimate_rates_single(t_, nx_, ny_, ev_)
    ests_single[m] = [e['a'], e['b'], e['c'], e['d']]

    # Parallel K=100, p=0.5
    res_ = simulate_with_splitting(**TRUE, nx0=1, ny0=0, N=N, K=K, p=0.5,
                                   mode="parallel", seed=seed_m)
    e = estimate_rates_parallel(res_)
    ests_par[m] = [e['a'], e['b'], e['c'], e['d']]

    if (m + 1) % 50 == 0:
        print(f"  {m+1}/{M} replicates done", flush=True)

param_names = ['a', 'b', 'c', 'd']
descriptions = [r'$X \!\to\! X{+}X$', r'$X \!\to\! X{+}Y$',
                r'$Y \!\to\! Y{+}X$', r'$Y \!\to\! Y{+}Y$']

fig, axes = plt.subplots(1, 4, figsize=(14, 5))
fig.suptitle(f"MLE estimates of division rates  ({M} replicates, $N={N}$)", fontsize=12)

for j, (pname, desc, true) in enumerate(zip(param_names, descriptions, TRUE_vals)):
    ax = axes[j]
    bp = ax.boxplot([ests_single[:, j], ests_par[:, j]],
                    tick_labels=["Single\npool", f"Parallel\n$K={K}$"],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", lw=2))
    for patch, col in zip(bp['boxes'], ["steelblue", "darkorange"]):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    ax.axhline(true, color="crimson", ls="--", lw=1.8,
               label=f"True = {true}")
    ax.set_title(f"${pname}$   ({desc})", fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(fig_path("fig_estimation.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig_estimation.pdf")

# Print numerical summary
print(f"\n{'Param':>5}  {'True':>6}  "
      f"{'Single: mean':>13}  {'Single: SD':>11}  "
      f"{'Parallel: mean':>15}  {'Parallel: SD':>13}")
print("-" * 72)
for j, (pname, true) in enumerate(zip(param_names, TRUE_vals)):
    sm, ss = ests_single[:, j].mean(), ests_single[:, j].std()
    pm, ps = ests_par[:, j].mean(),    ests_par[:, j].std()
    print(f"{pname:>5}  {true:>6.4f}  "
          f"{sm:>13.4f}  {ss:>11.4f}  {pm:>15.4f}  {ps:>13.4f}")

print("\nAll figures saved to ./figures/")

# ── 6. Missing-data estimation: counts only (Q3) ──────────────────────────────
print("\n── Counts-only MLE study (single pool) ──────────────────────────────────")
ests_co = np.empty((M, 4))

for m in range(M):
    seed_m = 1000 + m
    t_, nx_, ny_, _ = simulate_cells(**TRUE, nx0=1, ny0=0, N=N, seed=seed_m)
    e = estimate_rates_counts_only(t_, nx_, ny_)
    ests_co[m] = [e['a'], e['b'], e['c'], e['d']]
    if (m + 1) % 50 == 0:
        print(f"  {m+1}/{M} replicates done", flush=True)

fig, axes = plt.subplots(1, 4, figsize=(14, 5))
fig.suptitle(
    f"Full-data MLE vs counts-only MLE  ({M} replicates, $N={N}$, single pool)",
    fontsize=11)
for j, (pname, desc, true) in enumerate(zip(param_names, descriptions, TRUE_vals)):
    ax = axes[j]
    bp = ax.boxplot([ests_single[:, j], ests_co[:, j]],
                    labels=["Full data", "Counts only"],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", lw=2))
    for patch, col in zip(bp['boxes'], ["steelblue", "darkorange"]):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    ax.axhline(true, color="crimson", ls="--", lw=1.8)
    ax.set_title(f"${pname}$   ({desc})", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(fig_path("fig_estimation_counts.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig_estimation_counts.pdf")

# ── 7. First-event estimation (Q4) ────────────────────────────────────────────
print("\n── First-event MLE study (K=20, p=0.5) ──────────────────────────────────")
K_FE = 20
ests_fe = np.empty((M, 4))

for m in range(M):
    seed_m = 1000 + m
    res_ = simulate_with_splitting(**TRUE, nx0=1, ny0=0, N=N, K=K_FE, p=0.5,
                                   mode="parallel", seed=seed_m)
    e = estimate_rates_first_event(res_)
    ests_fe[m] = [e['a'], e['b'], e['c'], e['d']]
    if (m + 1) % 50 == 0:
        print(f"  {m+1}/{M} replicates done", flush=True)

fig, axes = plt.subplots(1, 4, figsize=(14, 5))
fig.suptitle(
    f"Full-data MLE vs first-event MLE  ({M} replicates, $N={N}$, $K={K_FE}$, $p=0.5$)",
    fontsize=11)
for j, (pname, desc, true) in enumerate(zip(param_names, descriptions, TRUE_vals)):
    ax = axes[j]
    bp = ax.boxplot([ests_single[:, j], ests_fe[:, j]],
                    labels=["Full data", f"First-event\n$K={K_FE}$"],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", lw=2))
    for patch, col in zip(bp['boxes'], ["steelblue", "seagreen"]):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    ax.axhline(true, color="crimson", ls="--", lw=1.8)
    ax.set_title(f"${pname}$   ({desc})", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(fig_path("fig_estimation_first_event.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig_estimation_first_event.pdf")

# SD vs K curve (Q4)
K_vals_fe = [2, 5, 10, 20, 50, 100]
sd_fe_K = {pname: [] for pname in param_names}
M_K = 100
for Kk in K_vals_fe:
    arr = np.empty((M_K, 4))
    for m in range(M_K):
        res_ = simulate_with_splitting(**TRUE, nx0=1, ny0=0, N=N, K=Kk, p=0.5,
                                       mode="parallel", seed=3000 + m)
        e = estimate_rates_first_event(res_)
        arr[m] = [e['a'], e['b'], e['c'], e['d']]
    for j, pname in enumerate(param_names):
        sd_fe_K[pname].append(float(np.nanstd(arr[:, j])))
    print(f"  K={Kk} done", flush=True)

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fig.suptitle(
    f"First-event MLE: SD vs $K$  ($N={N}$, $p=0.5$, {M_K} replicates each)",
    fontsize=11)
for j, (pname, desc, true) in enumerate(zip(param_names, descriptions, TRUE_vals)):
    ax = axes[j]
    ax.semilogx(K_vals_fe, sd_fe_K[pname], "o-", color="seagreen", lw=2,
                label="First-event MLE")
    ax.axhline(ests_single[:, j].std(), ls="--", color="steelblue", lw=1.5,
               label="Full-data SD")
    ax.set_xlabel("$K$ (pool size)"); ax.set_ylabel("SD of estimate")
    ax.set_title(f"${pname}$   ({desc})", fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xticks(K_vals_fe); ax.set_xticklabels(K_vals_fe)
plt.tight_layout()
fig.savefig(fig_path("fig_first_event_vs_K.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig_first_event_vs_K.pdf")
