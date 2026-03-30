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
                                 estimate_rates_counts_only, estimate_rates_first_event,
                                 simulate_pure_split, estimate_rates_pure_phase,
                                 extract_count_snapshots, estimate_rates_trajectory_ols,
                                 estimate_rates_trajectory_qrem,
                                 predict_final_counts)

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
                    tick_labels=["Full data", "Counts only"],
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
                    tick_labels=["Full data", f"First-event\n$K={K_FE}$"],
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
K_vals_fe = [2, 5, 10, 20, 50, 100, 200, 500, 900]
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

# ── 8. Pure-pool splitting study ───────────────────────────────────────────────
print("\n── Pure-pool splitting study ─────────────────────────────────────────────")
# Compare three estimators for K=10 and K=50:
#   (a) single-cell first-event MLE (simulate_with_splitting)
#   (b) pure-pool phase MLE (simulate_pure_split)
#   (c) full-data MLE (benchmark, from ests_single already computed above)

K_vals_pure = [10, 50]
M_pure = 200

for K_pure in K_vals_pure:
    ests_fe_K   = np.empty((M_pure, 4))
    ests_pp_K   = np.empty((M_pure, 4))
    pools_fe_K  = []
    pools_pp_K  = []

    for m in range(M_pure):
        s = 5000 + m
        # First-event
        res_fe = simulate_with_splitting(**TRUE, nx0=1, ny0=0, N=N,
                                         K=K_pure, p=0.5, mode="parallel", seed=s)
        e_fe = estimate_rates_first_event(res_fe)
        ests_fe_K[m]  = [e_fe['a'], e_fe['b'], e_fe['c'], e_fe['d']]
        pools_fe_K.append(e_fe['q_X'] + e_fe['q_Y'])
        # Pure-pool
        res_pp = simulate_pure_split(**TRUE, nx0=1, ny0=0, N=N, K=K_pure, seed=s)
        e_pp = estimate_rates_pure_phase(res_pp)
        ests_pp_K[m]  = [e_pp['a'], e_pp['b'], e_pp['c'], e_pp['d']]
        pools_pp_K.append(e_pp['n_pure_X'] + e_pp['n_pure_Y'])

    if (m + 1) % 50 == 0:
        print(f"  {m+1}/{M_pure} done (K={K_pure})", flush=True)

    print(f"\n  K={K_pure}  (mean pools: FE={np.mean(pools_fe_K):.0f}, "
          f"PP={np.mean(pools_pp_K):.0f})")
    print(f"  {'Param':>5}  {'True':>6}  "
          f"{'FE mean':>8}  {'FE SD':>7}  {'PP mean':>8}  {'PP SD':>7}")
    print("  " + "-" * 52)
    for j, (pname, true) in enumerate(zip(param_names, TRUE_vals)):
        print(f"  {pname:>5}  {true:>6.4f}  "
              f"{ests_fe_K[:,j].mean():>8.4f}  {ests_fe_K[:,j].std():>7.4f}  "
              f"{ests_pp_K[:,j].mean():>8.4f}  {ests_pp_K[:,j].std():>7.4f}")

    # Box-plot comparison: full-data vs first-event vs pure-pool
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    fig.suptitle(
        f"MLE comparison  ($K={K_pure}$, $N={N}$, {M_pure} replicates)",
        fontsize=11)
    for j, (pname, desc, true) in enumerate(
            zip(param_names, descriptions, TRUE_vals)):
        ax = axes[j]
        bp = ax.boxplot(
            [ests_single[:, j], ests_fe_K[:, j], ests_pp_K[:, j]],
            tick_labels=["Full\ndata", f"First-event\n$K={K_pure}$",
                         f"Pure-pool\n$K={K_pure}$"],
            patch_artist=True, widths=0.5,
            medianprops=dict(color="black", lw=2))
        for patch, col in zip(bp['boxes'],
                               ["steelblue", "seagreen", "darkorange"]):
            patch.set_facecolor(col); patch.set_alpha(0.6)
        ax.axhline(true, color="crimson", ls="--", lw=1.8)
        ax.set_title(f"${pname}$   ({desc})", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fname = f"fig_pure_pool_K{K_pure}.pdf"
    fig.savefig(fig_path(fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figures/{fname}")

# ── 9. Trajectory-OLS estimation (count-only, few pools) ─────────────────────
print("\n── Trajectory OLS estimation ────────────────────────────────────────────")

M_snap   = 50        # snapshots per pool (M+1 points, M intervals)
P_vals   = [1, 3, 10]  # number of independent pools per estimate
M_traj   = 200       # Monte Carlo replicates

# Storage: shape (M_traj, 4) per P value
ests_traj = {P: np.empty((M_traj, 4)) for P in P_vals}

for m in range(M_traj):
    base_seed = 8000 + m * max(P_vals)
    for P in P_vals:
        snap_list = []
        for p_idx in range(P):
            t_, nx_, ny_, _ = simulate_cells(**TRUE, nx0=1, ny0=0, N=N,
                                             seed=base_seed + p_idx)
            snap_list.append(extract_count_snapshots(t_, nx_, ny_, M=M_snap))
        e = estimate_rates_trajectory_ols(snap_list)
        ests_traj[P][m] = [e['a'], e['b'], e['c'], e['d']]
    if (m + 1) % 50 == 0:
        print(f"  {m+1}/{M_traj} replicates done", flush=True)

# Print summary table
print(f"\n  {'':>5}  {'True':>6}  ", end="")
print("  ".join(f"{'P='+str(P)+' mean':>10}  {'SD':>7}" for P in P_vals))
print("  " + "-" * (14 + 20 * len(P_vals)))
for j, (pname, true) in enumerate(zip(param_names, TRUE_vals)):
    row = f"  {pname:>5}  {true:>6.4f}  "
    for P in P_vals:
        row += f"  {ests_traj[P][:, j].mean():>10.4f}  {ests_traj[P][:, j].std():>7.4f}"
    print(row)

# Reference: full-data SD
print("\n  Full-data SD (benchmark):")
for j, (pname, true) in enumerate(zip(param_names, TRUE_vals)):
    print(f"    {pname}: {ests_single[:, j].std():.4f}")

# Box-plot: full-data, first-event K=10, trajectory P=1/3/10
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle(
    f"Trajectory OLS vs first-event ($N={N}$, {M_traj} replicates, "
    f"{M_snap} snapshots/pool)",
    fontsize=11)

# Re-use ests_fe from section 7 (K=10); if not available, recompute
try:
    _fe = ests_fe          # set in section 7
except NameError:
    _fe = np.empty((M_traj, 4))
    for m in range(M_traj):
        res_fe_ = simulate_with_splitting(**TRUE, nx0=1, ny0=0, N=N,
                                          K=10, p=0.5, mode="parallel",
                                          seed=1000 + m)
        e_fe_ = estimate_rates_first_event(res_fe_)
        _fe[m] = [e_fe_['a'], e_fe_['b'], e_fe_['c'], e_fe_['d']]

colors_traj = ["mediumpurple", "orchid", "deeppink"]

for j, (pname, desc, true) in enumerate(zip(param_names, descriptions, TRUE_vals)):
    ax = axes[j]
    data   = [ests_single[:, j], _fe[:, j]] + [ests_traj[P][:, j] for P in P_vals]
    labels = ["Full\ndata", "First-event\n$K=10$"] + [f"Traj OLS\n$P={P}$" for P in P_vals]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                    widths=0.5, medianprops=dict(color="black", lw=2))
    box_colors = ["steelblue", "seagreen"] + colors_traj
    for patch, col in zip(bp['boxes'], box_colors):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    ax.axhline(true, color="crimson", ls="--", lw=1.8)
    ax.set_title(f"${pname}$   ({desc})", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(fig_path("fig_trajectory_ols.pdf"), bbox_inches="tight")
plt.close(fig)
print("  Saved figures/fig_trajectory_ols.pdf")

# ── 9b. M-sweep: fix P=10, vary number of checkpoints ────────────────────────
print("\n── Trajectory OLS: M-sweep (P=10) ──────────────────────────────────────")

P_fixed = 10
M_vals  = [5, 10, 20, 50]

ests_msweep = {Mv: np.empty((M_traj, 4)) for Mv in M_vals}

for m in range(M_traj):
    base_seed = 9000 + m * P_fixed
    # Simulate P_fixed pools once; extract different snapshot densities
    trajectories = []
    for p_idx in range(P_fixed):
        t_, nx_, ny_, _ = simulate_cells(**TRUE, nx0=1, ny0=0, N=N,
                                         seed=base_seed + p_idx)
        trajectories.append((t_, nx_, ny_))

    for Mv in M_vals:
        snap_list = [extract_count_snapshots(t_, nx_, ny_, M=Mv)
                     for t_, nx_, ny_ in trajectories]
        e = estimate_rates_trajectory_ols(snap_list)
        ests_msweep[Mv][m] = [e['a'], e['b'], e['c'], e['d']]

    if (m + 1) % 50 == 0:
        print(f"  {m+1}/{M_traj} replicates done", flush=True)

print(f"\n  P={P_fixed} fixed, varying M (checkpoints per pool):")
print(f"\n  {'':>5}  {'True':>6}  ", end="")
print("  ".join(f"{'M='+str(Mv)+' mean':>10}  {'SD':>7}" for Mv in M_vals))
print("  " + "-" * (14 + 20 * len(M_vals)))
for j, (pname, true) in enumerate(zip(param_names, TRUE_vals)):
    row = f"  {pname:>5}  {true:>6.4f}  "
    for Mv in M_vals:
        row += f"  {ests_msweep[Mv][:, j].mean():>10.4f}  {ests_msweep[Mv][:, j].std():>7.4f}"
    print(row)

# Box-plot: full-data, first-event K=10, traj P=10 with M=5/10/20/50
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle(
    f"Trajectory OLS checkpoint density ($P={P_fixed}$, $N={N}$, "
    f"{M_traj} replicates)",
    fontsize=11)

colors_m = ["#c7b8ea", "#9370db", "#5b2c8d", "#2d0057"]  # light → dark purple

for j, (pname, desc, true) in enumerate(zip(param_names, descriptions, TRUE_vals)):
    ax = axes[j]
    data   = [ests_single[:, j], _fe[:, j]] + [ests_msweep[Mv][:, j] for Mv in M_vals]
    labels = ["Full\ndata", "First-event\n$K=10$"] + [f"Traj\n$M={Mv}$" for Mv in M_vals]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                    widths=0.5, medianprops=dict(color="black", lw=2))
    box_colors = ["steelblue", "seagreen"] + colors_m
    for patch, col in zip(bp['boxes'], box_colors):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.axhline(true, color="crimson", ls="--", lw=1.8)
    ax.set_title(f"${pname}$   ({desc})", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(fig_path("fig_trajectory_m_sweep.pdf"), bbox_inches="tight")
plt.close(fig)
print("  Saved figures/fig_trajectory_m_sweep.pdf")

# ── 9c. Export WLS input data for M=20, P=10 (200 replicates) ────────────────
print("\n── Exporting WLS snapshot data (M=20, P=10) ─────────────────────────────")

import csv

DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
csv_path = os.path.join(DATA_DIR, "wls_snapshots_M20_P10.csv")

M_export  = 20
P_export  = 10

with open(csv_path, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["replicate", "pool", "snapshot", "t", "nx", "ny"])
    for m in range(M_traj):
        base_seed = 9000 + m * P_fixed   # same seeds as M-sweep section
        for p_idx in range(P_export):
            t_, nx_, ny_, _ = simulate_cells(**TRUE, nx0=1, ny0=0, N=N,
                                             seed=base_seed + p_idx)
            snaps = extract_count_snapshots(t_, nx_, ny_, M=M_export)
            for s_idx, (ts, nxs, nys) in enumerate(snaps):
                writer.writerow([m, p_idx, s_idx, f"{ts:.6f}", nxs, nys])
        if (m + 1) % 50 == 0:
            print(f"  {m+1}/{M_traj} replicates exported", flush=True)

print(f"  Saved data/wls_snapshots_M20_P10.csv  "
      f"({M_traj} replicates × {P_export} pools × {M_export+1} snapshots = "
      f"{M_traj * P_export * (M_export + 1)} rows)")

# ── 9d. QREM vs WLS comparison (M=20, P=10, 200 replicates) ──────────────────
print("\n── QREM vs WLS comparison (M=20, P=10) ─────────────────────────────────")

M_cmp  = 20
P_cmp  = 10

ests_wls_cmp  = np.empty((M_traj, 4))
ests_qrem_cmp = np.empty((M_traj, 4))

for m in range(M_traj):
    base_seed = 9000 + m * P_fixed   # same seeds as M-sweep / export sections
    snap_list = []
    for p_idx in range(P_cmp):
        t_, nx_, ny_, _ = simulate_cells(**TRUE, nx0=1, ny0=0, N=N,
                                         seed=base_seed + p_idx)
        snap_list.append(extract_count_snapshots(t_, nx_, ny_, M=M_cmp))
    e_wls  = estimate_rates_trajectory_ols(snap_list)
    e_qrem = estimate_rates_trajectory_qrem(snap_list)
    ests_wls_cmp[m]  = [e_wls['a'],  e_wls['b'],  e_wls['c'],  e_wls['d']]
    ests_qrem_cmp[m] = [e_qrem['a'], e_qrem['b'], e_qrem['c'], e_qrem['d']]
    if (m + 1) % 50 == 0:
        print(f"  {m+1}/{M_traj} replicates done", flush=True)

# Summary table
print(f"\n  M={M_cmp}, P={P_cmp}  —  Bias (SD)  [true: a=0.5, b=0.1, c=0.05, d=0.4]")
print(f"  {'':8s}  {'a':>14s}  {'b':>14s}  {'c':>14s}  {'d':>14s}")
print("  " + "-" * 65)
for label, ests in [("WLS", ests_wls_cmp), ("QREM", ests_qrem_cmp)]:
    row = f"  {label:8s}"
    for j, true in enumerate(TRUE_vals):
        bias = ests[:, j].mean() - true
        sd   = ests[:, j].std()
        row += f"  {bias:+.4f} ({sd:.4f})"
    print(row)

# Box-plot comparison
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle(
    f"WLS vs QREM trajectory estimators ($M={M_cmp}$, $P={P_cmp}$, "
    f"{M_traj} replicates)",
    fontsize=11)

for j, (pname, desc, true) in enumerate(zip(param_names, descriptions, TRUE_vals)):
    ax = axes[j]
    bp = ax.boxplot(
        [ests_wls_cmp[:, j], ests_qrem_cmp[:, j]],
        tick_labels=["WLS\n(NNLS)", "QREM\n(median)"],
        patch_artist=True, widths=0.5,
        medianprops=dict(color="black", lw=2))
    for patch, col in zip(bp['boxes'], ["steelblue", "darkorange"]):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.axhline(true, color="crimson", ls="--", lw=1.8, label="true")
    ax.set_title(f"${pname}$   ({desc})", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(fig_path("fig_wls_vs_qrem.pdf"), bbox_inches="tight")
plt.close(fig)
print("  Saved figures/fig_wls_vs_qrem.pdf")

# ── 9e. Full P×M grid: rate accuracy + final-count prediction ─────────────────
print("\n── Full P×M grid (P×M sweep + count prediction) ─────────────────────────")

P_grid = [3, 5, 10, 20, 50]
M_grid = [5, 10, 20, 50]
P_max  = max(P_grid)   # 51 pools simulated per replicate (P_max estim. + 1 holdout)

# ests_grid[pi, mi, m, j]      — estimate of rate j at (P_grid[pi], M_grid[mi]), rep m
# count_ratios[pi, mi, m, k]   — k=0: X_hat/X_true, k=1: Y_hat/Y_true
ests_grid    = np.empty((len(P_grid), len(M_grid), M_traj, 4))
count_ratios = np.empty((len(P_grid), len(M_grid), M_traj, 2))

for m in range(M_traj):
    base_seed = 10000 + m * (P_max + 1)
    # Simulate P_max estimation pools + 1 independent holdout pool
    trajectories = []
    for p_idx in range(P_max + 1):
        t_, nx_, ny_, _ = simulate_cells(**TRUE, nx0=1, ny0=0, N=N,
                                         seed=base_seed + p_idx)
        trajectories.append((t_, nx_, ny_))
    # Holdout truth (last trajectory)
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

# ── Print summary tables ───────────────────────────────────────────────────────
print("\n  Rate SD across P×M grid  (P = columns, M = rows):")
for j, (pname, true) in enumerate(zip(param_names, TRUE_vals)):
    print(f"\n  {pname}  (true = {true:.3f}):")
    print("        " + "  ".join(f"  P={P:2d}" for P in P_grid))
    for mi, Mv in enumerate(M_grid):
        row = f"  M={Mv:2d}  "
        for pi in range(len(P_grid)):
            row += f"  {ests_grid[pi, mi, :, j].std():6.4f}"
        print(row)

print("\n  Rate Bias (mean - true) across P×M grid:")
for j, (pname, true) in enumerate(zip(param_names, TRUE_vals)):
    print(f"\n  {pname}  (true = {true:.3f}):")
    print("        " + "  ".join(f"  P={P:2d}" for P in P_grid))
    for mi, Mv in enumerate(M_grid):
        row = f"  M={Mv:2d}  "
        for pi in range(len(P_grid)):
            row += f"  {ests_grid[pi, mi, :, j].mean() - true:+.4f}"
        print(row)

print("\n  Mean of X_hat / X_true  (target = 1.000):")
print("        " + "  ".join(f"  P={P:2d}" for P in P_grid))
for mi, Mv in enumerate(M_grid):
    row = f"  M={Mv:2d}  "
    for pi in range(len(P_grid)):
        row += f"  {np.nanmean(count_ratios[pi, mi, :, 0]):6.4f}"
    print(row)

print("\n  SD of X_hat / X_true:")
print("        " + "  ".join(f"  P={P:2d}" for P in P_grid))
for mi, Mv in enumerate(M_grid):
    row = f"  M={Mv:2d}  "
    for pi in range(len(P_grid)):
        row += f"  {np.nanstd(count_ratios[pi, mi, :, 0]):6.4f}"
    print(row)

print("\n  Mean of Y_hat / Y_true  (target = 1.000):")
print("        " + "  ".join(f"  P={P:2d}" for P in P_grid))
for mi, Mv in enumerate(M_grid):
    row = f"  M={Mv:2d}  "
    for pi in range(len(P_grid)):
        row += f"  {np.nanmean(count_ratios[pi, mi, :, 1]):6.4f}"
    print(row)

print("\n  SD of Y_hat / Y_true:")
print("        " + "  ".join(f"  P={P:2d}" for P in P_grid))
for mi, Mv in enumerate(M_grid):
    row = f"  M={Mv:2d}  "
    for pi in range(len(P_grid)):
        row += f"  {np.nanstd(count_ratios[pi, mi, :, 1]):6.4f}"
    print(row)

# ── Figure 1: heatmap of rate SD across P×M grid ─────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle(
    f"Trajectory WLS rate-estimate SD across $P \\times M$ grid "
    f"($N={N}$, {M_traj} replicates)",
    fontsize=11)

P_labels = [str(P) for P in P_grid]
M_labels = [str(Mv) for Mv in M_grid]

for j, (pname, ax) in enumerate(zip(param_names,
                                    [axes[0,0], axes[0,1], axes[1,0], axes[1,1]])):
    # mat[mi, pi] = SD of parameter j for M_grid[mi], P_grid[pi]
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
                    fontsize=7, color='black')

plt.tight_layout()
fig.savefig(fig_path("fig_pm_grid_sd.pdf"), bbox_inches="tight")
plt.close(fig)
print("  Saved figures/fig_pm_grid_sd.pdf")

# ── Figure 2: heatmap of rate bias across P×M grid ────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle(
    f"Trajectory WLS rate-estimate bias across $P \\times M$ grid "
    f"($N={N}$, {M_traj} replicates)",
    fontsize=11)

for j, (pname, true, ax) in enumerate(zip(param_names, TRUE_vals,
                                           [axes[0,0], axes[0,1],
                                            axes[1,0], axes[1,1]])):
    mat = np.array([[ests_grid[pi, mi, :, j].mean() - true
                     for pi in range(len(P_grid))]
                    for mi in range(len(M_grid))])
    vmax = np.abs(mat).max()
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
                    fontsize=7, color='black')

plt.tight_layout()
fig.savefig(fig_path("fig_pm_grid_bias.pdf"), bbox_inches="tight")
plt.close(fig)
print("  Saved figures/fig_pm_grid_bias.pdf")

# ── Figure 3: heatmaps of count-ratio accuracy ────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle(
    f"Final-count prediction accuracy $\\hat{{n}}_X(T)/n_X(T)$ and "
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
        # Diverging around 1 for mean
        vmax = max(abs(mat.min() - 1), abs(mat.max() - 1))
        im = ax.imshow(mat, aspect='auto', cmap='RdBu_r',
                       vmin=1 - vmax, vmax=1 + vmax)
        fmt = ".3f"
    else:
        mat = np.array([[np.nanstd(count_ratios[pi, mi, :, k])
                         for pi in range(len(P_grid))]
                        for mi in range(len(M_grid))])
        im = ax.imshow(mat, aspect='auto', cmap='YlOrRd')
        fmt = ".3f"
    ax.set_xticks(range(len(P_grid))); ax.set_xticklabels(P_labels)
    ax.set_yticks(range(len(M_grid))); ax.set_yticklabels(M_labels)
    ax.set_xlabel("$P$ (pools)"); ax.set_ylabel("$M$ (checkpoints)")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.85)
    for mi in range(len(M_grid)):
        for pi in range(len(P_grid)):
            ax.text(pi, mi, format(mat[mi, pi], fmt), ha='center', va='center',
                    fontsize=7, color='black')

plt.tight_layout()
fig.savefig(fig_path("fig_pm_grid_counts.pdf"), bbox_inches="tight")
plt.close(fig)
print("  Saved figures/fig_pm_grid_counts.pdf")

# ── Print LaTeX-ready table rows ──────────────────────────────────────────────
print("\n  LaTeX table rows  (P block | M | a mean SD | b mean SD | c mean SD | d mean SD | Xratio mean SD | Yratio mean SD):")
for pi, P in enumerate(P_grid):
    first = True
    for mi, Mv in enumerate(M_grid):
        p_col = str(P) if first else ""
        first = False
        row_parts = [p_col, str(Mv)]
        for j in range(4):
            mn  = ests_grid[pi, mi, :, j].mean()
            sd  = ests_grid[pi, mi, :, j].std()
            row_parts += [f"{mn:.3f}", f"{sd:.3f}"]
        for k in range(2):
            mn  = np.nanmean(count_ratios[pi, mi, :, k])
            sd  = np.nanstd(count_ratios[pi, mi, :, k])
            row_parts += [f"{mn:.3f}", f"{sd:.3f}"]
        print("  " + " | ".join(row_parts))
