# Simulation scenario
# Two cell types - X, Y
# Initial population sizes are arguments to the function (default:
#   X=1, Y=0)
# A cell of type X can create a new X cell - such an event occurs
#   randomly with an exponential rate a
# A cell of type X can create a new Y cell - such an event occurs
#   randomly with an exponential rate b
# A cell of type Y can create a new X cell - such an event occurs
#   randomly with an exponential rate c
# A cell of type Y can create a new Y cell - such an event occurs
#   randomly with an exponential rate d
#
# All four distributions are independent.
# First step - track the counts of X, Y. Stop the process when
# the total number of cells reaches N (also an argument, default=100)
# Plot the counts of X and Y on the y-axis, and the time to event on
# the x-axis.

import csv
import heapq
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def simulate_cells(a=0.5, b=0.1, c=0.05, d=0.4, nx0=1, ny0=0, N=100, seed=None):
    """
    Gillespie simulation of two cell-type branching process.

    Events (each creates one additional cell):
        X -> X+X  rate a per X cell
        X -> X+Y  rate b per X cell
        Y -> Y+X  rate c per Y cell
        Y -> Y+Y  rate d per Y cell

    Parameters
    ----------
    a, b, c, d : float
        Per-cell division rates.
    nx0, ny0 : int
        Initial counts of X and Y cells.
    N : int
        Stop when total cells (nx + ny) reaches N.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    times, nx_trace, ny_trace : lists
        Event times and corresponding cell counts.
    """
    rng = np.random.default_rng(seed)

    nx, ny = nx0, ny0
    t = 0.0

    times     = [t]
    nx_trace  = [nx]
    ny_trace  = [ny]
    events    = ["initial"]   # no division event at t=0

    while nx + ny < N:
        # Total rates for each event type
        r_xx = a * nx   # X -> X+X
        r_xy = b * nx   # X -> X+Y
        r_yx = c * ny   # Y -> Y+X
        r_yy = d * ny   # Y -> Y+Y

        total_rate = r_xx + r_xy + r_yx + r_yy

        if total_rate == 0:
            break  # population extinct — cannot continue

        # Time to next event ~ Exp(total_rate)
        dt = rng.exponential(1.0 / total_rate)
        t += dt

        # Choose which event proportional to rates
        u = rng.uniform() * total_rate
        if u < r_xx:
            nx += 1
            event = "X->X"
        elif u < r_xx + r_xy:
            ny += 1
            event = "X->Y"
        elif u < r_xx + r_xy + r_yx:
            nx += 1
            event = "Y->X"
        else:
            ny += 1
            event = "Y->Y"

        times.append(t)
        nx_trace.append(nx)
        ny_trace.append(ny)
        events.append(event)

    return times, nx_trace, ny_trace, events


def simulate_with_splitting(a=0.5, b=0.1, c=0.05, d=0.4, nx0=1, ny0=0,
                             N=100, K=10, p=1.0, mode="sequential", seed=None):
    """
    Gillespie simulation with pool splitting.

    When a pool reaches K cells, one cell is selected (X with probability p,
    Y with probability 1-p) and removed; a new pool is seeded with that cell.
    The parent pool continues with K-1 cells.

    The process stops when the total number of cells across all pools exceeds N.

    Parameters
    ----------
    K    : int   — pool size that triggers a split.
    p    : float — probability that the transferred cell is type X.
    mode : str   — "sequential" (pools processed one at a time, local time axes)
                   "parallel"   (all pools advance on a shared global time axis).

    Returns
    -------
    list of dicts, one per pool:
        id, parent, times, nx_trace, ny_trace, events
    """
    rng = np.random.default_rng(seed)
    _next_id = [0]

    def new_id():
        i = _next_id[0]; _next_id[0] += 1; return i

    def make_pool(parent, nx, ny, t0=0.0):
        return dict(id=new_id(), parent=parent, nx=nx, ny=ny, t=t0,
                    times=[t0], nx_trace=[nx], ny_trace=[ny], events=["initial"])

    def draw_next(pool):
        """Draw (abs_time_of_next_event, event_label) for pool; None if extinct."""
        nx, ny = pool['nx'], pool['ny']
        r_xx, r_xy = a * nx, b * nx
        r_yx, r_yy = c * ny, d * ny
        total = r_xx + r_xy + r_yx + r_yy
        if total == 0:
            return None, None
        dt = rng.exponential(1.0 / total)
        u  = rng.uniform() * total
        if   u < r_xx:                 event = "X->X"
        elif u < r_xx + r_xy:          event = "X->Y"
        elif u < r_xx + r_xy + r_yx:   event = "Y->X"
        else:                          event = "Y->Y"
        return pool['t'] + dt, event

    def apply_event(pool, t, event):
        pool['t'] = t
        if event in ("X->X", "Y->X"):
            pool['nx'] += 1
        else:
            pool['ny'] += 1
        pool['times'].append(t)
        pool['nx_trace'].append(pool['nx'])
        pool['ny_trace'].append(pool['ny'])
        pool['events'].append(event)

    def do_split(pool):
        """Remove one cell from pool; return new child pool seeded with it."""
        nx, ny = pool['nx'], pool['ny']
        if   nx == 0: cell = "Y"
        elif ny == 0: cell = "X"
        else:         cell = "X" if rng.uniform() < p else "Y"
        if cell == "X":
            pool['nx'] -= 1;  child = make_pool(pool['id'], 1, 0, pool['t'])
        else:
            pool['ny'] -= 1;  child = make_pool(pool['id'], 0, 1, pool['t'])
        # record the split in the parent's trace
        pool['times'].append(pool['t'])
        pool['nx_trace'].append(pool['nx'])
        pool['ny_trace'].append(pool['ny'])
        pool['events'].append(f"split->{cell}")
        return child

    # ── global counters and trajectory ───────────────────────────────────────
    total_cells = nx0 + ny0
    total_X     = nx0          # running sum of X across all pools
    total_Y     = ny0          # running sum of Y across all pools
    all_pools   = []
    traj_x      = [0]          # x-axis values (event index or time)
    traj_X      = [total_X]
    traj_Y      = [total_Y]

    def record(x_val):
        traj_x.append(x_val)
        traj_X.append(total_X)
        traj_Y.append(total_Y)

    # ── Sequential mode ───────────────────────────────────────────────────────
    # Each pool runs until it either reaches K (splits) or goes extinct.
    # After a split the parent goes BACK onto the queue so it continues growing.
    # Only pools that go extinct or are still queued when N is reached are final.
    if mode == "sequential":
        queue       = deque([make_pool(None, nx0, ny0)])
        global_time = 0.0

        while queue and total_cells <= N:
            pool = queue.popleft()

            while pool['nx'] + pool['ny'] < K and total_cells <= N:
                prev_t = pool['t']
                t, event = draw_next(pool)
                if event is None:
                    break                           # pool extinct
                apply_event(pool, t, event)
                global_time += t - prev_t           # accumulate inter-event time
                total_cells += 1
                if event in ("X->X", "Y->X"):
                    total_X += 1
                else:
                    total_Y += 1
                record(global_time)

            if pool['nx'] + pool['ny'] >= K and total_cells <= N:
                child = do_split(pool)
                queue.append(child)                 # child grows independently
                queue.append(pool)                  # parent continues — BUG FIX
            else:
                all_pools.append(pool)              # extinct or limit reached

        all_pools.extend(queue)                     # capture pools still waiting
        traj = {'x': traj_x, 'total_X': traj_X, 'total_Y': traj_Y,
                'xlabel': 'Time (cumulative)'}

    # ── Parallel mode ─────────────────────────────────────────────────────────
    elif mode == "parallel":
        root     = make_pool(None, nx0, ny0)
        pool_map = {root['id']: root}
        heap     = []   # (next_event_time, pool_id, event_label)

        t, event = draw_next(root)
        if t is not None:
            heapq.heappush(heap, (t, root['id'], event))

        while heap and total_cells <= N:
            t, pid, event = heapq.heappop(heap)
            pool = pool_map[pid]

            apply_event(pool, t, event)
            total_cells += 1
            if event in ("X->X", "Y->X"):
                total_X += 1
            else:
                total_Y += 1
            record(t)

            if pool['nx'] + pool['ny'] >= K:
                child = do_split(pool)
                pool_map[child['id']] = child
                nt, ne = draw_next(child)
                if nt is not None:
                    heapq.heappush(heap, (nt, child['id'], ne))

            nt, ne = draw_next(pool)
            if nt is not None:
                heapq.heappush(heap, (nt, pid, ne))

        all_pools = list(pool_map.values())
        traj = {'x': traj_x, 'total_X': traj_X, 'total_Y': traj_Y,
                'xlabel': 'Time'}

    else:
        raise ValueError(f"mode must be 'sequential' or 'parallel', got {mode!r}")

    return {'pools': all_pools, 'traj': traj}


def estimate_rates_single(times, nx_trace, ny_trace, events):
    """
    Maximum-likelihood estimates of (a, b, c, d) from a single-pool
    Gillespie trajectory.

    The log-likelihood for a CTMC branching process factors as:

        L(a,b,c,d) = n_a*log(a) + n_b*log(b) - (a+b)*Λ_X
                   + n_c*log(c) + n_d*log(d) - (c+d)*Λ_Y

    where Λ_X = ∫ nx(t) dt and Λ_Y = ∫ ny(t) dt are total cell-time
    exposures, and n_a, n_b, n_c, n_d are event counts (split events
    and the initial record are excluded).

    Setting ∂L/∂a = 0, ..., ∂L/∂d = 0 gives the closed-form MLEs:

        â = n_a / Λ_X,   b̂ = n_b / Λ_X
        ĉ = n_c / Λ_Y,   d̂ = n_d / Λ_Y

    Parameters
    ----------
    times, nx_trace, ny_trace, events : lists
        Output of simulate_cells().

    Returns
    -------
    dict with keys a, b, c, d (estimates) and n_a, n_b, n_c, n_d,
    Lambda_X, Lambda_Y (sufficient statistics).
    """
    n_a = n_b = n_c = n_d = 0
    Lambda_X = Lambda_Y = 0.0

    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        Lambda_X += nx_trace[i] * dt
        Lambda_Y += ny_trace[i] * dt
        ev = events[i + 1]
        if   ev == "X->X": n_a += 1
        elif ev == "X->Y": n_b += 1
        elif ev == "Y->X": n_c += 1
        elif ev == "Y->Y": n_d += 1
        # "initial" is skipped; no split events in single-pool output

    a_hat = n_a / Lambda_X if Lambda_X > 0 else float('nan')
    b_hat = n_b / Lambda_X if Lambda_X > 0 else float('nan')
    c_hat = n_c / Lambda_Y if Lambda_Y > 0 else float('nan')
    d_hat = n_d / Lambda_Y if Lambda_Y > 0 else float('nan')

    return dict(a=a_hat, b=b_hat, c=c_hat, d=d_hat,
                n_a=n_a, n_b=n_b, n_c=n_c, n_d=n_d,
                Lambda_X=Lambda_X, Lambda_Y=Lambda_Y)


def estimate_rates_parallel(result):
    """
    Maximum-likelihood estimates of (a, b, c, d) from a parallel-splitting
    simulation, aggregated across all pools.

    The same closed-form MLEs apply as in estimate_rates_single():

        â = n_a / Λ_X,  b̂ = n_b / Λ_X,  ĉ = n_c / Λ_Y,  d̂ = n_d / Λ_Y

    where n_a, ..., n_d and Λ_X, Λ_Y are summed over all pools.

    Split events ("split->X" / "split->Y") are excluded from event counts
    and contribute dt=0 to exposure (they are recorded at the same time
    as the triggering division event).

    Tail correction: when the simulation terminates (at global time T_stop),
    all pools except the one that triggered termination have a final interval
    [t_last, T_stop] that is not yet recorded in their traces.  Omitting it
    would underestimate Λ and inflate the rate estimates.  This function
    explicitly adds those tail intervals using the terminal state of each pool.

    Parameters
    ----------
    result : dict
        Output of simulate_with_splitting(..., mode="parallel").

    Returns
    -------
    dict with keys a, b, c, d (estimates) and n_a, n_b, n_c, n_d,
    Lambda_X, Lambda_Y (sufficient statistics aggregated across pools).
    """
    n_a = n_b = n_c = n_d = 0
    Lambda_X = Lambda_Y = 0.0

    # T_stop: global time of the last processed event.  All pools were alive
    # up to T_stop, but pools that had no event near T_stop are missing the
    # tail interval [pool['times'][-1], T_stop] in their recorded traces.
    # We add this tail contribution explicitly to avoid underestimating Λ.
    T_stop = result['traj']['x'][-1]

    for pool in result['pools']:
        times    = pool['times']
        nx_trace = pool['nx_trace']
        ny_trace = pool['ny_trace']
        events   = pool['events']

        for i in range(len(times) - 1):
            dt = times[i + 1] - times[i]
            Lambda_X += nx_trace[i] * dt
            Lambda_Y += ny_trace[i] * dt
            ev = events[i + 1]
            if   ev == "X->X": n_a += 1
            elif ev == "X->Y": n_b += 1
            elif ev == "Y->X": n_c += 1
            elif ev == "Y->Y": n_d += 1
            # "initial" and "split->*" events contribute dt=0; counts ignored

        # Tail: exposure from last recorded event to global stopping time
        dt_tail = T_stop - times[-1]
        if dt_tail > 0:
            Lambda_X += nx_trace[-1] * dt_tail
            Lambda_Y += ny_trace[-1] * dt_tail

    a_hat = n_a / Lambda_X if Lambda_X > 0 else float('nan')
    b_hat = n_b / Lambda_X if Lambda_X > 0 else float('nan')
    c_hat = n_c / Lambda_Y if Lambda_Y > 0 else float('nan')
    d_hat = n_d / Lambda_Y if Lambda_Y > 0 else float('nan')

    return dict(a=a_hat, b=b_hat, c=c_hat, d=d_hat,
                n_a=n_a, n_b=n_b, n_c=n_c, n_d=n_d,
                Lambda_X=Lambda_X, Lambda_Y=Lambda_Y)


def save_csv_split(result, path="simulation_split.csv"):
    pools = result['pools']
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pool_id", "parent_id", "time", "X", "Y", "event"])
        for pool in pools:
            pid = pool['id']
            par = pool['parent'] if pool['parent'] is not None else ""
            for row in zip(pool['times'], pool['nx_trace'], pool['ny_trace'], pool['events']):
                writer.writerow([pid, par, *row])
    total_rows = sum(len(p['times']) for p in pools)
    print(f"Saved {total_rows} rows across {len(pools)} pools to {path}")


def plot_simulation_split(result):
    """Single-panel plot of cell counts per pool for the split simulation."""
    pools   = result['pools']
    cmap    = plt.get_cmap("tab10")
    n_pools = len(pools)

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, pool in enumerate(pools):
        col   = cmap(i % 10)
        pid   = pool['id']
        label = f"pool {pid}" + (f" (←{pool['parent']})" if pool['parent'] is not None else "")

        nx = np.array(pool['nx_trace'], dtype=float)
        ny = np.array(pool['ny_trace'], dtype=float)

        ax.step(pool['times'], nx, where="post", color=col, lw=1.8,
                label=f"X {label}")
        ax.step(pool['times'], ny, where="post", color=col, lw=1.8,
                ls="--", alpha=0.7)

    ax.set_xlabel("Time")
    ax.set_ylabel("Cell count")
    ax.set_title(f"Cell counts per pool  (solid=X, dashed=Y,  {n_pools} pool(s))")
    ax.legend(fontsize=7, ncol=max(1, n_pools // 5))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_combined_split(result):
    """
    Two-panel plot of combined totals across all pools.
    Panel 1: total X and Y counts.
    Panel 2: X/(X+Y) and Y/X ratios (Y/X is NaN when X=0).
    """
    traj    = result['traj']
    n_pools = len(result['pools'])

    nx    = np.array(traj['total_X'], dtype=float)
    ny    = np.array(traj['total_Y'], dtype=float)
    total = nx + ny
    ratio_x  = np.where(total > 0, nx / total, np.nan)
    ratio_yx = np.where(nx    > 0, ny / nx,    np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.step(traj['x'], nx, where="post", color="steelblue", lw=2, label="Total X")
    ax1.step(traj['x'], ny, where="post", color="tomato",    lw=2, label="Total Y")
    ax1.set_ylabel("Cell count")
    ax1.set_title(f"Combined cell counts across all pools  ({n_pools} pool(s))")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.step(traj['x'], ratio_x,  where="post", color="steelblue", lw=2, label="X / (X+Y)")
    ax2.step(traj['x'], ratio_yx, where="post", color="tomato",    lw=2, label="Y / X  (NaN when X=0)")
    ax2.set_xlabel(traj['xlabel'])
    ax2.set_ylabel("Ratio")
    ax2.set_title("Ratios of combined totals")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def save_csv(times, nx_trace, ny_trace, events, path="simulation.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "X", "Y", "event"])
        for row in zip(times, nx_trace, ny_trace, events):
            writer.writerow(row)
    print(f"Saved {len(times)} rows to {path}")


def plot_simulation(times, nx_trace, ny_trace):
    nx = np.array(nx_trace, dtype=float)
    ny = np.array(ny_trace, dtype=float)
    total = nx + ny

    # X/(X+Y): undefined only if total == 0 (only at t=0 if both start at 0)
    ratio_x = np.where(total > 0, nx / total, np.nan)

    # Y/X: undefined when nx == 0; use NaN so the line breaks naturally
    ratio_yx = np.where(nx > 0, ny / nx, np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # ── Panel 1: raw counts ───────────────────────────────────────────────────
    ax1.step(times, nx_trace, where="post", color="steelblue", lw=2, label="X cells")
    ax1.step(times, ny_trace, where="post", color="tomato",    lw=2, label="Y cells")
    ax1.set_ylabel("Cell count")
    ax1.set_title("Cell-type counts over time (Gillespie simulation)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: ratios ───────────────────────────────────────────────────────
    ax2.step(times, ratio_x,  where="post", color="steelblue", lw=2, label="X / (X+Y)")
    ax2.step(times, ratio_yx, where="post", color="tomato",    lw=2, label="Y / X  (NaN when X=0)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Ratio")
    ax2.set_title("Ratios over time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ── Demo run ──────────────────────────────────────────────────────────────────
def _print_estimates(label, est, true=None):
    """Pretty-print MLE estimates with optional comparison to true values."""
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    params = [('a', 'X→X+X'), ('b', 'X→X+Y'), ('c', 'Y→Y+X'), ('d', 'Y→Y+Y')]
    for sym, desc in params:
        hat = est[sym]
        n   = est[f'n_{sym}']
        row = f"  {sym}  ({desc})  n={n:6d}   est={hat:.4f}"
        if true is not None:
            row += f"   true={true[sym]:.4f}   err={hat - true[sym]:+.4f}"
        print(row)
    print(f"\n  Λ_X = {est['Lambda_X']:.2f}   Λ_Y = {est['Lambda_Y']:.2f}")


if __name__ == "__main__":
    TRUE = dict(a=0.5, b=0.1, c=0.05, d=0.4)

    # ── Single-pool estimation ────────────────────────────────────────────────
    print("\n══ Single pool (N=1000) ══")
    times, nx_trace, ny_trace, events = simulate_cells(
        **TRUE, nx0=1, ny0=0, N=1000, seed=42
    )
    print(f"Final counts : X={nx_trace[-1]}, Y={ny_trace[-1]}")
    print(f"Total events : {len(times) - 1}")
    print(f"Elapsed time : {times[-1]:.4f}")

    est_single = estimate_rates_single(times, nx_trace, ny_trace, events)
    _print_estimates("MLE — single pool", est_single, true=TRUE)

    save_csv(times, nx_trace, ny_trace, events, path="simulation.csv")
    plot_simulation(times, nx_trace, ny_trace)

    # ── Parallel-splitting estimation (K=100, p=0.5) ─────────────────────────
    print("\n══ Parallel splitting (K=100, p=0.5, N=1000) ══")
    pools_par = simulate_with_splitting(
        **TRUE, nx0=1, ny0=0, N=1000, K=100, p=0.5,
        mode="parallel", seed=42
    )
    print(f"Pools created : {len(pools_par['pools'])}")

    est_par = estimate_rates_parallel(pools_par)
    _print_estimates("MLE — parallel splitting", est_par, true=TRUE)

    save_csv_split(pools_par, path="simulation_split_par.csv")
    plot_simulation_split(pools_par)
    plt.suptitle("Parallel splitting (K=100, p=0.5)", fontsize=11)
    plot_combined_split(pools_par)
    plt.suptitle("Parallel splitting — combined (K=100, p=0.5)", fontsize=11)

    plt.show()
