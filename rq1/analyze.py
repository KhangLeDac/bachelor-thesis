"""Analyse the results of rq1/map.py. Works on partial results (missing jobs are ignored).

    uv run python rq1/analyze.py --config rq1/configs/map.toml

Prints checks (washout, memory cut-off) and the optimum per S, and saves to rq1/figures/:
    maps.png      mean total IPC over the devices per S (star: optimum), std over devices below
    optimum.png   optimum (lam*, p*) vs S with bootstrap spread over devices, and the IPC there
    ridge.png     memory function and degree split at the optimum of every finite S

All averages are over the devices with a complete map. The table also gives, at the optimum (ratios are taken of
the means, stabler than the mean of per-device ratios when the capacities are small):
    nonlinear fraction  = (mean C2 + mean C3) / mean C_tot, masked where mean C_tot < --min-total
    memory length       = sum_d d m_d / sum_d m_d over the delays with m_d > --mem-eps, m_d = mean memory
                          function over the devices; masked where no delay passes (no usable memory)
Caveat: degree 2 and 3 only include delays <= 10 and <= 7, so where the memory is long (small lam and p) the
nonlinear capacity, and with it the nonlinear fraction, is underestimated; only the sums per degree are saved,
so this cut-off cannot be checked from the results.
"""

import argparse
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent     # rq1/: configs, results and figures live here
FIGDIR = HERE / "figures"


def memory_length(mem: np.ndarray, eps: float) -> np.ndarray:
    """Centroid sum_d d m_d / sum_d m_d of memory functions (last axis = delay d = 0, 1, ...), using only the
    delays with m_d > eps (removes the noise tail and negative cross-fitted values). NaN if no delay passes."""
    w = np.where(mem > eps, mem, 0.0)
    norm = w.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 0, (w * np.arange(mem.shape[-1])).sum(axis=-1) / norm, np.nan)


def load(cfg: dict):
    """Arrays over (lam, p, seed, S): total (..), degree (.., 3), memory (.., K), washout (lam, p, seed)."""
    lams, ps, seeds, shots = cfg["lams"], cfg["ps"], cfg["seeds"], np.array(cfg["shots"], dtype=float)
    out = HERE / cfg["out"]
    K = cfg["delays"][0] + 1
    shape = (len(lams), len(ps), len(seeds), len(shots))
    total = np.full(shape, np.nan) # total capacity
    degree = np.full(shape + (3,), np.nan) # degree 1, 2, 3 capacities
    memory = np.full(shape + (K,), np.nan) # linear memory function
    washout = np.full(shape[:3], np.nan)
    n_warn = 0
    for i, lam in enumerate(lams):
        for j, p in enumerate(ps):
            for k, seed in enumerate(seeds):
                f = out / f"lam{lam:.2f}_p{p:.2f}_seed{seed}.npz"
                if not f.exists():
                    continue
                d = np.load(f)
                total[i, j, k], degree[i, j, k], memory[i, j, k] = d["total"], d["degree"], d["memory"]
                washout[i, j, k] = d["washout"]
                n_warn += len(d["warnings"]) > 0
    return np.array(lams), np.array(ps), np.array(seeds), shots, total, degree, memory, washout, n_warn


def label(S: float) -> str:
    return "S = inf" if np.isinf(S) else f"S = 1e{np.log10(S):.0f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="the job file used for the run")
    ap.add_argument("--boot", type=int, default=500, help="bootstrap resamples over devices")
    ap.add_argument("--min-total", type=float, default=0.1,
                    help="mask the nonlinear fraction where the mean total IPC is below this")
    ap.add_argument("--mem-eps", type=float, default=0.01,
                    help="delays with mean memory function below this are left out of the memory length")
    args = ap.parse_args()
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    lams, ps, seeds, shots, total, degree, memory, washout, n_warn = load(cfg)

    # ---- checks
    done = np.isfinite(washout)
    complete = done.all(axis=(0, 1))                     # devices with a complete map
    print(f"jobs done: {done.sum()} / {done.size}; complete devices: {seeds[complete].tolist()}")
    if not done.any():
        raise SystemExit("no results yet")
    print(f"washout: max trace distance {np.nanmax(washout):.1e}; jobs with warnings: {n_warn}")
    inf_idx = np.flatnonzero(np.isinf(shots))
    if inf_idx.size:
        tail = memory[:, :, :, inf_idx[0], -1]
        bad = np.argwhere(np.nanmean(tail, axis=2) > 0.01)
        msg = ", ".join(f"(lam={lams[i]}, p={ps[j]})" for i, j in bad) or "none"
        print(f"memory at the largest delay > 0.01 (S = inf; total underestimated there): {msg}")

    # ---- use devices with a complete map for averages (fair comparison between grid points)
    use = complete if complete.any() else done.any(axis=(0, 1))
    T = total[:, :, use]                                 # (lam, p, dev, S)
    # total ICP averaged
    n_dev = T.shape[2]
    mean, std = np.nanmean(T, axis=2), np.nanstd(T, axis=2)
    deg_m = np.nanmean(degree[:, :, use], axis=2)       # (lam, p, S, 3)
    mem_m = np.nanmean(memory[:, :, use], axis=2)       # (lam, p, S, K)
    with np.errstate(invalid="ignore", divide="ignore"):
        nl_frac = np.where(mean >= args.min_total, (deg_m[..., 1] + deg_m[..., 2]) / mean, np.nan)
    mem_len = memory_length(mem_m, args.mem_eps)        # (lam, p, S)

    print(f"\noptimum per S (mean over {n_dev} devices; sem = std / sqrt(n)):")
    print(f"{'S':>9} {'lam*':>5} {'p*':>5} {'IPC*':>7} {'sem':>6} {'2nd best':>9} {'nl frac':>8} {'mem len':>8}")
    opt = []
    for s, S in enumerate(shots):
        m = mean[:, :, s]
        i, j = np.unravel_index(np.nanargmax(m), m.shape)
        second = np.sort(m[np.isfinite(m)])[-2] if np.isfinite(m).sum() > 1 else np.nan
        opt.append((i, j))
        print(f"{S:>9.3g} {lams[i]:5.2f} {ps[j]:5.2f} {m[i, j]:7.3f} {std[i, j, s] / np.sqrt(n_dev):6.3f} "
              f"{second:9.3f} {nl_frac[i, j, s]:8.3f} {mem_len[i, j, s]:8.2f}")

    # ---- figure 1: maps C(p, lambda) per S
    nS = len(shots)
    fig, axes = plt.subplots(2, nS, figsize=(3.6 * nS, 6.4), squeeze=False)
    ext = dict(origin="lower", aspect="auto", cmap="viridis")
    for s, S in enumerate(shots):
        for row, (data, what) in enumerate([(mean, "mean"), (std, "std over devices")]):
            ax = axes[row, s]
            im = ax.imshow(data[:, :, s], **ext)
            fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_xticks(range(len(ps))[::3], [f"{p:g}" for p in ps[::3]])
            ax.set_yticks(range(len(lams))[::2], [f"{lam:g}" for lam in lams[::2]])
            ax.set_title(f"{label(S)}: {what}", fontsize=9)
            if row == 0:
                ax.plot(opt[s][1], opt[s][0], "*", color="red", ms=10)
            if s == 0:
                ax.set_ylabel("lam")
            if row == 1:
                ax.set_xlabel("p")
    fig.suptitle(f"total IPC, {n_dev} devices, M = {cfg['M']}")
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "maps.png", dpi=130)

    # ---- figure 2: optimum vs S with bootstrap over devices (finite S only)
    fin = np.flatnonzero(np.isfinite(shots))
    rng = np.random.default_rng(0)
    boot = np.empty((args.boot, len(fin), 2))
    for b in range(args.boot):
        # rng.integers(0, n_dev, n_dev) gives a bootstrap resample of the devices (with replacement)
        # the mean over the resample 
        Tb = np.nanmean(T[:, :, rng.integers(0, n_dev, n_dev)], axis=2)
        for c, s in enumerate(fin):
            # find the optimum (lam*, p*) for each S
            i, j = np.unravel_index(np.nanargmax(Tb[:, :, s]), Tb[:, :, s].shape)
            boot[b, c] = lams[i], ps[j]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, c, what in [(axes[0], 0, "lam*"), (axes[1], 1, "p*")]:
        lo, med, hi = np.percentile(boot[:, :, c], [16, 50, 84], axis=0)
        best = [(lams, ps)[c][opt[s][c]] for s in fin]
        ax.fill_between(shots[fin], lo, hi, alpha=0.3, label="bootstrap 16-84 %")
        ax.semilogx(shots[fin], best, "o-", label="optimum of the mean")
        ax.set(xlabel="shots per step S", ylabel=what)
        ax.legend(fontsize=8)
    ipc_opt = [mean[opt[s][0], opt[s][1], s] for s in fin]
    sem_opt = [std[opt[s][0], opt[s][1], s] / np.sqrt(n_dev) for s in fin]
    axes[2].errorbar(shots[fin], ipc_opt, yerr=sem_opt, fmt="o-", capsize=3)
    axes[2].set(xscale="log", xlabel="shots per step S", ylabel="total IPC at the optimum")
    fig.suptitle(f"best constant (lam, p) vs S, {n_dev} devices")
    fig.tight_layout()
    fig.savefig(FIGDIR / "optimum.png", dpi=130)

    # ---- figure 3: memory function and degree split at the optimum of every finite S
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.8 / len(fin)
    for c, s in enumerate(fin):
        i, j = opt[s]
        lab = f"{label(shots[s])}: lam={lams[i]:g}, p={ps[j]:g}"
        ax1.plot(np.nanmean(memory[i, j][use][:, s], axis=0), "o-", ms=3, label=lab)
        ax2.bar(np.arange(3) + c * width, np.nanmean(degree[i, j][use][:, s], axis=0), width, label=lab)
    ax1.set(xlabel="delay k", ylabel="degree-1 capacity", title="linear memory function at the optimum")
    ax2.set_xticks(np.arange(3) + 0.4 - width / 2, ["degree 1", "degree 2", "degree 3"])
    ax2.set(ylabel="capacity", title="degree split at the optimum")
    ax1.legend(fontsize=7)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / "ridge.png", dpi=130)
    print(f"\nsaved maps.png, optimum.png, ridge.png to {FIGDIR}")

if __name__ == "__main__":
    main()
