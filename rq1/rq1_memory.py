"""RQ1 memory: how fast does the reservoir forget its initial state, over the (lam, p) grid?

For every (lam, p, device) and every input sequence, two initial states (|0...0> and I/d) are driven by the
same inputs and the trace distance D(t) between them is recorded after every step
(qrc.reservoir.trace_distance_curve). No shots and no readout are involved, so the result does not depend on S.
From each curve:
    gamma        decay rate per step, fitted as log D(t) = c - gamma t where D lies inside ``fit_window``
    tau = 1/gamma   memory time in steps (D shrinks by a factor e every tau steps)
    t_1e-3       first step with D < 1e-3 (model-free alternative)

    uv run python rq1/rq1_memory.py --config rq1/configs/rq1_memory.toml            # compute (or load) and plot
    uv run python rq1/rq1_memory.py --config rq1/configs/rq1_memory.toml --force    # recompute

The curves are saved to <out>/curves.npz (about 20 MB for the default grid); a second call only loads them and
redraws. Figures go to rq1/figures/<name>/ (<name> = job file name):
    memory_map.png   mean tau over devices, relative spread over devices, mean t_1e-3
    memory_cuts.png  tau vs p for every lam, and example curves D(t)
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")     # small matrices: one BLAS thread is much faster; parallelise over processes

import argparse
import subprocess
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qrc.model import Params
from qrc.reservoir import trace_distance_curve

HERE = Path(__file__).resolve().parent     # rq1/: configs, results and figures live here
REQUIRED = ("lams", "ps", "seeds", "input_seeds", "T", "fit_window", "out")


def load_config(path: Path) -> dict:
    """Load and check a TOML job file."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    missing = [k for k in REQUIRED if k not in cfg]
    unknown = sorted(set(cfg) - set(REQUIRED))
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {missing}, unknown keys {unknown}")
    return cfg


def git_commit() -> str:
    """Short hash of the current git commit (label for the saved results)."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_job(job):
    """Trace-distance curves of one (lam, p, seed) for all input sequences. Returns (len(input_seeds), T)."""
    lam, p, seed, input_seeds, T = job
    params = replace(Params(), seed=seed)
    # same generator call as simulate(): input_seed 0 reproduces the first T inputs of the rq1_map runs
    return np.array([trace_distance_curve(params, lam, p, np.random.default_rng(k).uniform(-1.0, 1.0, T))
                     for k in input_seeds])


def compute(cfg: dict, workers: int) -> np.ndarray:
    """All curves, shape (lam, p, seed, input_seed, T)."""
    lams, ps, seeds = cfg["lams"], cfg["ps"], cfg["seeds"]
    jobs = [(lam, p, seed, cfg["input_seeds"], cfg["T"]) for lam, p, seed in product(lams, ps, seeds)]
    print(f"{len(jobs)} jobs x {len(cfg['input_seeds'])} input sequences x {cfg['T']} steps "
          f"with {workers} workers", flush=True)
    t0 = time.time()
    every = max(1, len(jobs) // 20)
    res = []
    with ProcessPoolExecutor(workers) as pool:
        for i, r in enumerate(pool.map(run_job, jobs, chunksize=4), start=1):
            res.append(r)
            if i % every == 0 or i == len(jobs):
                elapsed = time.time() - t0
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(jobs)} done, elapsed {elapsed / 60:.1f} min, "
                      f"remaining ~{elapsed / i * (len(jobs) - i) / 60:.1f} min", flush=True)
    # jobs were built in the order lam, p, seed -> reshape restores the grid
    return np.array(res).reshape(len(lams), len(ps), len(seeds), len(cfg["input_seeds"]), cfg["T"])


def decay_rate(curves: np.ndarray, window) -> np.ndarray:
    """Fitted decay rate gamma per step for every curve (last axis = time); NaN if < 3 points in the window."""
    lo, hi = window
    flat = curves.reshape(-1, curves.shape[-1])
    t = np.arange(1, flat.shape[1] + 1)       # D[0] is the distance after the first step
    gamma = np.full(len(flat), np.nan)
    for c, d in enumerate(flat):
        sel = (d > lo) & (d < hi)
        if sel.sum() >= 3:
            gamma[c] = -np.polyfit(t[sel], np.log(d[sel]), 1)[0]
    return gamma.reshape(curves.shape[:-1])


def first_below(curves: np.ndarray, eps: float) -> np.ndarray:
    """First step (1-based) with D < eps for every curve; NaN if never reached."""
    below = curves < eps
    return np.where(below.any(axis=-1), below.argmax(axis=-1) + 1.0, np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="TOML job file, e.g. rq1/configs/rq1_memory.toml")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="parallel processes")
    ap.add_argument("--force", action="store_true", help="recompute even if the curves are saved")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = args.config.stem
    out = HERE / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "curves.npz"
    lams, ps, seeds = np.array(cfg["lams"]), np.array(cfg["ps"]), np.array(cfg["seeds"])

    # ---- compute or load
    grid_keys = ("lams", "ps", "seeds", "input_seeds", "T")
    if path.exists() and not args.force:
        d = np.load(path)
        if any(not np.array_equal(d[k], np.array(cfg[k])) for k in grid_keys):
            raise SystemExit(f"{path} was computed with a different grid; use --force or a new 'out'.")
        curves = d["curves"].astype(float)
        print(f"loaded {path} (commit {d['commit']})")
    else:
        t0 = time.time()
        curves = compute(cfg, args.workers)
        tmp = path.with_name("curves.tmp.npz")
        np.savez(tmp, curves=curves.astype(np.float32), commit=git_commit(), seconds=time.time() - t0,
                 **{k: np.array(cfg[k]) for k in grid_keys})
        os.replace(tmp, path)
        print(f"saved {path}")

    # ---- per-curve measures, then average over input sequences -> (lam, p, seed)
    gamma = decay_rate(curves, cfg["fit_window"])
    tau = np.nanmean(1.0 / gamma, axis=3)
    t3 = np.nanmean(first_below(curves, 1e-3), axis=3)

    # ---- checks
    lo = cfg["fit_window"][0]
    not_decayed = ~(curves < lo).any(axis=-1)
    print(f"curves that never fall below {lo:g} within T = {cfg['T']}: {not_decayed.sum()} / {not_decayed.size}"
          + ("  -> increase T" if not_decayed.any() else ""))
    print(f"curves without a fit (< 3 points in the window): {np.isnan(gamma).sum()} / {gamma.size}")
    spread_in = np.nanstd(1.0 / gamma, axis=3) / np.nanmean(1.0 / gamma, axis=3)
    print(f"spread of tau over input sequences (std/mean): median {np.nanmedian(spread_in):.3f}, "
          f"max {np.nanmax(spread_in):.3f}")

    # ---- mean over devices
    tau_m, tau_s = np.nanmean(tau, axis=2), np.nanstd(tau, axis=2)
    t3_m = np.nanmean(t3, axis=2)
    i, j = np.unravel_index(np.nanargmax(tau_m), tau_m.shape)
    k, m = np.unravel_index(np.nanargmin(tau_m), tau_m.shape)
    print(f"\nmemory time tau (steps), mean over {len(seeds)} devices:")
    print(f"  longest  {tau_m[i, j]:7.2f} at lam = {lams[i]:g}, p = {ps[j]:g}")
    print(f"  shortest {tau_m[k, m]:7.2f} at lam = {lams[k]:g}, p = {ps[m]:g}")
    print(f"  relative spread over devices (std/mean): median {np.nanmedian(tau_s / tau_m):.3f}, "
          f"max {np.nanmax(tau_s / tau_m):.3f}")

    # ---- figure 1: maps
    figdir = HERE / "figures" / name
    figdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ext = dict(origin="lower", aspect="auto", cmap="viridis")
    for ax, data, title in [(axes[0], np.log10(tau_m), "log10 tau (steps), mean over devices"),
                            (axes[1], tau_s / tau_m, "tau: std / mean over devices"),
                            (axes[2], np.log10(t3_m), "log10 t(D < 1e-3) (steps), mean")]:
        im = ax.imshow(data, **ext)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(len(ps))[::3], [f"{p:g}" for p in ps[::3]])
        ax.set_yticks(range(len(lams))[::2], [f"{lam:g}" for lam in lams[::2]])
        ax.set(xlabel="p", ylabel="lam")
        ax.set_title(title, fontsize=9)
    fig.suptitle(f"memory time from the trace distance, {len(seeds)} devices x {len(cfg['input_seeds'])} inputs")
    fig.tight_layout()
    fig.savefig(figdir / "memory_map.png", dpi=130)

    # ---- figure 2: tau vs p per lam, and example curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(lams)))
    for a, lam in enumerate(lams):
        ax1.errorbar(ps, tau_m[a], yerr=tau_s[a] / np.sqrt(len(seeds)), color=colors[a], ms=3, fmt="o-",
                     capsize=2, label=f"lam = {lam:g}")
    ax1.set(xlabel="p", ylabel="tau (steps)", yscale="log", title="memory time vs reset strength")
    ax1.legend(fontsize=7, ncol=2)
    t = np.arange(1, cfg["T"] + 1)
    for a, b in [(0, 0), (len(lams) // 2, len(ps) // 2), (len(lams) - 1, len(ps) - 1), (len(lams) - 1, 0)]:
        ax2.semilogy(t, curves[a, b, 0, 0], label=f"lam = {lams[a]:g}, p = {ps[b]:g}")
    ax2.axhspan(*cfg["fit_window"], color="gray", alpha=0.15, label="fit window")
    ax2.set(xlabel="step t", ylabel="trace distance D(t)", ylim=(1e-16, 2),
            title=f"example curves (device {seeds[0]}, input {cfg['input_seeds'][0]})")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "memory_cuts.png", dpi=130)
    print(f"\nsaved rq1/figures/{name}/memory_map.png, memory_cuts.png")


if __name__ == "__main__":
    main()
