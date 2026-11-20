"""Cross-check of the analytic finite-shot capacity against a readout trained on actually sampled shots.

For a few settings (lam, p) the reservoir is simulated once. For every shot number S the features are then
estimated from S drawn shots per step (qrc.sampling.noisy_features), a linear readout is trained on the first
half and tested on the second half (qrc.sampling.readout_capacity), and the result is compared with the
analytic capacity C(S) = R^T (G + V/S)^+ R (qrc.ipc.capacities). Each S is sampled with several shot seeds.

    uv run python examples/shot_check.py --config examples/configs/shot_check.toml
    uv run python examples/shot_check.py --config examples/configs/shot_check.toml --M 100000
    uv run python examples/shot_check.py --points 0.8,0.7 0.8,0.1      # without a job file

The jobs (points, shot numbers, M, ...) are best listed in a TOML job file (--config); command-line options
override it. Saves examples/figures/<config name>.png and .npz (examples/figures/shot_check.* without a config
file) and prints a table.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")     # small matrices: one BLAS thread is much faster; parallelise over processes

import argparse
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qrc.ipc import capacities, legendre_targets, summarize
from qrc.model import Params
from qrc.reservoir import simulate

FIGURES = Path(__file__).resolve().parent / "figures"   # examples/figures, independent of the working directory
from qrc.sampling import noisy_features, readout_capacity

KEYS = ("total", "degree1", "degree2", "degree3")


def check_point(job):
    """Simulate one (lam, p) and return analytic and sampled capacities (per KEY) for all S."""
    lam, p, seed, M, T_wash, shots, reps = job
    params = replace(Params(), seed=seed)
    t0 = time.time()
    run = simulate(params, lam, p, M=M, T_wash=T_wash)
    Y, table = legendre_targets(run.u, run.T_wash)
    t_sim = time.time() - t0

    analytic = np.array([[summarize(capacities(run.x, run.cov, Y, S), table)[k] for k in KEYS] for S in shots])
    ideal = np.array([summarize(capacities(run.x, run.cov, Y, np.inf), table)[k] for k in KEYS])
    sampled = np.empty((len(shots), reps, len(KEYS)))
    for i, S in enumerate(shots):
        for r in range(reps):
            rng = np.random.default_rng([seed, r, int(S)])
            x_hat = noisy_features(run.pop, params.n, lam, int(S), rng)
            s = summarize(readout_capacity(x_hat, Y), table)
            sampled[i, r] = [s[k] for k in KEYS]
    return dict(lam=lam, p=p, analytic=analytic, ideal=ideal, sampled=sampled,
                t_sim=t_sim, t_total=time.time() - t0, washout=run.washout_distance)


def parse_point(text):
    lam, p = (float(v) for v in text.split(","))
    return lam, p


DEFAULTS = {"points": [(0.8, 0.7), (0.8, 0.1), (0.2, 0.5), (1.0, 0.5)], "shots": [10, 1e2, 1e3, 1e4, 1e5],
            "reps": 3, "seed": 0, "M": 50000, "T_wash": 1000}


def load_config(path):
    """Settings from a TOML job file (see examples/configs/shot_check.toml); points become (lam, p) tuples."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
    if "points" in cfg:
        cfg["points"] = [(float(pt["lam"]), float(pt["p"])) for pt in cfg["points"]]
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="TOML job file; command-line options override it")
    ap.add_argument("--points", type=parse_point, nargs="+", help="settings as lam,p (space separated)")
    ap.add_argument("--shots", type=float, nargs="+", help="shot numbers S")
    ap.add_argument("--reps", type=int, help="shot seeds per S")
    ap.add_argument("--seed", type=int, help="coupling seed (device)")
    ap.add_argument("--M", type=int, help="number of samples after the washout")
    ap.add_argument("--T_wash", type=int, help="washout steps")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes")
    cli = ap.parse_args()

    # priority: command line > config file > DEFAULTS
    settings = dict(DEFAULTS)
    if cli.config is not None:
        settings.update(load_config(cli.config))
    settings.update({k: v for k, v in vars(cli).items() if k in DEFAULTS and v is not None})
    args = argparse.Namespace(**settings, workers=cli.workers)
    name = cli.config.stem if cli.config is not None else "shot_check"

    shots = np.array(args.shots, dtype=float)
    jobs = [(lam, p, args.seed, args.M, args.T_wash, shots, args.reps) for lam, p in args.points]
    t0 = time.time()
    with ProcessPoolExecutor(min(args.workers, len(jobs))) as pool:
        results = list(pool.map(check_point, jobs))
    print(f"{len(jobs)} settings in {time.time() - t0:.0f} s (M = {args.M}, {args.reps} shot seeds per S)\n")

    # table: total IPC, analytic vs sampled (mean +- std over shot seeds)
    for res in results:
        print(f"lam = {res['lam']}, p = {res['p']}  (simulation {res['t_sim']:.0f} s, "
              f"washout distance {res['washout']:.1e}, S = inf: total {res['ideal'][0]:.3f})")
        print(f"{'S':>9} {'analytic':>9} {'sampled':>17} {'diff':>8}")
        for i, S in enumerate(shots):
            a = res["analytic"][i, 0]
            m, s = res["sampled"][i, :, 0].mean(), res["sampled"][i, :, 0].std()
            print(f"{S:>9.3g} {a:9.3f} {m:9.3f} +- {s:5.3f} {m - a:+8.3f}")
        print()

    # figure: one panel per setting, analytic lines and sampled points per degree
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4), sharey=True, squeeze=False)
    for ax, res in zip(axes[0], results):
        for j, key in enumerate(KEYS):
            line, = ax.semilogx(shots, res["analytic"][:, j], "-", label=key)
            ax.errorbar(shots, res["sampled"][:, :, j].mean(axis=1), yerr=res["sampled"][:, :, j].std(axis=1),
                        fmt="o", color=line.get_color(), ms=4, capsize=2)
            ax.axhline(res["ideal"][j], color=line.get_color(), ls=":", lw=1)
        ax.set(xlabel="shots per step S", title=f"lam = {res['lam']}, p = {res['p']}")
    axes[0, 0].set_ylabel("IPC")
    axes[0, 0].legend(title="line: analytic\npoints: sampled", fontsize=8)
    fig.suptitle(f"analytic vs sampled IPC, M = {args.M}, seed = {args.seed}, dotted: S = inf")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=150)
    np.savez(FIGURES / f"{name}.npz", shots=shots, keys=KEYS,
             points=np.array([(r["lam"], r["p"]) for r in results]),
             analytic=np.array([r["analytic"] for r in results]),
             sampled=np.array([r["sampled"] for r in results]),
             ideal=np.array([r["ideal"] for r in results]))
    print(f"saved examples/figures/{name}.png and examples/figures/{name}.npz")


if __name__ == "__main__":
    main()
