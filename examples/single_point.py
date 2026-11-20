"""IPC of one reservoir setting as a function of the number of shots S.

    uv run python examples/single_point.py --lam 1.0 --p 0.3
    uv run python examples/single_point.py --lam 0.8 --p 0.7 --eps 0.5 --delays 20 10 7 5 3

--delays gives the largest delay per degree (degree d = 1 .. len(delays)).

Saves examples/figures/single_point.png and prints the capacities.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")     # small matrices: one BLAS thread is much faster

import argparse
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qrc.ipc import capacities, legendre_targets, summarize
from qrc.model import Params
from qrc.reservoir import simulate

FIGURES = Path(__file__).resolve().parent / "figures"   # examples/figures, independent of the working directory


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lam", type=float, default=1.0, help="measurement strength")
    ap.add_argument("--p", type=float, default=0.3, help="reset strength")
    ap.add_argument("--eps", type=float, default=Params().eps, help="input amplitude")
    ap.add_argument("--seed", type=int, default=0, help="coupling seed (device)")
    ap.add_argument("--M", type=int, default=10000, help="number of samples after the washout")
    ap.add_argument("--T_wash", type=int, default=1000, help="washout steps")
    ap.add_argument("--delays", type=int, nargs="+", default=[20, 10, 7],
                    help="largest delay per degree, e.g. 20 10 7 5 3 for degrees 1 .. 5")
    args = ap.parse_args()

    params = replace(Params(), eps=args.eps, seed=args.seed)
    t0 = time.time()
    run = simulate(params, args.lam, args.p, M=args.M, T_wash=args.T_wash)
    Y, table = legendre_targets(run.u, run.T_wash, tuple(args.delays))
    print(f"simulation + targets: {time.time() - t0:.1f} s, washout trace distance {run.washout_distance:.1e}")

    shots = np.logspace(1, 6, 11)
    rows = [summarize(capacities(run.x, run.cov, Y, S), table) for S in shots]
    ideal = summarize(capacities(run.x, run.cov, Y, np.inf), table)
    keys = ["total"] + [f"degree{d}" for d in range(1, len(args.delays) + 1)]
    print(f"{len(table)} targets, delays {tuple(args.delays)}")
    print(f"{'S':>10}" + "".join(f" {k.replace('degree', 'deg'):>7}" for k in keys))
    for S, r in zip([*shots, np.inf], [*rows, ideal]):
        print(f"{S:>10.3g}" + "".join(f" {r[k]:7.3f}" for k in keys))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for key in keys:
        label = key.replace("degree", "degree ")
        line, = ax1.semilogx(shots, [r[key] for r in rows], "o-", label=label)
        ax1.axhline(ideal[key], color=line.get_color(), ls=":", lw=1)
    ax1.set(xlabel="shots per step S", ylabel="IPC", title="IPC vs S (dotted: S = inf)")
    ax1.legend()
    mem = summarize(capacities(run.x, run.cov, Y, 1e4), table)["memory"]
    ax2.bar(np.arange(len(mem)), mem)
    ax2.set(xlabel="delay k", ylabel="capacity", title="linear memory function, S = 1e4")
    fig.suptitle(f"N = {params.n}, lam = {args.lam}, p = {args.p}, eps = {params.eps}, tau = {params.tau}, "
                 f"seed = {params.seed}, delays = {tuple(args.delays)}")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "single_point.png", dpi=150)
    print("saved examples/figures/single_point.png")


if __name__ == "__main__":
    main()
