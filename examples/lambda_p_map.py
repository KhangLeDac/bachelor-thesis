"""Total IPC over a grid of measurement strengths lam and reset strengths p, at a fixed number of shots S,
averaged over several devices (coupling seeds). Grid points run in parallel processes.

    uv run python examples/lambda_p_map.py --S 1e4 --devices 3

Saves examples/figures/lambda_p_map.png.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")     # small matrices: one BLAS thread is much faster; parallelise over processes

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qrc.ipc import ipc, summarize
from qrc.model import Params
from qrc.reservoir import simulate

FIGURES = Path(__file__).resolve().parent / "figures"   # examples/figures, independent of the working directory

LAMS = (0.2, 0.4, 0.6, 0.8, 1.0)
PS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


def total_ipc(job):
    lam, p, seed, S, M = job
    run = simulate(replace(Params(), seed=seed), lam, p, M=M)
    C, table = ipc(run, S)
    return summarize(C, table)["total"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--S", type=float, default=1e4, help="shots per step (inf allowed)")
    ap.add_argument("--devices", type=int, default=3, help="number of coupling seeds to average over")
    ap.add_argument("--M", type=int, default=10000, help="number of samples after the washout")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes")
    args = ap.parse_args()

    jobs = [(lam, p, seed, args.S, args.M) for lam, p, seed in product(LAMS, PS, range(args.devices))]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        totals = np.array(list(pool.map(total_ipc, jobs))).reshape(len(LAMS), len(PS), args.devices)
    print(f"{len(jobs)} runs in {time.time() - t0:.0f} s")
    mean = totals.mean(axis=2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(mean, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(PS)), [str(p) for p in PS])
    ax.set_yticks(range(len(LAMS)), [str(lam) for lam in LAMS])
    ax.set(xlabel="reset strength p", ylabel="measurement strength lam",
           title=f"total IPC, S = {args.S:g}, mean over {args.devices} devices")
    for i, j in product(range(len(LAMS)), range(len(PS))):
        ax.text(j, i, f"{mean[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label="total IPC")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "lambda_p_map.png", dpi=150)
    print("saved examples/figures/lambda_p_map.png")


if __name__ == "__main__":
    main()
