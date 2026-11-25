"""RQ1 baseline: capacities over a (lam, p) grid, several devices, evaluated at several shot numbers S.

One job = one (lam, p, seed): simulate once, then evaluate total, degree 1..3 and the linear memory function
at every S of the job file. Each job is saved immediately as its own file

    <out>/lam0.80_p0.70_seed3.npz

so a long run can be interrupted and restarted: jobs whose file exists are skipped. ``out`` is relative to
the rq1/ folder, independent of the directory the script is started from. The job file is copied to
<out>/config.toml; a restart with different settings into the same ``out`` is refused.

    uv run python rq1/map.py --config rq1/configs/map.toml --dry-run   # count jobs only
    nohup uv run python rq1/map.py --config rq1/configs/map.toml > rq1/logs/map.log 2>&1 &

Analyse with rq1/analyze.py.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")     # small matrices: one BLAS thread is much faster; parallelise over processes

import argparse
import shutil
import subprocess
import time
import tomllib
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np

from qrc.ipc import capacities, legendre_targets, summarize
from qrc.model import Params
from qrc.reservoir import simulate

HERE = Path(__file__).resolve().parent     # rq1/: configs, results and figures live here
REQUIRED = ("lams", "ps", "seeds", "shots", "M", "T_wash", "delays", "out")


def load_config(path: Path) -> dict:
    """Load and check a TOML job file."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    missing = [k for k in REQUIRED if k not in cfg]
    unknown = sorted(set(cfg) - set(REQUIRED))
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {missing}, unknown keys {unknown}")
    return cfg


def job_path(out: Path, lam: float, p: float, seed: int) -> Path:
    """builds the path to the result file"""
    return out / f"lam{lam:.2f}_p{p:.2f}_seed{seed}.npz"


def git_commit() -> str:
    """reads the current git commit hash"""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_job(job):
    """Simulate one (lam, p, seed), evaluate it at all S and save the result. Returns (path, seconds)."""
    lam, p, seed, cfg, out, commit = job
    t0 = time.time()
    # run simulation
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = simulate(replace(Params(), seed=seed), lam, p, M=cfg["M"], T_wash=cfg["T_wash"])
    # legendre targets
    Y, table = legendre_targets(run.u, run.T_wash, tuple(cfg["delays"]))
    # capacities for all S:
    """
    run.x = (M, n) reservoir exact expectation values
    run.cov = (M, M, n) measurement-noise covariance
    outcome: res = []
    """
    res = [summarize(capacities(run.x, run.cov, Y, float(S)), table) for S in cfg["shots"]]

    # writes one job's result for one (lam, p, seed) to a disk
    path = job_path(out, lam, p, seed)
    tmp = path.with_name(path.stem + ".tmp.npz")       # write, then rename: no half-written result files
    # results: total, degree 1..3, memory function, washout distance, warnings, commit hash, runtime
    np.savez(tmp, lam=lam, p=p, seed=seed, shots=np.array(cfg["shots"], dtype=float),
             total=np.array([r["total"] for r in res]),
             degree=np.array([[r[f"degree{d}"] for d in (1, 2, 3)] for r in res]),
             memory=np.array([r["memory"] for r in res]),
             washout=run.washout_distance, M=cfg["M"], T_wash=cfg["T_wash"],
             warnings=np.array([str(w.message) for w in caught], dtype=str), commit=commit,
             seconds=time.time() - t0)
    os.replace(tmp, path)
    return path, time.time() - t0


def main():

    # parse command line arguments
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="TOML job file, e.g. rq1/configs/map.toml")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="parallel processes")
    ap.add_argument("--dry-run", action="store_true", help="only count jobs, run nothing")
    args = ap.parse_args()

    # load and check the job file
    cfg = load_config(args.config)
    out = HERE / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    saved_cfg = out / "config.toml"
    if saved_cfg.exists():
        if load_config(saved_cfg) != cfg:
            raise SystemExit(f"{out} was produced with different settings ({saved_cfg}). "
                             "Use a new 'out' or delete the folder.")
    else:
        shutil.copyfile(args.config, saved_cfg)

    # to do list for the run
    commit = git_commit()
    # seeds outermost: complete maps of device 0, 1, ... are finished first
    all_jobs = [(lam, p, seed, cfg, out, commit) for seed, lam, p in product(cfg["seeds"], cfg["lams"], cfg["ps"])]
    # filter out jobs whose result file already exists
    jobs = [j for j in all_jobs if not job_path(out, *j[:3]).exists()]
    print(f"{len(all_jobs)} jobs, {len(all_jobs) - len(jobs)} already done, {len(jobs)} to run "
          f"with {args.workers} workers -> {out}  (commit {commit})", flush=True)
    if args.dry_run or not jobs:
        return

    t0 = time.time()
    # report progress every 2% of the total number of jobs
    every = max(1, len(jobs) // 50)
    # run jobs in parallel
    with ProcessPoolExecutor(args.workers) as pool:
        # enumerate returns 1-based index, pool.map sends job to the workers
        for i, (path, sec) in enumerate(pool.map(run_job, jobs, chunksize=1), start=1):
            if i % every == 0 or i == len(jobs):
                # progess
                elapsed = time.time() - t0
                eta = elapsed / i * (len(jobs) - i)
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(jobs)} done, last job {sec:.0f} s, "
                      f"elapsed {elapsed / 60:.1f} min, remaining ~{eta / 60:.1f} min", flush=True)
    print(f"finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
