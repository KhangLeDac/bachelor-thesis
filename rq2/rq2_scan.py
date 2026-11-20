"""RQ2 phase A: fixed input-dependent reset p_t = f(u_t) around the RQ1 optimum.

For every shot number S of the job file the baseline is the RQ1 optimum (lam*_S, p*_S). The response
functions are f(u) = p*_S + a g(u) (see qrc/reset.py) with a = frac * a_max, so <p_t> = p*_S and every
frac in [-1, 1] keeps p_t in [0, 1]. frac = 0 is the constant baseline itself and is always run.

One job = one (baseline S, shape, frac, device seed): simulate once with lam*_S and f, evaluate the
capacities at that S, save to its own file

    <out>/S1e+04_magnitude_frac+0.50_seed3.npz

All jobs use the same input sequence (input_seed 0), so C[f] - C[p*_S] is a paired difference.
Restartable as rq1_map.py: existing files are skipped, the job file is copied to <out>/config.toml.

    uv run python rq2/rq2_scan.py --config rq2/configs/phaseA.toml --dry-run
    nohup uv run python rq2/rq2_scan.py --config rq2/configs/phaseA.toml > rq2/logs/phaseA.log 2>&1 &
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
from pathlib import Path

import numpy as np

from qrc.ipc import capacities, legendre_targets, summarize
from qrc.model import Params
from qrc.reservoir import simulate
from qrc.reset import Reset

HERE = Path(__file__).resolve().parent     # rq2/: configs, results and logs live here
REQUIRED = ("seeds", "M", "T_wash", "delays", "out", "fracs", "baseline", "shapes")


def load_config(path: Path) -> dict:
    """Load and check a TOML job file."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    missing = [k for k in REQUIRED if k not in cfg]
    unknown = sorted(set(cfg) - set(REQUIRED))
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {missing}, unknown keys {unknown}")
    for b in cfg["baseline"]:
        if set(b) != {"S", "lam", "p"}:
            raise ValueError(f"{path}: every [[baseline]] needs exactly S, lam, p; got {sorted(b)}")
    names = [shape_name(s) for s in cfg["shapes"]]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: shape names must be unique (set 'name' for repeated kinds): {names}")
    return cfg


def shape_name(shape: dict) -> str:
    return shape.get("name", shape["kind"])


def shape_params(shape: dict) -> dict:
    """Parameters of the shape (everything except kind and name), passed to Reset."""
    return {k: v for k, v in shape.items() if k not in ("kind", "name")}


def job_path(out: Path, S: float, name: str, frac: float, seed: int) -> Path:
    return out / f"S{S:.0e}_{name}_frac{frac:+.2f}_seed{seed}.npz"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def make_jobs(cfg: dict, out: Path, commit: str) -> list:
    """All jobs, devices outermost (complete results of device 0, 1, ... are finished first)."""
    jobs = []
    for seed in cfg["seeds"]:
        for b in cfg["baseline"]:
            jobs.append((b, {"kind": "constant"}, 0.0, seed, cfg, out, commit))
            for shape in cfg["shapes"]:
                for frac in cfg["fracs"]:
                    if frac != 0.0:
                        jobs.append((b, shape, float(frac), seed, cfg, out, commit))
    return jobs


def path_of(job) -> Path:
    b, shape, frac, seed, _, out, _ = job
    return job_path(out, float(b["S"]), shape_name(shape), frac, seed)


def run_job(job):
    """Simulate one (baseline, shape, frac, seed), evaluate at the baseline S, save. Returns (path, seconds)."""
    b, shape, frac, seed, cfg, out, commit = job
    S, lam, p0 = float(b["S"]), float(b["lam"]), float(b["p"])
    f = Reset.from_fraction(shape["kind"], p0, frac, **shape_params(shape))
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = simulate(replace(Params(), seed=seed), lam, f, M=cfg["M"], T_wash=cfg["T_wash"])
    Y, table = legendre_targets(run.u, run.T_wash, tuple(cfg["delays"]))
    r = summarize(capacities(run.x, run.cov, Y, S), table)
    D = len(cfg["delays"])
    p_used = run.p_seq[run.T_wash:]

    path = path_of(job)
    tmp = path.with_name(path.stem + ".tmp.npz")       # write, then rename: no half-written result files
    np.savez(tmp, S=S, lam=lam, p0=p0, kind=shape["kind"], name=shape_name(shape),
             shape=str(shape_params(shape)), frac=frac, a=f.a, a_max=f.a_max(+1 if frac >= 0 else -1),
             p_mean=p_used.mean(), p_std=p_used.std(), seed=seed,
             total=r["total"], degree=np.array([r[f"degree{d}"] for d in range(1, D + 1)]),
             memory=r["memory"], washout=run.washout_distance, M=cfg["M"], T_wash=cfg["T_wash"],
             warnings=np.array([str(w.message) for w in caught], dtype=str), commit=commit,
             seconds=time.time() - t0)
    os.replace(tmp, path)
    return path, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="TOML job file, e.g. rq2/configs/phaseA.toml")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="parallel processes")
    ap.add_argument("--dry-run", action="store_true", help="only count jobs, run nothing")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # build every response function once: fails here (not in a worker) if a shape is invalid
    for b in cfg["baseline"]:
        for shape in cfg["shapes"]:
            for frac in cfg["fracs"]:
                Reset.from_fraction(shape["kind"], float(b["p"]), float(frac), **shape_params(shape))

    out = HERE / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    saved_cfg = out / "config.toml"
    if saved_cfg.exists():
        if load_config(saved_cfg) != cfg:
            raise SystemExit(f"{out} was produced with different settings ({saved_cfg}). "
                             "Use a new 'out' or delete the folder.")
    else:
        shutil.copyfile(args.config, saved_cfg)

    commit = git_commit()
    all_jobs = make_jobs(cfg, out, commit)
    jobs = [j for j in all_jobs if not path_of(j).exists()]
    print(f"{len(all_jobs)} jobs, {len(all_jobs) - len(jobs)} already done, {len(jobs)} to run "
          f"with {args.workers} workers -> {out}  (commit {commit})", flush=True)
    if args.dry_run or not jobs:
        return

    t0 = time.time()
    every = max(1, len(jobs) // 50)
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (path, sec) in enumerate(pool.map(run_job, jobs, chunksize=1), start=1):
            if i % every == 0 or i == len(jobs):
                elapsed = time.time() - t0
                eta = elapsed / i * (len(jobs) - i)
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(jobs)} done, last job {sec:.0f} s, "
                      f"elapsed {elapsed / 60:.1f} min, remaining ~{eta / 60:.1f} min", flush=True)
    print(f"finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
