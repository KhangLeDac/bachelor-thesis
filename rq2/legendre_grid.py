"""Write the job file for a polar grid of Legendre responses in the (e_1, e_2) plane.

Every response is f(u) = p*_S + frac * a_max(theta, sign) * g_theta(u) with the normalised shape

    g_theta(u) = cos(theta) e_1(u) + sin(theta) e_2(u),      e_n = sqrt(2n+1) P_n,

i.e. kind "legendre" with coefs = [cos(theta), sin(theta)]. theta in [0, 180) and frac in [-1, 1] cover the
whole valid region once: a negative frac is the direction theta + 180 deg, frac = +-1 is the boundary
(p touches 0 or 1), frac = 0 the constant p*_S. theta = 0 is the linear, theta = 90 the quadratic shape.

    uv run python rq2/legendre_grid.py                       # defaults below
    uv run python rq2/legendre_grid.py --step 30 --S 1e5     # coarser test grid
    uv run python rq2/scan.py --config rq2/configs/phaseA_legendre_grid.toml --dry-run

Prints the number of jobs and a runtime estimate (seconds per job measured on phaseA_linquad with 18 workers).
"""

import argparse
from pathlib import Path

import numpy as np

from qrc.reset import Reset

HERE = Path(__file__).resolve().parent

# RQ1 optimum per S: argmax of the mean total IPC over 10 devices (rq1/results/map), as in phaseA.toml
BASELINES = {1e4: (1.0, 0.80), 1e5: (0.8, 0.70), 1e6: (0.4, 0.25)}


def fmt_S(S: float) -> str:
    return "inf" if np.isinf(S) else f"{S:.0e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=float, default=15.0, help="angle step in degrees (theta = 0, step, ... < 180)")
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
                    help="positive fracs; each is used with both signs")
    ap.add_argument("--S", type=float, nargs="+", default=[1e4, 1e5, 1e6],
                    help=f"shot numbers (must be in {[fmt_S(s) for s in BASELINES]})")
    ap.add_argument("--seeds", type=int, default=10, help="number of devices (seeds 0 .. n-1)")
    ap.add_argument("--M", type=int, default=50000)
    ap.add_argument("--T-wash", type=int, default=1000)
    ap.add_argument("--delays", type=int, nargs="+", default=[20, 10, 7], help="degrees 1..3, as in RQ1")
    ap.add_argument("--name", default="phaseA_legendre_grid", help="job file rq2/configs/<name>.toml, out results/<name>")
    ap.add_argument("--sec-per-job", type=float, default=98.0, help="for the runtime estimate")
    ap.add_argument("--workers", type=int, default=18, help="for the runtime estimate")
    args = ap.parse_args()

    unknown = [s for s in args.S if s not in BASELINES]
    if unknown:
        raise SystemExit(f"no RQ1 baseline for S = {unknown}")
    if args.step <= 0 or 180 % args.step:
        raise SystemExit("--step must divide 180")

    thetas = np.arange(0, 180, args.step)
    fracs = sorted({-f for f in args.fracs} | set(args.fracs))
    if any(abs(f) > 1 or f == 0 for f in fracs):
        raise SystemExit("fracs must be in (0, 1]")

    # check every response once (fails here, not after hours in the scan)
    shapes = []
    for th in thetas:
        t = np.deg2rad(th)
        coefs = [float(np.round(np.cos(t), 12)) + 0.0, float(np.round(np.sin(t), 12)) + 0.0]   # + 0.0: no -0.0
        for S in args.S:
            for fr in fracs:
                Reset.from_fraction("legendre", BASELINES[S][1], fr, coefs=coefs)
        shapes.append((th, coefs))

    lines = [
        "# RQ2 phase A: polar grid of Legendre responses in the (e_1, e_2) plane around the RQ1 optimum.",
        f"# Written by rq2/legendre_grid.py (theta step {args.step:g} deg). f(u) = p* + frac a_max(theta) g_theta(u),",
        "# g_theta = cos(theta) e_1 + sin(theta) e_2; theta in [0, 180) with frac in [-1, 1] covers the whole valid region",
        "# (frac < 0 = direction theta + 180 deg). theta = 0: linear, theta = 90: quadratic.",
        f"seeds  = {list(range(args.seeds))}",
        f"M      = {args.M}",
        f"T_wash = {args.T_wash}",
        f"delays = {args.delays}",
        f'out    = "results/{args.name}"',
        f"fracs  = [{', '.join(f'{f:g}' for f in fracs)}]   # frac = 0 (constant p*) always runs",
        "",
        "# RQ1 optimum per S (rq1/results/map)",
    ]
    for S in args.S:
        lam, p = BASELINES[S]
        lines += ["[[baseline]]", f"S = {fmt_S(S)}", f"lam = {lam}", f"p = {p}", ""]
    for th, (c1, c2) in shapes:
        lines += ["[[shapes]]", 'kind = "legendre"', f'name = "t{th:03.0f}"', f"coefs = [{c1!r}, {c2!r}]   # theta = {th:g} deg", ""]

    path = HERE / "configs" / f"{args.name}.toml"
    path.write_text("\n".join(lines))

    n_resp = len(thetas) * len(fracs) * len(args.S) * args.seeds
    n_const = len(args.S) * args.seeds
    total = n_resp + n_const
    hours = total * args.sec_per_job / args.workers / 3600
    print(f"wrote {path}")
    print(f"{len(thetas)} directions x {len(fracs)} fracs x {len(args.S)} S x {args.seeds} devices = {n_resp} "
          f"+ {n_const} constant = {total} jobs, ~{hours:.1f} h with {args.workers} workers")


if __name__ == "__main__":
    main()
