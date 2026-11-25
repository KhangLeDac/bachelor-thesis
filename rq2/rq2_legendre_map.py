"""Heat maps of Delta C over the (e_1, e_2) plane for a Legendre grid (job file from rq2_legendre_grid.py).

For every S one figure with two panels, both showing Delta C = mean over devices of C[f] - C[p*_S]
(paired, as in rq2_compare.py) at every grid point, interpolated between the points:

    left   absolute coefficients (a_1, a_2) = a (cos theta, sin theta), a = frac * a_max(theta, sign):
           the real response p = p* + a_1 e_1 + a_2 e_2; the outer points are the boundary of the valid region
    right  normalised disk (frac cos theta, frac sin theta): radius 1 = boundary in every direction

The constant p*_S is the origin (Delta C = 0). Black dots = grid points, star = best point.
Works with partial results (only points with both runs on some device are used).

    uv run python rq2/rq2_legendre_map.py --config rq2/configs/phaseA_legendre_grid.toml

Saves rq2/figures/<job file name>/heatmap_S<S>.png. --shared uses the same colour scale for every S.
"""

import argparse
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import CenteredNorm

from qrc.reset import Reset
from rq2_compare import load, paired

HERE = Path(__file__).resolve().parent


def grid_points(cfg: dict, res: dict, S: float, p0: float) -> dict:
    """Arrays a1, a2 (absolute), x, y (disk), dC, sem, n for every grid point with results, origin included."""
    pts = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)]                            # constant p*: Delta C = 0 by definition
    for shape in cfg["shapes"]:
        if shape["kind"] != "legendre" or len(shape["coefs"]) != 2:
            raise SystemExit(f"shape {shape} is not a Legendre direction in the (e_1, e_2) plane")
        c = np.asarray(shape["coefs"], dtype=float)
        c = c / np.linalg.norm(c)                                        # (cos theta, sin theta)
        for frac in cfg["fracs"]:
            r = paired(res, S, shape["name"], float(frac))
            if r is None:
                continue
            a = Reset.from_fraction("legendre", p0, float(frac), coefs=list(c)).a   # signed std of p_t
            pts.append((a * c[0], a * c[1], frac * c[0], frac * c[1], r["dC"], r["dC_sem"], r["n"]))
    keys = ("a1", "a2", "x", "y", "dC", "sem", "n")
    return {k: np.array(v) for k, v in zip(keys, zip(*pts))}


def panel(ax, x, y, z, norm, cmap, best, xlabel, ylabel, title):
    tri = mtri.Triangulation(x, y)
    im = ax.tripcolor(tri, z, shading="gouraud", cmap=cmap, norm=norm)
    ax.plot(x, y, "k.", ms=2)
    ax.plot(x[best], y[best], "*", color="gold", mec="k", ms=14)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title, aspect="equal")
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="job file written by rq2_legendre_grid.py")
    ap.add_argument("--shared", action="store_true", help="same colour scale for every S")
    ap.add_argument("--cmap", default="RdBu_r", help="diverging colour map, centred at Delta C = 0")
    args = ap.parse_args()
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    res = load(HERE / cfg["out"])
    if not res:
        raise SystemExit(f"no results in {HERE / cfg['out']}")

    data = {}
    for b in cfg["baseline"]:
        S, p0 = float(b["S"]), float(b["p"])
        g = grid_points(cfg, res, S, p0)
        if len(g["dC"]) < 4:
            print(f"S = {S:.0e}: only {len(g['dC']) - 1} grid points with results, skipped")
            continue
        data[S] = (p0, float(b["lam"]), g)
    if not data:
        raise SystemExit("not enough results for any S")

    vmax = max(np.abs(g["dC"]).max() for _, _, g in data.values())
    figdir = HERE / "figures" / args.config.stem
    figdir.mkdir(parents=True, exist_ok=True)
    for S, (p0, lam, g) in data.items():
        norm = CenteredNorm(0, vmax) if args.shared else CenteredNorm(0)
        best = int(np.argmax(g["dC"]))
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
        panel(axes[0], g["a1"], g["a2"], g["dC"], norm, args.cmap, best,
              "a_1  (coefficient of e_1 = sqrt3 u)", "a_2  (coefficient of e_2 = sqrt5 P_2(u))",
              "absolute coefficients")
        im = panel(axes[1], g["x"], g["y"], g["dC"], norm, args.cmap, best,
                   "frac cos(theta)", "frac sin(theta)", "normalised disk (radius 1 = boundary)")
        fig.colorbar(im, ax=axes, label="Delta C = C[f] - C[p*_S]", shrink=0.85)
        n = g["n"][1:]
        fig.suptitle(f"S = {S:.0e}: lam* = {lam:g}, p* = {p0:g};  {len(n)} grid points, devices per point "
                     f"{n.min()}-{n.max()}\nbest (star): Delta C = {g['dC'][best]:+.3f} +- {g['sem'][best]:.3f} "
                     f"at a = ({g['a1'][best]:+.3f}, {g['a2'][best]:+.3f})", fontsize=9)
        path = figdir / f"heatmap_S{S:.0e}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
