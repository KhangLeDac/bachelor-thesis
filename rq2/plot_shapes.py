"""Plot all response shapes of qrc/reset.py.

Every response function is f(u) = p0 + a g(u) with <g> = 0 and <g^2> = 1 for u ~ U(-1, 1) (see qrc/reset.py).
This script saves to rq2/figures/shapes/:

    all_shapes.png        normalised shapes g(u) of all kinds in one plot
    f_<kind>.png          one file per kind: f(u) = p0 + a g(u) for several frac = a / a_max (both signs),
                          with the allowed range [0, 1] and the mean p0
    best_<grid>.png       with --best CONFIG (a Legendre grid job file): for every S the best grid point
                          b = (b_1, b_2) (largest paired Delta C, as in rq2/legendre_map.py); left the direction
                          g(u) with c = b / |b|, right the response f(u) = p*_S + a g(u), a = |b| = std(p_t)
    best_<grid>_mirror.png  the same for the mirror point (-b_1, b_2), i.e. f(-u) (same |b|, also a grid point)
    <shape>_frac<frac>_<job>.png  with --fixed CONFIG SHAPE FRAC: the same plot for one shape of a job file at a
                          fixed frac, one curve per S, with its paired Delta C

    uv run python rq2/plot_shapes.py
    uv run python rq2/plot_shapes.py --p0 0.8 --k 10 --u0 0.5 --coefs 0 1 0.5
    uv run python rq2/plot_shapes.py --best rq2/configs/phaseA_legendre_grid.toml
    uv run python rq2/plot_shapes.py --fixed rq2/configs/phaseA_linquad.toml linear 1

Shapes with parameters: sigmoid (k, u0) and legendre (coefs = direction c_1, c_2, ...); the defaults below are examples.
"""

import argparse
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qrc.reset import SHAPES, Reset

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures" / "shapes"
FRACS = [-1.0, -0.5, 0.5, 1.0]


def shape_params(kind: str, args) -> dict:
    """Parameters of the shape (empty for kinds without parameters)."""
    if kind == "sigmoid":
        return {"k": args.k, "u0": args.u0}
    if kind == "legendre":
        return {"coefs": args.coefs}
    return {}


def label(kind: str, params: dict) -> str:
    return kind + (" (" + ", ".join(f"{k}={v}" for k, v in params.items()) + ")" if params else "")


def _point(S: float, p0: float, f: Reset, dC: float, sem: float) -> dict:
    """One curve of plot_points: S, p*_S, the response f and its paired Delta C."""
    if f.kind == "legendre":
        c = np.asarray(f.shape["coefs"], dtype=float)
        c = c / np.linalg.norm(c)
        theta = np.degrees(np.arctan2(c[1], c[0])) if len(c) == 2 else np.nan
        shape = f"c = ({', '.join(f'{x:+.2f}' for x in c)})" + (f", theta = {theta:+.0f} deg" if len(c) == 2 else "")
        strength = "b = (" + ", ".join(f"{x:+.3f}" for x in f.a * c) + ")"
    else:
        shape, strength = f.kind, f"a = {f.a:+.3f}"
    return {"S": S, "p0": p0, "f": f, "dC": dC, "sem": sem, "shape": shape, "strength": strength}


def best_points(cfg: dict, res: dict) -> tuple[list[dict], list[dict]]:
    """For every S: the best grid point b = (b_1, b_2) and its mirror (-b_1, b_2) (= f(-u), same |b|)."""
    from legendre_map import grid_points          # rq2/ is on sys.path when this script is run

    best, mirror = [], []
    for b in cfg["baseline"]:
        S, p0 = float(b["S"]), float(b["p"])
        g = grid_points(cfg, res, S, p0)
        i = int(np.argmax(g["dC"]))
        bvec = np.array([g["b1"][i], g["b2"][i]])
        a = float(np.linalg.norm(bvec))
        j = int(np.argmin(np.hypot(g["b1"] + bvec[0], g["b2"] - bvec[1])))
        if np.hypot(g["b1"][j] + bvec[0], g["b2"][j] - bvec[1]) > 1e-6 * a:
            raise SystemExit(f"S = {S:.0e}: mirror point (-b_1, b_2) is not on the grid")
        c = bvec / a                                                   # unit direction (c_1, c_2)
        best.append(_point(S, p0, Reset("legendre", p0, a, coefs=list(c)), g["dC"][i], g["sem"][i]))
        mirror.append(_point(S, p0, Reset("legendre", p0, a, coefs=[-c[0], c[1]]), g["dC"][j], g["sem"][j]))
    return best, mirror


def plot_points(points: list[dict], title: str, path: Path) -> None:
    """Left: normalised shape g(u); right: response f(u) = p*_S + a g(u), one curve per S."""
    u = np.linspace(-1, 1, 1001)
    fig, (ax_g, ax_f) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    ax_f.axhspan(0, 1, color="0.93", zorder=0)
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.85, len(points)))
    for pt, col in zip(points, colors):
        f = pt["f"]
        ax_g.plot(u, f.g(u), color=col, label=f"S = {pt['S']:.0e}: {pt['shape']}")
        ax_f.plot(u, f(u), color=col,
                  label=f"S = {pt['S']:.0e}: p* = {pt['p0']:g}, {pt['strength']}, "
                        f"Delta C = {pt['dC']:+.2f} +- {pt['sem']:.2f}")
        ax_f.axhline(pt["p0"], color=col, lw=0.8, ls="--")
    ax_g.axhline(0, color="gray", lw=0.8)
    ax_g.set(xlabel="input u", ylabel="g(u)", title=f"shape g(u), {title}")
    ax_f.set(xlabel="input u", ylabel="reset strength p = f(u)", ylim=(-0.05, 1.05),
             title="response f(u) = p*_S + a g(u)  (dashed: p*_S)")
    ax_g.legend(fontsize=7)
    ax_f.legend(fontsize=7, loc="best")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved {path}")


def _load(config: Path) -> tuple[dict, dict]:
    from compare import load                      # rq2/ is on sys.path when this script is run

    with open(config, "rb") as fh:
        cfg = tomllib.load(fh)
    res = load(HERE / cfg["out"])
    if not res:
        raise SystemExit(f"no results in {HERE / cfg['out']}")
    return cfg, res


def plot_best(config: Path) -> None:
    """For every S of a Legendre grid job file: best grid point and its mirror, in two figures."""
    cfg, res = _load(config)
    best, mirror = best_points(cfg, res)
    plot_points(best, "best grid point", FIGDIR / f"best_{config.stem}.png")
    plot_points(mirror, "mirror (-c_1, c_2) of the best point", FIGDIR / f"best_{config.stem}_mirror.png")


def plot_fixed(config: Path, name: str, frac: float) -> None:
    """For every S of a job file: the shape ``name`` at a fixed frac, with its paired Delta C."""
    from compare import paired

    cfg, res = _load(config)
    shapes = {sh.get("name", sh["kind"]): sh for sh in cfg["shapes"]}
    if name not in shapes:
        raise SystemExit(f"shape {name!r} not in {config}; known: {sorted(shapes)}")
    sh = shapes[name]
    params = {k: v for k, v in sh.items() if k not in ("kind", "name")}
    points = []
    for b in cfg["baseline"]:
        S, p0 = float(b["S"]), float(b["p"])
        r = paired(res, S, name, frac)
        if r is None:
            print(f"S = {S:.0e}: no paired results for {name} at frac = {frac:+g}, skipped")
            continue
        f = Reset.from_fraction(sh["kind"], p0, frac, **params)
        points.append(_point(S, p0, f, r["dC"], r["dC_sem"]))
    if not points:
        raise SystemExit("no results to plot")
    plot_points(points, f"{name} at frac = {frac:+g}", FIGDIR / f"{name}_frac{frac:+g}_{config.stem}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p0", type=float, default=0.5, help="mean reset strength for the f(u) plots")
    ap.add_argument("--k", type=float, default=10.0, help="sigmoid: steepness")
    ap.add_argument("--u0", type=float, default=0.5, help="sigmoid: threshold in |u|")
    ap.add_argument("--coefs", type=float, nargs="+", default=[0.0, 1.0, 0.5],
                    help="legendre: direction c_1, c_2, ... in the basis e_n = sqrt(2n+1) P_n(u) (any scale)")
    ap.add_argument("--best", type=Path, default=None,
                    help="Legendre grid job file: also plot the best grid point of every S")
    ap.add_argument("--fixed", nargs=3, metavar=("CONFIG", "SHAPE", "FRAC"), default=None,
                    help="job file, shape name and frac: plot that shape for every S with its Delta C")
    args = ap.parse_args()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    if args.best is not None:
        plot_best(args.best)
    if args.fixed is not None:
        plot_fixed(Path(args.fixed[0]), args.fixed[1], float(args.fixed[2]))
    u = np.linspace(-1, 1, 1001)
    kinds = [k for k in SHAPES if k != "constant"]

    # 1. all normalised shapes g(u)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for kind in kinds:
        params = shape_params(kind, args)
        ax.plot(u, Reset(kind, args.p0, **params).g(u), label=label(kind, params))
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(xlabel="input u", ylabel="g(u)", title="normalised response shapes: <g> = 0, <g^2> = 1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "all_shapes.png", dpi=130)
    plt.close(fig)
    print(f"saved {FIGDIR / 'all_shapes.png'}")

    # 2. one file per kind: f(u) for several frac
    for kind in kinds:
        params = shape_params(kind, args)
        fig, ax = plt.subplots(figsize=(6, 4.3))
        ax.axhspan(0, 1, color="0.93", zorder=0, label="allowed range [0, 1]")
        ax.axhline(args.p0, color="k", lw=1, ls="--", label=f"constant p0 = {args.p0:g} (frac = 0)")
        cmap = plt.get_cmap("coolwarm")
        for frac in FRACS:
            f = Reset.from_fraction(kind, args.p0, frac, **params)
            ax.plot(u, f(u), color=cmap((frac + 1) / 2), label=f"frac = {frac:+g}  (a = {f.a:+.3f})")
        ax.set(xlabel="input u", ylabel="reset strength p = f(u)", ylim=(-0.05, 1.05),
               title=f"f(u) = p0 + a g(u): {label(kind, params)}")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(FIGDIR / f"f_{kind}.png", dpi=130)
        plt.close(fig)
        print(f"saved {FIGDIR / f'f_{kind}.png'}")


if __name__ == "__main__":
    main()
