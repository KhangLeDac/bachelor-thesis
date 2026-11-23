"""Plot all response shapes of qrc/reset.py.

Every response function is f(u) = p0 + a g(u) with <g> = 0 and <g^2> = 1 for u ~ U(-1, 1) (see qrc/reset.py).
This script saves to rq2/figures/shapes/:

    all_shapes.png        normalised shapes g(u) of all kinds in one plot
    f_<kind>.png          one file per kind: f(u) = p0 + a g(u) for several frac = a / a_max (both signs),
                          with the allowed range [0, 1] and the mean p0

    uv run python rq2/plot_shapes.py
    uv run python rq2/plot_shapes.py --p0 0.8 --k 10 --u0 0.5 --coefs 0 1 0.5

Shapes with parameters: sigmoid (k, u0) and legendre (coefs c_1, c_2, ...); the defaults below are examples.
"""

import argparse
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p0", type=float, default=0.5, help="mean reset strength for the f(u) plots")
    ap.add_argument("--k", type=float, default=10.0, help="sigmoid: steepness")
    ap.add_argument("--u0", type=float, default=0.5, help="sigmoid: threshold in |u|")
    ap.add_argument("--coefs", type=float, nargs="+", default=[0.0, 1.0, 0.5],
                    help="legendre: coefficients c_1, c_2, ... of sqrt(2n+1) P_n(u)")
    args = ap.parse_args()

    FIGDIR.mkdir(parents=True, exist_ok=True)
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
