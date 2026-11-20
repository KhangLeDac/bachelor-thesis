"""Compare the IPC of the optimal constant reset p*_S with input-dependent resets p_t = f(u_t).

Reads the result files of rq2_scan.py for a job file (also partial results) and, for every S and every
shape in the results, compares C[f] with the constant baseline C[p*_S] (frac = 0). The difference is
paired: both runs use the same device and the same input sequence, so

    Delta C = mean over devices of (C[f] - C[p*_S]),    error = std / sqrt(n_devices).

Only devices for which the constant run and the run of f both exist are used.

    uv run python rq2/rq2_compare.py --config rq2/configs/linear.toml

Saves rq2/figures/<job file name>/compare.png and prints a table:
    left    IPC vs S: constant p*_S and the best frac of every shape (error bars: sem over devices)
    middle  Delta C vs frac, one line per S (and shape)
    right   degree split (1, 2, 3, ...) of the constant and of the best f per S
Note: the "best frac" is picked on the same data it is reported on, which favours it slightly; the
Delta C vs frac curve shows whether the gain is systematic or a single lucky point.
"""

import argparse
import tomllib
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def load(out: Path) -> dict:
    """{(S, name, frac): {seed: (total, degree array)}} from all result files in ``out``."""
    res = defaultdict(dict)
    for f in sorted(out.glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        d = np.load(f)
        res[(float(d["S"]), str(d["name"]), float(d["frac"]))][int(d["seed"])] = (float(d["total"]), d["degree"])
    return res


def paired(res: dict, S: float, name: str, frac: float):
    """Mean and sem over devices of C[f], C[p*] and Delta C, using devices with both runs."""
    base, run = res.get((S, "constant", 0.0), {}), res.get((S, name, frac), {})
    seeds = sorted(set(base) & set(run))
    if not seeds:
        return None
    c_f = np.array([run[s][0] for s in seeds])
    c_0 = np.array([base[s][0] for s in seeds])
    sem = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return dict(n=len(seeds), C=c_f.mean(), C_sem=sem(c_f), C0=c_0.mean(), C0_sem=sem(c_0),
                dC=(c_f - c_0).mean(), dC_sem=sem(c_f - c_0),
                deg=np.mean([run[s][1] for s in seeds], axis=0), deg0=np.mean([base[s][1] for s in seeds], axis=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="the job file used for rq2_scan.py")
    args = ap.parse_args()
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    res = load(HERE / cfg["out"])
    if not res:
        raise SystemExit(f"no results in {HERE / cfg['out']}")

    shots = sorted({k[0] for k in res})
    names = sorted({k[1] for k in res} - {"constant"})
    fracs = sorted({k[2] for k in res if k[1] != "constant"})

    # ---- table and best frac per (S, shape)
    best = {}
    print(f"{'S':>8} {'shape':>10} {'frac':>6} {'n':>3} {'C[f]':>7} {'C[p*]':>7} {'Delta C':>8} {'sem':>6}")
    for S in shots:
        for name in names:
            rows = {fr: paired(res, S, name, fr) for fr in fracs}
            rows = {fr: r for fr, r in rows.items() if r is not None}
            if not rows:
                continue
            fr = max(rows, key=lambda k: rows[k]["dC"])
            best[S, name] = (fr, rows)
            for k, r in rows.items():
                mark = " <- best" if k == fr else ""
                print(f"{S:>8.0e} {name:>10} {k:>+6.2f} {r['n']:>3} {r['C']:7.3f} {r['C0']:7.3f} "
                      f"{r['dC']:>+8.3f} {r['dC_sem']:6.3f}{mark}")
        print()

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    const = [(S, np.array([c for c, _ in res[S, "constant", 0.0].values()])) for S in shots
             if res.get((S, "constant", 0.0))]
    ax.errorbar([S for S, _ in const], [c.mean() for _, c in const],
                yerr=[c.std(ddof=1) / np.sqrt(len(c)) if len(c) > 1 else 0.0 for _, c in const],
                fmt="o-", capsize=3, color="k", label="constant p*_S (all devices)")
    for name in names:
        pts = [(S, best[S, name][1][best[S, name][0]]) for S in shots if (S, name) in best]
        ax.errorbar([S for S, _ in pts], [r["C"] for _, r in pts], yerr=[r["C_sem"] for _, r in pts],
                    fmt="s-", capsize=3, label=f"{name}, best frac")
    ax.set(xscale="log", xlabel="shots per step S", ylabel="total IPC (degrees 1-3)", title="IPC vs S")
    ax.legend(fontsize=8)

    ax = axes[1]
    for name in names:
        for S in shots:
            if (S, name) not in best:
                continue
            pts = {0.0: (0.0, 0.0)} | {k: (r["dC"], r["dC_sem"]) for k, r in best[S, name][1].items()}
            fr = sorted(pts)                                   # frac = 0 is the baseline itself: Delta C = 0
            ax.errorbar(fr, [pts[k][0] for k in fr], yerr=[pts[k][1] for k in fr], fmt="o-", capsize=2, ms=3,
                        label=f"S = {S:.0e}" + (f" ({name})" if len(names) > 1 else ""))
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(xlabel="frac = a / a_max", ylabel="Delta C = C[f] - C[p*_S]", title="gain vs response strength")
    ax.legend(fontsize=7)

    ax = axes[2]
    D = len(cfg["delays"])
    groups = [("const", None)] + [(n, n) for n in names]
    width = 0.8 / len(groups)
    x = np.arange(len(shots))
    for g, (label, name) in enumerate(groups):
        bottom = np.zeros(len(shots))
        for d in range(D):
            vals = []
            for S in shots:
                key = (S, name if name else names[0])
                if key not in best:
                    vals.append(0.0)
                    continue
                r = best[key][1][best[key][0]]
                vals.append((r["deg0"] if name is None else r["deg"])[d])
            ax.bar(x + (g - (len(groups) - 1) / 2) * width, vals, width, bottom=bottom, color=f"C{d}",
                   edgecolor="k", lw=0.5, hatch=None if name is None else "//",
                   label=f"degree {d + 1}" if g == 0 else None)
            bottom += vals
    ax.set_xticks(x, [f"{S:.0e}" for S in shots])
    ax.set(xlabel="shots per step S", ylabel="capacity",
           title="degree split: constant (plain) vs best f (hatched)")
    ax.legend(fontsize=8)

    fig.suptitle(f"constant p*_S vs input-dependent reset, {cfg['out']}")
    fig.tight_layout()
    figdir = HERE / "figures" / args.config.stem
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "compare.png", dpi=130)
    print(f"saved {figdir / 'compare.png'}")


if __name__ == "__main__":
    main()
