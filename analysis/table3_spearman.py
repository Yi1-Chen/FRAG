#!/usr/bin/env python3
"""Table 3: Spearman rho between each attack-free predictor and post-attack Delta-ES.

Each row of the correlation is one (model, method, forget split) checkpoint scored under the
forget+retain relearning attack -- the strongest of the three. Only healthy checkpoints are
included: GA is dropped because its pre-attack utility collapses to zero, so its Delta-ES
measures recovery from a broken model rather than robustness. That leaves 5 methods x 3
splits x 2 models = 30 pooled points. The "w/o FRP" column additionally drops every FRP
checkpoint (n = 24), which rules out the predictor being credited for ranking the method
built on the same principle.

More negative is better: a good predictor gives high scores to checkpoints that give back
little under attack.

Inputs (produced by analysis/compute_predictors.py and scripts/collect_results.py):
    --predictors  model,method,forget_split,L2,FRAG_pct
    --results     model,method,forget_split,stage,ES,dES,U

Usage:
    python analysis/table3_spearman.py \
        --predictors results/predictors_1b.csv results/predictors_3b.csv \
        --results results/table1_tofu_per_split.csv
"""
import argparse
import csv

# The five healthy Table-1 methods. GA is excluded as utility-collapsed (pre-attack U ~ 0),
# so its Delta-ES measures recovery from a broken model rather than robustness. The FRP-RV
# variant is excluded too: it shares FRP's checkpoints-by-construction and would double-count
# the same method.
METHODS = ["GradDiff", "NPO", "RMU", "SP", "FRP"]
ATTACK = "forget_retain"


def spearman(xs, ys):
    """Spearman rho with average ranks for ties."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictors", nargs="+", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--methods", nargs="+", default=METHODS,
                    help="methods to include (default: the five healthy Table-1 methods)")
    ap.add_argument("--attack", default=ATTACK,
                    choices=["retain", "forget", "forget_retain"])
    args = ap.parse_args()

    # FRAG is the reported predictor. cosF and cosR are the two terms of Eq. (5), reported
    # alongside it because Figure 2 turns on how each behaves on its own.
    cols, labels = ["L2", "FRAG_pct"], ["Global L2", "FRAG"]
    pred = {}
    for path in args.predictors:
        reader = csv.DictReader(open(path, encoding="utf-8"))
        for extra, label in (("cosF_pct", "cos(F, dW2)"), ("cosR_pct", "cos(R, dW2)")):
            if reader.fieldnames and extra in reader.fieldnames and extra not in cols:
                cols.append(extra); labels.append(label)
        for r in reader:
            pred[(r["model"], r["method"], r["forget_split"])] = tuple(
                float(r[c]) for c in cols)

    delta = {}
    for r in csv.DictReader(open(args.results, encoding="utf-8")):
        if r["stage"] == args.attack and r["dES"]:
            delta[(r["model"], r["method"], r["forget_split"])] = float(r["dES"])

    wanted = set(args.methods)
    keys = sorted(k for k in pred if k in delta and k[1] in wanted)
    if not keys:
        raise SystemExit("no overlapping (model, method, split) rows between the two files")

    def rho(subset, idx):
        return spearman([pred[k][idx] for k in subset], [delta[k] for k in subset])

    models = sorted({k[0] for k in keys})
    groups = [(m, [k for k in keys if k[0] == m]) for m in models]
    groups.append(("Pooled", keys))

    print(f"Spearman rho vs Delta-ES under the {args.attack} attack "
          f"(healthy checkpoints only; more negative is better)\n")
    header = f"{'':12s}" + "".join(f"{lab:>23s}" for lab in labels)
    sub = f"{'':12s}" + "".join(f"{'all':>11s}{'w/o FRP':>12s}" for _ in labels)
    print(header)
    print(sub)
    for label, subset in groups:
        no_frp = [k for k in subset if not k[1].startswith("FRP")]
        cells = "".join(f"{rho(subset, i):11.2f}{rho(no_frp, i):12.2f}"
                        for i in range(len(cols)))
        print(f"{label:12s}{cells}    (n={len(subset)}, w/o FRP n={len(no_frp)})")

    print("\nCheckpoints included:")
    for k in keys:
        vals = "  ".join(f"{lab}={pred[k][i]:.3f}" for i, lab in enumerate(labels))
        print(f"  {k[0]:14s} {k[1]:16s} {k[2]:9s} {vals}  dES={delta[k]:+.4f}")


if __name__ == "__main__":
    main()
