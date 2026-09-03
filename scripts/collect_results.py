#!/usr/bin/env python3
"""Collect Table 1 from the evaluation JSONs that the pipelines have written.

Walks the OpenUnlearning eval directory for the pre-attack and post-attack runs of every
(model, method, forget split), and writes two CSVs:

    table1_tofu_avg.csv        Table 1 as printed: averaged over forget01/05/10
    table1_tofu_per_split.csv  the same runs kept per split (Appendix C.2), which is
                               what analysis/table3_spearman.py correlates against

Delta-ES is post-attack ES minus the pre-attack ES of the same checkpoint.

Environment:
    OPEN_UNLEARNING  checkout holding saves/   (default ~/open-unlearning)

Usage:
    python scripts/collect_results.py --out results/
"""
import argparse
import collections
import json
import os

OU = os.environ.get("OPEN_UNLEARNING", os.path.expanduser("~/open-unlearning"))

STAGES = {"pure": "pure",
          "postrelearnretain1ep": "retain",
          "postrelearnforget1ep": "forget",
          "postrelearnforget_retain1ep": "forget_retain"}
SPLITS = ["forget01", "forget05", "forget10"]
SUFFIX = {"forget01": "_forget01", "forget05": "_forget05", "forget10": ""}
ATTACKS = ("retain", "forget", "forget_retain")
# (display name, checkpoint tag). FRP is stored under its original "RC" (rank-combo) tag.
METHODS = [("GA", "GradAscent"), ("GradDiff", "GradDiff"), ("NPO", "NPO"),
           ("RMU", "RMU"), ("SP", "SP"), ("FRP", "RC")]
# retain-veto fraction reported in Table 1 for each scale
RV_P = {"1B": "p015", "3B": "p005"}


def read(d):
    for name in ("TOFU_SUMMARY.json", "TOFU_EVAL.json"):
        path = os.path.join(d, name)
        if not os.path.exists(path):
            continue
        try:
            j = json.load(open(path, encoding="utf-8"))
        except (ValueError, OSError):
            return None
        es, u = j.get("extraction_strength"), j.get("model_utility")
        if es is not None:
            return float(es), (float(u) if u is not None else float("nan"))
    return None


def collect(canonical_dir, eval_dir, scales):
    data = collections.defaultdict(dict)
    order = []
    for scale in scales:
        for disp, tag in METHODS:
            order.append((scale, disp))
            for split in SPLITS:
                for stage_dir, stage in STAGES.items():
                    got = read(f"{canonical_dir}/canonical_{scale}_{tag}{SUFFIX[split]}_{stage_dir}")
                    if got:
                        data[(scale, disp, split)][stage] = got
        rv = RV_P.get(scale)
        if not rv:
            continue
        disp = f"FRP-RV(p={int(rv[1:]) / 100:.2f})"
        order.append((scale, disp))
        for split in SPLITS:
            for stage_dir, stage in STAGES.items():
                for prefix in ("frp", "frprv"):
                    got = read(f"{eval_dir}/{prefix}_{scale}_{split}_{rv}_{stage_dir}")
                    if got:
                        data[(scale, disp, split)][stage] = got
                        break
    return data, order


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results", help="directory for the two CSVs")
    ap.add_argument("--scales", nargs="+", default=["1B", "3B"])
    ap.add_argument("--canonical-dir",
                    default=f"{OU}/saves/table1_canonical/saves/eval",
                    help="eval dir for the baseline and FRP runs")
    ap.add_argument("--eval-dir", default=f"{OU}/saves/eval",
                    help="eval dir for the FRP-RV runs")
    args = ap.parse_args()

    data, order = collect(args.canonical_dir, args.eval_dir, args.scales)
    os.makedirs(args.out, exist_ok=True)
    avg_path = os.path.join(args.out, "table1_tofu_avg.csv")
    per_path = os.path.join(args.out, "table1_tofu_per_split.csv")
    n_avg = 0

    with open(avg_path, "w", encoding="utf-8") as f:
        f.write("model,method,ES,U,"
                + ",".join(f"{a}_ES,{a}_dES,{a}_U" for a in ATTACKS)
                + ",avg_ES,avg_dES,avg_U\n")
        for scale, disp in order:
            cells = {}
            for stage in ("pure",) + ATTACKS:
                vals = [data[(scale, disp, s)].get(stage) for s in SPLITS]
                if any(v is None for v in vals):
                    cells = None
                    break
                cells[stage] = (sum(v[0] for v in vals) / 3, sum(v[1] for v in vals) / 3)
            if cells is None:
                print(f"skip {scale}/{disp}: not all three splits are evaluated")
                continue
            pre = cells["pure"][0]
            row = [f"{pre:.3f}", f"{cells['pure'][1]:.3f}"]
            for a in ATTACKS:
                row += [f"{cells[a][0]:.3f}", f"{cells[a][0] - pre:.3f}", f"{cells[a][1]:.3f}"]
            row += [f"{sum(cells[a][0] for a in ATTACKS) / 3:.3f}",
                    f"{sum(cells[a][0] - pre for a in ATTACKS) / 3:.3f}",
                    f"{sum(cells[a][1] for a in ATTACKS) / 3:.3f}"]
            f.write(f"LLaMA-3.2-{scale},{disp}," + ",".join(row) + "\n")
            n_avg += 1

    n_per = 0
    with open(per_path, "w", encoding="utf-8") as f:
        f.write("model,method,forget_split,stage,ES,dES,U\n")
        for scale, disp in order:
            for split in SPLITS:
                stages = data[(scale, disp, split)]
                if "pure" not in stages:
                    continue
                pre, pre_u = stages["pure"]
                f.write(f"LLaMA-3.2-{scale},{disp},{split},pure,{pre:.4f},,{pre_u:.4f}\n")
                n_per += 1
                for a in ATTACKS:
                    if a in stages:
                        es, u = stages[a]
                        f.write(f"LLaMA-3.2-{scale},{disp},{split},{a},"
                                f"{es:.4f},{es - pre:+.4f},{u:.4f}\n")
                        n_per += 1

    print(f"wrote {avg_path} ({n_avg} rows)")
    print(f"wrote {per_path} ({n_per} rows)")


if __name__ == "__main__":
    main()
