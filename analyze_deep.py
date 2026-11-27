"""
analyze_deep.py -- extra analyses (printed to the log; also CSV):
  1. per-class Macro-F1 by (model, train_cond, test_cond), mean over seeds
  2. fertility (tokens/word) ratio cyrl/latn vs D_script (per model x train_cond), Spearman rho (n is tiny: descriptive only)
  3. error breakdown: share of items that flip correct->wrong / wrong->correct between latn and cyrl test (per model x train_cond)
usage: python analyze_deep.py results/encoder results/tables
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

IN = sys.argv[1] if len(sys.argv) > 1 else "results/encoder"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/tables"
os.makedirs(OUT, exist_ok=True)

seen = {}
for p in glob.glob(os.path.join(IN, "*", "runs.jsonl")):
    for line in open(p, encoding="utf-8"):
        r = json.loads(line); seen[(r["model"], r["train_cond"], r["seed"])] = r
runs = list(seen.values())
if not runs:
    sys.exit("no runs")

pc, flip, dd = [], [], []
for r in runs:
    names = r["label_names"]; k = len(names)
    for tc, t in r["tests"].items():
        f = f1_score(t["labels"], t["preds"], average=None, labels=list(range(k)), zero_division=0)
        for i, n in enumerate(names):
            pc.append({"model": r["model"], "train": r["train_cond"], "test": tc, "seed": r["seed"], "class": n, "f1": f[i]})
    a = np.array(r["tests"]["latn"]["preds"]) == np.array(r["tests"]["latn"]["labels"])
    b = np.array(r["tests"]["cyrl"]["preds"]) == np.array(r["tests"]["cyrl"]["labels"])
    flip.append({"model": r["model"], "train": r["train_cond"], "seed": r["seed"],
                 "both_correct": (a & b).mean(), "lost_in_cyrl": (a & ~b).mean(),
                 "gained_in_cyrl": (~a & b).mean(), "both_wrong": (~a & ~b).mean(),
                 "d_script": r["tests"]["cyrl"]["macro_f1"] - r["tests"]["latn"]["macro_f1"]})

pc = pd.DataFrame(pc)
t1 = pc.groupby(["model", "train", "test", "class"]).f1.mean().unstack("test").round(3)
t1["cyrl-latn"] = (t1["cyrl"] - t1["latn"]).round(3)
t1.reset_index().to_csv(f"{OUT}/deep_per_class.csv", index=False)
fl = pd.DataFrame(flip)
t3 = fl.groupby(["model", "train"]).mean(numeric_only=True).drop(columns="seed").round(3)
t3.reset_index().to_csv(f"{OUT}/deep_error_flips.csv", index=False)

print("\n== per-class F1 (mean over seeds)"); print(t1.to_string())
print("\n== item-level flips latn->cyrl test (mean over seeds)"); print(t3.to_string())

diag = os.path.join(OUT, "diagnostics_encoders.csv")
if os.path.exists(diag):
    d = pd.read_csv(diag)
    w = d.pivot(index="model", columns="cond", values="tokens_per_word")
    if {"latn", "cyrl"} <= set(w.columns):
        w["fertility_ratio_cyrl/latn"] = w["cyrl"] / w["latn"]
        m = fl.groupby(["model", "train"]).d_script.mean().reset_index().merge(
            w[["fertility_ratio_cyrl/latn"]].reset_index(), on="model")
        m.round(4).to_csv(f"{OUT}/deep_fertility_vs_delta.csv", index=False)
        print("\n== fertility ratio vs D_script"); print(m.round(4).to_string(index=False))
        for tr, g in m.groupby("train"):
            if len(g) >= 3:
                rho, p = stats.spearmanr(g["fertility_ratio_cyrl/latn"], g["d_script"])
                print(f"train={tr}: Spearman rho={rho:.2f} p={p:.2f} (n={len(g)} models; descriptive only)")
