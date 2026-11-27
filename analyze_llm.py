"""
analyze_llm.py -- accuracy per condition, Delta, paired bootstrap CI and exact McNemar test
(items are paired: same 900 questions in every script condition).

python analyze_llm.py results/llm results/tables
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest

IN = sys.argv[1] if len(sys.argv) > 1 else "results/llm"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/tables"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj


rows = []
for path in sorted(glob.glob(os.path.join(IN, "*.jsonl"))):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    df = pd.DataFrame(recs)
    model, shots = df["model"].iloc[0], int(df["shots"].iloc[0])
    corr = {c: (g.sort_values("i")["pred"].values == g.sort_values("i")["gold"].values).astype(int)
            for c, g in df.groupby("cond")}
    for a, b, name in [("cyrl", "latn", "D_script"), ("latn_norm", "latn", "D_typo"),
                       ("cyrl", "latn_norm", "D_pure")]:
        if a not in corr or b not in corr:
            continue
        x, y = corr[a], corr[b]
        d = x.mean() - y.mean()
        idx = rng.integers(0, len(x), (10000, len(x)))
        boots = x[idx].mean(1) - y[idx].mean(1)
        n01 = int(((x == 1) & (y == 0)).sum()); n10 = int(((x == 0) & (y == 1)).sum())
        p = binomtest(n01, n01 + n10, 0.5).pvalue if (n01 + n10) > 0 else 1.0
        tr = df.groupby("cond")["truncated"].sum().to_dict() if "truncated" in df else {}
        rows.append({"model": model, "shots": shots, "delta": name, "acc_a": x.mean(), "acc_b": y.mean(),
                     "truncated_a": tr.get(a, -1), "truncated_b": tr.get(b, -1),
                     "diff": d, "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
                     "mcnemar_p": p, "only_a_correct": n01, "only_b_correct": n10, "n": len(x)})
res = pd.DataFrame(rows)
if len(res):
    res["p_holm"] = np.nan
    for dname, g in res.groupby("delta"):
        res.loc[g.index, "p_holm"] = holm(g["mcnemar_p"].values)
res.to_csv(os.path.join(OUT, "llm_belebele_deltas.csv"), index=False)
pd.set_option("display.width", 220)
print(res.round(4).to_string(index=False))
