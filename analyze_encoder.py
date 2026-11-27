"""
analyze_encoder.py -- aggregate encoder runs into paper tables.

Reads results/encoder/*/runs.jsonl and writes results/tables/*.csv

Per model x train_cond x test_cond:  mean +- std Accuracy and Macro-F1 over seeds
Delta (script effect) per model/train_cond:
    D_script  = F1(test=cyrl)      - F1(test=latn)        # total effect
    D_typo    = F1(test=latn_norm) - F1(test=latn)        # typographic control
    D_pure    = F1(test=cyrl)      - F1(test=latn_norm)   # script effect net of apostrophe normalisation
Statistics (reported with their limitations):
  * paired t-test over seeds (n = number of seeds; low power)
  * item-level paired bootstrap CI of Delta (10,000 resamples of the 204 test items,
    applied to each seed and averaged)
  * Holm correction across models within each (train_cond, delta)
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

IN = sys.argv[1] if len(sys.argv) > 1 else "results/encoder"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/tables"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)


def load():
    """Load all runs; if a (model, train_cond, seed) appears twice keep the LAST one."""
    seen = {}
    for p in glob.glob(os.path.join(IN, "*", "runs.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            seen[(r["model"], r["train_cond"], r["seed"])] = r
    return list(seen.values())


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def fast_macro_f1(y, p, k):
    """Macro-F1 over k classes from integer arrays (sklearn convention: classes absent from
    both y and p contribute 0 only if present in the label set; here all k classes are used)."""
    tp = np.bincount(y[p == y], minlength=k).astype(float)
    pred = np.bincount(p, minlength=k).astype(float)
    true = np.bincount(y, minlength=k).astype(float)
    denom = pred + true
    # a class absent from BOTH y and p in a bootstrap resample is skipped, not counted as 0
    present = denom > 0
    f1 = 2 * tp[present] / denom[present]
    return f1.mean()


def boot_delta(preds_a, preds_b, labels, n_boot=10000):
    """Paired bootstrap over items: mean of Macro-F1(a) - Macro-F1(b) across seeds."""
    n = len(labels)
    labels = np.asarray(labels)
    k = int(labels.max()) + 1
    A = [np.asarray(p) for p in preds_a]
    B = [np.asarray(p) for p in preds_b]
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        y = labels[idx]
        deltas[i] = np.mean([fast_macro_f1(y, a[idx], k) - fast_macro_f1(y, b[idx], k)
                             for a, b in zip(A, B)])
    return np.percentile(deltas, [2.5, 97.5])


def main():
    recs = load()
    if not recs:
        sys.exit("no runs found")
    rows = []
    for r in recs:
        for tc, m in r["tests"].items():
            rows.append({"model": r["model"], "train": r["train_cond"], "test": tc,
                         "seed": r["seed"], "acc": m["acc"], "macro_f1": m["macro_f1"],
                         "never_pred": m["n_classes_never_predicted"],
                         "n_truncated": m.get("n_truncated", -1),
                         "val_f1": r.get("val_macro_f1", np.nan), "collapsed": int(r.get("collapsed", False))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "encoder_all_runs.csv"), index=False)

    agg = (df.groupby(["model", "train", "test"])
             .agg(n=("seed", "nunique"), acc_mean=("acc", "mean"), acc_std=("acc", "std"),
                  f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
                  never_pred=("never_pred", "max"),
                  n_truncated=("n_truncated", "max"),
                  val_f1_mean=("val_f1", "mean"), n_collapsed=("collapsed", "sum")).reset_index())
    agg.to_csv(os.path.join(OUT, "encoder_summary.csv"), index=False)

    # --- deltas
    piv = df.pivot_table(index=["model", "train", "seed"], columns="test", values="macro_f1").reset_index()
    out = []
    for name, (a, b) in {"D_script": ("cyrl", "latn"), "D_typo": ("latn_norm", "latn"),
                         "D_pure": ("cyrl", "latn_norm")}.items():
        for (model, train), g in piv.groupby(["model", "train"]):
            d = (g[a] - g[b]).values
            t, p = (stats.ttest_1samp(d, 0.0) if len(d) > 1 else (np.nan, np.nan))
            # bootstrap
            sub = [r for r in recs if r["model"] == model and r["train_cond"] == train]
            labels = sub[0]["tests"]["latn"]["labels"]
            ci = boot_delta([r["tests"][a]["preds"] for r in sub],
                            [r["tests"][b]["preds"] for r in sub], labels)
            out.append({"delta": name, "model": model, "train": train, "n_seeds": len(d),
                        "mean": d.mean(), "std": d.std(ddof=1) if len(d) > 1 else np.nan,
                        "p_raw_ttest": p, "boot_ci_lo": ci[0], "boot_ci_hi": ci[1]})
    od = pd.DataFrame(out)
    od["p_holm"] = np.nan
    for (dname, train), g in od.groupby(["delta", "train"]):
        od.loc[g.index, "p_holm"] = holm(g["p_raw_ttest"].fillna(1.0).values)
    od.to_csv(os.path.join(OUT, "encoder_deltas.csv"), index=False)
    pd.set_option("display.width", 200)
    print(agg.round(3).to_string(index=False))
    print(od.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
