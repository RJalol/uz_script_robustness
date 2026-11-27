"""
validate_feruza.py -- aggregate-only validation of lat2cyr against FeruzaSpeech transcripts.

Terms of the dataset: academic/non-commercial use, no redistribution of text -> this script prints
ONLY aggregate statistics and character-level confusion counts (single characters), never sentences/words.
Needs an HF token with granted access (Kaggle secret HF_TOKEN).
usage: python validate_feruza.py <hf_dataset_id> [max_rows]
"""
import re, sys, difflib, statistics
from collections import Counter
from translit_uz import lat2cyr, char_error_rate

DS = sys.argv[1]
MAXR = int(sys.argv[2]) if len(sys.argv) > 2 else 4000   # rows per split
tok = None
import os
tok = os.environ.get("HF_TOKEN")
if not tok:
    try:
        from kaggle_secrets import UserSecretsClient
        us = UserSecretsClient()
        for label in ("HF_TOKEN", "uz_script_robustness"):   # secret label may be either
            try:
                tok = us.get_secret(label); print("token read from secret:", label); break
            except Exception:
                pass
    except Exception:
        pass
if not tok:
    sys.exit("No token found: add a Kaggle secret (label HF_TOKEN or uz_script_robustness) and attach it to the notebook.")
from datasets import load_dataset
from itertools import islice
ds = load_dataset(DS, token=tok, streaming=True)   # streaming: avoids downloading the audio of the whole corpus
print("splits:", list(ds.keys()))
split0 = list(ds.keys())[0]
print("features:", {k: str(v)[:60] for k, v in ds[split0].features.items()})
CYR = re.compile(r"[\u0400-\u04FF]"); LAT = re.compile(r"[A-Za-z]")

def kind(s):
    c, l = len(CYR.findall(s)), len(LAT.findall(s))
    return "cyr" if c > 2 * l else "lat" if l > 2 * c else "mixed"

head = list(islice(ds[split0], 200))
cols = [c for c in head[0] if isinstance(head[0][c], str)]
votes = {c: Counter(kind(r[c]) for r in head if isinstance(r[c], str)) for c in cols}
print("string columns and script votes:", {c: dict(v) for c, v in votes.items()})
cyr_col = max(cols, key=lambda c: votes[c]["cyr"], default=None)
lat_col = max(cols, key=lambda c: votes[c]["lat"], default=None)
print("using Latin column:", lat_col, "| Cyrillic column:", cyr_col)
if not cyr_col or not lat_col or votes[cyr_col]["cyr"] == 0 or votes[lat_col]["lat"] == 0:
    sys.exit("Could not find both a Latin and a Cyrillic text column; inspect the printed columns.")

for sp in ds.keys():
    d = ds[sp]
    cers, exact, wok, wall, n = [], 0, 0, 0, 0
    conf = Counter()
    for i, r in enumerate(d):
        if i >= MAXR: break
        lat, cyr = r.get(lat_col), r.get(cyr_col)
        if not lat or not cyr: continue
        h = lat2cyr(lat); n += 1
        cers.append(char_error_rate(cyr, h)); exact += h == cyr
        hw, rw = h.split(), cyr.split()
        if len(hw) == len(rw):
            wall += len(rw); wok += sum(a == b for a, b in zip(hw, rw))
        sm = difflib.SequenceMatcher(None, cyr, h, autojunk=False)
        for op, a0, a1, b0, b1 in sm.get_opcodes():
            if op == "replace" and (a1 - a0) == (b1 - b0):
                for x, y in zip(cyr[a0:a1], h[b0:b1]): conf[(x, y)] += 1
            elif op == "delete":   # present in human, missing in hypothesis
                for x in cyr[a0:a1]: conf[(x, "<none>")] += 1
            elif op == "insert":   # extra in hypothesis
                for y in h[b0:b1]: conf[("<none>", y)] += 1
    if not n: continue
    q = sorted(cers)
    print(f"\n[{sp}] n={n} meanCER={sum(cers)/n:.4f} medianCER={statistics.median(cers):.4f} "
          f"p90={q[int(.9*(n-1))]:.4f} sentExact={exact/n:.3f} wordExact={wok/max(wall,1):.3f} (over {wall} words)")
    print("top single-character confusions (human -> rule-based): count")
    for (x, y), c in conf.most_common(15):
        print(f"  {x!r} -> {y!r}: {c}")
