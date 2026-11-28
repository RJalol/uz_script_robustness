"""
check_apostrophes.py -- counts apostrophe-like code points in the Latin Uzbek texts
(SIB-200 uzn_Latn train/validation/test and Belebele uzn_Latn: passage, question, answers).
Purpose: show whether the latn_norm (apostrophe-normalised) control can differ from latn at all.
usage: python check_apostrophes.py data
"""
import sys, glob, csv, unicodedata
from collections import Counter
D = sys.argv[1] if len(sys.argv) > 1 else "data"
APOS = ["'", "ʻ", "ʼ", "‘", "’", "`", "´", "ʹ", "′"]

def report(name, texts):
    c = Counter(); sent = Counter(); n = 0
    for t in texts:
        n += 1
        for ch in APOS:
            k = t.count(ch)
            if k: c[ch] += k; sent[ch] += 1
    print(f"\n[{name}] texts={n}")
    for ch in APOS:
        print(f"  U+{ord(ch):04X} {unicodedata.name(ch):<32} occurrences={c[ch]:6d}  texts_with={sent[ch]:5d}")
    return c

sib = []
for p in sorted(glob.glob(f"{D}/sib200_*_latn.tsv")):
    with open(p, encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        sib += [row["text"] for row in r]
report("SIB-200 uzn_Latn (as used: latn)", sib)

try:
    from datasets import load_dataset
    ds = load_dataset("facebook/belebele", "uzn_Latn", split="test")
    fields = {"passage": "flores_passage", "question": "question",
              "answers": None}
    report("Belebele passage", [r["flores_passage"] for r in ds])
    report("Belebele question", [r["question"] for r in ds])
    report("Belebele answers", [r[f"mc_answer{i}"] for r in ds for i in range(1, 5)])
except Exception as e:
    print("Belebele check failed:", repr(e))
