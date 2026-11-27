"""
validate_translit.py -- external validity check of the rule-based transliterator.

Takes a TSV with two columns:  <latin_sentence> TAB <human_cyrillic_sentence>
(e.g. the Latin/Cyrillic transcripts of FeruzaSpeech -- access and licence must be
checked: [VERIFICATION REQUIRED]) and reports how close lat2cyr(latin) is to the human Cyrillic.

Reports: CER, sentence exact-match, word exact-match (when token counts agree),
and the most frequent mismatching word pairs (error analysis for the paper).

python validate_translit.py pairs.tsv
"""
import sys
from collections import Counter
from translit_uz import lat2cyr, char_error_rate

path = sys.argv[1]
cers, exact, w_ok, w_all = [], 0, 0, 0
bad = Counter()
n = 0
for line in open(path, encoding="utf-8"):
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 2:
        continue
    lat, cyr = parts[0], parts[1]
    hyp = lat2cyr(lat)
    n += 1
    cers.append(char_error_rate(cyr, hyp))
    exact += int(hyp == cyr)
    hw, rw = hyp.split(), cyr.split()
    if len(hw) == len(rw):
        for h, r in zip(hw, rw):
            w_all += 1
            if h == r:
                w_ok += 1
            else:
                bad[(r, h)] += 1
print(f"sentences={n}  mean CER={sum(cers)/n:.4f}  sentence exact={exact/n:.3f}")
print(f"word exact (length-aligned sentences)={w_ok/max(w_all,1):.3f}  over {w_all} words")
print("top mismatches (human -> rule-based):")
for (r, h), c in bad.most_common(30):
    print(f"  {c:4d}  {r}  ->  {h}")
