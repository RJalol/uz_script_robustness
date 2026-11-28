"""
compare_tumlu.py -- agreement of our Latin->Cyrillic transliteration with the Cyrillic Uzbek questions of TUMLU-mini.
Data: github.com/ceferisbarov/TUMLU, data/uzbek/test and data/uzbek-cyrillic/test (700 questions each, 7 subjects).
Licence of the repo data: not stated in the repo [VERIFICATION REQUIRED]; only aggregate numbers are printed.

Procedure (deterministic):
 1. text = question + " " + " ".join(choices)   (so that choices are compared too)
 2. per subject, hyp = lat2cyr(Latin text); cost[i][j] = char_error_rate(Cyrillic_j, hyp_i)
 3. one-to-one assignment (Hungarian algorithm) minimising total cost
 4. a pair counts as 'parallel' if its CER <= TH (default 0.30); others are reported as unmatched
 5. report on parallel pairs: mean/median CER, exact match, word exact match (equal word counts only)
 Also reported: how many pairs are on the same index, and the answer-letter agreement of assigned pairs.
usage: python compare_tumlu.py <path to TUMLU clone>/data [threshold]
"""
import sys, os, json, glob, statistics
import numpy as np
from scipy.optimize import linear_sum_assignment
from translit_uz import lat2cyr, char_error_rate

D = sys.argv[1]
TH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30

def load(path):
    out = []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l); out.append(r)
    return out

def text(r): return r["question"] + " " + " ".join(r["choices"])

tot = dict(n=0, par=0, same_idx=0, ans_eq=0, exact=0, wok=0, wall=0); cers = []
for f in sorted(glob.glob(os.path.join(D, "uzbek", "test", "*.jsonl"))):
    s = os.path.basename(f)
    A = load(f); B = load(os.path.join(D, "uzbek-cyrillic", "test", s))
    H = [lat2cyr(text(a)) for a in A]; T = [text(b) for b in B]
    C = np.array([[char_error_rate(t, h) for t in T] for h in H])
    ri, ci = linear_sum_assignment(C)
    sub = []
    for i, j in zip(ri, ci):
        tot["n"] += 1
        tot["same_idx"] += int(i == j)
        if C[i, j] <= TH:
            tot["par"] += 1; sub.append(C[i, j]); cers.append(C[i, j])
            tot["ans_eq"] += int(A[i]["answer"] == B[j]["answer"])
            tot["exact"] += int(H[i] == T[j])
            hw, rw = H[i].split(), T[j].split()
            if len(hw) == len(rw):
                tot["wall"] += len(rw); tot["wok"] += sum(a == b for a, b in zip(hw, rw))
    print(f"{s}: questions={len(A)} parallel={len(sub)} meanCER={np.mean(sub):.4f} medianCER={np.median(sub):.4f}")
print(f"\nALL: n={tot['n']} parallel(CER<={TH})={tot['par']} same-index assignments={tot['same_idx']}")
print(f"on parallel pairs: meanCER={np.mean(cers):.4f} medianCER={np.median(cers):.4f} p90={np.percentile(cers,90):.4f} "
      f"exact={tot['exact']/tot['par']:.3f} wordExact={tot['wok']/max(tot['wall'],1):.3f} (over {tot['wall']} words) "
      f"answer-letter agreement={tot['ans_eq']/tot['par']:.3f}")
