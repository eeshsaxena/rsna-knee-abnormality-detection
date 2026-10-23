# sim_r7_oof.py (r7teacher/labels, 2026-10-14): writes a SIMULATED R7-OOF teacher file in the exact format the label
# builder expects from the real R7 out-of-fold run, so the whole chain can be dry-run before the real file exists.
# Source of the simulated values: the K14 fleet member set DT (TTR224 frac75 + OAI T123v00, s1043): its 5-fold OOF on
# the non-gold pool rows and the 5-fold mean of its gold74 pass-A predictions on the gold rows. DT is NOT one of the
# OAI-free sets that form the real fleet_oof component, so the simulated T_b has three distinct sources like the real one.
# The file is tagged teacher_tag = SIM:K14-DT on every row; the builder names its outputs SIM_* and the queue gate refuses
# any SIM table. A few pool rows are dropped on purpose (--drop) to exercise the builder's missing-row path.
#
# TEACHER FILE FORMAT = job 1's merged r7_oof.csv (session_2026-10-15/r7teacher/merge_sessions.py): StudyInstanceUID,
#   r7_fold (int = int(sha1(uid)[:8], 16) % 4, the hash fold whose model scored the study = the fold that did NOT train on it;
#   gold studies carry their hash fold too: gold58 was never in R7's pool, so any fold model is out-of-fold there), shard, job,
#   precision ('full' when logits were dumped), p|<label> x 12 (per-study PROBABILITIES, the R7 bundle's final output),
#   z|<label> x 12 and fluid / stk / state extras (ignored here). The builder also accepts plain <label> columns.
#   This simulation writes r7_fold, precision, p|<label> x 12 and teacher_tag ('SIM:...'; the real file has no tag column).
import argparse, hashlib, os
import numpy as np, pandas as pd

H = os.path.expanduser('~')
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
ap = argparse.ArgumentParser()
ap.add_argument('--oof', required=True)          # collect_fleet_oof.py oof_<set>.csv (pool rows)
ap.add_argument('--gold_mean', required=True)    # collect_fleet_oof.py gold_<name>_mean.csv (gold74 rows)
ap.add_argument('--out', required=True)
ap.add_argument('--tag', default='SIM:K14-DT')
ap.add_argument('--drop', type=int, default=3)
A = ap.parse_args()
assert A.tag.startswith('SIM:'), 'a simulated teacher must carry a SIM: tag'
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
o = rd(A.oof)[T]; g = rd(A.gold_mean)[T]
assert not set(o.index) & set(g.index)
rng = np.random.default_rng(7)
drop = set(rng.choice(np.array(sorted(o.index)), A.drop, replace=False)) if A.drop else set()
o = o.drop(index=sorted(drop))
hf = lambda u: int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4
d = pd.concat([o, g])
d['r7_fold'] = [hf(u) for u in d.index]                  # job-1 layout: every study (gold too) carries its hash fold
d = d.rename(columns={c: 'p|' + c for c in T})
d['precision'] = 'full'; d['teacher_tag'] = A.tag
d = d[['r7_fold', 'precision'] + ['p|' + c for c in T] + ['teacher_tag']]
d.index.name = 'StudyInstanceUID'
d.to_csv(A.out)
print(f'wrote {A.out}: {len(d)} rows ({len(o)} pool rows, {len(g)} gold rows, {len(drop)} pool rows dropped on purpose) '
      f'| hash-fold counts {np.bincount(d.r7_fold, minlength=4).tolist()} | tag {A.tag}')
