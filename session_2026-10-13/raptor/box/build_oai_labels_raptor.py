# build_oai_labels_raptor.py (2026-10-13, box CPU only, OAI stays on the box): nartaa's published OAI rules for the Raptor
# 0.949 recipe (discussion 746792, 10-13), applied to our merged V00 MOAKS reads (~/oai_build/oai_label_sources.csv):
#   PF OA      = 1 if full-thickness cartilage loss >= 1 in any PF subregion (PM, PL, FMA, FLA = m_cm_mp lp amf alf)
#   Lateral OA = same rule over FLC, FLP, TLA, TLC, TLP (= m_cm_clf plf alt clt plt)
#   Synovitis  = Hoffa-synovitis m_syic >= 1
#   hard 0/1; 1 if any valid subregion meets the cut; 0 only if EVERY needed subregion is valid and none meets it; else NaN.
#   All other 9 findings NaN (zero loss weight). Expected counts (nartaa): PF OA 2,396, Lateral OA 2,397, Synovitis 1,659.
# Output: ~/oai_build/oai_labels_raptorT1.csv (mode 600). Prints aggregate counts only.
import os
import numpy as np, pandas as pd
H = os.path.expanduser('~'); W = f'{H}/oai_build'
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
CUT = {'PF OA': ['mp', 'lp', 'amf', 'alf'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt']}
src = pd.read_csv(f'{W}/oai_label_sources.csv').set_index('kid')
out = pd.DataFrame(np.nan, index=src.index, columns=T)
for lab, regs in CUT.items():
    c = src[[f'm_cm_{r}' for r in regs]].astype(float).values
    ft = np.round((c - np.floor(c + 1e-6)) * 10)                 # MOAKS "area.fullthickness": first decimal = FT grade
    valid = np.isfinite(c); hit = valid & (ft >= 1)
    y = np.where(hit.any(1), 1.0, np.where(valid.all(1), 0.0, np.nan))
    out[lab] = y
s = src['m_syic'].astype(float)
out['Synovitis'] = np.where(s.notna(), (s >= 1).astype(float), np.nan)
out = out[out[T].notna().any(axis=1)]
for lab in ('PF OA', 'Lateral OA', 'Synovitis'):
    v = out[lab].dropna()
    print(f'{lab:11s} labelled {len(v):5d} (nartaa 2396/2397/1659) | positive rate {v.mean():.3f}')
print('knees with >= 1 label', len(out))
p = f'{W}/oai_labels_raptorT1.csv'
out.reset_index().rename(columns={'kid': 'StudyInstanceUID'}).to_csv(p, index=False)
os.chmod(p, 0o600)
print('wrote', p)
