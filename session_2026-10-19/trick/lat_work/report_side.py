# Report-side extraction (multilingual) vs DICOM side (kc_side_train.csv). CPU only, reads local files.
import re, sys, json
import numpy as np, pandas as pd

D = 'C:/Users/eeshs/Downloads/rsna-knee/'
OUT = D + 'session_2026-10-19/trick/lat_work/'
rep = pd.read_csv(D + '_data/_reports_lang.csv')
kc = pd.read_csv(D + 'session_2026-10-15/kcrop/out_kc_side_v1/kc_side_train.csv')

# knee-context side words. Keep it strict: side word adjacent (within a few tokens) to a knee word, or a header style
# "MR Knie Rechts", "RIGHT KNEE", "left knee", "rodilla izquierda", "sol diz", "desno koljeno", ...
KNEE = r'(knee|knie|rodilla|diz|dizin|dizde|koljen\w*|koleno|коляно|коляното|γόνατο\w*|γονάτου|genou|knieg\w*|kniegewricht)'
LEFT = r'(left|lt|links|linker|linke|linken|linkes|izquierd[ao]|izda|izq|sol|lijev\w*|ляв\w*|лява|лявото|левия|αριστερ\w*|gauche)'
RIGHT = r'(right|rt|rechts|rechter|rechte|rechten|rechtes|derech[ao]|dcha|dch|sağ|sag|desn\w*|дясн\w*|десн\w*|δεξ\w*|droit|droite)'
W = r'[\s\-:,\.\(\)]{1,3}(?:\S+[\s\-:,\.\(\)]{1,3}){0,1}'   # at most one token between
pat_l = re.compile(r'\b' + LEFT + W + KNEE + r'|\b' + KNEE + W + LEFT + r'\b', re.I | re.U)
pat_r = re.compile(r'\b' + RIGHT + W + KNEE + r'|\b' + KNEE + W + RIGHT + r'\b', re.I | re.U)
# Turkish 'sag' (ascii) is ambiguous with sagittal: only accept 'sağ'
def count(p, txt):
    n = 0
    for m in p.finditer(txt):
        s = m.group(0).lower()
        if re.search(r'\bsag\b', s) and 'sağ' not in s:
            continue
        if re.search(r'\b(lt|rt)\b', s) and not re.search(r'\b(lt|rt)\.?\s*(knee)', s):
            continue
        n += 1
    return n

rows = []
for sid, txt, lang in zip(rep.StudyInstanceUID, rep.Report.fillna(''), rep.lang):
    nl, nr = count(pat_l, txt), count(pat_r, txt)
    side = 'L' if nl > 0 and nr == 0 else 'R' if nr > 0 and nl == 0 else ('B' if nl and nr else 'U')
    rows.append((sid, lang, nl, nr, side))
rs = pd.DataFrame(rows, columns=['StudyInstanceUID', 'lang', 'n_left', 'n_right', 'rep_side'])
m = rs.merge(kc[['StudyInstanceUID', 'mean_x', 'tag', 'side', 'rule', 'man']], on='StudyInstanceUID', how='left')
m['x_side'] = np.where(m.mean_x >= 0, 'L', 'R')
m.to_csv(OUT + 'report_side.csv', index=False)

out = {}
print('report side counts by lang')
print(pd.crosstab(m.lang, m.rep_side))
known = m[m.rep_side.isin(['L', 'R'])]
print('\nreport-sided studies', len(known), 'of', len(m))
for col in ['x_side', 'side', 'tag']:
    k2 = known[known[col].isin(['L', 'R'])]
    agree = (k2.rep_side == k2[col]).mean()
    print(f'report vs {col}: n={len(k2)} agree={agree:.4f} disagree={int((k2.rep_side != k2[col]).sum())}')
    out[f'rep_vs_{col}'] = dict(n=int(len(k2)), agree=float(agree), disagree=int((k2.rep_side != k2[col]).sum()))
print('\nby lang: report vs x_side')
g = known.groupby('lang').apply(lambda d: pd.Series(dict(n=len(d), disagree=int((d.rep_side != d.x_side).sum()))))
print(g)
print('\nby |x| bin: report vs x_side')
known = known.assign(absx=pd.cut(known.mean_x.abs(), [0, 10, 20, 30, 50, 80, 200]))
print(known.groupby('absx', observed=True).apply(lambda d: pd.Series(dict(n=len(d), disagree=int((d.rep_side != d.x_side).sum())))))
print('\ntag vs x disagreement, with report adjudication')
t = m[m.tag.isin(['L', 'R'])]
td = t[t.tag != t.x_side]
print('tag!=x', len(td), ' report says tag:', int((td.rep_side == td.tag).sum()), ' report says x:', int((td.rep_side == td.x_side).sum()), ' report unknown:', int((~td.rep_side.isin(['L','R'])).sum()))
print(td[['StudyInstanceUID','lang','mean_x','tag','rep_side','man']].to_string())
out['tag_vs_x'] = dict(n_tag=int(len(t)), n_disagree=int(len(td)), rep_says_tag=int((td.rep_side == td.tag).sum()),
                       rep_says_x=int((td.rep_side == td.x_side).sum()))
json.dump(out, open(OUT + 'report_side.json', 'w'), indent=1)
