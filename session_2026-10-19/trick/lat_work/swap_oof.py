# Medial/lateral swap evidence on R7 OOF (heliosli family, canonicalised, position-based M/L) by laterality-suspect group.
# Labels: LLM report verdicts (report_labels_v2: YES / NO / UNK). CPU only.
import json
import numpy as np, pandas as pd

D = 'C:/Users/eeshs/Downloads/rsna-knee/'
OUT = D + 'session_2026-10-19/trick/lat_work/'
m = pd.read_csv(OUT + 'report_side.csv')
m['tagged'] = m.tag.isin(['L', 'R'])
m['x_side'] = np.where(m.mean_x >= 0, 'L', 'R')
m['hs'] = np.where(m.tagged, m.tag, m.x_side)          # heliosli decide_study_side without the image check
m['absx'] = m.mean_x.abs()
rep_ok = m.rep_side.isin(['L', 'R']) & (m.lang != 'bg')  # bg reports carry a 'left knee' boilerplate line (measured)
m['rep_use'] = np.where(rep_ok, m.rep_side, 'U')

lab = pd.read_csv(D + '_labels/rsna-knee-llm-labels/report_labels_v2.csv')
r7 = pd.read_csv(D + 'session_2026-10-18/next/g58/R7OOF.csv')
PAIRS = [('Medial Meniscus', 'Lateral Meniscus'), ('Medial OA', 'Lateral OA')]
cols = ['StudyInstanceUID']
for a, b in PAIRS:
    cols += [a + '__verdict', b + '__verdict']
df = m.merge(lab[cols], on='StudyInstanceUID').merge(
    r7[['StudyInstanceUID'] + ['z|' + c for p in PAIRS for c in p] + ['z|MCL']], on='StudyInstanceUID')

def pair_stats(d):
    """for each study and pair where exactly one side is YES and the other NO: is the model's M-L logit gap on the YES side?"""
    n = c = 0
    ll = []
    for a, b in PAIRS:
        ya, yb = d[a + '__verdict'], d[b + '__verdict']
        sel = ((ya == 'YES') & (yb == 'NO')) | ((ya == 'NO') & (yb == 'YES'))
        s = np.where(ya[sel] == 'YES', 1.0, -1.0)
        gap = (d['z|' + a][sel] - d['z|' + b][sel]).values
        n += int(sel.sum()); c += int((np.sign(gap) == s).sum())
    return n, c

def swap_llr(d):
    """per-study log-likelihood ratio (swapped vs as-is) of the YES/NO verdicts under the model's sigmoid probabilities"""
    out = np.zeros(len(d))
    for a, b in PAIRS:
        for (u, v) in [(a, b), (b, a)]:
            y = d[u + '__verdict'].values
            z_as, z_sw = d['z|' + u].values, d['z|' + v].values
            known = (y == 'YES') | (y == 'NO')
            yy = (y == 'YES').astype(float)
            lp = lambda z: -np.logaddexp(0, -z) * yy - np.logaddexp(0, z) * (1 - yy)
            out += np.where(known, lp(z_sw) - lp(z_as), 0.0)
    return out

df['llr_swap'] = swap_llr(df)
groups = {
    'all': df.index == df.index,
    'clean: tagged, tag==x, |x|>50': df.tagged & (df.tag == df.x_side) & (df.absx > 50),
    'untagged |x|>50': (~df.tagged) & (df.absx > 50),
    'untagged |x|<20': (~df.tagged) & (df.absx < 20),
    'untagged |x|<20, report != x': (~df.tagged) & (df.absx < 20) & (df.rep_use != 'U') & (df.rep_use != df.x_side),
    'untagged |x|<20, report == x': (~df.tagged) & (df.absx < 20) & (df.rep_use != 'U') & (df.rep_use == df.x_side),
    'tag != x': df.tagged & (df.tag != df.x_side),
    'tag != x, |x|>=20': df.tagged & (df.tag != df.x_side) & (df.absx >= 20),
    'report != heliosli side': (df.rep_use != 'U') & (df.rep_use != df.hs),
    'report != heliosli side, |x|>=30': (df.rep_use != 'U') & (df.rep_use != df.hs) & (df.absx >= 30),
    'report != x side, |x|>=30': (df.rep_use != 'U') & (df.rep_use != df.x_side) & (df.absx >= 30),
}
res = {}
print(f"{'group':45s} {'studies':>7s} {'pairs':>6s} {'correct':>8s} {'LLR>2':>6s} {'LLR>2 %':>7s}")
for g, sel in groups.items():
    d = df[sel]
    n, c = pair_stats(d)
    hi = int((d.llr_swap > 2).sum())
    res[g] = dict(studies=int(len(d)), pairs=n, correct=(c / n if n else None), llr_gt2=hi)
    print(f"{g:45s} {len(d):7d} {n:6d} {(c / n if n else float('nan')):8.3f} {hi:6d} {100 * hi / max(1, len(d)):7.2f}")
# list the strongest swap candidates overall
top = df.sort_values('llr_swap', ascending=False).head(25)
print(top[['StudyInstanceUID', 'lang', 'mean_x', 'tag', 'rep_use', 'hs', 'llr_swap']].to_string())
df[['StudyInstanceUID', 'lang', 'mean_x', 'tag', 'x_side', 'hs', 'rep_use', 'llr_swap']].to_csv(OUT + 'swap_oof.csv', index=False)
json.dump(res, open(OUT + 'swap_oof.json', 'w'), indent=1)
