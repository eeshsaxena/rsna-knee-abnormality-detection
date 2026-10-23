# Per-language / per-vendor medial-lateral cross-talk on R7 OOF vs LLM report verdicts, and the visible test geometry.
import numpy as np, pandas as pd, json
from scipy.stats import rankdata
D = 'C:/Users/eeshs/Downloads/rsna-knee/'
OUT = D + 'session_2026-10-19/trick/lat_work/'
sw = pd.read_csv(OUT + 'swap_oof.csv')
kc = pd.read_csv(D + 'session_2026-10-15/kcrop/out_kc_side_v1/kc_side_train.csv')[['StudyInstanceUID', 'man']]
lab = pd.read_csv(D + '_labels/rsna-knee-llm-labels/report_labels_v2.csv')
r7 = pd.read_csv(D + 'session_2026-10-18/next/g58/R7OOF.csv')
L4 = ['Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA']
df = sw.merge(kc, on='StudyInstanceUID').merge(lab[['StudyInstanceUID'] + [c + '__verdict' for c in L4]], on='StudyInstanceUID') \
       .merge(r7[['StudyInstanceUID'] + ['z|' + c for c in L4]], on='StudyInstanceUID')
def auc(s, y):
    y = np.asarray(y); p = y.sum(); q = len(y) - p
    if p == 0 or q == 0: return np.nan
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
def block(d):
    out = {'n': len(d)}
    for a, b in [('Medial Meniscus', 'Lateral Meniscus'), ('Medial OA', 'Lateral OA')]:
        for u, v in [(a, b), (b, a)]:
            k = d[u + '__verdict'].isin(['YES', 'NO'])
            y = (d.loc[k, u + '__verdict'] == 'YES').astype(int)
            out[f'{u[:3]}{u.split()[1][:3]} own'] = auc(d.loc[k, 'z|' + u], y)
            out[f'{u[:3]}{u.split()[1][:3]} cross'] = auc(d.loc[k, 'z|' + v], y)
        # discordant-pair ordering
        ya, yb = d[a + '__verdict'], d[b + '__verdict']
        sel = ((ya == 'YES') & (yb == 'NO')) | ((ya == 'NO') & (yb == 'YES'))
        s = np.where(ya[sel] == 'YES', 1, -1); gap = d.loc[sel, 'z|' + a] - d.loc[sel, 'z|' + b]
        out[f'{a.split()[1][:4]} disc n'] = int(sel.sum()); out[f'{a.split()[1][:4]} disc ok'] = float((np.sign(gap) == s).mean()) if sel.sum() else np.nan
    return out
pd.set_option('display.width', 250)
rows = {('lang', g): block(d) for g, d in df.groupby('lang')}
rows.update({('vendor', g): block(d) for g, d in df.groupby('man') if len(d) >= 40})
df['neariso_untag'] = (df.mean_x.abs() < 20) & ~df.tag.isin(['L', 'R'])
rows.update({('near-iso untagged', str(g)): block(d) for g, d in df.groupby('neariso_untag')})
rows[('all', '')] = block(df)
t = pd.DataFrame(rows).T
print(t.round(3).to_string())
t.round(4).to_csv(OUT + 'lang_cross.csv')
print('\nvisible test studies (KC side table):')
print(pd.read_csv(D + 'session_2026-10-15/kcrop/out_kc_side_v1/kc_side_test.csv')[['mean_x', 'tag', 'side', 'rule', 'man']].to_string())
