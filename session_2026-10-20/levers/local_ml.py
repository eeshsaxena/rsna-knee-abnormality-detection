# local_ml.py (10-20 levers, Claude; research only, CPU, competition data): multilingual spot check of the two largest census
# findings (meniscal degeneration treated as negative; small effusion treated as positive) in Spanish and Turkish reports.
import re, numpy as np, pandas as pd
R = 'C:/Users/eeshs/Downloads/rsna-knee'
lang = pd.read_csv(R + '/_data/_reports_lang.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
srcs = {'v4': R + '/_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv', 'jev': R + '/session_2026-10-05/jump2/labels_in/jev_hybrid_scores.csv'}
LBL = {k: pd.read_csv(p, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID') for k, p in srcs.items()}
t = lang.Report.str.lower()
rules = {
 ('es', 'Medial Meniscus', 'degeneration, no tear word'): (r'degenera', r'(rotura|desgarro|ruptura|fisura|lesi[oó]n meniscal)', r'menisco (medial|interno)'),
 ('es', 'Effusion', 'small only'): (r'derrame[^.]{0,30}(leve|peque|escaso|m[ií]nimo|discreto)|(leve|peque|escaso|m[ií]nimo|discreto)[^.]{0,20}derrame', r'(moderad|abundante|importante|significativ|cuant)', r'derrame'),
 ('tr', 'Medial Meniscus', 'degeneration, no tear word'): (r'dejenera', r'(y[ıi]rt[ıi]k|r[uü]pt[uü]r)', r'medial men[ıi]s'),
 ('tr', 'Effusion', 'small only'): (r'(minimal|az miktar|hafif|minim)[^.]{0,30}(s[ıi]v[ıi]|ef[uü]zyon)|(s[ıi]v[ıi]|ef[uü]zyon)[^.]{0,30}(minimal|az|hafif)', r'(orta|belirgin|fazla|yo[gğ]un|massif|masif)', r'(s[ıi]v[ıi]|ef[uü]zyon)'),
}
for (lg, lab, nm), (pat, excl, ctx) in rules.items():
    m = (lang.lang == lg) & t.str.contains(pat, regex=True) & ~t.str.contains(excl, regex=True) & t.str.contains(ctx, regex=True)
    ids = lang.index[m]
    print(f'{lg} {lab:16s} {nm:28s} n {len(ids):4d} | ' + ' '.join(f'{k} pos {(d.reindex(ids)[lab] >= .5).mean():.2f}' for k, d in LBL.items()))
