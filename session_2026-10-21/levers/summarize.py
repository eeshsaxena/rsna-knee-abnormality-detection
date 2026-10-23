# summarize.py (10-21 levers, Claude). LOCAL ONLY. Joins regime_sweep.json (raw-gold74 regimes) and regime_dlt_sweep.json (population-matched)
# into one classification per direction x a (label AUC x1e4; macro = /12). Writes regime_summary.log.
import json
import numpy as np
AB = {"ACL": "ACL", "MCL": "MCL", "Medial Meniscus": "MM", "Lateral Meniscus": "LM", "Medial OA": "MOA", "Lateral OA": "LOA",
      "PF OA": "PF", "Effusion": "Eff", "Synovitis": "Syn", "Baker's": "Bak", "Contusion": "Cont", "Fracture": "Frac"}
a1 = json.load(open('regime_sweep.json')); a2 = json.load(open('regime_dlt_sweep.json'))
M = a2['MODELS']; AV = a2['AV']
out = open('regime_summary.log', 'w')
def P(s): print(s); out.write(s + '\n')
cls = {'BOTH (H_train EB+RAW+EBg and H_gold G74+IPF)': [], 'H_train only (EB+RAW+EBg, G74 not all +)': [], 'H_gold only (G74+IPF, EB not all +)': []}
for k in a2['RES']:
    for ia, a in enumerate(AV):
        v = lambda src, r: np.array([src['RES'][k]['res'][m + '|' + r][ia] for m in M])
        c = {r: v(a2, r) for r in ('REP', 'EB', 'RAW', 'EBg', 'GEN')}; c.update({r: v(a1, r) for r in ('G74', 'IPF', 'I_FT')})
        ap = lambda r: bool((c[r] > 0).all())
        ht = ap('EB') and ap('RAW') and ap('EBg'); hg = ap('G74') and ap('IPF')
        n = a2['RES'][k]['n']
        row = (k, a, c, n)
        if ht and hg: cls['BOTH (H_train EB+RAW+EBg and H_gold G74+IPF)'].append(row)
        elif ht: cls['H_train only (EB+RAW+EBg, G74 not all +)'].append(row)
        elif hg and not ap('EB'): cls['H_gold only (G74+IPF, EB not all +)'].append(row)
f = lambda x: '%+6.1f/%+6.1f' % (x[3] * 1e4, x.min() * 1e4)
for nm, rows in cls.items():
    P('== %s: %d cells' % (nm, len(rows)))
    P('%-12s %5s %5s | %-15s %-15s %-15s %-15s %-15s | %s' % ('dir', 'a', 'n', 'REP', 'EB', 'RAW', 'G74', 'I_FT', 'flags'))
    for k, a, c, n in sorted(rows, key=lambda r: -min(r[2]['EB'][3], r[2]['G74'][3])):
        t, o = k.split(' <- ')
        fl = []
        if 'Synovitis' in (t, o): fl.append('SYN-subset')
        if n < 1200: fl.append('n<1200')
        if (c['REP'] > 0).all(): fl.append('REP+ (report-label gain too)')
        if (c['REP'] < 0).all(): fl.append('REP- (mechanism-consistent)')
        P('%-12s %+5.2f %5d | %-15s %-15s %-15s %-15s %-15s | %s' % (AB[t] + '<-' + AB[o], a, n, f(c['REP']), f(c['EB']), f(c['RAW']), f(c['G74']), f(c['I_FT']), ', '.join(fl)))
