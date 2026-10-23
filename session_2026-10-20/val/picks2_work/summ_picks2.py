# summ_picks2.py: sensitivity tables + regret summary from picks2_<tag>.json (MODEL outputs). pts = percentage points, E in e-4.
import json, os, numpy as np
H = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/val/picks2_work/'
order = ['base']
for l in open(H + 'run_sens2.sh'):
    for part in l.split('"'):
        if part.startswith('tag='): order.append(part.split()[0][4:])
ASK = ['R6u', 'R6pw30', 'R6e20', 'eoa20', 'c4']; EXT = ['pw20', 'R6pw20', 'R6e20|pw20', 'R6pw20|pw20', 'R6pw30|pw20']
BB = ASK + EXT
R = {t: json.load(open(H + 'picks2_%s.json' % t)) for t in order if os.path.exists(H + 'picks2_%s.json' % t)}
for an in ('pw30', 'R1{pw30,pw25}'):
    for metric, sc in (('silver', 100), ('r150', 100), ('E', 1e4)):
        print('== A = %s | %s (%s) | asked: %s | extra: %s' % (an, metric, 'pts' if sc == 100 else 'e-4', ', '.join(ASK), ', '.join(EXT)))
        print('%-22s %7s ' % ('row', 'A') + ' '.join('%11s' % b for b in BB) + '   best(asked)  best(all)')
        reg = {b: [] for b in BB}
        for t, J in R.items():
            if an not in J['A']: continue
            S = J['A'][an]; v = {b: S[b][metric] * sc for b in BB if b in S}
            if len(v) < len(BB): continue
            a5 = max(ASK, key=lambda b: v[b]); a6 = max(BB, key=lambda b: v[b]); m = max(v.values())
            for b in BB: reg[b].append(m - v[b])
            print('%-22s %7.2f ' % (t, S['A alone'][metric] * sc) + ' '.join('%11.2f' % v[b] for b in BB) + '   %-11s  %s' % (a5, a6))
        n = len(reg[BB[0]])
        print('%-22s %7s ' % ('mean regret (%d rows)' % n, '') + ' '.join('%11.2f' % np.mean(reg[b]) for b in BB))
        print('%-22s %7s ' % ('max regret', '') + ' '.join('%11.2f' % np.max(reg[b]) for b in BB))
        print('%-22s %7s ' % ('rows best', '') + ' '.join('%11d' % int(sum(np.array(reg[b]) < 1e-9)) for b in BB))
        ra = {b: [] for b in ASK}
        for t, J in R.items():
            if an not in J['A']: continue
            S = J['A'][an]; v = {b: S[b][metric] * sc for b in ASK}; m = max(v.values())
            for b in ASK: ra[b].append(m - v[b])
        print('%-22s %7s ' % ('asked-only mean regret', '') + ' '.join('%11.2f' % np.mean(ra[b]) for b in ASK) + '   (max ' + ' '.join('%.2f' % np.max(ra[b]) for b in ASK) + ')')
        print()
