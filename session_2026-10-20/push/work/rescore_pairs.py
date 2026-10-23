# rescore_pairs.py: re-score the 72 directional pair transforms (ex/box_coupling.log JSON, OAI aggregates) at the regime
# implied by the public E-OA reads (abc_local.py: population shift sh ~ -0.2 [-1.0, +0.85] in gold74 log-OR sd units).
# Interpolates each transform's 3-model mean d(.1) / d(.2) between R2 (sh 0), R2lo (sh -1), R2hi (sh +1). Single-label AUC
# units; macro = / 12. Adds c4 gold58 gap and the report-OOF control from coupling_rows.json for context.
import json, numpy as np
E = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/'
line = [l for l in open(E + 'box_coupling.log', encoding='utf-8').read().splitlines() if l.startswith('{"ACL | Medial Meniscus"')][0]
D = json.loads(line)
rows = []
for pair, pd_ in D.items():
    for dn, dd in pd_['dir'].items():
        ms = [m for m in dd if isinstance(dd[m], dict) and 'rw' in dd[m]]
        g = lambda sch, k: np.mean([dd[m]['rw'][sch][k] for m in ms])
        mn = lambda sch, k: np.min([dd[m]['rw'][sch][k] for m in ms])
        r = dict(dir=dn, raw1=np.mean([dd[m]['d']['0.1'] for m in ms]), R2_1=g('R2', 'd.1'), lo1=g('R2lo', 'd.1'), hi1=g('R2hi', 'd.1'),
                 R2_2=g('R2', 'd.2'), lo2=g('R2lo', 'd.2'), min_lo1=mn('R2lo', 'd.1'), min_R2_1=mn('R2', 'd.1'))
        for sh in (-1.0, -0.5, -0.2, 0.0):
            r['at%.1f' % sh] = r['R2_1'] + (r['R2_1'] - r['lo1']) * sh if sh <= 0 else r['R2_1'] + (r['hi1'] - r['R2_1']) * sh
        rows.append(r)
rows.sort(key=lambda r: -r['at-0.2'])
print('%-38s %8s %8s %8s %8s | %8s %8s %8s | %9s' % ('direction (a .1, 3-model mean AUC delta)', 'raw', 'R2(0)', 'R2lo(-1)', 'R2hi(+1)', 'sh-0.2', 'sh-0.5', 'min lo', 'R2lo a.2'))
for r in rows[:22]:
    print('%-38s %+.4f  %+.4f  %+.4f  %+.4f  | %+.4f  %+.4f  %+.4f | %+.4f' % (
        r['dir'][:38], r['raw1'], r['R2_1'], r['lo1'], r['hi1'], r['at-0.2'], r['at-0.5'], r['min_lo1'], r['lo2']))
print('...'); print('n directions with all-3-model R2lo d.1 > 0:', sum(r['min_lo1'] > 0 for r in rows), '| with R2 (gold74) all > 0:', sum(r['min_R2_1'] > 0 for r in rows))
json.dump(rows, open('rescore_pairs.json', 'w'), indent=0, default=float)
