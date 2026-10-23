# prior_ladder.py (10-20 r5): parse the MEASURED skeptic ladder (OAI-free models on OAI expert MOAKS, reweighted to RSNA coupling;
# aggregates only, already local) into cells (definition x model x coupling) and summarise the joint (a.1, a.2) macro deltas.
import re, json, numpy as np
B = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/more/skeptic_ladder/'
cells = []
for f in ('box_ladder_transfer.log', 'box_ladder_transfer_blend.log'):
    dfn = None
    for ln in open(B + f):
        m = re.match(r'== (\S+)', ln)
        if m: dfn = m.group(1); continue
        m = re.match(r'\s+(\S+)\s+(R0|R1g74|tilt1\.0|tilt1\.5)\s+ESS\s+(\d+) phi ([\d.]+) rho ([\d.]+) q ([\d.]+) gap ([+-][\d.]+) \| a0\.1 ([+-][\d.]+)\(sd ([\d.]+),P [\d.]+\) a0\.2 ([+-][\d.]+)\(sd ([\d.]+)', ln)
        if m:
            cells.append(dict(defn=dfn, model=m.group(1), cpl=m.group(2), phi=float(m.group(4)), rho=float(m.group(5)), q=float(m.group(6)),
                              gap=float(m.group(7)), d1=float(m.group(8)) / 4, s1=float(m.group(9)) / 4, d2=float(m.group(10)) / 4, s2=float(m.group(11)) / 4))
print('cells', len(cells))
json.dump(cells, open('ladder_cells.json', 'w'), indent=0)
W = {'R1g74': .375, 'tilt1.0': .375, 'tilt1.5': .25}
for cp in ('R0', 'R1g74', 'tilt1.0', 'tilt1.5'):
    c = [x for x in cells if x['cpl'] == cp]
    d1 = np.array([x['d1'] for x in c]); d2 = np.array([x['d2'] for x in c])
    print('%-8s n %2d phi %.2f | macro a.1 mean %+.5f [min %+.5f max %+.5f] P>0 %.2f | a.2 mean %+.5f [%+.5f %+.5f] P>0 %.2f | a.2>a.1 %.2f' % (
        cp, len(c), np.mean([x['phi'] for x in c]), d1.mean(), d1.min(), d1.max(), (d1 > 0).mean(), d2.mean(), d2.min(), d2.max(), (d2 > 0).mean(), (d2 > d1).mean()))
c = [x for x in cells if x['cpl'] != 'R0']
w = np.array([W[x['cpl']] for x in c]); w /= w.sum()
d1 = np.array([x['d1'] for x in c]); d2 = np.array([x['d2'] for x in c]); g = np.array([x['gap'] for x in c])
m1, m2 = (w * d1).sum(), (w * d2).sum(); v1 = (w * (d1 - m1) ** 2).sum(); v2 = (w * (d2 - m2) ** 2).sum(); cv = (w * (d1 - m1) * (d2 - m2)).sum()
print('weighted mixture (R1 .375, tilt1.0 .375, tilt1.5 .25): a.1 %+.5f sd %.5f P>0 %.2f | a.2 %+.5f sd %.5f P>0 %.2f | corr %.3f | P(a.2>a.1) %.2f' % (
    m1, v1 ** .5, (w * (d1 > 0)).sum(), m2, v2 ** .5, (w * (d2 > 0)).sum(), cv / (v1 * v2) ** .5, (w * (d2 > d1)).sum()))
# delta vs gap regression across non-R0 cells (macro): d_a = k_a (gap - g0_a)
for nm, d in (('a.1', d1), ('a.2', d2)):
    k, b = np.polyfit(g, d, 1, w=np.sqrt(w)); r = np.corrcoef(g, d)[0, 1]
    print('%s: d = %.4f * gap %+.5f  (break-even gap %.3f, corr %.2f)' % (nm, k, b, -b / k, r))
print('per model (non-R0 weighted):')
for mo in ('D5', 's1044', 's1045', 'blend3'):
    ii = np.array([x['model'] == mo for x in c]); ww = w[ii] / w[ii].sum()
    print('  %-7s a.1 %+.5f a.2 %+.5f mean gap %.3f' % (mo, (ww * d1[ii]).sum(), (ww * d2[ii]).sum(), (ww * g[ii]).sum()))
