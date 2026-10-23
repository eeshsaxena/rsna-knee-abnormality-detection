# box_sink_summ.py: parse box_sink.log (aggregates) into mean-over-nests tables per proxy x regime.
import re, numpy as np, collections
rows = collections.defaultdict(list); lad = collections.defaultdict(list)
for line in open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/box_sink.log'):
    m = re.match(r'\s+(P_FT|P_AREA) (\w+) (.+?) \| (marg on|I on) (\w+) \| (.*)', line)
    if m:
        px, rg, nest, kind, base, rest = m.groups()
        for it in rest.split(' | '):
            mm = re.match(r'(\S+) ([+-][0-9.]+) \(sd ([0-9.]+) P ([0-9.]+)\)', it.strip())
            if mm:
                rows[(px, rg, kind, base, mm.group(1))].append((float(mm.group(2)), float(mm.group(3)), float(mm.group(4)), nest))
        continue
    m = re.match(r'\s+(P_FT|P_AREA) (\w+) (.+?) \| n_boot \d+ \| ladder vs c4 (.*?) \| bases vs pw30: pw25 ([+-][0-9.]+) .*? \| spl ([+-][0-9.]+)', line)
    if m:
        px, rg, nest, ld, p25, spl = m.groups()
        lad[(px, rg)].append((nest, ld, float(p25), float(spl)))
for k, v in lad.items():
    print('LADDER %s %s' % k)
    for x in v: print('   %-20s %s | pw25 %+.2f | spl %+.2f | LOMM on pw25 %+.2f' % (x[0], x[1], x[2], x[3], x[3] - x[2]))
for kind in ('marg on', 'I on'):
    for px in ('P_FT', 'P_AREA'):
        for rg in ('g74', 'pub', 'raw'):
            ks = sorted({k for k in rows if k[0] == px and k[1] == rg and k[2] == kind})
            if not ks: continue
            print('\n== %s %s %s (mean over 5 nests [min, max], n nests > 0, mean P) ==' % (px, rg, kind))
            for k in ks:
                v = rows[k]; ms = [a[0] for a in v]
                if k[4] in ('S6up', 'RPF', 'E1', 'BOR', 'C6p', 'C14') and kind == 'marg on' and k[3] != 'pw25': continue
                print('  %-5s %-10s %+6.2f [%+6.2f, %+6.2f] %d/5 P %.2f' % (k[3], k[4], np.mean(ms), min(ms), max(ms), sum(x > 0 for x in ms), np.mean([a[2] for a in v])))
