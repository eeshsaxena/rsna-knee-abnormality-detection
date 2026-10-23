#!/usr/bin/env python
# gate_pl4.py (2026-10-11, Claude): applies the PROBE4_PREREG.md bars to the report written by
# score_oai_probe2.py --tier T12 --report_only for the T123PL probe. Writes ~/oai_build/out/pl4.verdict (PASS / FAIL)
# and ~/oai_build/out/gate_pl4.txt. Exit 0 PASS, 1 FAIL, 2 integrity, 3 not ready.
#   --selftest: the arithmetic on the T123 probe's own report (must reproduce the frozen numbers; no verdict written)
import os, sys, json, argparse, hashlib
H = os.path.expanduser
ap = argparse.ArgumentParser()
ap.add_argument('--report', default=H('~/oai_build/out/gate2_REPORT_T12_f0_s1043_ttr_frac75oaiT123PL.json'))
ap.add_argument('--selftest', action='store_true')
A = ap.parse_args()
PREREG_SHA = open(H('~/oai_build/PROBE4_PREREG.sha256')).read().split()[0]
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
P3 = ['MCL', 'Contusion', 'Fracture']
R9 = [c for c in T if c not in P3]
FROZEN = dict(p3=0.868693, r9=0.893982, oof12=0.887660, g58=0.919434)
BARS = dict(G1=FROZEN['p3'] + 0.003, G2=FROZEN['oof12'] - 0.0003, G3=FROZEN['r9'] - 0.002, G4=FROZEN['g58'] - 0.005)
OUT = H('~/oai_build/out')

if hashlib.sha256(open(H('~/oai_build/PROBE4_PREREG.md'), 'rb').read()).hexdigest() != PREREG_SHA:
    print('INTEGRITY: PROBE4_PREREG.md changed'); sys.exit(2)
if A.selftest:
    A.report = H('~/oai_build/out/gate2_REPORT_T12_f0_s1043_ttr_frac75oaiT123v00.json')
if not os.path.exists(A.report):
    print('NOT READY: no report', A.report); sys.exit(3)
d = json.load(open(A.report))
if not A.selftest:
    if d.get('labels') != 'labels_frac75oai_T123PLf0.csv' or not d['run'].endswith('f0_s1043_ttr_frac75oaiT123PL'):
        print('INTEGRITY: report is not the T123PL probe', d.get('run'), d.get('labels')); sys.exit(2)
oof, g = d['oof_per_label'][0], d['gold58_per_label'][0]
mean = lambda v, names: sum(v[T.index(n)] for n in names) / len(names)
p3, r9, oof12, g58, p3g = mean(oof, P3), mean(oof, R9), d['oof12'][0], d['G3'][0], mean(g, P3)
res = dict(G1=p3 >= BARS['G1'], G2=oof12 >= BARS['G2'], G3=r9 >= BARS['G3'], G4=g58 >= BARS['G4'])
ok = all(res.values())
pf = lambda b: 'PASS' if b else 'FAIL'
lines = [f'gate_pl4 | report {A.report} | prereg sha {PREREG_SHA[:8]}',
         f'  G1 P3 OOF mean  {p3:.6f} >= {BARS["G1"]:.6f} -> {pf(res["G1"])}',
         f'  G2 oof12        {oof12:.6f} >= {BARS["G2"]:.6f} -> {pf(res["G2"])}',
         f'  G3 R9 OOF mean  {r9:.6f} >= {BARS["G3"]:.6f} -> {pf(res["G3"])}',
         f'  G4 gold58 12    {g58:.6f} >= {BARS["G4"]:.6f} -> {pf(res["G4"])}',
         f'  report: gold58 P3 mean {p3g:.6f} (T123 ref .924135)',
         '  per-label OOF probe: ' + ' '.join(f'{n} {oof[i]:.4f}' for i, n in enumerate(T)),
         '  per-label gold58 probe: ' + ' '.join(f'{n} {g[i]:.4f}' for i, n in enumerate(T)),
         f'PL4 GATE: {pf(ok)}' + (' (SELFTEST on the T123 probe: no verdict written)' if A.selftest else '')]
print('\n'.join(lines))
if A.selftest:
    assert abs(p3 - FROZEN['p3']) < 5e-7 and abs(r9 - FROZEN['r9']) < 5e-7 and abs(oof12 - FROZEN['oof12']) < 5e-7 \
        and abs(g58 - FROZEN['g58']) < 5e-7, 'frozen mismatch'
    print('SELFTEST OK: frozen numbers reproduced')
    sys.exit(0)
open(f'{OUT}/gate_pl4.txt', 'w').write('\n'.join(lines) + '\n')
tmp = f'{OUT}/pl4.verdict.tmp'; open(tmp, 'w').write(pf(ok) + '\n'); os.replace(tmp, f'{OUT}/pl4.verdict')
sys.exit(0 if ok else 1)
