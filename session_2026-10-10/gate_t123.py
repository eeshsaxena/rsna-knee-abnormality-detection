#!/usr/bin/env python
# gate_t123.py (2026-10-10, Claude): applies the PROBE3_PREREG.md (sha 3534b34b...) bars to the report written by
# score_oai_probe2.py --tier T12 --report_only for the T123v00 probe. Writes ~/oai_build/out/t123.verdict (PASS / FAIL)
# and ~/oai_build/out/gate_t123.txt. Exit 0 PASS, 1 FAIL, 2 integrity, 3 not ready.
#   --selftest: run the arithmetic on the T12 probe's own report (must reproduce the frozen numbers; no verdict written)
import os, sys, json, argparse, hashlib
H = os.path.expanduser
ap = argparse.ArgumentParser()
ap.add_argument('--report', default=H('~/oai_build/out/gate2_REPORT_T12_f0_s1043_ttr_frac75oaiT123v00.json'))
ap.add_argument('--selftest', action='store_true')
A = ap.parse_args()
PREREG_SHA = '3534b34bdcaff90b82e190e309ab5d22ddd0a2ef19969cf2536da3daa6241ba2'
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
T3 = ['ACL', 'Medial Meniscus', 'Lateral Meniscus']
S12 = ['PF OA', 'Lateral OA', 'Synovitis', 'Medial OA', 'Effusion', "Baker's"]
FROZEN = dict(t3=0.903992, s12=0.886364, oof12=0.887764)
BARS = dict(G1=FROZEN['t3'] + 0.002, G2=FROZEN['oof12'], G3=FROZEN['s12'] - 0.002, G4=0.9113)
OUT = H('~/oai_build/out')

pp = H('~/oai_build/PROBE3_PREREG.md')
if hashlib.sha256(open(pp, 'rb').read()).hexdigest() != PREREG_SHA:
    print('INTEGRITY: PROBE3_PREREG.md changed'); sys.exit(2)
if A.selftest:
    A.report = H('~/oai_build/out/gate2_T12_f0_s1043_ttr_frac75oaiT12v00.json')
if not os.path.exists(A.report):
    print('NOT READY: no report', A.report); sys.exit(3)
d = json.load(open(A.report))
if not A.selftest:
    if d.get('labels') != 'labels_frac75oai_T123v00.csv' or not d['run'].endswith('f0_s1043_ttr_frac75oaiT123v00'):
        print('INTEGRITY: report is not the T123v00 probe', d.get('run'), d.get('labels')); sys.exit(2)
oof = d['oof_per_label'][0]
mean = lambda names: sum(oof[T.index(n)] for n in names) / len(names)
t3, s12, oof12, g58 = mean(T3), mean(S12), d['oof12'][0], d['G3'][0]
res = dict(G1=(t3, t3 >= BARS['G1']), G2=(oof12, oof12 >= BARS['G2']), G3=(s12, s12 >= BARS['G3']),
           G4=(g58, g58 >= BARS['G4']))
ok = all(v[1] for v in res.values())
lines = [f'gate_t123 | report {A.report} | prereg sha {PREREG_SHA[:8]}',
         f'  G1 T3 OOF mean {t3:.6f} >= {BARS["G1"]:.6f} -> {"PASS" if res["G1"][1] else "FAIL"}',
         f'  G2 oof12      {oof12:.6f} >= {BARS["G2"]:.6f} -> {"PASS" if res["G2"][1] else "FAIL"}',
         f'  G3 S12 OOF mean {s12:.6f} >= {BARS["G3"]:.6f} -> {"PASS" if res["G3"][1] else "FAIL"}',
         f'  G4 gold58 12  {g58:.6f} >= {BARS["G4"]:.4f} -> {"PASS" if res["G4"][1] else "FAIL"}',
         '  per-label OOF probe: ' + ' '.join(f'{n} {oof[i]:.4f}' for i, n in enumerate(T)),
         f'T123 GATE: {"PASS" if ok else "FAIL"}' + (' (SELFTEST on the T12 probe: no verdict written)' if A.selftest else '')]
print('\n'.join(lines))
if A.selftest:
    assert abs(t3 - FROZEN['t3']) < 5e-7 and abs(s12 - FROZEN['s12']) < 5e-7 and abs(oof12 - FROZEN['oof12']) < 5e-7, 'frozen mismatch'
    print('SELFTEST OK: frozen numbers reproduced')
    sys.exit(0)
open(f'{OUT}/gate_t123.txt', 'w').write('\n'.join(lines) + '\n')
tmp = f'{OUT}/t123.verdict.tmp'; open(tmp, 'w').write(('PASS' if ok else 'FAIL') + '\n'); os.replace(tmp, f'{OUT}/t123.verdict')
sys.exit(0 if ok else 1)
