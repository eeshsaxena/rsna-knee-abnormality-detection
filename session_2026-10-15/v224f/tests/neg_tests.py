# Negative tests for v224f_check.py: each mutated copy of a real commit output must FAIL the gate (and the unmutated ones PASS).
import os, sys, shutil, subprocess, tempfile, json
import pandas as pd, numpy as np
D = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-15/v224f'
CHK = D + '/v224f_check.py'
CLEAN, CTL = D + '/out_v224f20_v1', D + '/out_v224fctl_v1'
REFV = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/v224/out_v224_v1'
REFF = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-13/hr7rr/out_hfr7rr_v1'


def gate(d, *extra):
    r = subprocess.run([sys.executable, CHK, d, '--wf', '0.2', '--ref-v224', REFV, '--ref-fleet', REFF, *extra],
                       capture_output=True, text=True, encoding='utf-8')
    last = [l for l in r.stdout.splitlines() if l.startswith('V224F_CHECK ') and ('PASS' in l or 'FAIL' in l)]
    fails = [l.strip() for l in r.stdout.splitlines() if l.strip().startswith('FAIL')]
    return r.returncode, (last[-1] if last else r.stdout[-500:] + r.stderr[-500:]), fails


def mutate(src, fn):
    t = tempfile.mkdtemp(prefix='v224fneg_')
    shutil.rmtree(t); shutil.copytree(src, t)
    fn(t)
    return t


def bump(path, col, row=0, dv=0.01):
    s = pd.read_csv(path, dtype={'StudyInstanceUID': str})
    s.loc[row, col] = float(s.loc[row, col]) + dv if float(s.loc[row, col]) + dv <= 1 else float(s.loc[row, col]) - dv
    s.to_csv(path, index=False)


cases = [
    ('clean as clean', CLEAN, [], 0),
    ('control as control', CTL, ['--control'], 0),
    ('control output, clean gate', CTL, [], 1),
    ('clean output, control gate', CLEAN, ['--control'], 1),
    ('clean, wrong w_f .25', CLEAN, ['--wf', '0.25'], 1),
    ('clean, submission cell +.01', mutate(CLEAN, lambda t: bump(t + '/submission.csv', 'ACL')), [], 1),
    ('clean, submission_V224 cell +.01', mutate(CLEAN, lambda t: bump(t + '/submission_V224.csv', 'MCL')), [], 1),
    ('clean, fleet solo cell +.01', mutate(CLEAN, lambda t: bump(t + '/submission_fleetsolo.csv', 'Effusion', dv=0.001)), [], 1),
    ('control, Synovitis not blank', mutate(CTL, lambda t: bump(t + '/submission.csv', 'Synovitis')), ['--control'], 1),
    ('control, ACL changed by 1e-6', mutate(CTL, lambda t: bump(t + '/submission.csv', 'ACL', dv=1e-6)), ['--control'], 1),
    ('clean, fleet receipt hard_stopped', mutate(CLEAN, lambda t: json.dump(dict(json.load(open(t + '/fleet_receipt.json')),
                                                hard_stopped=[True, False]), open(t + '/fleet_receipt.json', 'w'))), [], 1),
]
ok_all = True
for name, d, extra, want in cases:
    # --wf in extra overrides the default (argparse takes the last)
    rc, last, fails = gate(d, *extra)
    good = (want is None) or (rc == want)
    ok_all &= good
    print(f'NEG {name:<36} rc={rc} want={want} {"OK" if good else "WRONG"} | {last} | first fails: {fails[:2]}')
print('NEG', 'PASS' if ok_all else 'FAIL')
