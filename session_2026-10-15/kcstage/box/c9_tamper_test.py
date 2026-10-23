#!/usr/bin/env python
# c9_tamper_test.py (10-15, Claude subagent, kcstage; KCS2 fail-before / pass-after of the C9 TOOL). Box CPU only.
# A serve-side side-table drift (the table says the other knee for one strong-zone study, internally consistent: mean_x negated,
# side and s<k>_flip re-derived by kc_common) is the train / serve skew KCS2 describes. Three runs of c9_served.py on that study:
#   T1 --path header (= the A6-frozen kc_prob_parity.py as it stands, kc_worker's own header side): it cannot see the drift -> PASS
#      on probabilities (this is the fail-before: the pre-fix C9 passes a mis-fed serve path)
#   T2 --path served with the drifted table: the shipped serve path is mis-fed -> max |dp| > 1e-3 and side != ref -> FAIL (detected)
#   T3 --path served with the fresh side-kernel rows: probabilities PASS and side == ref (pass-after on the true table)
# Prints KC_C9_TAMPER PASS n/3.
#   python c9_tamper_test.py --tool kcstage_1015/tools/c9_served.py --members a.pt b.pt --workdir kcstage_1015/c9tamper [common args]
import os, sys, json, argparse, subprocess
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--tool', required=True); ap.add_argument('--code', required=True); ap.add_argument('--serve', required=True)
ap.add_argument('--kpp', required=True); ap.add_argument('--members', nargs='+', required=True); ap.add_argument('--store', required=True)
ap.add_argument('--dicom', required=True); ap.add_argument('--side-ref', default=os.path.expanduser('~/kc_store/side_train.csv'))
ap.add_argument('--workdir', required=True); ap.add_argument('--threads', type=int, default=6)
A = ap.parse_args()
os.makedirs(A.workdir, exist_ok=True)
sys.path.insert(0, os.path.abspath(A.code))
import kc_common as kc
ref = pd.read_csv(A.side_ref, dtype={'StudyInstanceUID': str, 'tag': str})
ROOT = A.dicom if os.path.exists(f'{A.dicom}/train_series.csv') else f'{A.dicom}/rsna-knee-abnormality-detection'
have = set(os.listdir(f'{ROOT}/train_series'))
ip = kc.VIEW_NAMES.index('sag_pd')
sid = None
for s in np.random.default_rng(1015).permutation(sorted(have)):              # a strong-zone, single-canon study with every view
    r = ref[ref['StudyInstanceUID'] == s]
    if len(r) == 1 and r.iloc[0]['rule'] == 'mean_x' and r.iloc[0]['serve'] == 'single' and os.path.exists(f'{A.store}/{s}.npz'):
        with np.load(f'{A.store}/{s}.npz') as z:
            if f'v{ip}_blob' in z.files:
                sid = s
                break
assert sid, 'no strong-zone study found'
row = ref[ref['StudyInstanceUID'] == sid].copy()
row['mean_x'] = -row['mean_x']
side, rule = kc.side_rule(float(row.iloc[0]['mean_x']), row.iloc[0]['tag'] if isinstance(row.iloc[0]['tag'], str) else '')
assert rule == 'mean_x' and side != ref[ref['StudyInstanceUID'] == sid].iloc[0]['side']
row['side'] = side
SLOTK = {'sagfs': 0, 'sagpd': 1, 'corfs': 2, 'ax': 4}
geom = {}
for b in kc.GEOM_DEFAULT:
    k = SLOTK.get(b)
    sg = row.iloc[0].get(f's{k}_sign') if k is not None else None
    mg = row.iloc[0].get(f's{k}_mag') if k is not None else None
    ok = k is not None and pd.notna(sg) and pd.notna(mg) and float(sg) != 0
    geom[b] = float(sg) * float(mg) if ok else kc.GEOM_DEFAULT[b]
fl = kc.block_flips(side, geom)
for b, k in SLOTK.items():
    if pd.notna(row.iloc[0].get(f's{k}_flip')):
        row[f's{k}_flip'] = int(bool(fl[b]))
tam = os.path.join(A.workdir, 'side_drifted.csv')
row.to_csv(tam, index=False)
common = ['--code', A.code, '--serve', A.serve, '--kpp', A.kpp, '--members', *A.members, '--store', A.store, '--dicom', A.dicom,
          '--threads', str(A.threads), '--side-ref', A.side_ref, '--sids', sid]
R = []


def run(name, extra):
    out = os.path.join(A.workdir, name + '.json')
    p = subprocess.run([sys.executable, '-u', A.tool] + common + extra + ['--out', out], capture_output=True, text=True)
    print('   ', name, (p.stdout.strip().splitlines() or ['(no output)'])[-1], flush=True)
    return json.load(open(out))


print('KC_C9_TAMPER study %s: table side %s -> drifted %s' % (sid, ref[ref['StudyInstanceUID'] == sid].iloc[0]['side'], side), flush=True)
t1 = run('T1_header_drifted', ['--path', 'header', '--side-table', tam])
ok = t1.get('error') is None and t1.get('prob_pass') is True and t1.get('side_equal_ref') is False
R.append(ok)
print('  %s T1 pre-fix C9 (kc_worker header path) does NOT see the drifted serve table: probabilities PASS (max |dp| %s), side != ref'
      % ('OK  ' if ok else 'FAIL', t1.get('max_dp')), flush=True)
t2 = run('T2_served_drifted', ['--path', 'served', '--side-table', tam])
ok = t2.get('error') is None and t2.get('prob_pass') is False and t2.get('max_dp', 0) > 1e-3 and t2.get('PASS') is False \
    and t2.get('substitution_live') is True
R.append(ok)
print('  %s T2 served C9 with the drifted table: max |dp| %s > 1e-3 -> FAIL (the skew is detected)' % ('OK  ' if ok else 'FAIL', t2.get('max_dp')),
      flush=True)
t3 = run('T3_served_fresh', ['--path', 'served'])
ok = t3.get('error') is None and t3.get('prob_pass') is True and t3.get('side_equal_ref') is True and t3.get('side_source') == 'fresh' \
    and t3.get('substitution_live') is True
R.append(ok)
print('  %s T3 served C9 with the fresh side-kernel rows: probabilities PASS (max |dp| %s), side == ref' % ('OK  ' if ok else 'FAIL',
                                                                                                          t3.get('max_dp')), flush=True)
print('KC_C9_TAMPER %s %d/%d' % ('PASS' if all(R) else 'FAIL', sum(R), len(R)), flush=True)
sys.exit(0 if all(R) else 1)
