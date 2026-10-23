#!/usr/bin/env python3
"""t_k12_clean.py (box, CPU; writes only ~/k12_test/tclean; 2026-10-09, Claude): K12-OAI-CLEAN tests. The real D-OAI f0
raw best checkpoint + its OOF (vs set D f0's OOF) must PASS; planted copies must FAIL naming the cause: an OAI id string
in args, a pandas label table in args, an extra top-level key, a non-RSNA decode_probe key, an OOF with an oai id row,
an OOF whose id set differs from the reference. Last line: TK12CLEAN PASS|FAIL n/m"""
import copy, glob, os, subprocess, sys, torch, pandas as pd
H = os.path.expanduser('~'); T = f'{H}/k12_test/tclean'; os.makedirs(T, exist_ok=True)
PY = sys.executable; SC = f'{H}/k12_test/k12/box/k12_oai_clean.py'
ck = glob.glob(f'{H}/fleet_runs/f0_s1043_ttr_frac75oaiT12v00/f0_s1043_raw_*_best.pt')[0]
oof = f'{H}/fleet_runs/f0_s1043_ttr_frac75oaiT12v00/oof_f0_s1043_raw.csv'
ref = glob.glob(f'{H}/k10_test/oof/oof_f0_s1043f75_raw_cv*.csv')[0]
res = []
def run(spec, name, want_pass, needle=''):
    r = subprocess.run([PY, SC, '--out', f'{T}/{name}.json', '--ckpt', spec], capture_output=True, text=True)
    good = (r.returncode == 0) == want_pass and needle in r.stdout
    res.append(good); print(f'{"PASS" if good else "FAIL"}  {name}: rc {r.returncode} | {r.stdout.strip().splitlines()[0][:220]}')
run(f'{ck}={oof}={ref}', 'real_f0', True)
base = torch.load(ck, map_location='cpu', weights_only=False)
def planted(name, edit):
    c = copy.deepcopy(base); edit(c); p = f'{T}/{name}.pt'; torch.save(c, p); return p
run(planted('oai_id', lambda c: c['args'].update(note='trained with oai_9012345_R_v00')), 'neg_oai_id', False, 'OAI id pattern')
run(planted('table', lambda c: c['args'].update(lbl=pd.DataFrame({'a': range(3)}))), 'neg_pandas', False, 'pandas')
run(planted('key', lambda c: c.update(labels={'x': 1})), 'neg_extra_key', False, 'top-level keys')
run(planted('probe', lambda c: c['args']['decode_probe'].update({'oai_9000001_L': 'x'})), 'neg_probe', False, 'OAI id pattern')
run(planted('ids', lambda c: c['args'].update(ids=[f'1.2.3.{i}' for i in range(100)])), 'neg_id_list', False, '> 64 strings')
o = pd.read_csv(oof, dtype={'StudyInstanceUID': str}); o2 = o.copy(); o2.loc[0, 'StudyInstanceUID'] = 'oai_9000001_R'
o2.to_csv(f'{T}/oof_oai.csv', index=False); run(f'{ck}={T}/oof_oai.csv', 'neg_oof_oai', False, 'oai ids')
o.iloc[1:].to_csv(f'{T}/oof_short.csv', index=False); run(f'{ck}={T}/oof_short.csv={ref}', 'neg_oof_ids', False, 'differ')
for f in glob.glob(f'{T}/*.pt'):
    os.remove(f)
print(f'TK12CLEAN {"PASS" if all(res) else "FAIL"} {sum(res)}/{len(res)}')
