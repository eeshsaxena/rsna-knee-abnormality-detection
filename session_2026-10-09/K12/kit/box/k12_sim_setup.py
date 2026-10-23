#!/usr/bin/env python3
"""k12_sim_setup.py (box, CPU; TEST ONLY, never part of a real build; 2026-10-09, Claude). Simulated D-OAI folds for the
stub build + A0 while the real D-OAI folds 1-4 train: fold F (F in --folds, default 1,2,3,4) = set D's fold-F pick (the
FN11 member, ~/k10_test/ckpt) with ckpt args train_labels -> labels_frac75oai_T12v00.csv (weights untouched), saved as
~/k12_test/sim/root/f{F}_s1043_ttr_frac75oaiT12v00/<D file name>; its OOF = set D's OOF copy; its gold prediction = set D's
cached prediction copied to ~/k12_test/sim/preds with a sidecar = the simulated ckpt sha256. Writes
~/k12_test/sim/members_DO_sim.json (records marked simulated). Real runs are only read. Writes only ~/k12_test/sim.
  ~/miniconda3/envs/kmimic/bin/python k12_sim_setup.py [--folds 1,2,3,4]
"""
import argparse, glob, hashlib, json, os, shutil, torch

H = os.path.expanduser('~')
S = f'{H}/k12_test/sim'
READ_ONLY = [f'{H}/novel/k9gold/preds', f'{H}/novel/k8gold/preds', f'{H}/novel/gold74_preds']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser(); ap.add_argument('--folds', default='1,2,3,4'); a = ap.parse_args()
base = json.load(open(f'{H}/k11_test/members.json'))
D = {r['fold']: r for r in base['members'] if r['set'] == 'D'}
os.makedirs(f'{S}/preds', exist_ok=True)
out = []
for F in [int(x) for x in a.folds.split(',') if x.strip()]:
    r = D[F]
    run = f'f{F}_s1043_ttr_frac75oaiT12v00'
    d = f'{S}/root/{run}'
    os.makedirs(d, exist_ok=True)
    ck = torch.load(r['copy'], map_location='cpu', weights_only=False)
    ck['args']['train_labels'] = 'labels_frac75oai_T12v00.csv'
    p = f'{d}/{r["file"]}'
    torch.save(ck, p)
    s = sha(p)
    dr = os.path.basename(os.path.dirname(r['src']))                      # the D run name (gold cache key)
    q = next(f'{x}/{dr}__raw__A.csv' for x in READ_ONLY if os.path.isfile(f'{x}/{dr}__raw__A.csv'))
    shutil.copy2(q, f'{S}/preds/{run}__raw__A.csv'); open(f'{S}/preds/{run}__raw__A.csv.sha256', 'w').write(s + '\n')
    x = dict(r)
    x.update(set='DO', simulated=True, src=p, copy=p, sha256=s, train_labels='labels_frac75oai_T12v00.csv',
             suffix='ttr_frac75oaiT12v00', run=f'{S}/root/{run}', key_tag='f75oai', source='SIMULATED (set D weights)',
             kaggle_receipt=None)
    out.append(x)
    print(f'SIM DO f{F}: {p} sha256 {s[:16]} (D {r["sha256"][:16]}), OOF {r["oof"]}, gold pred <- {q}')
json.dump(dict(members=out, simulated=True), open(f'{S}/members_DO_sim.json', 'w'), indent=1)
print(f'K12SIM OK {len(out)} simulated DO fold(s) -> {S}/members_DO_sim.json')
