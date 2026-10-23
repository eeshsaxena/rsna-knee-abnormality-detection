# _sim_setup.py (laptop, TEST ONLY, 2026-10-09): k11/_sim = C + D + E (hard links of the real K10 picks) + a SIMULATED E2
# (set E fold F weights, ckpt args seed 1046, renamed _s1046_). members.json run paths carry '/k10_test/sim/' so every build
# from it is marked DO_NOT_PUSH_SIMULATION. Never used for a real build.
import json, os, sys, hashlib, copy, torch
SP = sys.argv[1]; K10 = f'{SP}/k10'; S = f'{SP}/k11/_sim'
m10 = json.load(open(f'{K10}/members.json'))
out = []
for r in m10['members']:
    dst = f'{S}/ckpt/{r["file"]}'
    if not os.path.exists(dst):
        os.link(f'{K10}/ckpt/{r["file"]}', dst)
    out.append(dict(r))
for r in m10['members']:
    if r['set'] != 'E':
        continue
    ck = torch.load(f'{K10}/ckpt/{r["file"]}', map_location='cpu', weights_only=False)
    ck['args']['seed'] = 1046
    f = r['file'].replace('_s1045_', '_s1046_')
    torch.save(ck, f'{S}/ckpt/{f}')
    x = copy.deepcopy(r)
    x.update(set='E2', seed=1046, file=f, sha256=hashlib.sha256(open(f'{S}/ckpt/{f}', 'rb').read()).hexdigest(),
             run=f'/home/training/k10_test/sim/root/f{r["fold"]}_s1046_ttr288frac75', source='modal',
             modal_receipt='SIM', vast_receipt=None, suffix='ttr288frac75')
    out.append(x)
m = dict(m10); m.update(members=out, k11=True, generated='SIM', interim=False, do_not_submit=False, e2_ship=None,
                        set_e2=dict(status='SIM'))
json.dump(m, open(f'{S}/members.json', 'w'), indent=1)
print('SIM members', len(out))
