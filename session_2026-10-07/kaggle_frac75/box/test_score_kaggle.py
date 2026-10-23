# test_score_kaggle.py (kaggle_frac75): CPU tests of ~/kt_frac75/box/score_frac75_kaggle.py. Writes ONLY under
# ~/kt_frac75/test (KT_F75_ROOT redirects the scorer's out/ and gold74_preds/); reads J1 (~/fleet_runs/f0_s1043_ttr_mix50),
# ~/novel/frac75 and ~/queue/gates/frac75_gate.sh. Synthetic "Kaggle probe" = J1's best raw ckpt + OOF + log cut to 17
# epochs, args re-tagged epochs 17 / save_epochs 15-17 / train_labels frac75 (what the Kaggle run's args will look like).
import os, sys, json, glob, shutil, subprocess, importlib.util, hashlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

H = os.path.expanduser('~')
KT = f'{H}/kt_frac75'
T = f'{KT}/test'
ROOT = f'{T}/root'
FR = f'{T}/fleet_runs'
PY = f'{H}/miniconda3/envs/rsna/bin/python'
SC = f'{KT}/box/score_frac75_kaggle.py'
J1 = f'{H}/fleet_runs/f0_s1043_ttr_mix50'
J1G = f'{H}/jump2/gold74_preds/f0_s1043_ttr_mix50__raw.csv'
FAKEPASS = f'{H}/novel/frac75/dry/fakepass_gold_preds.csv'
RES = {}


def check(name, cond, info=''):
    RES[name] = bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{info}]' if info else ''), flush=True)


shutil.rmtree(T, ignore_errors=True)
os.makedirs(f'{ROOT}/out')
os.makedirs(FR)
ck = glob.glob(f'{J1}/f0_s1043_raw_*_best.pt')[0]
c = torch.load(ck, map_location='cpu', weights_only=False)
j1log = open(f'{J1}.log').read().splitlines()
closing = json.loads(j1log[-1])


def make_run(name, E=17, save=None, extra=None):
    d = f'{FR}/{name}'
    os.makedirs(f'{d}/epochs')
    a = dict(c['args'], epochs=E, save_epochs=save or f'{E - 2}-{E}', train_labels='labels_frac75_logit.csv')
    a.update(extra or {})
    torch.save(dict(c, args=a), f'{d}/{os.path.basename(ck)}')
    shutil.copyfile(f'{J1}/oof_f0_s1043_raw.csv', f'{d}/oof_f0_s1043_raw.csv')
    for k in range(E - 2, E + 1):
        shutil.copyfile(f'{J1}/epochs/oof_f0_s1043_raw_ep{k}.csv', f'{d}/epochs/oof_f0_s1043_raw_ep{k}.csv')
    keep, cut = [], f'ep {E - 1} ema |'
    for l in j1log:
        if l.startswith('{'):
            continue
        keep.append(l)
        if l.startswith(cut):
            break
    keep.append(json.dumps(closing))
    open(f'{d}.log', 'w').write('\n'.join(keep) + '\n')
    return d


def run(args, env_extra=None, timeout=3600):
    env = dict(os.environ, KT_F75_ROOT=ROOT, F75_KAGGLE_RECEIPT=f'{T}/no_receipt.json')
    env.update(env_extra or {})
    r = subprocess.run(['nice', '-n', '19', PY, SC] + args, capture_output=True, text=True, env=env, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


RK = make_run('f0_s1043_ttr_frac75')

# 1 the ORIGINAL scorer logic (only ROOT redirected, no pairing patch) refuses a Kaggle-shaped probe
spec = importlib.util.spec_from_file_location('sk', SC)
sk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sk)                                  # defines functions only (main() is guarded)
orig = open(sk.ORIG, 'rb').read()
check('pinned sha256 == the live ~/novel/frac75/score_frac75.py', hashlib.sha256(orig).hexdigest() == sk.ORIG_SHA256)
src_root_only = orig.decode().replace(sk.PATCH_ROOT_OLD, sk.PATCH_ROOT_NEW)
open(f'{T}/orig_rootonly.py', 'w').write(src_root_only)
r = subprocess.run([PY, f'{T}/orig_rootonly.py', '--run', RK, '--gold_preds', J1G, '--tag', 'orig_on_kaggle'],
                   capture_output=True, text=True, env=dict(os.environ, KT_F75_ROOT=ROOT))
check('original scorer on a Kaggle-shaped probe: exit 2, refuses epochs/save_epochs', r.returncode == 2 and
      "beyond train_labels: ['epochs', 'save_epochs']" in r.stdout, r.stdout.strip().splitlines()[-1][:160])

# 2 J1 as a null probe through the wrapper (E 24): same numbers as the original selftest, FAIL (G2)
rc, out = run(['--run', J1, '--expect_labels', 'labels_mix50_logit.csv', '--gold_preds', J1G, '--tag', 'j1_null'])
j = json.load(open(f'{ROOT}/out/gate_j1_null.json'))
check('wrapper on J1 (null): exit 1, G1 .8803 G2 .8936 G3 .9092 (== original selftest), read FAIL',
      rc == 1 and round(j['G1'], 4) == 0.8803 and round(j['G2'], 4) == 0.8936 and round(j['G3'], 4) == 0.9092 and
      j['kaggle_read'] == 'FAIL', (rc, j.get('kaggle_read'), out.strip().splitlines()[-1][:120]))
check('wrapper on J1: schedule-end read == 0 deltas, internal control 0', j['schedule_end_last3'] and
      all(abs(v['d']) < 1e-12 for v in j['schedule_end_last3'].values()) and abs(j['gold74_internal_control']['frac_specific']) < 1e-12)

# 3 the Kaggle-shaped probe (E 17): pairing accepted with exactly epochs/save_epochs, FAIL (G2) like J1, json for the queue
rc, out = run(['--run', RK, '--gold_preds', J1G])
print(out[-3500:], flush=True)
j = json.load(open(f'{ROOT}/out/gate_f0_s1043_ttr_frac75.json'))
check('wrapper on the Kaggle-shaped probe: exit 1, pairing allowed {epochs, save_epochs}, read FAIL, run/dry/pass_ for the queue',
      rc == 1 and set(j['kaggle_diff']) == {'epochs', 'save_epochs'} and j['kaggle_read'] == 'FAIL' and j['run'] == RK and
      j['dry'] is False and j['pass_'] is False and j['fails'] == ['G2'] and j['kaggle_E'] == 17, (rc, j.get('kaggle_diff')))
check('wrapper txt has the registered verdict + the addendum block', 'FRAC75 GATE: FAIL (G2)' in out and
      'KAGGLE ADDENDUM READ: FAIL' in out and 'schedule-end read' in out and 'matched schedule fraction' in out)

# 4 synthetic PASS (J1 + Fracture preds pulled toward the experts; dry, so it can never unlock anything)
rc, out = run(['--run', RK, '--gold_preds', FAKEPASS, '--dry', '--tag', 'fakepass'])
j = json.load(open(f'{ROOT}/out/gate_DRY_fakepass.json'))
check('wrapper synthetic PASS (dry): exit 0, read PASS, pass_ False (dry never unlocks)', rc == 0 and
      j['kaggle_read'] == 'PASS' and j['pass_'] is False and j['dry'] is True, (rc, j.get('kaggle_read'), j.get('G2')))

# 5 integrity refusals
for nm, kw, frag in (('amp', dict(extra=dict(amp='fp32')), "beyond train_labels: ['amp']"),
                     ('seed', dict(extra=dict(seed=1044)), 'fold/seed/model'),
                     ('save', dict(save='6-17'), 'is not the Kaggle notebook rule 15-17'),
                     ('short', dict(E=12), 'outside [14, 24]')):
    d = make_run(f'f0_s1043_ttr_frac75_{nm}', **kw)
    rc, out = run(['--run', d, '--gold_preds', J1G])
    check(f'wrapper refuses a probe with bad {nm}: exit 2', rc == 2 and frag in out, out.strip().splitlines()[-1][:160])
bad = f'{T}/score_frac75_modified.py'
open(bad, 'wb').write(orig.replace(b"BAR = {'G1': 0.8774", b"BAR = {'G1': 0.8700"))
rc, out = run(['--run', RK, '--gold_preds', J1G], dict(F75_ORIG_SCORER=bad))
check('wrapper refuses a changed pre-registered scorer (sha256 pin): exit 2', rc == 2 and 'pinned' in out, out.strip()[:160])

# 6 the addendum read, unit level
B = {'G1': 0.8774, 'G2': 0.912, 'G3': 0.9062}
kr = sk.kaggle_read
check('kaggle_read: all pass -> PASS', kr(dict(G1=.8800, G2=.9150, G3=.9080), B, 17)[0] == 'PASS')
check('kaggle_read: G1 short by .0014 only -> INCONCLUSIVE-NARROW', kr(dict(G1=.8760, G2=.9150, G3=.9080), B, 17)[0] == 'INCONCLUSIVE-NARROW')
check('kaggle_read: G2 short by .0015 + G3 short by .0012 -> INCONCLUSIVE-NARROW',
      kr(dict(G1=.8800, G2=.9105, G3=.9050), B, 17)[0] == 'INCONCLUSIVE-NARROW')
check('kaggle_read: G2 short by .0184 (J1-like) -> FAIL', kr(dict(G1=.8803, G2=.8936, G3=.9092), B, 17)[0] == 'FAIL')
check('kaggle_read: one narrow + one far -> FAIL', kr(dict(G1=.8760, G2=.9000, G3=.9080), B, 17)[0] == 'FAIL')
check('kaggle_read: exactly .003 short -> FAIL (narrow is strictly < .003)', kr(dict(G1=.8744, G2=.915, G3=.908), B, 17)[0] == 'FAIL')
check('kaggle_read: NaN gate -> FAIL', kr(dict(G1=.8800, G2=float('nan'), G3=.9080), B, 17)[0] == 'FAIL')
check('kaggle_read: E 15 PASS carries the weaker-read note', 'E 15 < 16' in kr(dict(G1=.88, G2=.915, G3=.908), B, 15)[1])

# 7 end to end through the queue adapter (real gold_infer5 on CPU into the test root, ~10 min)
env = dict(os.environ, KT_F75_ROOT=ROOT, F75_SCORER=SC, F75_RUN=RK, F75_JSON=f'{ROOT}/out/gate_f0_s1043_ttr_frac75.json',
           F75_VERDICT=f'{T}/frac75.verdict', F75_KAGGLE_RECEIPT=f'{T}/no_receipt.json')
os.remove(f'{ROOT}/out/gate_f0_s1043_ttr_frac75.json')
r = subprocess.run(['bash', f'{H}/queue/gates/frac75_gate.sh'], capture_output=True, text=True, env=env, timeout=5400)
print(r.stdout[-3000:], r.stderr[-1500:], flush=True)
v = open(f'{T}/frac75.verdict').read() if os.path.exists(f'{T}/frac75.verdict') else ''
gp = f'{ROOT}/gold74_preds/f0_s1043_ttr_frac75__raw__A.csv'
import pandas as pd, numpy as np
dmax = float(np.abs(pd.read_csv(gp, index_col=0).sort_index().values - pd.read_csv(J1G, index_col=0).sort_index().values).max()) \
    if os.path.exists(gp) else -1
check('frac75_gate.sh + F75_SCORER=wrapper: gold_infer5 ran into the test root, preds == J1 (same weights), verdict FAIL G2',
      r.returncode == 0 and v.startswith('FAIL frac75 probe fails G2') and 0 <= dmax < 1e-4, (r.returncode, v.strip()[:100], dmax))
check('nothing written outside ~/kt_frac75/test by these tests', not os.path.exists(f'{H}/novel/frac75/gold74_preds') and
      not glob.glob(f'{H}/novel/frac75/out/gate_*kaggle*') and not glob.glob(f'{H}/novel/frac75/out/gate_j1_null*') and
      not os.path.exists(f'{H}/novel/frac75/out/gate_f0_s1043_ttr_frac75.json'))
print(f'\nSCORE-KAGGLE TESTS: {sum(RES.values())}/{len(RES)} PASS', flush=True)
sys.exit(0 if all(RES.values()) else 1)
