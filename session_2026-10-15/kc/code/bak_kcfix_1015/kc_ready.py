# kc_ready.py -- writes ~/kc/KC_READY.verdict, the gate of the KC1 f0 queue line (BOX_PLAN_1015 sec 3.3 as TRIMMED by the
# adopted BOX_PLAN_CRITIQUE_1015). Called by ~/kc/kc_ready.sh. Box CPU, reads only; writes only the verdict file.
# PASS iff ALL of (C3 serve parity, C5 dread labels and C9 are NOT in KC_READY; C3 must PASS before the G3 pack):
#   SIDE   ~/kc_store/side_train.csv: 4,407 rows, v2 per-slot geometry columns (or --allow_geom_const with a const store)
#   STORE  ~/kc_store/store_summary.json PASS: coverage >= 99 %, U <= 2 %, contract == the installed kc_common, side table
#          sha == side_train.csv, test458 subset array-equal; and >= 99 % of 4,407 npz present
#   C2B    ~/kc/contact/C2B_OK first token PASS (main, after the visual canon check of the contact sheets)
#   C4     ~/kc/smoke/C4_RECEIPT.txt 'C4 PASS' with md5s == the code2 copies of fleet_train_kc.py / kc_common.py / kc_g1.py
#   C7     ~/kc/gate/refs.json with code_md5 == ~/kc/code/kc_oof_kit.py
#   C8     code2 copies == ~/kc/code copies (md5)
#   M2     Amendment 6 sha256 (64 hex; --a6_sha or ~/kc/A6_SHA256.txt)
#   M4     code2 trainer md5 == 8d12515f (shimmed) and ~/kc/regress/REGRESS.txt ends 'REGRESS PASS trainer_md5=8d12515f...'
# Usage: kc_ready.sh [--status] [--a6_sha SHA] [--allow_geom_const] | kc_ready.sh --fail "<reason>"
import os, sys, json, hashlib, argparse, time, glob

H = os.path.expanduser('~')
CODE = os.path.join(H, 'fast_build', 'code2')
TRAINER = os.path.join(CODE, 'fleet_' + 'train' + '.py')       # path built in code (never on a command line)
SHIM_MD5 = '8d12515f237d204ad63ca93bd74b36b9'
OUT = os.path.join(H, 'kc', 'KC_READY.verdict')
ap = argparse.ArgumentParser()
ap.add_argument('--status', action='store_true'); ap.add_argument('--a6_sha', default='')
ap.add_argument('--allow_geom_const', action='store_true'); ap.add_argument('--fail', default='')
ap.add_argument('--force', action='store_true'); ap.add_argument('--out', default=OUT)
A = ap.parse_args()
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
sha256 = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def have_verdict(p):
    if not os.path.exists(p):
        return None
    for ln in open(p):
        s = ln.strip()
        if s.startswith('PASS') or s.startswith('FAIL'):
            return s
    return None


def write(lines):
    old = have_verdict(A.out)
    if old and not A.force:
        print(f'REFUSED: {A.out} already has a verdict line ({old[:80]}); --force to replace'); sys.exit(2)
    if os.path.exists(A.out):
        os.replace(A.out, A.out + time.strftime('.prev_%m%d_%H%M%S'))
    tmp = A.out + '.tmp'
    open(tmp, 'w').write('\n'.join(lines) + '\n')
    os.replace(tmp, A.out)
    print('wrote', A.out, '|', lines[0])


if A.fail:
    write([f'FAIL kc_ready {A.fail}', f'# written by kc_ready.py --fail {time.strftime("%Y-%m-%d %H:%M:%S")}'])
    sys.exit(0)

C, info = {}, {}
# SIDE
side = os.path.join(H, 'kc_store', 'side_train.csv')
if os.path.exists(side):
    import pandas as pd
    d = pd.read_csv(side, dtype={'StudyInstanceUID': str})
    v2 = all(f'g_{b}' in d.columns for b in ('sagfs', 'sagpd', 'corfs', 'ax')) or (           # kc_side_kernel_v2 schema
        all(f's{k}_{c}' in d.columns for k in (0, 1, 2, 4) for c in ('sign', 'mag', 'flip'))     # or main's kernel schema
        and {'weak', 'geo_ok'} <= set(d.columns))
    info['side'] = dict(rows=len(d), v2=v2, sha=sha256(side)[:12], u=int((d.side == 'U').sum()))
    C['SIDE'] = len(d) == 4407 and (v2 or A.allow_geom_const)
else:
    C['SIDE'] = False; info['side'] = 'missing ~/kc_store/side_train.csv'
# STORE
summ_p = os.path.join(H, 'kc_store', 'store_summary.json')
sys.path.insert(0, CODE)
sys.dont_write_bytecode = True
try:
    import kc_common as kc
    contract = kc.CONTRACT_HASH
except Exception as e:
    contract = None; info['kc_common_import'] = str(e)
if os.path.exists(summ_p):
    s = json.load(open(summ_p))
    n_npz = len(glob.glob(os.path.join(H, 'kc_store', 'store', '*.npz')))
    info['store'] = dict(n_ok=s.get('n_ok'), n_exists=s.get('n_exists'), n_err=s.get('n_error'), u_rate=s.get('u_rate'),
                         weak=s.get('weak'), geom_ok=s.get('geom_ok'), geom=s.get('geom_mode'), contract=s.get('contract_hash'),
                         checks=s.get('checks'), gb=s.get('bytes_total_gb'), n_npz=n_npz)
    side_ok = os.path.exists(side) and s.get('side_table_sha256') == sha256(side)
    C['STORE'] = bool(s.get('PASS') and s.get('checks', {}).get('subset_equal') and s.get('contract_hash') == contract
                      and side_ok and n_npz >= 0.99 * 4407 and (s.get('u_rate', 1) <= 0.02)
                      and (s.get('geom_mode') == 'table' or A.allow_geom_const))
else:
    C['STORE'] = False; info['store'] = 'missing ~/kc_store/store_summary.json'
# C2B
c2b = os.path.join(H, 'kc', 'contact', 'C2B_OK')
C['C2B'] = os.path.exists(c2b) and open(c2b).read().split()[:1] == ['PASS']
# C8 + C4
files = ('fleet_train_kc.py', 'kc_common.py', 'kc_g1.py')
c8 = {f: (md5(os.path.join(CODE, f)) if os.path.exists(os.path.join(CODE, f)) else None) for f in files}
src = {f: md5(os.path.join(H, 'kc', 'code', f)) for f in files}
C['C8'] = all(c8[f] == src[f] for f in files)
rec4 = os.path.join(H, 'kc', 'smoke', 'C4_RECEIPT.txt')
if os.path.exists(rec4):
    t = open(rec4).read()
    C['C4'] = t.startswith('C4 PASS') and all(f'{f}={c8[f]}' in t for f in files if c8[f]) and all(c8.values())
else:
    C['C4'] = False
# C7
refs = os.path.join(H, 'kc', 'gate', 'refs.json')
C['C7'] = os.path.exists(refs) and json.load(open(refs)).get('code_md5') == md5(os.path.join(H, 'kc', 'code', 'kc_oof_kit.py'))
# M2
a6 = A.a6_sha or (open(os.path.join(H, 'kc', 'A6_SHA256.txt')).read().split()[0]
                  if os.path.exists(os.path.join(H, 'kc', 'A6_SHA256.txt')) else '')
C['M2'] = len(a6) == 64 and all(ch in '0123456789abcdef' for ch in a6.lower())
# M4
tm = md5(TRAINER)
reg = os.path.join(H, 'kc', 'regress', 'REGRESS.txt')
reg_last = open(reg).read().strip().splitlines()[-1] if os.path.exists(reg) else ''
C['M4'] = tm == SHIM_MD5 and reg_last == f'REGRESS PASS trainer_md5={SHIM_MD5}'
info.update(trainer_md5=tm, regress_last=reg_last, a6_sha=a6[:16], c8=c8)

for k, v in C.items():
    print(f'{k:6s} {"ok" if v else "MISSING / FAIL"}')
print(json.dumps(info, default=str))
if A.status:
    sys.exit(0)
if not all(C.values()):
    print('KC_READY not written: ' + ', '.join(k for k, v in C.items() if not v)); sys.exit(1)
st = info['store']
write([f"PASS kc_ready store={st['n_npz']}/4407 U={st['u_rate']:.4f} weak={st['weak']} geom={st['geom']} "
       f"geom_ok={st['geom_ok']} smoke=ok shim_md5={SHIM_MD5[:8]} a6_sha={a6[:16]} side_sha={info['side']['sha']} "
       f"contract={contract} c2b=ok",
       f"# C3 serve parity is NOT part of KC_READY (critique, adopted): it must PASS before the G3 pack",
       f"# written by kc_ready.py {time.strftime('%Y-%m-%d %H:%M:%S')} | {json.dumps(info, default=str)[:900]}"])
