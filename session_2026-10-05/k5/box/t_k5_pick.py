#!/usr/bin/env python3
"""t_k5_pick.py (box, CPU, test only; writes only under ~/k5_test/pick_sim): k5_pick.py on SIMULATED Kaggle-filled runs
built from the real J1 fold-0 run (copies, never the originals; ~/fleet_runs is only read).

  CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/kmimic/bin/python -W ignore ~/k5_test/k5/box/t_k5_pick.py

Scenarios (each its own root under ~/k5_test/pick_sim/<name>, fold 0 only, --dest a scratch dir):
  kag_ok        copies in download order (OOF written BEFORE the ckpt: the mtime rule would fail) + a run receipt with
                the right sha256 -> PICK (mtime rule skipped, receipt sha + OOF rescored cv checked)
  kag_noreceipt same files, no receipt -> FAIL (mtime rule, message names --kaggle_receipt / --kaggle_folds)
  kag_folds     same files, no receipt, --kaggle_folds 0 -> PICK (no sha check, OOF content proof still required)
  kag_badsha    receipt sha256 of the ckpt wrong -> FAIL
  kag_swapoof   the ema OOF saved under the raw OOF name (+ receipt matching those bytes) -> FAIL (OOF rescored cv)
  kag_e18       ckpts re-saved with args epochs 18, log header 'epochs 18', last-epoch lines 'ep 17' -> PICK with E 18
  kag_e18_hdr24 the E-18 ckpts with a log header saying 24 -> SKIP (not finished: no 'ep 23' lines)
  kag_e12       E 12 -> FAIL (E < --min_epochs 16); with --min_epochs 12 -> PICK
  extra_root    --root = a dir with only an EMPTY placeholder, --extra_root = kag_ok's root -> PICK from the extra root
"""
import hashlib, json, os, shutil, subprocess, sys, time

H = os.path.expanduser('~')
PYK = f'{H}/miniconda3/envs/kmimic/bin/python'
PICK = f'{H}/k5_test/k5/box/k5_pick.py'
SRC = f'{H}/fleet_runs/f0_s1043_ttr_mix50'
LOG = f'{H}/fleet_runs/f0_s1043_ttr_mix50.log'
SIM = f'{H}/k5_test/pick_sim'
R = 'f0_s1043_ttr_mix50'
RAW, EMA = 'f0_s1043_raw_ttr224-320x24_cv8835_best.pt', 'f0_s1043_ema_ttr224-320x24_cv8823_best.pt'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def build(name, epochs=None, hdr=None, swap_oof=False, receipt=True, bad_sha=False):
    root = f'{SIM}/{name}'
    d = f'{root}/{R}'
    os.makedirs(d)
    # download order: OOFs first, then the ckpts (new mtimes; the OOF is OLDER than its ckpt -> mtime rule fails)
    for o in ('oof_f0_s1043_raw.csv', 'oof_f0_s1043_ema.csv'):
        src = f'{SRC}/{"oof_f0_s1043_ema.csv" if (swap_oof and o.endswith("raw.csv")) else o}'
        shutil.copyfile(src, f'{d}/{o}')
    time.sleep(1.2)
    import torch
    for c in (RAW, EMA):
        if epochs is None:
            shutil.copyfile(f'{SRC}/{c}', f'{d}/{c}')
        else:
            ck = torch.load(f'{SRC}/{c}', map_location='cpu', weights_only=False)
            ck['args'] = dict(ck['args'], epochs=epochs, save_epochs=f'{epochs - 1}-{epochs}')
            torch.save(ck, f'{d}/{c}')
    lines = open(LOG, encoding='utf-8').read().splitlines()
    E = epochs or 24
    out = []
    for l in lines:
        if l.startswith('BJ opt-in:'):
            l = l.replace('| epochs 24 |', f'| epochs {hdr or E} |')
        if epochs is not None and l.startswith('ep 23 '):
            l = l.replace('ep 23 ', f'ep {E - 1} ', 1)
        out.append(l)
    open(f'{root}/{R}.log', 'w', encoding='utf-8').write('\n'.join(out) + '\n')
    if receipt:
        files = [dict(path=f'{R}/{f}', bytes=os.path.getsize(f'{d}/{f}'), sha256=sha(f'{d}/{f}')) for f in sorted(os.listdir(d))]
        if bad_sha:
            files = [dict(e, sha256='0' * 64) if e['path'].endswith(RAW) else e for e in files]
        json.dump({'status': 'DONE', 'train': {'folds': {'0': {'files': files, 'closing_line': None}}}},
                  open(f'{root}/kaggle_f34_run_receipt.json', 'w'), indent=1)
    return root


def pick(name, root, *extra):
    dest = f'{SIM}/dest_{name}'
    cmd = [PYK, '-W', 'ignore', PICK, '--folds', '0', '--root', root, '--dest', dest, '--copy', *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    open(f'{SIM}/log_{name}.txt', 'w').write(r.stdout + '\n--- stderr ---\n' + r.stderr)
    return r.returncode, r.stdout


def main():
    assert not os.path.exists(SIM), f'{SIM} exists (fresh dir per run)'
    os.makedirs(SIM)
    res = []

    def ok(name, cond, detail=''):
        res.append((name, bool(cond)))
        print(f'{"PASS" if cond else "FAIL"}  {name}  {detail[:300]}', flush=True)

    r_ok = build('kag_ok')
    rc, out = pick('kag_ok', r_ok)
    m = json.load(open(f'{SIM}/dest_kag_ok/members.json')) if rc == 0 else {}
    mm = (m.get('members') or [{}])[0]
    ok('kag_ok: Kaggle-filled run picked (mtime rule skipped, receipt sha256 + OOF rescored cv checked)',
       rc == 0 and mm.get('source') == 'kaggle' and mm.get('file') == RAW and 'mtime rule skipped' in out
       and 'ckpt + OOF sha256 == receipt' in out and mm.get('epochs') == 24, out.splitlines()[-1] if out else '')
    r_nr = build('kag_noreceipt', receipt=False)
    rc, out = pick('kag_noreceipt', r_nr)
    ok('kag_noreceipt: FAIL on the mtime rule, message names the receipt', rc == 1 and 'oof mtime' in out
       and '--kaggle_receipt' in out, [l for l in out.splitlines() if l.startswith('FAIL')][:1].__str__())
    rc, out = pick('kag_folds', r_nr, '--kaggle_folds', '0')
    ok('kag_folds: --kaggle_folds 0 without a receipt -> PICK (content proof only)', rc == 0
       and 'NO receipt sha check' in out and 'rescored cv' in out)
    r_bs = build('kag_badsha', bad_sha=True)
    rc, out = pick('kag_badsha', r_bs)
    ok('kag_badsha: ckpt sha256 != receipt -> FAIL', rc == 1 and 'sha256 != the Kaggle run receipt' in out)
    r_sw = build('kag_swapoof', swap_oof=True)
    rc, out = pick('kag_swapoof', r_sw)
    ok('kag_swapoof: wrong OOF under the raw name (receipt matches the bytes) -> FAIL on the rescored cv', rc == 1
       and 'OOF rescored cv' in out and 'not this checkpoint' in out)
    r18 = build('kag_e18', epochs=18)
    rc, out = pick('kag_e18', r18)
    m = json.load(open(f'{SIM}/dest_kag_e18/members.json')) if rc == 0 else {}
    ok('kag_e18: E 18 from the log header == ckpt args, last-epoch lines ep 17 -> PICK with epochs 18',
       rc == 0 and (m.get('members') or [{}])[0].get('epochs') == 18)
    r18b = build('kag_e18_hdr24', epochs=18, hdr=24)
    rc, out = pick('kag_e18_hdr24', r18b)
    ok('kag_e18_hdr24: header says 24 but the run has ep 17 lines -> SKIP (not finished), exit 3',
       rc == 3 and 'not finished' in out)
    r12 = build('kag_e12', epochs=12)
    rc, out = pick('kag_e12', r12)
    ok('kag_e12: E 12 < --min_epochs 16 -> FAIL', rc == 1 and 'min_epochs' in out)
    rc, out = pick('kag_e12_ok', r12, '--min_epochs', '12')
    ok('kag_e12 with --min_epochs 12 -> PICK', rc == 0)
    ph = f'{SIM}/placeholder_root'
    os.makedirs(f'{ph}/{R}')                                        # EMPTY placeholder, no log
    rc, out = pick('extra_root', ph, '--extra_root', r_ok)
    ok('extra_root: empty placeholder under --root, run found under --extra_root -> PICK', rc == 0
       and f'{r_ok}/{R}' in open(f'{SIM}/dest_extra_root/members.json').read())
    print(f'TK5PICK {"PASS" if all(g for _, g in res) else "FAIL"} {sum(g for _, g in res)}/{len(res)}', flush=True)
    return 0 if all(g for _, g in res) else 1


if __name__ == '__main__':
    sys.exit(main())
