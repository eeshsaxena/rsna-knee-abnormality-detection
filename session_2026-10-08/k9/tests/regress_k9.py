#!/usr/bin/env python3
"""regress_k9.py (laptop; TEST ONLY; 2026-10-08, Claude; == regress_k8.py run on the k9 copies, + K9-SAME: every shared
tool of k9 (build_k2.py, make_k2_kernel.py, make_kf_cell.py, k2_check.py, code/, cells/, box/k2_box_stage.py) is
byte-identical to k8's). Original text: the K8 kit's shared tools (k8/build_k2.py, make_k2_kernel.py with
the new --outer_flat option, pack_fleet_ds.py, cells/, code/) must rebuild the two SUBMITTED kits byte for byte when the
new option is not used:
  K5   (.946, refs 56858295 / 56858769 / 56858818): SP/k5/build  = main strict W .25 (+ .30/.50), solo, w50 strict W .50
       (+ .20/.35), 5 TTR224 members, TTA AB
  K7i  (refs 56896918 / 56896921): SP/k7/build_i10 = main strict W .50 (+ .40/.60), solo, 10 members (A 5, B f0-3,
       C f0 re-keyed f0_s1044t288), TTA AB
For each: every upload-folder file sha256 == the shipped build's, and every kernel folder (notebook + kernel-metadata.json)
byte-identical. The rebuilt folders go to k8/_regress/<kit> and are deleted after a PASS (scratch).
  python tests/regress_k9.py            -> 'REGRESS_K9 PASS'
"""
import hashlib, json, os, shutil, sys

T_ = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
K8 = os.path.dirname(T_)
SP = os.path.dirname(K8)
sys.path.insert(0, K8)
import build_k2 as bk  # noqa: E402  (k8's copy, which imports k8's make_k2_kernel)

R = []


def ok(name, cond, extra=''):
    R.append(bool(cond)); print(f'{"PASS" if cond else "FAIL"} {name} {extra}', flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def same_dir(a, b):
    fa, fb = sorted(os.listdir(a)), sorted(os.listdir(b))
    if fa != fb:
        return False, f'file lists differ: {sorted(set(fa) ^ set(fb))}'
    bad = [f for f in fa if sha(f'{a}/{f}') != sha(f'{b}/{f}')]
    return not bad, (f'{len(fa)} files identical' if not bad else f'differ: {bad}')


def compare(tag, out, ref, slugs):
    good, det = same_dir(f'{out}/upload/rsna-knee-cnxt-fleet', f'{ref}/upload/rsna-knee-cnxt-fleet')
    ok(f'{tag} upload folder == the shipped build (sha256 per file)', good, det)
    for s in slugs:
        good, det = same_dir(f'{out}/kernel/{s}', f'{ref}/kernel/{s}')
        nb = [f for f in os.listdir(f'{out}/kernel/{s}') if f.endswith('.ipynb')]
        ok(f'{tag} kernel {s} (notebook + kernel-metadata.json) byte-identical', good,
           f'{det}; notebook sha256 {sha(f"{out}/kernel/{s}/{nb[0]}")[:16] if nb else None}')


def main():
    os.makedirs(f'{K8}/_regress', exist_ok=True)
    # ---- K5
    ref = f'{SP}/k5/build'
    kb = json.load(open(f'{ref}/K5_BUILD.json'))
    names = kb['members']
    specs = [f'{SP}/k5/ckpt/{n}' for n in names]
    assert all(os.path.isfile(p) for p in specs), specs
    tta = ','.join(f'{n}={m}' for n, m in kb['tta'].items())
    out = f'{K8}/_regress/k5'
    bk.main(sum((['--ckpt', p] for p in specs), []) + ['--store', f'{SP}/k5/_real/store', '--slug', 'rsna-sub-k5-fleet',
            '--main_mode', 'strict', '--members', ','.join(names), '--w_main', '0.25', '--w_variants', '0.30,0.50',
            '--tta', tta, '--out', out, '--overwrite'])
    bk.main(['--kernel_only', '--slug', 'rsna-sub-k5-solo', '--main_mode', 'solo', '--members', ','.join(names),
             '--w_main', '0.25', '--w_variants', '0.30,0.50', '--out', out])
    bk.main(['--kernel_only', '--slug', 'rsna-sub-k5-w50', '--main_mode', 'strict', '--members', ','.join(names),
             '--w_main', '0.50', '--w_variants', '0.20,0.35', '--out', out])
    compare('K5', out, ref, ['rsna-sub-k5-fleet', 'rsna-sub-k5-solo', 'rsna-sub-k5-w50'])
    # ---- K7i
    ref = f'{SP}/k7/build_i10'
    kb = json.load(open(f'{ref}/K5_BUILD.json'))
    names = kb['members']
    specs = []
    for p in kb['picks']:
        src = f'{SP}/k7/ckpt/{p["file"]}'
        assert os.path.isfile(src) and sha(src) == p['sha256'], src
        key = p['staged'][:len(p['staged']) - len(p['file']) + len(f'f{p["fold"]}_s{p["seed"]}')] if p['staged'] != p['file'] else None
        specs.append(src + (f'={key}' if key else ''))
    assert [s.split('=')[-1] for s in specs if '=' in os.path.basename(s)] == ['f0_s1044t288'], specs
    tta = ','.join(f'{n}={m}' for n, m in kb['tta'].items())
    out = f'{K8}/_regress/k7i'
    bk.main(sum((['--ckpt', s] for s in specs), []) + ['--store', f'{SP}/k7/_real/store', '--slug', 'rsna-sub-k7-fleet',
            '--main_mode', 'strict', '--members', ','.join(names), '--w_main', '0.50', '--w_variants', '0.40,0.60',
            '--tta', tta, '--out', out, '--overwrite'])
    bk.main(['--kernel_only', '--slug', 'rsna-sub-k7-solo', '--main_mode', 'solo', '--members', ','.join(names),
             '--w_main', '0.50', '--w_variants', '0.40,0.60', '--out', out])
    compare('K7i', out, ref, ['rsna-sub-k7-fleet', 'rsna-sub-k7-solo'])
    allok = all(R)
    if allok:
        for k in ('k5', 'k7i'):
            shutil.rmtree(f'{K8}/_regress/{k}', ignore_errors=True)
    # ---- K9-SAME: the shared tools are byte-identical to the K8 kit's (so every K8 test of them applies to K9)
    import hashlib
    same = []
    for rel in (['build_k2.py', 'make_k2_kernel.py', 'make_kf_cell.py', 'k2_check.py', 'box/k2_box_stage.py']
                + [f'code/{f}' for f in sorted(os.listdir(f'{K8}/code')) if f.endswith('.py')]
                + [f'cells/{f}' for f in sorted(os.listdir(f'{K8}/cells')) if f.endswith('.py')]):
        a_, b_ = f'{K8}/{rel}', f'{SP}/k8/{rel}'
        same.append((rel, os.path.isfile(b_) and hashlib.sha256(open(a_, 'rb').read()).digest()
                     == hashlib.sha256(open(b_, 'rb').read()).digest()))
    ok('K9-SAME shared tools byte-identical to k8 (build_k2, make_k2_kernel, make_kf_cell, k2_check, k2_box_stage, code/, cells/)',
       all(g for _, g in same), [r for r, g in same if not g] or f'{len(same)} files')
    allok = all(R)
    print(f'REGRESS_K9 {"PASS" if allok else "FAIL"} {sum(R)}/{len(R)}')
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
