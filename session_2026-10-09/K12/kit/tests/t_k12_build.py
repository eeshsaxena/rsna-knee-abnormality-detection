#!/usr/bin/env python3
"""t_k12_build.py (laptop, no model run; 2026-10-09, Claude; TK11BUILD-style): K12 build tests. Never pushes anything.
  python -B -W ignore tests/t_k12_build.py --w k12/build_w12[_sim] --fn k12/build_fn12_R[_sim] --fn70 ..._n70 --variant R [--test]
--test: TEST builds (simulated / interim): DO_NOT_* markers must be PRESENT; otherwise ABSENT. Last line: TK12BUILD PASS|FAIL n/m
"""
import argparse, contextlib, copy, hashlib, io, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/')
SP = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, f'{HERE}/box')
import make_k12_tools as mt, make_k10_tools as mt10, k12_oai_clean as clean, fn12_check, fnw_check, wdiff, build_k12   # noqa: E402
PY = sys.executable
K11 = f'{SP}/k11'
res = []


def ok(name, cond, detail=''):
    res.append((name, bool(cond)))
    print(f'{"PASS" if cond else "FAIL"}  {name}  {str(detail)[:260]}', flush=True)


def raises(fn, needle):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
    except AssertionError as e:
        return needle in str(e), str(e)[:200]
    except Exception as e:                                             # noqa: BLE001
        return False, f'{type(e).__name__}: {e}'[:200]
    return False, 'no error'


def fsha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def last(r):
    return ((r.stdout + r.stderr).strip().splitlines() or [''])[-1][:260]


def main():
    ap = argparse.ArgumentParser()
    for k in ('w', 'fn', 'fn70'):
        ap.add_argument(f'--{k}', required=True)
    ap.add_argument('--variant', required=True, choices=('R', 'A')); ap.add_argument('--test', action='store_true')
    a = ap.parse_args()
    W, FN, F70 = (os.path.abspath(x).replace('\\', '/') for x in (a.w, a.fn, a.fn70))
    TEST = a.test
    # ---- T1 tools
    r = subprocess.run([PY, '-B', f'{HERE}/make_k12_tools.py', '--check'], capture_output=True, text=True)
    ok('T1 make_k12_tools --check (generated cell / checks / harness == make(); k11 byte copies identical)', r.returncode == 0, last(r))
    k5 = open(f'{HERE}/cells/k2_fleet_cell.py', encoding='utf-8').read()
    c12 = open(f'{HERE}/cells/k12_fleet_cell.py', encoding='utf-8').read()
    ok('T1 k12 fleet cell: reverse == the K5 cell e37f0604997cd40f; no fleet slug left; reads /tmp/k12_fleet only',
       mt.unpatch_k12(c12) == k5 and 'rsna-knee-cnxt-fleet' not in c12 and c12.count('/tmp/k12_fleet') == 2)
    # ---- T2 the k12w folder
    up = f'{W}/upload/rsna-knee-cnxt-fleet-k12w'
    rw = json.load(open(f'{W}/K12W_BUILD.json'))
    man = json.load(open(f'{up}/k12w_manifest.json'))
    dm = json.load(open(f'{up}/dataset-metadata.json'))
    files = sorted(os.listdir(up))
    ok('T2 k12w: id / title eeshsaxena/rsna-knee-cnxt-fleet-k12w; files == 5 D-OAI ckpts + k12w_manifest.json + dataset-metadata.json '
       '(no code, no csv, no npz, nothing else)', dm['id'] == 'eeshsaxena/rsna-knee-cnxt-fleet-k12w' and dm['title'] == 'rsna-knee-cnxt-fleet-k12w'
       and files == sorted(man['checkpoints'] + ['k12w_manifest.json', 'dataset-metadata.json']) and len(man['checkpoints']) == 5
       and all(f.startswith(f'f{F}_s1043f75oai_raw_ttr224-320x24_cv') for F, f in enumerate(man['checkpoints'])), files)
    ok('T2 every D-OAI file sha256 == the manifest (full + [:16]) == the collected pick; UPLOAD_SHA256SUMS == the folder',
       all(fsha(f'{up}/{n}') == man['sources'][n]['sha256_full'] and man['sources'][n]['sha256'] == man['sources'][n]['sha256_full'][:16]
           for n in man['checkpoints']) and [p['sha256'] for p in rw['picks']] == [man['sources'][n]['sha256_full'] for n in man['checkpoints']]
       and sorted(l.split()[1] for l in open(f'{W}/UPLOAD_SHA256SUMS.txt')) == files)
    sc = [clean.scan(f'{up}/{n}', None, None) for n in man['checkpoints']]
    ok('T2 K12-OAI-CLEAN on the laptop copies: 0 hits in all 5 (no OAI id / table / extra key; decode_probe RSNA only)',
       all(not s['hits'] for s in sc), [s['hits'] for s in sc if s['hits']])
    m11 = json.load(open(f'{K11}/build_w11/upload/rsna-knee-cnxt-fleet-k11w/fleet_manifest.json'))
    mg = json.load(open(f'{W}/merged_manifest.json'))
    ok('T2 merged manifest == the live k11w manifest + the 5 D-OAI entries (pack format: original / run_dir / sha256[:16] / '
       'model_mod / model_sha / tta AB)', mg['checkpoints'] == m11['checkpoints'] + man['checkpoints']
       and all(mg['sources'][k] == v for k, v in m11['sources'].items())
       and all(mg['sources'][n] == {k: v for k, v in man['sources'][n].items() if k != 'sha256_full'} for n in man['checkpoints'])
       and set(mg) == set(m11))
    # ---- T3 the kernels
    fb, f7 = json.load(open(f'{FN}/FN_BUILD.json')), json.load(open(f'{F70}/FN_BUILD.json'))
    f11 = json.load(open(f'{K11}/build_fn11/FN_BUILD.json'))
    k, k7 = fb['kernels']['rsna-sub-k12fn'], f7['kernels']['rsna-sub-k12fn-n70']
    ok('T3 kernels rsna-sub-k12fn (W3 0/.40/.60) + rsna-sub-k12fn-n70 (0/.30/.70), TAG K12, MODE FN, files as FN11',
       k['tag'] == k7['tag'] == 'K12' and k['W3'] == [0.0, 0.4, 0.6] and k7['W3'] == [0.0, 0.3, 0.7]
       and k['files'] == f11['kernels']['rsna-sub-k11fn']['files'])
    B = f11['members']
    mem = fb['members']
    if a.variant == 'R':
        good = len(mem) == 20 and sum(x.split('_')[1] == 's1043f75oai' for x in mem) == 5 and not any(x.split('_')[1] == 's1043f75' for x in mem)
    else:
        good = mem[:20] == B and len(mem) == 25
    ok(f'T3 variant {a.variant}: members {len(mem)} ({"D replaced by D-OAI in place" if a.variant == "R" else "FN11 + D-OAI appended"}); '
       'k2_cfg MEMBERS / TTA == members, every member AB; K12_BUILD tta == the same', good and fb['k2_cfg']['MEMBERS'] == mem
       and fb['k2_cfg']['TTA'] == {n: 'AB' for n in mem} == json.load(open(f'{FN}/K12_BUILD.json'))['tta'] and f7['members'] == mem)
    keys = ('child_sha256', 'src_sha256', 'weights_file', 'weights_sha256', 'weights_ds', 'k9n_n_cell_sha16', 'fn_cell_sha256', 'ids_cell_sha256')
    ok('T3 provenance == FN11 (child, N weights, N cell, FN-IDS cell)', all(fb[x] == f11[x] == f7[x] for x in keys))
    ok('T3 datasets [k11w, k12w, N weights]; merged manifest sha recorded', k['dataset_sources'] == ['eeshsaxena/rsna-knee-cnxt-fleet-k11w',
       'eeshsaxena/rsna-knee-cnxt-fleet-k12w', 'nartaa/rsna-knee-publication-swa-weights-20261007'] == k7['dataset_sources']
       and fb['fleet_manifest_sha256'] == fsha(f'{W}/merged_manifest.json'))
    for d in (W, FN, F70):
        mk = sorted(f for f in os.listdir(d) if f.startswith('DO_NOT_'))
        ok(f'T3 {os.path.basename(d)}: DO_NOT_* markers {"present" if TEST else "absent"} ({mk})', bool(mk) == TEST)
    # ---- T4 byte diffs
    for new, ref, tag in ((f'{FN}/kernel/rsna-sub-k12fn', f'{K11}/build_fn11/kernel/rsna-sub-k11fn', 'K12 vs FN11'),
                          (f'{F70}/kernel/rsna-sub-k12fn-n70', f'{K11}/build_fn11_n70/kernel/rsna-sub-k11fn-n70', 'K12-N70 vs FN11-N70')):
        r = subprocess.run([PY, '-B', f'{HERE}/k12_nbdiff.py', new, ref, '--variant', a.variant], capture_output=True, text=True)
        ok(f'T4 k12_nbdiff {tag}: only md / the NEW merge cell / MEMBERS+TTA / fleet-dir line / TAG / metadata differ',
           r.returncode == 0 and 'K12NBDIFF OK' in r.stdout, last(r))
    try:
        dtxt, summ = wdiff.check(f'{F70}/kernel/rsna-sub-k12fn-n70/rsna_sub_k12fn_n70.ipynb', f'{FN}/kernel/rsna-sub-k12fn/rsna_sub_k12fn.ipynb',
                                 'rsna-sub-k12fn-n70', 'rsna-sub-k12fn', [0.0, 0.3, 0.7], [0.0, 0.4, 0.6],
                                 f'{F70}/kernel/rsna-sub-k12fn-n70/kernel-metadata.json', f'{FN}/kernel/rsna-sub-k12fn/kernel-metadata.json')
        ok('T4 wdiff K12-N70 vs K12: only the md weights + K9N_CFG W3 / W3_VARIANTS + naming', True, summ.get('changed_text_lines'))
    except AssertionError as e:
        ok('T4 wdiff K12-N70 vs K12', False, e)
    # ---- T5 determinism
    with tempfile.TemporaryDirectory() as td:
        args = ['--variant', a.variant, '--out_w', f'{td}/w', '--out', f'{td}/f'] + (['--test'] if TEST else [])
        with contextlib.redirect_stdout(io.StringIO()):
            build_k12.main(args)
        same = all(fsha(f'{td}/f{s}/kernel/rsna-sub-k12fn{s.replace("_", "-")}/rsna_sub_k12fn{s}.ipynb') ==
                   fsha(f'{(FN if not s else F70)}/kernel/rsna-sub-k12fn{s.replace("_", "-")}/rsna_sub_k12fn{s}.ipynb') for s in ('', '_n70')) \
            and fsha(f'{td}/w/merged_manifest.json') == fsha(f'{W}/merged_manifest.json')
        ok('T5 rebuild into a temp dir: both notebooks + the merged manifest byte-identical', same)
        # ---- T6 refusals
        mj = json.load(open(f'{HERE}/members.json'))
        if mj.get('do_not_submit'):
            ok('T6 refuse an INTERIM / SIMULATED members.json without --test',
               *raises(lambda: build_k12.main(['--variant', a.variant, '--out_w', f'{td}/x1', '--out', f'{td}/y1']), 'pass --test'))
        m4 = copy.deepcopy(mj); m4['members'] = [r for r in m4['members'] if not (r['set'] == 'DO' and r['fold'] == 4)]
        m4.update(do_not_submit=False, interim=False, simulated=False)
        json.dump(m4, open(f'{td}/m4.json', 'w'))
        ok('T6 refuse a FINAL build with only 4 D-OAI folds',
           *raises(lambda: build_k12.main(['--variant', a.variant, '--members_json', f'{td}/m4.json', '--out_w', f'{td}/x2', '--out', f'{td}/y2']), 'needs 5/5'))
        mt_ = copy.deepcopy(mj)
        for r in mt_['members']:
            if r['set'] == 'DO' and r['fold'] == 1:
                r['sha256'] = '0' * 64
        json.dump(mt_, open(f'{td}/mt.json', 'w'))
        ok('T6 refuse a D-OAI file whose bytes != members.json',
           *raises(lambda: build_k12.main(['--variant', a.variant, '--members_json', f'{td}/mt.json', '--out_w', f'{td}/x3', '--out', f'{td}/y3', '--test']), 'sha256 != members.json'))
        ok('T6 refuse a TEST FN11 build as the base (k11/build_fn11_i19)',
           *raises(lambda: build_k12.main(['--variant', a.variant, '--fn11', f'{K11}/build_fn11_i19', '--out_w', f'{td}/x4', '--out', f'{td}/y4', '--test']), 'not a FINAL FN11 build'))
    # ---- T7 gate slugs + runtime
    ok('T7 fn12_check / fnw_check accept the K12 slugs; fn12_check refuses the FN11 slug',
       fn12_check.set_slug(f'{FN}/FN_BUILD.json') == 'rsna-sub-k12fn' and fnw_check.set_slug(f'{F70}/FN_BUILD.json')[0] == 'rsna-sub-k12fn-n70'
       and raises(lambda: fn12_check.set_slug(f'{K11}/build_fn11/FN_BUILD.json'), 'not a K12 FN kernel')[0])
    lines = open(f'{HERE}/tests/runtime_k11.log').read()
    ok('T8 runtime rule (PREREG_K12 sec 5): R == FN11 3.08 h, A == FN11-D2 composition 3.47 h slow, both <= 5.85 h',
       'FN11 ~2.31 h central / ~3.08 h slow' in lines and 'FN11-D2 slow 3.47 h (PASS)' in lines)
    n = sum(g for _, g in res)
    print(f'TK12BUILD {"PASS" if n == len(res) else "FAIL"} {n}/{len(res)}')
    return 0 if n == len(res) else 1


if __name__ == '__main__':
    sys.exit(main())
