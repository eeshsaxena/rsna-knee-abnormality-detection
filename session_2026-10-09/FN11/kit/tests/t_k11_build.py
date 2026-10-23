#!/usr/bin/env python3
"""t_k11_build.py (laptop, no model; 2026-10-09, Claude; TK10BUILD-style): FN11 kit build tests on the REAL builds
(k11/build_w11[_iN] + build_fn11[_iN] + build_fn11[_iN]_n70). Never pushes anything.
  python -B -W ignore tests/t_k11_build.py --w k11/build_w11 --fn k11/build_fn11 --fn70 k11/build_fn11_n70 [--test]
--test: the builds are TEST builds (E2 4/5 interim): DO_NOT_* markers must be PRESENT; otherwise they must be ABSENT.
Last line: TK11BUILD PASS|FAIL n/m
"""
import argparse, ast, contextlib, copy, glob, hashlib, io, json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/')
SP = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, f'{HERE}/n949')
import make_k11_tools as mt, build_k11fn, fn11_check, fnw_check      # noqa: E402
PY = sys.executable
K10 = f'{SP}/k10'
res = []


def ok(name, cond, detail=''):
    res.append((name, bool(cond)))
    print(f'{"PASS" if cond else "FAIL"}  {name}  {str(detail)[:300]}', flush=True)


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


def cfg_of(src, key):
    return ast.literal_eval(src.split('\n', 1)[0][len(key):].split('   #', 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--w', required=True); ap.add_argument('--fn', required=True); ap.add_argument('--fn70', required=True)
    ap.add_argument('--test', action='store_true')
    a = ap.parse_args()
    W, FN, F70 = (os.path.abspath(x).replace('\\', '/') for x in (a.w, a.fn, a.fn70))
    TEST = a.test
    DS = 'rsna-knee-cnxt-fleet-k11w'
    # ---- T1 generated tools
    r = subprocess.run([PY, '-B', f'{HERE}/make_k11_tools.py', '--check'], capture_output=True, text=True)
    ok('T1 make_k11_tools --check: every generated file == make(), k10 byte copies identical', r.returncode == 0, (r.stdout.strip().splitlines() or [''])[-1])
    k5 = open(f'{HERE}/cells/k2_fleet_cell.py', encoding='utf-8').read()
    for slug, p in ((DS, 'cells/k11w_fleet_cell.py'), ('rsna-knee-cnxt-fleet-k11w2', 'cells/k11w2_fleet_cell.py')):
        c = open(f'{HERE}/{p}', encoding='utf-8').read()
        ok(f'T1 {p}: 3 x {slug}, reverse == the K5 cell e37f0604997cd40f', c.count(slug) == 3 and mt.unpatch_cell(c, slug) == k5
           and hashlib.sha256(k5.encode()).hexdigest()[:16] == 'e37f0604997cd40f')
    # ---- T2 the k11w upload build
    kb = json.load(open(f'{W}/K5_BUILD.json'))
    b10 = json.load(open(f'{K10}/build_w/K5_BUILD.json'))
    names = kb['members']
    n2 = kb['sets'].get('E2', 0)
    ok(f'T2 K5_BUILD: k11, kit k11w, sets C5 D5 E5 E2 {n2}, {"TEST (interim, do_not_submit, marker)" if TEST else "FINAL (no marker)"}',
       kb['k11'] and kb['kit'] == 'k11w' and {s: kb['sets'][s] for s in 'CDE'} == {'C': 5, 'D': 5, 'E': 5} and n2 >= 4
       and bool(kb['do_not_submit']) == TEST and os.path.isfile(f'{W}/DO_NOT_SUBMIT_INTERIM.txt') == TEST
       and not os.path.isfile(f'{W}/DO_NOT_PUSH_SIMULATION.txt') and not kb['simulated'], kb['sets'])
    e2 = names[len(b10['members']):]
    ok('T2 members == the k10w members (same order) + E2 f{F}_s1046t288f75 keys', names[:len(b10['members'])] == b10['members']
       and len(e2) == n2 and all(x.split('_')[1] == 's1046t288f75' and '_raw_ttr288-320x24_' in x for x in e2), e2)
    ok('T2 every member TTA AB; C / D / E picks (staged name, sha256) == k10w', kb['tta'] == {x: 'AB' for x in names}
       and {p['staged']: p['sha256'] for p in kb['picks'] if p['set'] in 'CDE'} == {p['staged']: p['sha256'] for p in b10['picks']})
    ok('T2 E2 picks: raw, seed 1046, ttr 288, frac75, 21 ep, source modal + modal receipt',
       all(p['tag'] == 'raw' and p['seed'] == 1046 and int(p['ttr_size']) == 288 and p['train_labels'] == 'labels_frac75_logit.csv'
           and p['epochs'] == 21 and p['source'] == 'modal' and p.get('modal_receipt') for p in kb['picks'] if p['set'] == 'E2'))
    up = f'{W}/upload/{DS}'
    u10 = f'{K10}/build_w/upload/rsna-knee-cnxt-fleet-k10w'
    dm = json.load(open(f'{up}/dataset-metadata.json'))
    man = json.load(open(f'{up}/fleet_manifest.json'))
    nprobe = len([f for f in os.listdir(up) if f.startswith('parity_')])
    ok(f'T2 upload: id / title eeshsaxena/{DS}, {len(os.listdir(up))} files = 5 code + {len(names)} members + manifest + metadata + '
       f'{nprobe} parity npz, manifest == members, sources sha16 == picks',
       dm['id'] == f'eeshsaxena/{DS}' and dm['title'] == DS and len(os.listdir(up)) == 5 + len(names) + 2 + nprobe
       and man['checkpoints'] == names and all(man['sources'][p['staged']]['sha256'][:16] == p['sha256'][:16] for p in kb['picks']))
    same = [f for f in os.listdir(u10) if f not in ('fleet_manifest.json', 'dataset-metadata.json')]
    diff = [f for f in same if not os.path.isfile(f'{up}/{f}') or fsha(f'{up}/{f}') != fsha(f'{u10}/{f}')]
    ok(f'T2 every non-manifest k10w file ({len(same)}: code, 15 members, parity npz) is in k11w byte for byte', not diff, diff)
    man10 = json.load(open(f'{u10}/fleet_manifest.json'))
    ok('T2 manifest == the k10w manifest + the E2 entries (other keys and the 15 old sources unchanged)',
       {k: v for k, v in man.items() if k not in ('checkpoints', 'sources')} == {k: v for k, v in man10.items() if k not in ('checkpoints', 'sources')}
       and all(man['sources'][k] == v for k, v in man10['sources'].items()), sorted(set(man) ^ set(man10)))
    # ---- T3 the FN11 kernels
    fb, f7 = json.load(open(f'{FN}/FN_BUILD.json')), json.load(open(f'{F70}/FN_BUILD.json'))
    f3 = json.load(open(f'{K10}/build_fn3/FN_BUILD.json'))
    k, k7 = fb['kernels']['rsna-sub-k11fn'], f7['kernels']['rsna-sub-k11fn-n70']
    k3 = f3['kernels']['rsna-sub-k10fn3']
    ok('T3 FN11: one kernel rsna-sub-k11fn, TAG FN11, MODE FN, W3 0/.40/.60, variants as FN10-3, files as FN10-3',
       list(fb['kernels']) == ['rsna-sub-k11fn'] and k['tag'] == 'FN11' and k['mode'] == 'FN' and k['W3'] == [0.0, 0.4, 0.6]
       and k['W3_VARIANTS'] == k3['W3_VARIANTS'] and k['files'] == k3['files'])
    ok('T3 FN11-N70: rsna-sub-k11fn-n70, TAG FN11, W3 0/.30/.70, ref_build == build_fn11, wdiff OK',
       list(f7['kernels']) == ['rsna-sub-k11fn-n70'] and k7['tag'] == 'FN11' and k7['W3'] == [0.0, 0.3, 0.7]
       and f7.get('ref_build') == FN and f7.get('ref_nb_sha256') == k['nb_sha256'] and os.path.isfile(f'{F70}/NB_DIFF_vs_tested.txt'))
    ok('T3 FN11 datasets == [k11w, the N weights]; FN10-3 had [k10w, the same N weights]',
       k['dataset_sources'] == [f'eeshsaxena/{DS}', k3['dataset_sources'][1]] == k7['dataset_sources']
       and k3['dataset_sources'][0] == 'eeshsaxena/rsna-knee-cnxt-fleet-k10w')
    same_keys = ('child_sha256', 'src_sha256', 'weights_file', 'weights_sha256', 'weights_ds', 'k9n_n_cell_sha16', 'fn_cell_sha256',
                 'ids_cell_sha256', 'k5_cell_sha16', 'k10_main_nb_sha16', 'k10_k5_build_sha256')
    ok(f'T3 FN11 == FN10-3 in {len(same_keys)} provenance keys (child / weights / N cell / FN-IDS / K5 cell / K10-main base)',
       all(fb[x] == f3[x] for x in same_keys), [x for x in same_keys if fb[x] != f3[x]])
    ok('T3 k2_cfg == FN10-3 except MEMBERS / TTA; MEMBERS == the k11w members; final flags', {x: v for x, v in fb['k2_cfg'].items() if x not in ('MEMBERS', 'TTA')}
       == {x: v for x, v in f3['k2_cfg'].items() if x not in ('MEMBERS', 'TTA')} and fb['k2_cfg']['MEMBERS'] == names == fb['members']
       and fb['fleet_manifest_sha256'] == fsha(f'{up}/fleet_manifest.json') and fb['do_not_push'] == TEST == f7['do_not_push']
       and os.path.isfile(f'{FN}/DO_NOT_PUSH_TEST.txt') == TEST == os.path.isfile(f'{F70}/DO_NOT_PUSH_TEST.txt'))
    meta = json.load(open(f'{FN}/kernel/rsna-sub-k11fn/kernel-metadata.json'))
    ok('T3 kernel-metadata: private, GPU T4, no internet, competition source, no kernel / model sources',
       meta['is_private'] is True and meta['enable_gpu'] and not meta['enable_internet'] and meta['machine_shape'] == 'NvidiaTeslaT4'
       and meta['competition_sources'] == ['rsna-knee-abnormality-detection'] and meta['kernel_sources'] == [] and meta['model_sources'] == [])
    # ---- T4 byte diffs vs the TESTED twins
    for new, ref, tag in ((f'{FN}/kernel/rsna-sub-k11fn', f'{K10}/build_fn3/kernel/rsna-sub-k10fn3', 'FN11 vs FN10-3'),
                          (f'{F70}/kernel/rsna-sub-k11fn-n70', f'{K10}/build_fn3_n70/kernel/rsna-sub-k10fn3-n70', 'FN11-N70 vs FN10-3-N70')):
        r = subprocess.run([PY, '-B', f'{HERE}/k11_nbdiff.py', new, ref], capture_output=True, text=True)
        ok(f'T4 k11_nbdiff {tag}: only members / dataset slug / naming / TAG differ', r.returncode == 0 and 'K11NBDIFF OK' in r.stdout,
           ((r.stdout + r.stderr).strip().splitlines() or [''])[-1][:300])
    # ---- T5 rebuild byte-identical
    with tempfile.TemporaryDirectory() as td:
        args = ['--kit', 'fn11', '--k10w_build', W, '--out', f'{td}/b', '--compare', FN] + (['--test'] if TEST else [])
        r = subprocess.run([PY, '-B', '-W', 'ignore', f'{HERE}/build_k11fn.py'] + args, capture_output=True, text=True)
        ok('T5 build_k11fn --compare: FN11 rebuilds byte-identical (notebook + metadata)', r.returncode == 0 and 'COMPARE OK' in r.stdout,
           ((r.stdout + r.stderr).strip().splitlines() or [''])[-1][:300])
    # ---- T6 refusals
    with tempfile.TemporaryDirectory() as td:
        def mk(edit):                                                    # a JSON-only copy of the k11w build, edited
            d = f'{td}/w{len(os.listdir(td))}'
            os.makedirs(f'{d}/upload/{DS}')
            b = copy.deepcopy(kb); edit(b)
            json.dump(b, open(f'{d}/K5_BUILD.json', 'w'), indent=1)
            shutil.copy2(f'{up}/fleet_manifest.json', f'{d}/upload/{DS}/fleet_manifest.json')
            if TEST:
                open(f'{d}/DO_NOT_SUBMIT_INTERIM.txt', 'w').write('x')
            return d
        T_ = ['--test'] if TEST else []
        go = lambda d, extra=(): build_k11fn.main(['--kit', 'fn11', '--k10w_build', d, '--out', f'{td}/o{len(os.listdir(td))}'] + T_ + list(extra))
        ok('T6 refuse a C / D / E pick with other bytes', *raises(lambda: go(mk(lambda b: b['picks'][0].update(sha256='0' * 64))), 'C / D / E bytes'))
        ok('T6 refuse a k10w build as the source (not a build_k11.py build)',
           *raises(lambda: build_k11fn.main(['--kit', 'fn11', '--k10w_build', f'{K10}/build_w', '--out', f'{td}/x1'] + T_), 'is not a build_k11.py'))
        ok('T6 refuse dropping a k10w member (C f0 missing)', *raises(lambda: go(mk(lambda b: b.update(members=b['members'][1:]))), 'k10w members in order'))
        ok('T6 refuse a non-AB TTA', *raises(lambda: go(mk(lambda b: b['tta'].update({b['members'][-1]: 'A'}))), ''))
        ok('T6 refuse F/N .30/.70 without --slug_suffix', *raises(lambda: go(W, ['--w_fn', '0.30,0.70']), 'need --slug_suffix'))
        if TEST:
            ok('T6 refuse the TEST (interim) source without --test',
               *raises(lambda: build_k11fn.main(['--kit', 'fn11', '--k10w_build', W, '--out', f'{td}/x2']), 'not a FINAL K11w build'))
        else:
            ok('T6 refuse a do_not_submit source without --test',
               *raises(lambda: build_k11fn.main(['--kit', 'fn11', '--k10w_build', mk(lambda b: b.update(do_not_submit=True)),
                                                 '--out', f'{td}/x2']), 'not a FINAL K11w build'))
    # ---- T7 gate-script slugs
    ok('T7 fn11_check / fnw_check accept the K11 slugs; fn11_check refuses a K10 slug',
       fn11_check.set_slug(f'{FN}/FN_BUILD.json') == 'rsna-sub-k11fn' and fnw_check.set_slug(f'{F70}/FN_BUILD.json')[0] == 'rsna-sub-k11fn-n70'
       and raises(lambda: fn11_check.set_slug(f'{K10}/build_fn3/FN_BUILD.json'), 'not a K11 FN kernel')[0])
    lines = open(f'{HERE}/tests/runtime_k11.log').read()
    ok('T8 runtime rule (PREREG_FN11 sec 5): FN11 slow end <= 5.85 h', 'rule <= 5.85 h: PASS' in lines, lines.splitlines()[-1][:200])
    n = sum(g for _, g in res)
    print(f'TK11BUILD {"PASS" if n == len(res) else "FAIL"} {n}/{len(res)}')
    return 0 if n == len(res) else 1


if __name__ == '__main__':
    sys.exit(main())
