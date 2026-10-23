#!/usr/bin/env python3
"""t_k10_build.py (laptop, no model; 2026-10-09, Claude): K10 kit build tests on the SIMULATION builds in k10/_sim
(build, build_w, build_n, build_fn, build_fn3 from k10_collect.sh on ~/k10_test/sim/root). Never pushes anything.
  python -B -W ignore tests/t_k10_build.py [--root k10/_sim]          last line: TK10BUILD PASS|FAIL n/m
"""
import argparse, ast, copy, glob, hashlib, io, json, os, shutil, subprocess, sys, tempfile, contextlib

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/')
SP = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, f'{HERE}/n949')
import build_k10, make_k10_tools as mt                               # noqa: E402
PY = sys.executable
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


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def nbj(p):
    return json.load(open(p, encoding='utf-8'))


def cfg_of(src, key='K2_CFG = '):
    return ast.literal_eval(src.split('\n', 1)[0][len(key):].split('   #', 1)[0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default=f'{HERE}/_sim')
    ap.add_argument('--real', action='store_true', help='the root holds the REAL (FINAL) builds: simulation markers must be absent')
    a = ap.parse_args()
    R = a.root.replace('\\', '/')
    SIM = not a.real
    TEST = ['--test'] if SIM else []
    # ---- T1 generated tools
    r = subprocess.run([PY, '-B', f'{HERE}/make_k10_tools.py', '--check'], capture_output=True, text=True)
    ok('T1 make_k10_tools --check: every generated file == make() (cells, k10n_check, 3 box harnesses)', r.returncode == 0, r.stdout[-200:])
    k5 = open(f'{HERE}/cells/k2_fleet_cell.py', encoding='utf-8').read()
    for slug, p in (('rsna-knee-cnxt-fleet-k10', 'cells/k10_fleet_cell.py'), ('rsna-knee-cnxt-fleet-k10w', 'cells/k10w_fleet_cell.py')):
        c = open(f'{HERE}/{p}', encoding='utf-8').read()
        ok(f'T1 {p}: 3 x {slug}, minus K10-DS == the K5 cell e37f0604997cd40f, no bare K9 slug left',
           c.count(slug) == 3 and mt.unpatch_cell(c, slug) == k5 and sha(k5)[:16] == 'e37f0604997cd40f'
           and "rsna-knee-cnxt-fleet'" not in c and 'rsna-knee-cnxt-fleet)' not in c)
    ok('T1 K5 cell == the k9 kit cell (bytes)', open(f'{HERE}/cells/k2_fleet_cell.py', 'rb').read() == open(f'{SP}/k9/cells/k2_fleet_cell.py', 'rb').read())
    # ---- T2 the K10 (E + D) sim build
    kb = json.load(open(f'{R}/build/K5_BUILD.json'))
    names = kb['members']
    ok(f'T2 K5_BUILD: k8 + k10, kit k10, {"SIMULATION -> do_not_submit, marker file present" if SIM else "FINAL: not simulated, not do_not_submit, no marker"}',
       kb['k8'] and kb['k10'] and kb['kit'] == 'k10' and kb['simulated'] == SIM and kb['do_not_submit'] == SIM and not kb['interim']
       and os.path.isfile(f'{R}/build/DO_NOT_PUSH_SIMULATION.txt') == SIM and not os.path.isfile(f'{R}/build/DO_NOT_SUBMIT_INTERIM.txt'))
    ok('T2 sets E 5 + D 5, members D (224) then E (288), keys f{F}_s1043f75 / f{F}_s1045t288f75, no C',
       kb['sets'] == {'E': 5, 'D': 5} and [n.split('_')[1] for n in names] == ['s1043f75'] * 5 + ['s1045t288f75'] * 5
       and [n.split('_')[0] for n in names] == [f'f{F}' for F in range(5)] * 2, names[:2])
    ok('T2 every member TTA AB; cell_sha16 == the K10 cell', kb['tta'] == {n: 'AB' for n in names}
       and kb['cell_sha16'] == sha(mt.make()['cells/k10_fleet_cell.py'])[:16])
    up = f'{R}/build/upload/rsna-knee-cnxt-fleet-k10'
    dm = json.load(open(f'{up}/dataset-metadata.json'))
    man = json.load(open(f'{up}/fleet_manifest.json'))
    ok('T2 upload: id / title eeshsaxena/rsna-knee-cnxt-fleet-k10, 21 files, manifest == members, sources sha16 == picks',
       dm['id'] == 'eeshsaxena/rsna-knee-cnxt-fleet-k10' and dm['title'] == 'rsna-knee-cnxt-fleet-k10' and len(os.listdir(up)) == 21
       and man['checkpoints'] == names and all(man['sources'][p['staged']]['sha256'][:16] == p['sha256'][:16] for p in kb['picks']))
    k9nb = nbj(f'{SP}/k9/build/kernel/rsna-sub-k9-fleet/rsna_sub_k9_fleet.ipynb')
    k9meta = json.load(open(f'{SP}/k9/build/kernel/rsna-sub-k9-fleet/kernel-metadata.json'))
    k9cfg = cfg_of(''.join(k9nb['cells'][31]['source']))
    for slug, mode in (('rsna-sub-k10-fleet', 'strict'), ('rsna-sub-k10-solo', 'solo')):
        kd = f'{R}/build/kernel/{slug}'
        meta = json.load(open(f'{kd}/kernel-metadata.json')); nb = nbj(f'{kd}/{meta["code_file"]}')
        c31 = ''.join(nb['cells'][31]['source']); cfg = cfg_of(c31)
        ok(f'T2 {slug}: 32 cells, cells 0-30 == K9-main byte for byte, cell 31 metadata == K9',
           len(nb['cells']) == 32 and nb['cells'][:31] == k9nb['cells'][:31] and nb['cells'][31]['metadata'] == k9nb['cells'][31]['metadata'])
        ok(f'T2 {slug}: cell 31 body == the K10-DS cell; K2_CFG == K9-main except MEMBERS / TTA / MAIN_MODE ({mode})',
           c31.split('\n', 1)[1] == mt.make()['cells/k10_fleet_cell.py'] and cfg['MAIN_MODE'] == mode and cfg['MEMBERS'] == names
           and cfg['TTA'] == kb['tta'] and {k: v for k, v in cfg.items() if k not in ('MEMBERS', 'TTA', 'MAIN_MODE')}
           == {k: v for k, v in k9cfg.items() if k not in ('MEMBERS', 'TTA', 'MAIN_MODE')}, {k: v for k, v in cfg.items() if k != 'MEMBERS'})
        ok(f'T2 {slug}: metadata == K9-main with only the fleet slug (-> -k10) and id / title / code_file changed',
           meta['dataset_sources'] == [d if d != 'eeshsaxena/rsna-knee-cnxt-fleet' else 'eeshsaxena/rsna-knee-cnxt-fleet-k10'
                                       for d in k9meta['dataset_sources']]
           and {k: v for k, v in meta.items() if k not in ('dataset_sources', 'id', 'title', 'code_file')}
           == {k: v for k, v in k9meta.items() if k not in ('dataset_sources', 'id', 'title', 'code_file')}
           and meta['id'] == f'eeshsaxena/{slug}', meta['dataset_sources'][-2:])
    # ---- T3 load_members refusals (hardlinked ckpt dirs, edited members.json)
    mj0 = json.load(open(f'{R}/members.json'))
    tmp = tempfile.mkdtemp(prefix='t_k10_')
    try:
        def variant(name, mut, drop=()):
            d = f'{tmp}/{name}'; os.makedirs(f'{d}/ckpt')
            m = mut(copy.deepcopy(mj0))
            for r in mj0['members']:
                if r['file'] not in drop:
                    os.link(f'{R}/ckpt/{r["file"]}', f'{d}/ckpt/{r["file"]}')
            json.dump(m, open(f'{d}/members.json', 'w'))
            return d
        d = variant('noflag', lambda m: (m.pop('k10'), m)[1])
        g, w = raises(lambda: build_k10.load_members(f'{d}/members.json', f'{d}/ckpt', 'k10'), 'is not a k10_collect.sh')
        ok('T3 members.json without k10 flag refused', g, w)

        def to_a(m):
            for r in m['members']:
                if r['set'] == 'E' and r['fold'] == 2:
                    r.update(seed=1043, ttr_size=224, train_labels='labels_mix50_logit.csv')
            return m
        d = variant('seta', to_a)
        g, w = raises(lambda: build_k10.load_members(f'{d}/members.json', f'{d}/ckpt', 'k10'), 'set A')
        ok('T3 a set A identity (s1043 TTR224 mix50) refused', g, w)
        d = variant('wrongset', lambda m: ([r.update(set='C') for r in m['members'] if r['set'] == 'E' and r['fold'] == 0], m)[1])
        g, w = raises(lambda: build_k10.load_members(f'{d}/members.json', f'{d}/ckpt', 'k10'), 'its identity is set E')
        ok('T3 a member whose set label != its identity refused', g, w)
        d = variant('ema', lambda m: ([r.update(tag='ema') for r in m['members'] if r['set'] == 'E' and r['fold'] == 1], m)[1])
        g, w = raises(lambda: build_k10.load_members(f'{d}/members.json', f'{d}/ckpt', 'k10'), 'RAW best')
        ok('T3 an ema pick refused', g, w)
        d = variant('stale', lambda m: m)
        shutil.copyfile(f'{R}/ckpt/{mj0["members"][0]["file"]}', f'{d}/ckpt/f9_s1045_raw_ttr288-320x24_cv0000_best.pt')
        g, w = raises(lambda: build_k10.load_members(f'{d}/members.json', f'{d}/ckpt', 'k10'), 'stale file')
        ok('T3 a stale checkpoint in ckpt/ refused', g, w)
        _, mem, _ = build_k10.load_members(f'{R}/members.json', f'{R}/ckpt', 'k10')
        ok('T3 kit k10 takes E + D only (members.json holds C too)', sorted({r['set'] for r in mem}) == ['D', 'E'] and len(mem) == 10)
        _, memw, _ = build_k10.load_members(f'{R}/members.json', f'{R}/ckpt', 'k10w')
        ok('T3 kit k10w takes C + D + E (15)', sorted({r['set'] for r in memw}) == ['C', 'D', 'E'] and len(memw) == 15)
        old = build_k10.older_keys
        try:
            build_k10.older_keys = lambda: {'f3_s1045t288f75': {'0123456789abcdef'}}
            g, w = raises(lambda: build_k10.load_members(f'{R}/members.json', f'{R}/ckpt', 'k10'), 'already shipped')
            ok('T3 K10-KEY: an E key seen in an older manifest is refused', g, w)
            build_k10.older_keys = lambda: {'f2_s1043f75': {'0123456789abcdef'}}
            g, w = raises(lambda: build_k10.load_members(f'{R}/members.json', f'{R}/ckpt', 'k10'), 'other bytes')
            ok('T3 K10-KEY: a D key shipped before with other bytes is refused', g, w)
            build_k10.older_keys = lambda: {'f2_s1043f75': {'SIM:0123456789abcdef'}}
            g, w = raises(lambda: build_k10.load_members(f'{R}/members.json', f'{R}/ckpt', 'k10'), 'nothing')
            ok('T3 K10-KEY: simulation-build bytes under a D key are ignored (sim relabels other weights)', w == 'no error', w)
        finally:
            build_k10.older_keys = old
        ok12 = old()
        ok('T3 K10-KEY on disk: no E key in any older manifest; every D / C key shipped before carries the same bytes',
           all(f'f{F}_s1045t288f75' not in ok12 for F in range(5))
           and all({x for x in ok12.get(r['key'], set()) if not x.startswith('SIM:')} <= {r['sha256'][:16]} for r in memw if r['set'] != 'E'))
        d = variant('interim', lambda m: (m.__setitem__('members', [r for r in m['members'] if not (r['set'] == 'E' and r['fold'] == 4)]), m)[1],
                    drop=[r['file'] for r in mj0['members'] if r['set'] == 'E' and r['fold'] == 4])
        g, w = raises(lambda: build_k10.main(['--kit', 'k10', '--members_json', f'{d}/members.json', '--store', f'{R}/_real/store',
                                              '--out', f'{d}/b']), 'INTERIM build needs')
        ok('T3 E 4/5 without do_not_submit: the build refuses', g, w)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # ---- T4 the k10w upload-only build
    kw = json.load(open(f'{R}/build_w/K5_BUILD.json'))
    upw = f'{R}/build_w/upload/rsna-knee-cnxt-fleet-k10w'
    ok('T4 k10w: 15 members D, C, E; no kernel at all; dataset -k10w; simulation marker',
       kw['kit'] == 'k10w' and kw['sets'] == {'C': 5, 'D': 5, 'E': 5} and kw['kernels'] == {} and not os.path.exists(f'{R}/build_w/kernel')
       and json.load(open(f'{upw}/dataset-metadata.json'))['id'] == 'eeshsaxena/rsna-knee-cnxt-fleet-k10w'
       and json.load(open(f'{upw}/fleet_manifest.json'))['checkpoints'] == kw['members'] and len(os.listdir(upw)) == 26
       and os.path.isfile(f'{R}/build_w/DO_NOT_PUSH_SIMULATION.txt') == SIM and kw['do_not_submit'] == SIM)
    ok('T4 k10w E / D members == the K10 build members (same bytes)', [m for m in kw['members'] if '_s1044t288_' not in m] == names
       and {p['sha256'] for p in kw['picks'] if p['set'] != 'C'} == {p['sha256'] for p in kb['picks']})
    # ---- T5 K10N-a
    nbn = json.load(open(f'{R}/build_n/K10N_BUILD.json'))
    kd = f'{R}/build_n/kernel/rsna-sub-k10n-a'
    nb = nbj(f'{kd}/rsna_sub_k10n_a.ipynb'); meta = json.load(open(f'{kd}/kernel-metadata.json'))
    main_nb = nbj(f'{R}/build/kernel/rsna-sub-k10-fleet/rsna_sub_k10_fleet.ipynb')
    main_meta = json.load(open(f'{R}/build/kernel/rsna-sub-k10-fleet/kernel-metadata.json'))
    k9n = nbj(f'{SP}/k9n/build/kernel/rsna-sub-k9n-a/rsna_sub_k9n_a.ipynb')
    c32, k32 = ''.join(nb['cells'][32]['source']), ''.join(k9n['cells'][32]['source'])
    ok('T5 K10N-a: 33 cells, 0-31 == K10-main, cell 32 after its config line == the K9N-a cell 32, metadata == K9N-a',
       len(nb['cells']) == 33 and nb['cells'][:32] == main_nb['cells'] and c32.split('\n', 1)[1] == k32.split('\n', 1)[1]
       and nb['cells'][32]['metadata'] == k9n['cells'][32]['metadata'])
    n10, n9 = cfg_of(c32, 'K9N_CFG = '), cfg_of(k32, 'K9N_CFG = ')
    ok('T5 K10N-a config == K9N-a except TAG (W3 .10/.30/.60, variants, child / weights sha)', n10['TAG'] == 'K10N-a'
       and n10['W3'] == [0.1, 0.3, 0.6] and {k: v for k, v in n10.items() if k != 'TAG'} == {k: v for k, v in n9.items() if k != 'TAG'})
    ok(f'T5 K10N-a metadata == K10-main + the N weights dataset; do_not_push {SIM}', meta['dataset_sources'] ==
       main_meta['dataset_sources'] + ['nartaa/rsna-knee-publication-swa-weights-20261007'] and nbn['do_not_push'] == SIM
       and os.path.isfile(f'{R}/build_n/DO_NOT_PUSH_TEST.txt') == SIM and meta['id'] == 'eeshsaxena/rsna-sub-k10n-a')
    if SIM:
        r = subprocess.run([PY, '-B', f'{HERE}/build_k10n.py', '--k10_build', f'{R}/build', '--out', f'{R}/_t_build_n'],
                           capture_output=True, text=True)
        ok('T5 build_k10n refuses a simulation K10 build without --test', r.returncode != 0 and 'not a FINAL K10 build' in r.stderr, r.stderr[-150:])
    r = subprocess.run([PY, '-B', f'{HERE}/build_k10n.py', '--k10_build', f'{R}/build', '--out', f'{R}/_t_build_n',
                        '--overwrite', '--compare', f'{R}/build_n'] + TEST, capture_output=True, text=True)
    ok('T5 build_k10n rebuild is byte-identical (--compare)', r.returncode == 0 and 'COMPARE OK' in r.stdout, r.stdout[-150:] + r.stderr[-150:])
    shutil.rmtree(f'{R}/_t_build_n', ignore_errors=True)
    # ---- T6 FN10 / FN10-3
    kfn = nbj(f'{SP}/kfn/build/kernel/rsna-sub-k9fn/rsna_sub_k9fn.ipynb')
    main_cfg = cfg_of(''.join(main_nb['cells'][31]['source']))
    for kit, bdir, slug, ds, cell, src_members, n in (
            ('fn10', 'build_fn', 'rsna-sub-k10fn', 'rsna-knee-cnxt-fleet-k10', 'cells/k10_fleet_cell.py', kb, 10),
            ('fn10_3', 'build_fn3', 'rsna-sub-k10fn3', 'rsna-knee-cnxt-fleet-k10w', 'cells/k10w_fleet_cell.py', kw, 15)):
        fb = json.load(open(f'{R}/{bdir}/FN_BUILD.json'))
        kd = f'{R}/{bdir}/kernel/{slug}'
        nb = nbj(f'{kd}/{slug.replace("-", "_")}.ipynb'); meta = json.load(open(f'{kd}/kernel-metadata.json'))
        c2, c3 = ''.join(nb['cells'][2]['source']), ''.join(nb['cells'][3]['source'])
        f2 = cfg_of(c2)
        ok(f'T6 {kit}: md / FN-IDS (== kfn) / fleet / N; N cell after its config line == kfn FN; config == kfn FN except TAG',
           len(nb['cells']) == 4 and nb['cells'][1] == kfn['cells'][1] and c3.split('\n', 1)[1] == ''.join(kfn['cells'][3]['source']).split('\n', 1)[1]
           and {k: v for k, v in cfg_of(c3, 'K9N_CFG = ').items() if k != 'TAG'}
           == {k: v for k, v in cfg_of(''.join(kfn['cells'][3]['source']), 'K9N_CFG = ').items() if k != 'TAG'})
        ok(f'T6 {kit}: fleet body == {cell}; K2_CFG == K10-main literal with MEMBERS / TTA of {n} members + RHO_MIN_N 1e9',
           c2.split('\n', 1)[1] == mt.make()[cell] and f2['MEMBERS'] == src_members['members'] and len(f2['MEMBERS']) == n
           and f2['TTA'] == src_members['tta'] and f2['RHO_MIN_N'] == 10 ** 9
           and {k: v for k, v in f2.items() if k not in ('MEMBERS', 'TTA', 'RHO_MIN_N')} == {k: v for k, v in main_cfg.items() if k not in ('MEMBERS', 'TTA')}
           and fb['k2_cfg'] == f2)
        ok(f'T6 {kit}: metadata = competition + eeshsaxena/{ds} + N weights; no kernel / model source; private T4 GPU no internet',
           meta['dataset_sources'] == [f'eeshsaxena/{ds}', 'nartaa/rsna-knee-publication-swa-weights-20261007']
           and meta['kernel_sources'] == [] and meta['model_sources'] == [] and meta['is_private'] and meta['enable_gpu']
           and not meta['enable_internet'] and meta['machine_shape'] == 'NvidiaTeslaT4'
           and meta['competition_sources'] == ['rsna-knee-abnormality-detection'])
        txt = open(f'{kd}/{slug.replace("-", "_")}.ipynb', encoding='utf-8').read()
        ok(f'T6 {kit}: no K9 fleet mount path, no parent source in the notebook', "rsna-knee-cnxt-fleet'" not in txt
           and 'rsna-knee-cnxt-fleet)' not in txt and not [p for p in fb['parent_sources_excluded'] if p in txt]
           and fb['do_not_push'] == SIM and os.path.isfile(f'{R}/{bdir}/DO_NOT_PUSH_TEST.txt') == SIM)
        import fn10_check
        ok(f'T6 fn10_check takes the slug {slug} from its FN_BUILD.json', fn10_check.set_slug(f'{R}/{bdir}/FN_BUILD.json') == slug)
        if SIM:
            r = subprocess.run([PY, '-B', f'{HERE}/build_k10fn.py', '--kit', kit, '--k10_build', f'{R}/build', '--k10w_build',
                                f'{R}/build_w', '--out', f'{R}/_t_fn'], capture_output=True, text=True)
            ok(f'T6 build_k10fn {kit} refuses simulation builds without --test', r.returncode != 0 and 'not a FINAL' in r.stderr, r.stderr[-120:])
        r = subprocess.run([PY, '-B', f'{HERE}/build_k10fn.py', '--kit', kit, '--k10_build', f'{R}/build', '--k10w_build',
                            f'{R}/build_w', '--out', f'{R}/_t_fn', '--overwrite', '--compare', f'{R}/{bdir}'] + TEST,
                           capture_output=True, text=True)
        ok(f'T6 build_k10fn {kit} rebuild byte-identical (--compare)', r.returncode == 0 and 'COMPARE OK' in r.stdout, r.stderr[-150:])
        shutil.rmtree(f'{R}/_t_fn', ignore_errors=True)
    two = {'kernels': {'rsna-sub-k10fn': {'mode': 'FN'}, 'rsna-sub-k10fn3': {'mode': 'FN'}}}
    p2 = f'{R}/_t_two.json'; json.dump(two, open(p2, 'w'))
    g, w = raises(lambda: fn10_check.set_slug(p2), 'exactly one MODE FN')
    os.remove(p2)
    ok('T6 fn10_check refuses a build record with two FN kernels', g, w)
    # ---- T7 inference cap + slug isolation across every kernel
    parents = [f'{R}/build/kernel/rsna-sub-k10-fleet/rsna_sub_k10_fleet.ipynb', f'{R}/build/kernel/rsna-sub-k10-solo/rsna_sub_k10_solo.ipynb',
               f'{R}/build_n/kernel/rsna-sub-k10n-a/rsna_sub_k10n_a.ipynb']
    capok = True
    for p in parents:
        cfg = cfg_of(''.join(nbj(p)['cells'][31]['source']))
        tt = [m.split('_')[3].split('-')[0] for m in cfg['MEMBERS']]
        capok &= (len(cfg['MEMBERS']) == 10 and tt.count('ttr224') == 5 and tt.count('ttr288') == 5
                  and not any('_s1044t288_' in m for m in cfg['MEMBERS']))
    ok('T7 inference cap: every parent kernel runs exactly 5 x TTR224 (D) + 5 x TTR288 (E), no set C (== K9 composition)', capok)
    allnb = parents + [f'{R}/build_fn/kernel/rsna-sub-k10fn/rsna_sub_k10fn.ipynb', f'{R}/build_fn3/kernel/rsna-sub-k10fn3/rsna_sub_k10fn3.ipynb']
    ok('T7 no K10 notebook can mount the live K9 dataset (no bare rsna-knee-cnxt-fleet path in any of the 5)',
       all("rsna-knee-cnxt-fleet'" not in open(p, encoding='utf-8').read() and 'rsna-knee-cnxt-fleet)' not in open(p, encoding='utf-8').read()
           for p in allnb))
    metas = [json.load(open(os.path.join(os.path.dirname(p), 'kernel-metadata.json'))) for p in allnb]
    ok('T7 no K10 kernel attaches eeshsaxena/rsna-knee-cnxt-fleet (the K9 version stays untouched)',
       all('eeshsaxena/rsna-knee-cnxt-fleet' not in m['dataset_sources'] for m in metas), [m['id'] for m in metas])
    ok('T7 the five slugs are new (not a K9 / K9N / KFN slug)', sorted(m['id'] for m in metas) == sorted(
        ['eeshsaxena/rsna-sub-k10-fleet', 'eeshsaxena/rsna-sub-k10-solo', 'eeshsaxena/rsna-sub-k10n-a', 'eeshsaxena/rsna-sub-k10fn',
         'eeshsaxena/rsna-sub-k10fn3']))
    npass = sum(g for _, g in res)
    print(f'TK10BUILD {"PASS" if npass == len(res) else "FAIL"} {npass}/{len(res)}')
    return 0 if npass == len(res) else 1


if __name__ == '__main__':
    sys.exit(main())
