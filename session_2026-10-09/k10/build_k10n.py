#!/usr/bin/env python3
"""build_k10n.py (2026-10-09, Claude; from k9n/build_k9n.py). Local only: nothing is pushed, versioned or submitted.

K10N-a = the K10-main kernel (parent repro_0946 cells 0-30 + the K10-DS fleet cell 31, STRICT W .50, the 10 K10 members
         E + D, TTA AB) + cell 32 = the K9N N stage (cells/k9n_n949_cell.py, byte-identical to the k9n kit's, sha16
         bd87efb8fa89bbac; child n949_child.py == the K9N child) with the K9N-a weights: submission.csv =
         rank_pct(.10 r(P) + .30 r(F) + .60 r(N)), strict fail-loud. TAG 'K10N-a'. Diagnostics: the other K9N triples
         (.30/.30/.40, .40/.30/.30) as submission_pfn_<PPFFNN>.csv (never scored).
  K10N-a  eeshsaxena/rsna-sub-k10n-a  metadata = K10-main metadata + dataset nartaa/rsna-knee-publication-swa-weights-20261007
Runtime (MODEL): = K9N-a (E replaces C: same 5 x TTR288 + 5 x TTR224 composition): 7.40 h central / 7.95 h slow at a 6.0 h
parent (k9n tests/runtime_k9n.py); the N stage's 20-min start guard and 8.6 h kill are unchanged.

  python build_k10n.py [--k10_build SP/k10/build] [--out SP/k10/build_n] [--overwrite] [--test] [--compare DIR]
--k10_build must be a FINAL build_k10.py --kit k10 build unless --test (then DO_NOT_PUSH_TEST.txt + do_not_push true).
"""
import argparse, ast, base64, copy, hashlib, json, os, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
SP = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, f'{HERE}/n949')
import make_child                                                     # noqa: E402
import make_k10_tools as mt                                           # noqa: E402

MAIN = 'rsna-sub-k10-fleet'
SLUG = 'rsna-sub-k10n-a'
W3_A = [0.1, 0.3, 0.6]
W3_VAR = [[0.3, 0.3, 0.4], [0.4, 0.3, 0.3]]                          # == the K9N-a kernel's variants
K5_CELL_SHA16 = 'e37f0604997cd40f'
K9N_CELL_SHA16 = 'bd87efb8fa89bbac'
N_CELL = f'{HERE}/cells/k9n_n949_cell.py'
FLEET_SLUG = 'rsna-knee-cnxt-fleet-k10'
FLEET_DS = f'eeshsaxena/{FLEET_SLUG}'
K9N_BUILD = f'{SP}/k9n/build/K9N_BUILD.json'


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def fsha(p):
    return sha(open(p, 'rb').read())


def tname(w):
    return 'submission_pfn_' + ''.join(f'{int(round(float(x) * 100)):02d}' for x in w) + '.csv'


def load_k10(k10b, test):
    kb = json.load(open(f'{k10b}/K5_BUILD.json', encoding='utf-8'))
    assert kb.get('k10') is True and kb.get('kit') == 'k10', f'{k10b} is not a build_k10.py --kit k10 build'
    markers = [m for m in ('DO_NOT_PUSH_SIMULATION.txt', 'DO_NOT_SUBMIT_INTERIM.txt') if os.path.exists(f'{k10b}/{m}')]
    final = not (kb.get('interim') or kb.get('simulated') or kb.get('do_not_submit') or markers)
    if not test:
        assert final, f'{k10b}: not a FINAL K10 build (markers {markers}); --test builds a DO_NOT_PUSH test kit'
        assert kb['sets'] == {'E': 5, 'D': 5} and kb['n_members'] == 10, kb['sets']
    kdir = f'{k10b}/kernel/{MAIN}'
    meta = json.load(open(f'{kdir}/kernel-metadata.json', encoding='utf-8'))
    nbb = open(f'{kdir}/{meta["code_file"]}', 'rb').read()
    assert sha(nbb)[:16] == kb['kernels'][MAIN]['nb_sha16'], 'K10-main notebook != its K5_BUILD.json sha'
    nb = json.loads(nbb.decode('utf-8'))
    assert len(nb['cells']) == 32, len(nb['cells'])
    hdr, body = ''.join(nb['cells'][31]['source']).split('\n', 1)
    k10cell = open(f'{HERE}/cells/k10_fleet_cell.py', encoding='utf-8').read()
    assert body == k10cell == mt.make()['cells/k10_fleet_cell.py'] and sha(body)[:16] == kb['cell_sha16'], 'cell 31 != K10 cell'
    assert sha(mt.unpatch_cell(body, FLEET_SLUG))[:16] == K5_CELL_SHA16, 'K10 cell minus K10-DS != the K5 cell'
    cfg = ast.literal_eval(hdr[len('K2_CFG = '):].split('   #', 1)[0])
    assert cfg['MAIN_MODE'] == 'strict' and abs(cfg['W_MAIN'] - 0.5) < 1e-9 and 'OUTER_FLAT' not in cfg, cfg
    assert cfg['MEMBERS'] == kb['members'] and cfg.get('TTA') == kb['tta'], 'cell 31 K2_CFG members / TTA != K5_BUILD.json'
    assert FLEET_DS in meta['dataset_sources'] and 'eeshsaxena/rsna-knee-cnxt-fleet' not in meta['dataset_sources']
    assert meta['id'] == f'eeshsaxena/{MAIN}'
    man_p = f'{k10b}/upload/{FLEET_SLUG}/fleet_manifest.json'
    man = json.load(open(man_p, encoding='utf-8'))
    assert man['checkpoints'] == kb['members'], 'manifest checkpoints != K5_BUILD members'
    return kb, meta, nb, cfg, man_p, final


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--k10_build', default=f'{HERE}/build')
    ap.add_argument('--out', default=f'{HERE}/build_n')
    ap.add_argument('--overwrite', action='store_true'); ap.add_argument('--test', action='store_true')
    ap.add_argument('--compare', default=None)
    a = ap.parse_args(argv)
    k10b, out = os.path.abspath(a.k10_build).replace('\\', '/'), os.path.abspath(a.out).replace('\\', '/')
    kb, meta10, nb10, cfg10, man_p, final = load_k10(k10b, a.test)
    child = make_child.make()
    assert open(make_child.OUT, encoding='utf-8', newline='').read() == child, 'n949/n949_child.py != make_child.make()'
    csha = sha(child)
    ncell = open(N_CELL, encoding='utf-8').read()
    assert sha(ncell)[:16] == K9N_CELL_SHA16, 'cells/k9n_n949_cell.py != the K9N N cell'
    assert fsha(N_CELL) == fsha(f'{SP}/k9n/cells/k9n_n949_cell.py'), 'N cell file != the k9n kit file (bytes)'
    if os.path.isfile(K9N_BUILD):
        k9nb = json.load(open(K9N_BUILD))
        assert k9nb['child_sha256'] == csha and k9nb['n_cell_sha16'] == K9N_CELL_SHA16 and \
            k9nb['weights_sha256'] == make_child.WEIGHTS_SHA256, 'child / N cell / weights != the K9N kit'
        assert k9nb['kernels']['rsna-sub-k9n-a']['W3'] == W3_A and k9nb['kernels']['rsna-sub-k9n-a']['W3_VARIANTS'] == W3_VAR, \
            'K10N-a weights != the K9N-a kernel weights'
    if os.path.exists(out) and os.listdir(out):
        assert a.overwrite, f'{out} exists (pass --overwrite)'
        shutil.rmtree(out)
    os.makedirs(out)
    cfg_n = {'W3': W3_A, 'W3_VARIANTS': W3_VAR, 'TAG': 'K10N-a', 'CHILD_SHA256': csha,
             'WEIGHTS_FILE': make_child.WEIGHTS_FILE, 'WEIGHTS_SHA256': make_child.WEIGHTS_SHA256,
             'WEIGHTS_DS': make_child.WEIGHTS_DS, 'SRC_SHA256': make_child.SRC_SHA256}
    hdr = (f'K9N_CFG = {cfg_n!r}   # literal, written by k10/build_k10n.py (pushed kernels get no env vars)\n'
           f"_K9N_CHILD_B64 = '{base64.b64encode(child.encode()).decode()}'   # n949_child.py (K9N-P patched public "
           f'.949 cell 4), sha256 {csha}\n')
    src = hdr + ncell
    compile(src, 'k9n_n949_stage', 'exec')
    bad = [l for l in src.splitlines() if '/kaggle/' in l and 'kaggle/input' not in l and 'kaggle/working' not in l
           and not l.lstrip().startswith('#')]
    assert not bad, bad[:3]
    nb = copy.deepcopy(nb10)
    nb['cells'].append({'cell_type': 'code', 'execution_count': None, 'metadata': {'k2': 'k9n_n949_stage'},
                        'outputs': [], 'source': src.splitlines(keepends=True)})
    kd = f'{out}/kernel/{SLUG}'
    os.makedirs(kd)
    code_file = SLUG.replace('-', '_') + '.ipynb'
    with open(f'{kd}/{code_file}', 'w', encoding='utf-8') as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    m = dict(meta10)
    m.update(id=f'eeshsaxena/{SLUG}', title=SLUG.replace('-', ' '), code_file=code_file, is_private=True)
    m['dataset_sources'] = list(dict.fromkeys(list(meta10['dataset_sources']) + [make_child.WEIGHTS_DS]))
    assert m.get('enable_gpu') and not m.get('enable_internet') and m.get('machine_shape') == 'NvidiaTeslaT4'
    assert 'docker_image' not in m
    with open(f'{kd}/kernel-metadata.json', 'w', encoding='utf-8') as fh:
        json.dump(m, fh, indent=2)
    chk = json.load(open(f'{kd}/{code_file}', encoding='utf-8'))
    assert len(chk['cells']) == 33
    for i in range(32):
        assert chk['cells'][i] == nb10['cells'][i], f'cell {i} != K10-main'
    assert ''.join(chk['cells'][32]['source']) == src
    mm = json.load(open(f'{kd}/kernel-metadata.json', encoding='utf-8'))
    for k in ('kernel_sources', 'competition_sources', 'model_sources', 'machine_shape', 'enable_gpu', 'enable_internet'):
        assert mm.get(k) == meta10.get(k), (SLUG, k)
    assert mm['dataset_sources'][:-1] == meta10['dataset_sources'] and mm['dataset_sources'][-1] == make_child.WEIGHTS_DS
    k9n_nb = f'{SP}/k9n/build/kernel/rsna-sub-k9n-a/rsna_sub_k9n_a.ipynb'
    if os.path.isfile(k9n_nb):                                           # cell 32 == K9N-a's except the config literal line
        c32 = ''.join(json.load(open(k9n_nb, encoding='utf-8'))['cells'][32]['source'])
        assert c32.split('\n', 1)[1] == src.split('\n', 1)[1], 'cell 32 (after the K9N_CFG line) != the K9N-a kernel'
        k9cfg = ast.literal_eval(c32.split('\n', 1)[0][len('K9N_CFG = '):].split('   #', 1)[0])
        assert {k: v for k, v in k9cfg.items() if k != 'TAG'} == {k: v for k, v in cfg_n.items() if k != 'TAG'}, \
            'K9N_CFG != the K9N-a config (TAG aside)'
    info = dict(built=time.strftime('%Y-%m-%d %H:%M:%S'), k10_build=k10b, k10_final=final, do_not_push=bool(a.test or not final),
                k10_k5_build_sha256=fsha(f'{k10b}/K5_BUILD.json'), k10_main_nb_sha16=kb['kernels'][MAIN]['nb_sha16'],
                fleet_manifest=man_p, fleet_manifest_sha256=fsha(man_p), fleet_dataset=FLEET_DS, members=kb['members'],
                tta=kb['tta'], child_sha256=csha, src_sha256=make_child.SRC_SHA256, weights_file=make_child.WEIGHTS_FILE,
                weights_sha256=make_child.WEIGHTS_SHA256, weights_ds=make_child.WEIGHTS_DS, n_cell_sha16=K9N_CELL_SHA16,
                kernels={SLUG: dict(tag='K10N-a', W3=W3_A, W3_VARIANTS=W3_VAR, dir=kd, code_file=code_file,
                                    nb_sha256=fsha(f'{kd}/{code_file}'), meta_sha256=fsha(f'{kd}/kernel-metadata.json'),
                                    cell32_sha16=sha(src)[:16], main_file=tname(W3_A),
                                    files=['submission.csv', 'submission_n949.csv'] + [tname(w) for w in [W3_A] + W3_VAR])})
    info['simulated'] = bool(kb.get('simulated'))
    if info['do_not_push']:
        open(f'{out}/DO_NOT_PUSH_TEST.txt', 'w').write('TEST K10N BUILD (from a non-final K10 build or --test). NEVER push, '
                                                      'NEVER submit.\n')
    if info['simulated']:
        open(f'{out}/DO_NOT_PUSH_SIMULATION.txt', 'w').write('SIMULATION BUILD (K10 members from the ~/k10_test/sim root). '
                                                            'TEST ONLY. NEVER push, NEVER submit.\n')
    json.dump(info, open(f'{out}/K10N_BUILD.json', 'w'), indent=1)
    if a.compare:
        cmpd = a.compare.replace('\\', '/')
        other = json.load(open(f'{cmpd}/K10N_BUILD.json'))
        for key in ('k10_main_nb_sha16', 'fleet_manifest_sha256', 'members', 'tta', 'child_sha256', 'weights_sha256'):
            assert other.get(key) == info[key], f'--compare: K10N_BUILD.json {key} differs from {cmpd}'
        for f in (code_file, 'kernel-metadata.json'):
            assert fsha(f'{kd}/{f}') == fsha(f'{cmpd}/kernel/{SLUG}/{f}'), f'--compare: {f} differs from {cmpd}'
        print(f'COMPARE OK: {SLUG} byte-identical to {cmpd}')
    tag = 'FINAL' if not info['do_not_push'] else 'TEST ONLY (DO_NOT_PUSH)'
    print(f'{SLUG}: 33 cells (0-31 == K10-main {kb["kernels"][MAIN]["nb_sha16"]}), cell 32 N stage W3 P/F/N {W3_A} variants '
          f'{W3_VAR}, notebook sha256 {info["kernels"][SLUG]["nb_sha256"][:16]}, datasets {mm["dataset_sources"]}')
    print(f'\nK10N BUILD OK {tag}: {out} | child {csha[:16]} | weights {make_child.WEIGHTS_SHA256[:16]} | K10 members '
          f'{len(kb["members"])} | K10N-a {W3_A}')
    return info


if __name__ == '__main__':
    main()
