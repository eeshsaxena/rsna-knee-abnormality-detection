#!/usr/bin/env python3
"""build_k9n.py (2026-10-08, Claude). Local only: nothing is pushed, versioned or submitted.

K9N = the K9-main kernel (parent repro_0946 cells 0-30 + the K5 fleet cell 31, STRICT W .50, the 10 K9 members, TTA AB)
      + cell 32 = the K9N N stage (cells/k9n_n949_cell.py): the public .949 CoAtNet (nartaa) as a child process after the
      fleet stage, then submission.csv = rank_pct(wP r(P) + wF r(F) + wN r(N)), strict fail-loud.
Two kernels from ONE K9 FINAL build (they read the SAME dataset version of eeshsaxena/rsna-knee-cnxt-fleet as K9; K9N
ships no dataset of its own):
  K9N-a  eeshsaxena/rsna-sub-k9n-a   W3 P/F/N .10/.30/.60  (k9n_weights.py: ~minimax-regret, N-heavy; MODEL)
  K9N-b  eeshsaxena/rsna-sub-k9n-b   W3 P/F/N .30/.30/.40  (hedge: less N if its public .949 does not carry; MODEL)
  each also writes the other triples as submission_pfn_<PPFFNN>.csv (diagnostics; only submission.csv is scored).
The kernel metadata = K9-main metadata + dataset nartaa/rsna-knee-publication-swa-weights-20261007.

  python build_k9n.py [--k9_build SP/k9/build] [--out SP/k9n/build] [--w3_a .10,.30,.60] [--w3_b .30,.30,.40]
                      [--overwrite] [--test] [--compare DIR]
--k9_build must be a FINAL build_k9.py build (K5_BUILD.json k9, not interim / simulated / do_not_submit, no DO_NOT_*
marker) unless --test (then OUT gets DO_NOT_PUSH_TEST.txt and K9N_BUILD.json do_not_push true).
--compare DIR: after building, every kernel file must be byte-identical to DIR/kernel/<slug>/ (e.g. the K9N build made
from the k9n/k9base copy vs a rebuild from the main session's k9/build).
"""
import argparse, base64, copy, hashlib, json, os, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
SP = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, f'{HERE}/n949')
import make_child                                                     # noqa: E402

MAIN = 'rsna-sub-k9-fleet'
SLUGS = {'a': 'rsna-sub-k9n-a', 'b': 'rsna-sub-k9n-b'}
W3_DEFAULT = {'a': '0.10,0.30,0.60', 'b': '0.30,0.30,0.40'}
W3_EXTRA = [[0.40, 0.30, 0.30]]                                       # the originally requested K9N-b triple (diagnostic file)
K5_CELL_SHA16 = 'e37f0604997cd40f'
N_CELL = f'{HERE}/cells/k9n_n949_cell.py'
FLEET_DS = 'eeshsaxena/rsna-knee-cnxt-fleet'


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def fsha(p):
    return sha(open(p, 'rb').read())


def w3(s):
    w = [float(x) for x in str(s).split(',') if x.strip()]
    assert len(w) == 3 and min(w) >= 0 and abs(sum(w) - 1) < 1e-9 and w[2] > 0, f'W3 {s}: 3 weights P,F,N summing to 1, N > 0'
    return [float(f'{x:.4f}') for x in w]


def tname(w):
    return 'submission_pfn_' + ''.join(f'{int(round(float(x) * 100)):02d}' for x in w) + '.csv'


def load_k9(k9b, test):
    kb = json.load(open(f'{k9b}/K5_BUILD.json', encoding='utf-8'))
    assert kb.get('k9') is True, f'{k9b} is not a build_k9.py build'
    markers = [m for m in ('DO_NOT_PUSH_SIMULATION.txt', 'DO_NOT_SUBMIT_INTERIM.txt') if os.path.exists(f'{k9b}/{m}')]
    final = not (kb.get('interim') or kb.get('simulated') or kb.get('do_not_submit') or markers)
    if not test:
        assert final, f'{k9b}: not a FINAL K9 build (interim {kb.get("interim")}, simulated {kb.get("simulated")}, ' \
                      f'do_not_submit {kb.get("do_not_submit")}, markers {markers}); --test builds a DO_NOT_PUSH test kit'
        assert kb['sets'] == {'C': 5, 'D': 5} and kb['n_members'] == 10, kb['sets']
    kdir = f'{k9b}/kernel/{MAIN}'
    meta = json.load(open(f'{kdir}/kernel-metadata.json', encoding='utf-8'))
    nbp = f'{kdir}/{meta["code_file"]}'
    nbb = open(nbp, 'rb').read()
    assert sha(nbb)[:16] == kb['kernels'][MAIN]['nb_sha16'], 'K9-main notebook != its K5_BUILD.json sha'
    nb = json.loads(nbb.decode('utf-8'))
    assert len(nb['cells']) == 32, len(nb['cells'])
    src31 = ''.join(nb['cells'][31]['source'])
    hdr, body = src31.split('\n', 1)
    assert hdr.startswith('K2_CFG = ') and sha(body)[:16] == K5_CELL_SHA16 == kb['cell_sha16'], 'cell 31 != the K5 cell'
    assert body == open(f'{HERE}/cells/k2_fleet_cell.py', encoding='utf-8').read(), 'cell 31 != k9n/cells/k2_fleet_cell.py'
    import ast
    cfg = ast.literal_eval(hdr[len('K2_CFG = '):].split('   #', 1)[0])
    assert cfg['MAIN_MODE'] == 'strict' and abs(cfg['W_MAIN'] - 0.5) < 1e-9 and 'OUTER_FLAT' not in cfg, cfg
    assert cfg['MEMBERS'] == kb['members'] and cfg.get('TTA') == kb['tta'], 'cell 31 K2_CFG members / TTA != K5_BUILD.json'
    assert FLEET_DS in meta['dataset_sources'] and meta['id'] == f'eeshsaxena/{MAIN}'
    man_p = f'{k9b}/upload/rsna-knee-cnxt-fleet/fleet_manifest.json'
    man = json.load(open(man_p, encoding='utf-8'))
    assert man['checkpoints'] == kb['members'], 'manifest checkpoints != K5_BUILD members'
    return kb, meta, nb, cfg, man_p, final


def n_cell(cfg_n, child_src):
    body = open(N_CELL, encoding='utf-8').read()
    hdr = (f'K9N_CFG = {cfg_n!r}   # literal, written by k9n/build_k9n.py (pushed kernels get no env vars)\n'
           f"_K9N_CHILD_B64 = '{base64.b64encode(child_src.encode()).decode()}'   # n949_child.py (K9N-P patched public "
           f'.949 cell 4), sha256 {cfg_n["CHILD_SHA256"]}\n')
    src = hdr + body
    compile(src, 'k9n_n949_stage', 'exec')
    bad = [l for l in src.splitlines() if '/kaggle/' in l and 'kaggle/input' not in l and 'kaggle/working' not in l
           and not l.lstrip().startswith('#')]
    assert not bad, bad[:3]
    return src, body


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--k9_build', default=f'{SP}/k9/build')
    ap.add_argument('--out', default=f'{HERE}/build')
    ap.add_argument('--w3_a', default=W3_DEFAULT['a']); ap.add_argument('--w3_b', default=W3_DEFAULT['b'])
    ap.add_argument('--overwrite', action='store_true'); ap.add_argument('--test', action='store_true')
    ap.add_argument('--compare', default=None)
    a = ap.parse_args(argv)
    k9b, out = os.path.abspath(a.k9_build).replace('\\', '/'), os.path.abspath(a.out).replace('\\', '/')
    kb, meta9, nb9, cfg9, man_p, final = load_k9(k9b, a.test)
    child = make_child.make()
    disk = open(make_child.OUT, encoding='utf-8', newline='').read()
    assert disk == child, 'n949/n949_child.py != make_child.make(): run python n949/make_child.py'
    csha = sha(child)
    W = {'a': w3(a.w3_a), 'b': w3(a.w3_b)}
    assert W['a'] != W['b']
    if os.path.exists(out) and os.listdir(out):
        assert a.overwrite, f'{out} exists (pass --overwrite)'
        shutil.rmtree(out)
    os.makedirs(out)
    info = dict(built=time.strftime('%Y-%m-%d %H:%M:%S'), k9_build=k9b, k9_final=final, do_not_push=bool(a.test or not final),
                k9_k5_build_sha256=fsha(f'{k9b}/K5_BUILD.json'), k9_main_nb_sha16=kb['kernels'][MAIN]['nb_sha16'],
                fleet_manifest=man_p, fleet_manifest_sha256=fsha(man_p), members=kb['members'], tta=kb['tta'],
                child_sha256=csha, src_sha256=make_child.SRC_SHA256, weights_file=make_child.WEIGHTS_FILE,
                weights_sha256=make_child.WEIGHTS_SHA256, weights_ds=make_child.WEIGHTS_DS,
                n_cell_sha16=sha(open(N_CELL, encoding='utf-8').read())[:16], kernels={})
    for tag in ('a', 'b'):
        slug = SLUGS[tag]
        var = [W[t] for t in ('a', 'b') if t != tag] + [w for w in W3_EXTRA if w not in (W['a'], W['b'])]
        cfg_n = {'W3': W[tag], 'W3_VARIANTS': var, 'TAG': f'K9N-{tag}', 'CHILD_SHA256': csha,
                 'WEIGHTS_FILE': make_child.WEIGHTS_FILE, 'WEIGHTS_SHA256': make_child.WEIGHTS_SHA256,
                 'WEIGHTS_DS': make_child.WEIGHTS_DS, 'SRC_SHA256': make_child.SRC_SHA256}
        src, body = n_cell(cfg_n, child)
        nb = copy.deepcopy(nb9)
        nb['cells'].append({'cell_type': 'code', 'execution_count': None, 'metadata': {'k2': 'k9n_n949_stage'},
                            'outputs': [], 'source': src.splitlines(keepends=True)})
        kd = f'{out}/kernel/{slug}'
        os.makedirs(kd)
        code_file = slug.replace('-', '_') + '.ipynb'
        with open(f'{kd}/{code_file}', 'w', encoding='utf-8') as fh:
            json.dump(nb, fh, indent=1, ensure_ascii=False)
        m = dict(meta9)
        m.update(id=f'eeshsaxena/{slug}', title=slug.replace('-', ' '), code_file=code_file, is_private=True)
        m['dataset_sources'] = list(dict.fromkeys(list(meta9['dataset_sources']) + [make_child.WEIGHTS_DS]))
        assert m.get('enable_gpu') and not m.get('enable_internet') and m.get('machine_shape') == 'NvidiaTeslaT4'
        assert 'docker_image' not in m, 'K9N keeps the K9 environment (no docker pin), like every scored kit'
        with open(f'{kd}/kernel-metadata.json', 'w', encoding='utf-8') as fh:
            json.dump(m, fh, indent=2)
        chk = json.load(open(f'{kd}/{code_file}', encoding='utf-8'))
        assert len(chk['cells']) == 33
        for i in range(32):
            assert chk['cells'][i] == nb9['cells'][i], f'{slug}: cell {i} != K9-main'
        assert ''.join(chk['cells'][32]['source']) == src
        mm = json.load(open(f'{kd}/kernel-metadata.json', encoding='utf-8'))
        for k in ('kernel_sources', 'competition_sources', 'model_sources', 'machine_shape', 'enable_gpu', 'enable_internet'):
            assert mm.get(k) == meta9.get(k), (slug, k)
        assert mm['dataset_sources'][:-1] == meta9['dataset_sources'] and mm['dataset_sources'][-1] == make_child.WEIGHTS_DS
        info['kernels'][slug] = dict(tag=f'K9N-{tag}', W3=W[tag], W3_VARIANTS=var, dir=kd, code_file=code_file,
                                     nb_sha256=fsha(f'{kd}/{code_file}'), meta_sha256=fsha(f'{kd}/kernel-metadata.json'),
                                     cell32_sha16=sha(src)[:16], main_file=tname(W[tag]),
                                     files=['submission.csv', 'submission_n949.csv'] + [tname(w) for w in [W[tag]] + var])
        print(f'{slug}: 33 cells (0-31 == K9-main {kb["kernels"][MAIN]["nb_sha16"]}), cell 32 N stage W3 P/F/N {W[tag]} '
              f'variants {var}, notebook sha256 {info["kernels"][slug]["nb_sha256"][:16]}, + dataset {make_child.WEIGHTS_DS}')
    if info['do_not_push']:
        open(f'{out}/DO_NOT_PUSH_TEST.txt', 'w').write('TEST K9N BUILD (from a non-final K9 build or --test). NEVER push, '
                                                      'NEVER submit.\n')
    json.dump(info, open(f'{out}/K9N_BUILD.json', 'w'), indent=1)
    if a.compare:
        cmpd = a.compare.replace('\\', '/')
        other = json.load(open(f'{cmpd}/K9N_BUILD.json'))
        for key in ('k9_main_nb_sha16', 'fleet_manifest_sha256', 'members', 'tta', 'child_sha256', 'weights_sha256'):
            assert other.get(key) == info[key], f'--compare: K9N_BUILD.json {key} differs from {cmpd} ({other.get(key)} vs {info[key]})'
        for slug, k in info['kernels'].items():
            for f in (k['code_file'], 'kernel-metadata.json'):
                x, y = f'{out}/kernel/{slug}/{f}', f'{cmpd}/kernel/{slug}/{f}'
                assert os.path.isfile(y) and fsha(x) == fsha(y), f'--compare: {slug}/{f} differs from {cmpd}'
        print(f'COMPARE OK: both kernels byte-identical to {cmpd}')
    tag = 'FINAL' if not info['do_not_push'] else 'TEST ONLY (DO_NOT_PUSH)'
    print(f'\nK9N BUILD OK {tag}: {out} | child sha256 {csha[:16]} (public cell 4 {make_child.SRC_SHA256[:16]} + '
          f'{len(make_child.P)} K9N-P patches) | weights {make_child.WEIGHTS_FILE} {make_child.WEIGHTS_SHA256[:16]} | '
          f'K9 members {len(kb["members"])} | a {W["a"]} b {W["b"]}')
    return info


if __name__ == '__main__':
    main()
