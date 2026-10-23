#!/usr/bin/env python3
"""build_kfn.py (2026-10-09, Claude). Local only: nothing is pushed, versioned or submitted.

Two PARENT-FREE kernels from ONE K9 FINAL build (they read the SAME, already live, version of eeshsaxena/rsna-knee-cnxt-fleet
as K9 / K9N; no dataset of their own; no parent cell, dataset, kernel source or model source):
  FN     eeshsaxena/rsna-sub-k9fn   cells: md | FN-IDS | K9 fleet stage (the K5 cell, byte-identical body; K2_CFG = the
                                    K9-main literal (strict, W .50, the 10 members, TTA AB) + RHO_MIN_N 1e9) | N stage
                                    (cells/fn_n949_cell.py, MODE FN, RHO_REF F) -> submission.csv = rank_pct(.40 r(F) + .60 r(N))
                                    (P/F/N 0/.40/.60; P = the id-only placeholder, weight 0)
  N-only eeshsaxena/rsna-sub-nonly  cells: md | FN-IDS | N stage (MODE NONLY, RHO_REF NONE) -> submission.csv =
                                    rank_pct(r(N)) (P/F/N 0/0/1), no fleet stage
Metadata: competition + eeshsaxena/rsna-knee-cnxt-fleet (FN only) + nartaa/rsna-knee-publication-swa-weights-20261007;
private, T4, GPU, no internet, no docker pin (as K9 / K9N).

  python build_kfn.py [--k9_build SP/k9/build] [--k9n_build SP/k9n/build] [--out SP/kfn/build] [--w_fn .40,.60]
                      [--overwrite] [--test] [--compare DIR]
"""
import argparse, ast, base64, copy, hashlib, json, os, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
SP = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, f'{HERE}/n949')
import make_child, make_fn_cell                                       # noqa: E402

MAIN = 'rsna-sub-k9-fleet'
SLUG_FN, SLUG_NO = 'rsna-sub-k9fn', 'rsna-sub-nonly'
K5_CELL_SHA16 = 'e37f0604997cd40f'
K9N_CELL_SHA16 = 'bd87efb8fa89bbac'
FLEET_DS = 'eeshsaxena/rsna-knee-cnxt-fleet'
COMP = 'rsna-knee-abnormality-detection'
IDS_CELL = f'{HERE}/cells/fn_ids_cell.py'
FLEET_CELL = f'{HERE}/cells/k2_fleet_cell.py'
RHO_OFF = 10 ** 9
MD_FN = """# rsna-sub-k9fn: parent-free fleet + N (rank blend F .40 / N .60)

* **F**: our 10-member ConvNeXt-T fleet (dataset `eeshsaxena/rsna-knee-cnxt-fleet`, 5 x TTR288 + 5 x TTR224, test-time
  augmentation A+B), run by our own worker.
* **N**: the public .949 CoAtNet, notebook `nartaa/rsna-knee-0949-anatomical-mirror`, weights dataset
  `nartaa/rsna-knee-publication-swa-weights-20261007`. Inference code and weights by **nartaa**, built on
  **dreaddevelopment**'s Raptor CoAtNet/MIL recipe and **timm**; model-use terms: research and educational use, including
  the RSNA Knee competition. Its inference cell runs as a child process, verbatim except marked I/O patches.
* No parent pipeline: the first code cell only reads the test ids. submission.csv = per-label average-tie percentile rank
  of .40 rank(F) + .60 rank(N).
* Strict fail-loud: any stage failure leaves no submission.csv.
"""
MD_NO = """# rsna-sub-nonly: N alone in our code path (calibration of N)

* **N**: the public .949 CoAtNet, notebook `nartaa/rsna-knee-0949-anatomical-mirror`, weights dataset
  `nartaa/rsna-knee-publication-swa-weights-20261007`. Inference code and weights by **nartaa**, built on
  **dreaddevelopment**'s Raptor CoAtNet/MIL recipe and **timm**; model-use terms: research and educational use, including
  the RSNA Knee competition. Its inference cell runs as a child process, verbatim except marked I/O patches.
* submission.csv = per-label average-tie percentile rank of N's probabilities (the same N path as rsna-sub-k9fn).
* Strict fail-loud: any failure leaves no submission.csv.
"""


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def fsha(p):
    return sha(open(p, 'rb').read())


def tname(w):
    return 'submission_pfn_' + ''.join(f'{int(round(float(x) * 100)):02d}' for x in w) + '.csv'


def parent_slugs(meta9):
    """Every parent-only source of the K9-main kernel (must never appear in an FN / N-only kernel)."""
    ds = [d for d in meta9['dataset_sources'] if d != FLEET_DS]
    return ds + list(meta9.get('kernel_sources') or []) + list(meta9.get('model_sources') or [])


def load_k9(k9b, test):
    kb = json.load(open(f'{k9b}/K5_BUILD.json', encoding='utf-8'))
    assert kb.get('k9') is True, f'{k9b} is not a build_k9.py build'
    markers = [m for m in ('DO_NOT_PUSH_SIMULATION.txt', 'DO_NOT_SUBMIT_INTERIM.txt') if os.path.exists(f'{k9b}/{m}')]
    final = not (kb.get('interim') or kb.get('simulated') or kb.get('do_not_submit') or markers)
    if not test:
        assert final, f'{k9b}: not a FINAL K9 build (markers {markers}); --test builds a DO_NOT_PUSH test kit'
        assert kb['sets'] == {'C': 5, 'D': 5} and kb['n_members'] == 10, kb['sets']
    kdir = f'{k9b}/kernel/{MAIN}'
    meta = json.load(open(f'{kdir}/kernel-metadata.json', encoding='utf-8'))
    nbb = open(f'{kdir}/{meta["code_file"]}', 'rb').read()
    assert sha(nbb)[:16] == kb['kernels'][MAIN]['nb_sha16'], 'K9-main notebook != its K5_BUILD.json sha'
    nb = json.loads(nbb.decode('utf-8'))
    assert len(nb['cells']) == 32, len(nb['cells'])
    c31 = nb['cells'][31]
    hdr, body = ''.join(c31['source']).split('\n', 1)
    assert hdr.startswith('K2_CFG = ') and sha(body)[:16] == K5_CELL_SHA16 == kb['cell_sha16'], 'cell 31 != the K5 cell'
    assert body == open(FLEET_CELL, encoding='utf-8').read(), 'cell 31 != kfn/cells/k2_fleet_cell.py'
    cfg = ast.literal_eval(hdr[len('K2_CFG = '):].split('   #', 1)[0])
    assert cfg['MAIN_MODE'] == 'strict' and abs(cfg['W_MAIN'] - 0.5) < 1e-9 and 'OUTER_FLAT' not in cfg, cfg
    assert 'RHO_MIN_N' not in cfg, cfg
    assert cfg['MEMBERS'] == kb['members'] and cfg.get('TTA') == kb['tta'], 'cell 31 K2_CFG members / TTA != K5_BUILD.json'
    assert FLEET_DS in meta['dataset_sources'] and meta['id'] == f'eeshsaxena/{MAIN}'
    man_p = f'{k9b}/upload/rsna-knee-cnxt-fleet/fleet_manifest.json'
    man = json.load(open(man_p, encoding='utf-8'))
    assert man['checkpoints'] == kb['members'], 'manifest checkpoints != K5_BUILD members'
    return kb, meta, nb, c31, body, cfg, man_p, final


def n_cell(cfg_n, child_src):
    body = make_fn_cell.make()
    assert body == open(make_fn_cell.OUT, encoding='utf-8', newline='').read(), 'cells/fn_n949_cell.py != make_fn_cell.make()'
    hdr = (f'K9N_CFG = {cfg_n!r}   # literal, written by kfn/build_kfn.py (pushed kernels get no env vars)\n'
           f"_K9N_CHILD_B64 = '{base64.b64encode(child_src.encode()).decode()}'   # n949_child.py (K9N-P patched public "
           f'.949 cell 4), sha256 {cfg_n["CHILD_SHA256"]}\n')
    src = hdr + body
    compile(src, 'fn_n949_stage', 'exec')
    return src, body


def nb_cell(kind, src, meta):
    c = {'cell_type': kind, 'metadata': meta, 'source': src.splitlines(keepends=True)}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--k9_build', default=f'{SP}/k9/build')
    ap.add_argument('--k9n_build', default=f'{SP}/k9n/build')
    ap.add_argument('--out', default=f'{HERE}/build')
    ap.add_argument('--w_fn', default='0.40,0.60', help='F,N weights of the FN kernel (P is 0)')
    ap.add_argument('--overwrite', action='store_true'); ap.add_argument('--test', action='store_true')
    ap.add_argument('--compare', default=None)
    a = ap.parse_args(argv)
    k9b, out = os.path.abspath(a.k9_build).replace('\\', '/'), os.path.abspath(a.out).replace('\\', '/')
    kb, meta9, nb9, c31, body31, cfg9, man_p, final = load_k9(k9b, a.test)
    # the N path must be the tested K9N one: same child bytes, same N cell source before the FN-P patches
    k9nb = json.load(open(f'{os.path.abspath(a.k9n_build)}/K9N_BUILD.json'))
    child = make_child.make()
    assert open(make_child.OUT, encoding='utf-8', newline='').read() == child, 'n949/n949_child.py != make_child.make()'
    csha = sha(child)
    assert csha == k9nb['child_sha256'], f'child {csha[:16]} != the K9N kit child {k9nb["child_sha256"][:16]}'
    assert sha(open(make_fn_cell.SRC, encoding='utf-8').read())[:16] == K9N_CELL_SHA16 == k9nb['n_cell_sha16'], 'K9N N cell changed'
    if not a.test:
        assert k9nb['fleet_manifest_sha256'] == fsha(man_p), 'fleet manifest != the one the K9N kit (live K9 version) used'
    wf, wn = [float(x) for x in a.w_fn.split(',')]
    assert abs(wf + wn - 1) < 1e-9 and wf > 0 and wn > 0
    W_FN = [0.0, round(wf, 4), round(wn, 4)]
    VAR_FN = [w for w in ([0.0, 0.3, 0.7], [0.0, 0.5, 0.5]) if w != W_FN]
    if os.path.exists(out) and os.listdir(out):
        assert a.overwrite, f'{out} exists (pass --overwrite)'
        shutil.rmtree(out)
    os.makedirs(out)
    ids_src = open(IDS_CELL, encoding='utf-8').read()
    compile(ids_src, 'fn_ids', 'exec')
    k2cfg = dict(cfg9); k2cfg['RHO_MIN_N'] = RHO_OFF
    fleet_src = (f'K2_CFG = {k2cfg!r}   # literal, written by kfn/build_kfn.py = the K9-main K2_CFG + RHO_MIN_N 1e9 (FN: the '
                 f'parent is the FN-IDS id-only placeholder; the F~N Spearman gate runs in the N stage)\n' + body31)
    base_cfg = {'CHILD_SHA256': csha, 'WEIGHTS_FILE': make_child.WEIGHTS_FILE, 'WEIGHTS_SHA256': make_child.WEIGHTS_SHA256,
                'WEIGHTS_DS': make_child.WEIGHTS_DS, 'SRC_SHA256': make_child.SRC_SHA256}
    specs = {
        SLUG_FN: dict(tag='FN', W3=W_FN, VAR=VAR_FN, mode='FN', rho='F', md=MD_FN,
                      ds=[FLEET_DS, make_child.WEIGHTS_DS]),
        SLUG_NO: dict(tag='NONLY', W3=[0.0, 0.0, 1.0], VAR=[], mode='NONLY', rho='NONE', md=MD_NO,
                      ds=[make_child.WEIGHTS_DS]),
    }
    psl = parent_slugs(meta9)
    info = dict(built=time.strftime('%Y-%m-%d %H:%M:%S'), k9_build=k9b, k9_final=final, do_not_push=bool(a.test or not final),
                k9_k5_build_sha256=fsha(f'{k9b}/K5_BUILD.json'), k9_main_nb_sha16=kb['kernels'][MAIN]['nb_sha16'],
                fleet_manifest=man_p, fleet_manifest_sha256=fsha(man_p), members=kb['members'], tta=kb['tta'],
                child_sha256=csha, src_sha256=make_child.SRC_SHA256, weights_file=make_child.WEIGHTS_FILE,
                weights_sha256=make_child.WEIGHTS_SHA256, weights_ds=make_child.WEIGHTS_DS,
                k9n_n_cell_sha16=K9N_CELL_SHA16, fn_cell_sha256=sha(make_fn_cell.make()), ids_cell_sha256=sha(ids_src),
                fleet_cell_sha16=K5_CELL_SHA16, k2_cfg=k2cfg, parent_sources_excluded=psl, kernels={})
    for slug, s in specs.items():
        cfg_n = dict(W3=s['W3'], W3_VARIANTS=s['VAR'], TAG=s['tag'], MODE=s['mode'], RHO_REF=s['rho'], **base_cfg)
        nsrc, _ = n_cell(cfg_n, child)
        cells = [nb_cell('markdown', s['md'], {}), nb_cell('code', ids_src, {'kfn': 'fn_ids'})]
        if s['mode'] == 'FN':
            c = copy.deepcopy(c31); c['source'] = fleet_src.splitlines(keepends=True)
            cells.append(c)
        cells.append(nb_cell('code', nsrc, {'k2': 'fn_n949_stage'}))
        nb = {k: copy.deepcopy(v) for k, v in nb9.items() if k != 'cells'}
        nb['cells'] = cells
        kd = f'{out}/kernel/{slug}'
        os.makedirs(kd)
        code_file = slug.replace('-', '_') + '.ipynb'
        with open(f'{kd}/{code_file}', 'w', encoding='utf-8') as fh:
            json.dump(nb, fh, indent=1, ensure_ascii=False)
        m = {'id': f'eeshsaxena/{slug}', 'title': slug.replace('-', ' '), 'code_file': code_file, 'language': 'python',
             'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
             'enable_internet': False, 'dataset_sources': s['ds'], 'kernel_sources': [], 'competition_sources': [COMP],
             'model_sources': [], 'machine_shape': 'NvidiaTeslaT4'}
        assert meta9['machine_shape'] == m['machine_shape'] and meta9['competition_sources'] == [COMP]
        with open(f'{kd}/kernel-metadata.json', 'w', encoding='utf-8') as fh:
            json.dump(m, fh, indent=2)
        # self-checks: no parent source anywhere in the notebook text (incl. the decoded child), cell layout
        txt = open(f'{kd}/{code_file}', encoding='utf-8').read() + '\n' + child
        hits = [p for p in psl if '/' + p.split('/')[1] in txt or p in txt]   # a mount path /NAME or the slug owner/NAME
        assert not hits, f'{slug}: parent sources referenced {hits}'
        chk = json.load(open(f'{kd}/{code_file}', encoding='utf-8'))
        assert len(chk['cells']) == (4 if s['mode'] == 'FN' else 3)
        info['kernels'][slug] = dict(tag=s['tag'], mode=s['mode'], rho_ref=s['rho'], W3=s['W3'], W3_VARIANTS=s['VAR'],
                                     dir=kd, code_file=code_file, nb_sha256=fsha(f'{kd}/{code_file}'),
                                     meta_sha256=fsha(f'{kd}/kernel-metadata.json'), n_cell_sha16=sha(nsrc)[:16],
                                     dataset_sources=s['ds'], main_file=tname(s['W3']),
                                     files=['submission.csv', 'submission_n949.csv'] + [tname(w) for w in [s['W3']] + s['VAR']])
        print(f'{slug}: {len(chk["cells"])} cells, MODE {s["mode"]}, W3 P/F/N {s["W3"]} variants {s["VAR"]}, notebook sha256 '
              f'{info["kernels"][slug]["nb_sha256"][:16]}, datasets {s["ds"]}')
    if info['do_not_push']:
        open(f'{out}/DO_NOT_PUSH_TEST.txt', 'w').write('TEST KFN BUILD (non-final K9 build or --test). NEVER push, NEVER submit.\n')
    json.dump(info, open(f'{out}/FN_BUILD.json', 'w'), indent=1)
    if a.compare:
        cmpd = a.compare.replace('\\', '/')
        other = json.load(open(f'{cmpd}/FN_BUILD.json'))
        for key in ('k9_main_nb_sha16', 'fleet_manifest_sha256', 'members', 'tta', 'child_sha256', 'weights_sha256',
                    'fn_cell_sha256', 'ids_cell_sha256'):
            assert other.get(key) == info[key], f'--compare: FN_BUILD.json {key} differs from {cmpd}'
        for slug, k in info['kernels'].items():
            for f in (k['code_file'], 'kernel-metadata.json'):
                x, y = f'{out}/kernel/{slug}/{f}', f'{cmpd}/kernel/{slug}/{f}'
                assert os.path.isfile(y) and fsha(x) == fsha(y), f'--compare: {slug}/{f} differs from {cmpd}'
        print(f'COMPARE OK: both kernels byte-identical to {cmpd}')
    tag = 'FINAL' if not info['do_not_push'] else 'TEST ONLY (DO_NOT_PUSH)'
    print(f'\nKFN BUILD OK {tag}: {out} | child {csha[:16]} | weights {make_child.WEIGHTS_SHA256[:16]} | K9 members '
          f'{len(kb["members"])} | FN P/F/N {W_FN} | N-only [0, 0, 1] | parent sources excluded: {len(psl)}')
    return info


if __name__ == '__main__':
    main()
