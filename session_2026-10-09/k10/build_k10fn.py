#!/usr/bin/env python3
"""build_k10fn.py (2026-10-09, Claude; from kfn/build_kfn.py). Local only: nothing is pushed, versioned or submitted.

PARENT-FREE K10 kernels (no parent cell, dataset, kernel source or model source), F .40 / N .60 as FN (kfn kit):
  --kit fn10    eeshsaxena/rsna-sub-k10fn   fleet = the 10 K10 members (E + D), dataset eeshsaxena/rsna-knee-cnxt-fleet-k10
  --kit fn10_3  eeshsaxena/rsna-sub-k10fn3  fleet = the 15 members C + D + E,   dataset eeshsaxena/rsna-knee-cnxt-fleet-k10w
cells: md | FN-IDS (cells/fn_ids_cell.py == kfn) | fleet stage (the K10-DS cell of the kit's dataset; K2_CFG = the K10-main
literal (strict, W .50, TTA AB) with MEMBERS / TTA of the kit + RHO_MIN_N 1e9, as kfn) | N stage (cells/fn_n949_cell.py ==
kfn's FN cell == the K9N cell + 10 FN-P patches; MODE FN, RHO_REF F, P/F/N 0/.40/.60, variants 0/.30/.70 and 0/.50/.50).
submission.csv = rank_pct(.40 r(F) + .60 r(N)). Strict fail-loud. One kernel per output dir (FN_BUILD.json for
fn10_check.py, which takes the slug from it).
Runtime MODEL (tests/runtime_k10fn.py): fleet stage + N stage + ~3 min boot; FN10 == FN (1.45 h central / 2.00 h slow);
FN10-3 adds 5 x TTR288 A+B members: ~1.9 h central / ~2.6 h slow, both far inside the 8.6 h wall.

  python build_k10fn.py --kit fn10|fn10_3 [--k10_build SP/k10/build] [--k10w_build SP/k10/build_w] [--out DIR]
                        [--overwrite] [--test] [--compare DIR]
"""
import argparse, ast, base64, copy, hashlib, json, os, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
SP = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, f'{HERE}/n949')
import make_child, make_fn_cell, make_k10_tools as mt                # noqa: E402

MAIN = 'rsna-sub-k10-fleet'
K5_CELL_SHA16 = 'e37f0604997cd40f'
K9N_CELL_SHA16 = 'bd87efb8fa89bbac'
COMP = 'rsna-knee-abnormality-detection'
IDS_CELL = f'{HERE}/cells/fn_ids_cell.py'
KFN_BUILD = f'{SP}/kfn/build/FN_BUILD.json'
KFN_NB = f'{SP}/kfn/build/kernel/rsna-sub-k9fn/rsna_sub_k9fn.ipynb'
RHO_OFF = 10 ** 9
KITS = {'fn10': dict(slug='rsna-sub-k10fn', ds='rsna-knee-cnxt-fleet-k10', cell='cells/k10_fleet_cell.py', sets='E + D',
                     n=10, out='build_fn', desc='10-member fleet: 5 x TTR288 frac75 s1045 (set E) + 5 x TTR224 frac75 s1043 '
                                                '(set D)'),
        'fn10_3': dict(slug='rsna-sub-k10fn3', ds='rsna-knee-cnxt-fleet-k10w', cell='cells/k10w_fleet_cell.py',
                       sets='C + D + E', n=15, out='build_fn3',
                       desc='15-member fleet: 5 x TTR288 mix50 s1044 (set C) + 5 x TTR224 frac75 s1043 (set D) + 5 x TTR288 '
                            'frac75 s1045 (set E)')}
MD = """# {slug}: parent-free fleet + N (rank blend F .40 / N .60)

* **F**: our {desc}, ConvNeXt-T, test-time augmentation A+B (dataset `eeshsaxena/{ds}`), run by our own worker.
* **N**: the public .949 CoAtNet, notebook `nartaa/rsna-knee-0949-anatomical-mirror`, weights dataset
  `nartaa/rsna-knee-publication-swa-weights-20261007`. Inference code and weights by **nartaa**, built on
  **dreaddevelopment**'s Raptor CoAtNet/MIL recipe and **timm**; model-use terms: research and educational use, including
  the RSNA Knee competition. Its inference cell runs as a child process, verbatim except marked I/O patches.
* No parent pipeline: the first code cell only reads the test ids. submission.csv = per-label average-tie percentile rank
  of .40 rank(F) + .60 rank(N).
* Strict fail-loud: any stage failure leaves no submission.csv.
"""


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def fsha(p):
    return sha(open(p, 'rb').read())


def tname(w):
    return 'submission_pfn_' + ''.join(f'{int(round(float(x) * 100)):02d}' for x in w) + '.csv'


def nb_cell(kind, src, meta):
    c = {'cell_type': kind, 'metadata': meta, 'source': src.splitlines(keepends=True)}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


def final_of(b):
    markers = [m for m in ('DO_NOT_PUSH_SIMULATION.txt', 'DO_NOT_SUBMIT_INTERIM.txt') if os.path.exists(f'{b}/{m}')]
    kb = json.load(open(f'{b}/K5_BUILD.json', encoding='utf-8'))
    return kb, not (kb.get('interim') or kb.get('simulated') or kb.get('do_not_submit') or markers), markers


def load(k10b, k10wb, kit, test):
    kb, fin, mk = final_of(k10b)
    assert kb.get('k10') is True and kb.get('kit') == 'k10', f'{k10b} is not a build_k10.py --kit k10 build'
    if not test:
        assert fin, f'{k10b}: not a FINAL K10 build (markers {mk}); --test builds a DO_NOT_PUSH test kit'
        assert kb['sets'] == {'E': 5, 'D': 5}
    meta = json.load(open(f'{k10b}/kernel/{MAIN}/kernel-metadata.json', encoding='utf-8'))
    nbb = open(f'{k10b}/kernel/{MAIN}/{meta["code_file"]}', 'rb').read()
    assert sha(nbb)[:16] == kb['kernels'][MAIN]['nb_sha16'], 'K10-main notebook != its K5_BUILD.json sha'
    nb = json.loads(nbb.decode('utf-8'))
    c31 = nb['cells'][31]
    hdr, body = ''.join(c31['source']).split('\n', 1)
    assert body == mt.make()['cells/k10_fleet_cell.py'] and sha(body)[:16] == kb['cell_sha16'], 'cell 31 != the K10 cell'
    cfg = ast.literal_eval(hdr[len('K2_CFG = '):].split('   #', 1)[0])
    assert cfg['MAIN_MODE'] == 'strict' and abs(cfg['W_MAIN'] - 0.5) < 1e-9 and 'OUTER_FLAT' not in cfg and 'RHO_MIN_N' not in cfg
    assert cfg['MEMBERS'] == kb['members'] and cfg.get('TTA') == kb['tta']
    K = KITS[kit]
    if kit == 'fn10':
        src_b, srckb, final = k10b, kb, fin
    else:
        srckb, finw, mkw = final_of(k10wb)
        assert srckb.get('k10') is True and srckb.get('kit') == 'k10w', f'{k10wb} is not a build_k10.py --kit k10w build'
        if not test:
            assert finw, f'{k10wb}: not a FINAL K10w build (markers {mkw})'
            assert srckb['sets'] == {'C': 5, 'D': 5, 'E': 5}
        assert not srckb['kernels'] and not os.path.exists(f'{k10wb}/kernel'), 'the k10w build must hold no kernel'
        e_d = [m for m in srckb['members'] if '_s1044t288_' not in m]
        assert e_d == kb['members'], 'k10w members minus set C != the K10 members (different E / D picks?)'
        assert {p['sha256'] for p in srckb['picks'] if p['set'] != 'C'} == {p['sha256'] for p in kb['picks']}, 'E / D bytes differ'
        src_b, final = k10wb, fin and finw
    man_p = f'{src_b}/upload/{K["ds"]}/fleet_manifest.json'
    man = json.load(open(man_p, encoding='utf-8'))
    assert man['checkpoints'] == srckb['members'] and len(srckb['members']) == K['n'] or test, (len(srckb['members']), K['n'])
    return kb, meta, nb, c31, cfg, srckb, man_p, final


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--kit', required=True, choices=sorted(KITS))
    ap.add_argument('--k10_build', default=f'{HERE}/build')
    ap.add_argument('--k10w_build', default=f'{HERE}/build_w')
    ap.add_argument('--out', default=None)
    ap.add_argument('--w_fn', default='0.40,0.60')
    ap.add_argument('--overwrite', action='store_true'); ap.add_argument('--test', action='store_true')
    ap.add_argument('--compare', default=None)
    a = ap.parse_args(argv)
    K = KITS[a.kit]
    k10b = os.path.abspath(a.k10_build).replace('\\', '/')
    k10wb = os.path.abspath(a.k10w_build).replace('\\', '/')
    out = os.path.abspath(a.out or f'{os.path.dirname(k10b)}/{K["out"]}').replace('\\', '/')
    kb, meta10, nb10, c31, cfg10, srckb, man_p, final = load(k10b, k10wb, a.kit, a.test)
    child = make_child.make()
    assert open(make_child.OUT, encoding='utf-8', newline='').read() == child, 'n949/n949_child.py != make_child.make()'
    csha = sha(child)
    kfnb = json.load(open(KFN_BUILD)) if os.path.isfile(KFN_BUILD) else None
    fn_body = make_fn_cell.make()
    assert fn_body == open(make_fn_cell.OUT, encoding='utf-8', newline='').read(), 'cells/fn_n949_cell.py != make_fn_cell.make()'
    assert sha(open(make_fn_cell.SRC, encoding='utf-8').read())[:16] == K9N_CELL_SHA16, 'K9N N cell changed'
    ids_src = open(IDS_CELL, encoding='utf-8').read()
    compile(ids_src, 'fn_ids', 'exec')
    if kfnb:
        assert kfnb['child_sha256'] == csha and kfnb['fn_cell_sha256'] == sha(fn_body) and kfnb['ids_cell_sha256'] == sha(ids_src), \
            'child / FN cell / FN-IDS cell != the kfn kit (tested 82/82 + box A0)'
    wf, wn = [float(x) for x in a.w_fn.split(',')]
    assert abs(wf + wn - 1) < 1e-9 and wf > 0 and wn > 0
    W_FN = [0.0, round(wf, 4), round(wn, 4)]
    VAR_FN = [w for w in ([0.0, 0.3, 0.7], [0.0, 0.5, 0.5]) if w != W_FN]
    if os.path.exists(out) and os.listdir(out):
        assert a.overwrite, f'{out} exists (pass --overwrite)'
        shutil.rmtree(out)
    os.makedirs(out)
    cell_body = mt.make()[K['cell']]
    assert open(f'{HERE}/{K["cell"]}', encoding='utf-8', newline='').read() == cell_body
    assert sha(mt.unpatch_cell(cell_body, K['ds']))[:16] == K5_CELL_SHA16
    k2cfg = dict(cfg10); k2cfg['MEMBERS'] = list(srckb['members']); k2cfg['TTA'] = dict(srckb['tta']); k2cfg['RHO_MIN_N'] = RHO_OFF
    assert set(k2cfg) == set(cfg10) | {'RHO_MIN_N'} and all(k2cfg[k] == cfg10[k] for k in cfg10 if k not in ('MEMBERS', 'TTA'))
    fleet_src = (f'K2_CFG = {k2cfg!r}   # literal, written by k10/build_k10fn.py = the K10-main K2_CFG (MEMBERS / TTA of '
                 f'{K["sets"]}) + RHO_MIN_N 1e9 (parent-free: the F~N Spearman gate runs in the N stage)\n' + cell_body)
    fleet_ds = f'eeshsaxena/{K["ds"]}'
    cfg_n = dict(W3=W_FN, W3_VARIANTS=VAR_FN, TAG=('FN10' if a.kit == 'fn10' else 'FN10-3'), MODE='FN', RHO_REF='F',
                 CHILD_SHA256=csha, WEIGHTS_FILE=make_child.WEIGHTS_FILE, WEIGHTS_SHA256=make_child.WEIGHTS_SHA256,
                 WEIGHTS_DS=make_child.WEIGHTS_DS, SRC_SHA256=make_child.SRC_SHA256)
    nsrc = (f'K9N_CFG = {cfg_n!r}   # literal, written by k10/build_k10fn.py (pushed kernels get no env vars)\n'
            f"_K9N_CHILD_B64 = '{base64.b64encode(child.encode()).decode()}'   # n949_child.py (K9N-P patched public "
            f'.949 cell 4), sha256 {cfg_n["CHILD_SHA256"]}\n') + fn_body
    compile(nsrc, 'fn_n949_stage', 'exec')
    c = copy.deepcopy(c31); c['source'] = fleet_src.splitlines(keepends=True)
    cells = [nb_cell('markdown', MD.format(slug=K['slug'], desc=K['desc'], ds=K['ds']), {}),
             nb_cell('code', ids_src, {'kfn': 'fn_ids'}), c, nb_cell('code', nsrc, {'k2': 'fn_n949_stage'})]
    nb = {k: copy.deepcopy(v) for k, v in nb10.items() if k != 'cells'}
    nb['cells'] = cells
    slug = K['slug']
    kd = f'{out}/kernel/{slug}'
    os.makedirs(kd)
    code_file = slug.replace('-', '_') + '.ipynb'
    with open(f'{kd}/{code_file}', 'w', encoding='utf-8') as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    m = {'id': f'eeshsaxena/{slug}', 'title': slug.replace('-', ' '), 'code_file': code_file, 'language': 'python',
         'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
         'enable_internet': False, 'dataset_sources': [fleet_ds, make_child.WEIGHTS_DS], 'kernel_sources': [],
         'competition_sources': [COMP], 'model_sources': [], 'machine_shape': 'NvidiaTeslaT4'}
    assert meta10['machine_shape'] == m['machine_shape'] and meta10['competition_sources'] == [COMP]
    with open(f'{kd}/kernel-metadata.json', 'w', encoding='utf-8') as fh:
        json.dump(m, fh, indent=2)
    psl = [d for d in meta10['dataset_sources'] if d != 'eeshsaxena/rsna-knee-cnxt-fleet-k10'] + \
        list(meta10.get('kernel_sources') or []) + list(meta10.get('model_sources') or [])
    txt = open(f'{kd}/{code_file}', encoding='utf-8').read() + '\n' + child
    hits = [p for p in psl if '/' + p.split('/')[1] in txt or p in txt]
    assert not hits, f'{slug}: parent sources referenced {hits}'
    assert 'rsna-knee-cnxt-fleet' not in txt.replace(K['ds'], ''), 'another fleet dataset name in the notebook'
    chk = json.load(open(f'{kd}/{code_file}', encoding='utf-8'))
    assert len(chk['cells']) == 4
    if os.path.isfile(KFN_NB):                                          # == the tested kfn FN kernel except the K10 parts
        kn = json.load(open(KFN_NB, encoding='utf-8'))
        assert chk['cells'][1] == kn['cells'][1], 'FN-IDS cell != kfn'
        a3, b3 = ''.join(chk['cells'][3]['source']).split('\n', 2), ''.join(kn['cells'][3]['source']).split('\n', 2)
        assert a3[1:] == b3[1:], 'N cell (after the K9N_CFG line) != kfn'
        kcfg = ast.literal_eval(b3[0][len('K9N_CFG = '):].split('   #', 1)[0])
        assert {k: v for k, v in kcfg.items() if k != 'TAG'} == {k: v for k, v in cfg_n.items() if k != 'TAG'}, 'N config != kfn FN'
        assert chk['cells'][2]['metadata'] == kn['cells'][2]['metadata'] and chk['cells'][3]['metadata'] == kn['cells'][3]['metadata']
        kb2 = ''.join(kn['cells'][2]['source']).split('\n', 1)[1]
        assert mt.unpatch_cell(''.join(chk['cells'][2]['source']).split('\n', 1)[1], K['ds']) == kb2, 'fleet body != kfn minus K10-DS'
        assert {k: v for k, v in nb.items() if k != 'cells'} == {k: v for k, v in kn.items() if k != 'cells'}
    info = dict(built=time.strftime('%Y-%m-%d %H:%M:%S'), kit=a.kit, k10_build=k10b,
                k10w_build=(k10wb if a.kit == 'fn10_3' else None), final=final, do_not_push=bool(a.test or not final),
                k10_k5_build_sha256=fsha(f'{k10b}/K5_BUILD.json'), k10_main_nb_sha16=kb['kernels'][MAIN]['nb_sha16'],
                fleet_dataset=fleet_ds, fleet_manifest=man_p, fleet_manifest_sha256=fsha(man_p), members=srckb['members'],
                tta=srckb['tta'], sets=srckb['sets'], child_sha256=csha, src_sha256=make_child.SRC_SHA256,
                weights_file=make_child.WEIGHTS_FILE, weights_sha256=make_child.WEIGHTS_SHA256,
                weights_ds=make_child.WEIGHTS_DS, k9n_n_cell_sha16=K9N_CELL_SHA16, fn_cell_sha256=sha(fn_body),
                ids_cell_sha256=sha(ids_src), fleet_cell_sha16=sha(cell_body)[:16], k5_cell_sha16=K5_CELL_SHA16,
                k2_cfg=k2cfg, parent_sources_excluded=psl,
                kernels={slug: dict(tag=cfg_n['TAG'], mode='FN', rho_ref='F', W3=W_FN, W3_VARIANTS=VAR_FN, dir=kd,
                                    code_file=code_file, nb_sha256=fsha(f'{kd}/{code_file}'),
                                    meta_sha256=fsha(f'{kd}/kernel-metadata.json'), n_cell_sha16=sha(nsrc)[:16],
                                    dataset_sources=m['dataset_sources'], main_file=tname(W_FN),
                                    files=['submission.csv', 'submission_n949.csv'] + [tname(w) for w in [W_FN] + VAR_FN])})
    info['simulated'] = bool(kb.get('simulated') or srckb.get('simulated'))
    if info['do_not_push']:
        open(f'{out}/DO_NOT_PUSH_TEST.txt', 'w').write('TEST K10 FN BUILD (non-final K10 build or --test). NEVER push, '
                                                      'NEVER submit.\n')
    if info['simulated']:
        open(f'{out}/DO_NOT_PUSH_SIMULATION.txt', 'w').write('SIMULATION BUILD (K10 members from the ~/k10_test/sim root). '
                                                            'TEST ONLY. NEVER push, NEVER submit.\n')
    json.dump(info, open(f'{out}/FN_BUILD.json', 'w'), indent=1)
    if a.compare:
        cmpd = a.compare.replace('\\', '/')
        other = json.load(open(f'{cmpd}/FN_BUILD.json'))
        for key in ('k10_main_nb_sha16', 'fleet_manifest_sha256', 'members', 'tta', 'child_sha256', 'fn_cell_sha256'):
            assert other.get(key) == info[key], f'--compare: FN_BUILD.json {key} differs from {cmpd}'
        for f in (code_file, 'kernel-metadata.json'):
            assert fsha(f'{kd}/{f}') == fsha(f'{cmpd}/kernel/{slug}/{f}'), f'--compare: {f} differs'
        print(f'COMPARE OK: {slug} byte-identical to {cmpd}')
    tag = 'FINAL' if not info['do_not_push'] else 'TEST ONLY (DO_NOT_PUSH)'
    print(f'{slug}: 4 cells, MODE FN, W3 P/F/N {W_FN} variants {VAR_FN}, {len(srckb["members"])} members ({K["sets"]}), '
          f'notebook sha256 {info["kernels"][slug]["nb_sha256"][:16]}, datasets {m["dataset_sources"]}')
    print(f'\nK10FN BUILD OK {tag} [{a.kit}]: {out} | child {csha[:16]} | fleet {fleet_ds} {len(srckb["members"])} members | '
          f'parent sources excluded: {len(psl)}')
    return info


if __name__ == '__main__':
    main()
