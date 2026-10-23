#!/usr/bin/env python3
"""build_k12.py (2026-10-09, Claude). Local only: nothing is pushed, created, versioned or submitted.

K12 = the FN11 kernels (A0-tested, FINAL) with set D-OAI (DO = f{0..4}_s1043_ttr_frac75oaiT12v00, PREREG_K12.md) in the
fleet, as R (REPLACE: C + DO + E + E2, 20 members) or A (ADD: C + D + DO + E + E2, 25 members). Two datasets:
  eeshsaxena/rsna-knee-cnxt-fleet-k11w  the LIVE FN11 dataset, unchanged (worker code, 20 members, parity npz, manifest)
  eeshsaxena/rsna-knee-cnxt-fleet-k12w  NEW: ONLY the 5 DO raw best checkpoints (re-keyed f{F}_s1043f75oai_raw_ttr224-...)
                                        + k12w_manifest.json (sha256 full + [:16], model_mod / model_sha, tta AB)
Kernel = the FN11 notebook with exactly these changes (k12_nbdiff.py proves it): md title + F bullet; a NEW K12-MERGE cell
after FN-IDS (cells/k12_merge_cell.py + its K12M_CFG literal: the k11w manifest, the k12w manifest, the merged manifest);
the fleet cell's K2_CFG MEMBERS / TTA (R or A) and its body = cells/k12_fleet_cell.py (the K5 cell reading the merged
dir /tmp/k12_fleet; reverses to the K5 cell); the N cell's TAG K12; metadata id / title / code_file / dataset_sources.
Every DO checkpoint is re-checked here (identity args, contract == the k11w D members, decode_probe within the k11w
parity npz, K12-OAI-CLEAN scan = box/k12_oai_clean.py) before it is staged.
  python build_k12.py --variant R|A [--members_json k12/members.json] [--k11w SP/k11/build_w11] [--fn11 SP/k11/build_fn11]
                      [--fn11_n70 SP/k11/build_fn11_n70] [--out_w DIR] [--out DIR] [--test] [--overwrite]
Outputs: <out_w>/upload/rsna-knee-cnxt-fleet-k12w + merged_manifest.json + K12W_BUILD.json + UPLOAD_SHA256SUMS.txt;
<out>/kernel/rsna-sub-k12fn (+ <out>_n70/kernel/rsna-sub-k12fn-n70) each with FN_BUILD.json, K12_BUILD.json,
NB_DIFF_vs_FN11.txt. Defaults: out_w k12/build_w12, out k12/build_fn12_<variant>; a TEST (interim / simulated) members.json
needs --test and every output gets DO_NOT_PUSH_TEST.txt (+ DO_NOT_PUSH_SIMULATION.txt when simulated).
"""
import argparse, ast, copy, hashlib, json, os, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
SP = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, f'{HERE}/box')
import make_k12_tools as mt, make_k10_tools as mt10, k12_oai_clean as clean      # noqa: E402

K11W, K12W = 'rsna-knee-cnxt-fleet-k11w', 'rsna-knee-cnxt-fleet-k12w'
DOLAB = 'labels_frac75oai_T12v00.csv'
MERGE_CELL = f'{HERE}/cells/k12_merge_cell.py'
MD_F = ('* **F**: our {n}-member fleet: {desc}, ConvNeXt-T, test-time augmentation A+B (datasets `eeshsaxena/{a}` + '
        '`eeshsaxena/{b}`; set D-OAI was trained on the RSNA training folds + OAI MOAKS knees, labels T12 V00), run by our '
        'own worker.')
SETDESC = {'C': 'TTR288 mix50 s1044 (set C)', 'D': 'TTR224 frac75 s1043 (set D)', 'DO': 'TTR224 frac75+OAI-T12 s1043 (set D-OAI)',
           'E': 'TTR288 frac75 s1045 (set E)', 'E2': 'TTR288 frac75 s1046 (set E2)'}


def fsha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def sha(s):
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def cfg_line(src, key):
    l1, rest = src.split('\n', 1)
    assert l1.startswith(key), l1[:60]
    lit, sep, comment = l1[len(key):].partition('   #')
    return ast.literal_eval(lit), comment, rest


def nb_cell(kind, src, meta):
    c = {'cell_type': kind, 'metadata': meta, 'source': src.splitlines(keepends=True)}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


def load_do(mj, k11w_up, test):
    import torch
    do = sorted([r for r in mj['members'] if r['set'] == 'DO'], key=lambda r: r['fold'])
    folds = [r['fold'] for r in do]
    assert folds == [0, 1, 2, 3, 4] or test, f'DO folds {folds} (a FINAL K12 needs 5/5)'
    man11 = json.load(open(f'{k11w_up}/fleet_manifest.json'))
    parity = {f[len('parity_'):-len('.npz')] for f in os.listdir(k11w_up) if f.startswith('parity_')}
    dname = [n for n in man11['checkpoints'] if n.startswith('f0_s1043f75_')][0]
    ref = torch.load(f'{k11w_up}/{dname}', map_location='cpu', weights_only=False)['args']
    out = []
    for r in do:
        p = r['laptop_path']
        assert fsha(p) == r['sha256'], f'{p}: sha256 != members.json'
        sc = clean.scan(p, None, None)
        assert not sc['hits'], f'{p}: K12-OAI-CLEAN hits {sc["hits"]}'
        ck = torch.load(p, map_location='cpu', weights_only=False)
        g, F = ck['args'], r['fold']
        want = dict(fold=F, seed=1043, ttr_size=224, model_mod='fleet_model_ttr', model_sha='f141aeed136dfc01',
                    train_labels=DOLAB, sel_labels='llm_labels_v4_blend.csv', amp='fp16')
        mism = {k: g.get(k) for k, v in want.items() if g.get(k) != v}
        assert not mism, f'DO f{F} args {mism}'
        for k in ('contract', 'views', 'backbone', 'dim', 'n_slices', 'n_views', 'n_contrast', 'store', 'crop_mm',
                  'src_slices', 'jpeg_fp', 'select_sha'):
            assert g.get(k) == ref.get(k), f'DO f{F} args {k} {g.get(k)!r} != the k11w D member {ref.get(k)!r}'
        assert set(g['decode_probe']) <= parity, f'DO f{F} decode_probe outside the k11w parity npz'
        assert abs(float(ck['cv']) - r['cv']) < 1e-12 and int(ck['epoch']) + 1 == r['epoch']
        assert int(g.get('epochs', -1)) >= 14
        staged = f'f{F}_s1043f75oai' + r['file'][len(f'f{F}_s1043'):]
        assert staged.startswith(f'f{F}_s1043f75oai_raw_ttr224-320x24_cv') and staged not in man11['checkpoints']
        out.append(dict(r, staged=staged, sha16=r['sha256'][:16], clean_names=sc['oai_named_files']))
    return do, out, man11


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=('R', 'A'))
    ap.add_argument('--members_json', default=f'{HERE}/members.json')
    ap.add_argument('--k11w', default=f'{SP}/k11/build_w11')
    ap.add_argument('--fn11', default=f'{SP}/k11/build_fn11')
    ap.add_argument('--fn11_n70', default=f'{SP}/k11/build_fn11_n70')
    ap.add_argument('--out_w', default=None); ap.add_argument('--out', default=None)
    ap.add_argument('--test', action='store_true'); ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args(argv)
    gen = mt.make()
    for p in ('cells/k12_fleet_cell.py',):
        assert open(f'{HERE}/{p}', encoding='utf-8', newline='').read() == gen[p], f'{p} != make_k12_tools.make()'
    mj = json.load(open(a.members_json))
    assert mj.get('k12') is True, 'not a k12_collect.sh members.json'
    nonfinal = bool(mj.get('do_not_submit') or mj.get('interim') or mj.get('simulated'))
    assert a.test or not nonfinal, 'members.json is INTERIM / SIMULATED: pass --test (TEST build, never pushed)'
    sim = bool(mj.get('simulated'))
    b11 = json.load(open(f'{a.k11w}/K5_BUILD.json'))
    assert b11.get('k11') and b11.get('kit') == 'k11w' and not b11.get('do_not_submit') and \
        b11['sets'] == {'C': 5, 'D': 5, 'E': 5, 'E2': 5}, 'the k11w build is not the FINAL FN11 build'
    up11 = f'{a.k11w}/upload/{K11W}'
    do, dos, man11 = load_do(mj, up11, a.test)
    for bd in (a.fn11, a.fn11_n70):
        fbase = json.load(open(f'{bd}/FN_BUILD.json'))
        assert not fbase.get('do_not_push') and fbase.get('final'), f'{bd} is not a FINAL FN11 build'
        assert fsha(f'{up11}/fleet_manifest.json') == fbase['fleet_manifest_sha256'], f'{bd}: fleet manifest != the k11w build'
    out_w = (a.out_w or f'{HERE}/build_w12' + ('_sim' if sim else ('_test' if nonfinal else ''))).replace('\\', '/')
    out = (a.out or f'{HERE}/build_fn12_{a.variant}' + ('_sim' if sim else ('_test' if nonfinal else ''))).replace('\\', '/')
    for d in (out_w, out, out + '_n70'):
        if os.path.exists(d) and os.listdir(d):
            assert a.overwrite, f'{d} exists (pass --overwrite)'
            shutil.rmtree(d)
    # ---------------- the k12w dataset folder (ONLY the DO checkpoints) ----------------
    upw = f'{out_w}/upload/{K12W}'
    os.makedirs(upw)
    srcs = {}
    for r in dos:
        shutil.copy2(r['laptop_path'], f'{upw}/{r["staged"]}')
        assert fsha(f'{upw}/{r["staged"]}') == r['sha256']
        srcs[r['staged']] = {'original': r['staged'], 'run_dir': 'members', 'sha256': r['sha16'], 'sha256_full': r['sha256'],
                             'model_mod': 'fleet_model_ttr', 'model_sha': 'f141aeed136dfc01', 'tta': 'AB'}
    manw = {'checkpoints': [r['staged'] for r in dos], 'sources': srcs}
    json.dump(manw, open(f'{upw}/k12w_manifest.json', 'w'), indent=1)
    json.dump({'title': K12W, 'id': f'eeshsaxena/{K12W}', 'licenses': [{'name': 'CC0-1.0'}]},
              open(f'{upw}/dataset-metadata.json', 'w'), indent=1)
    assert sorted(os.listdir(upw)) == sorted(manw['checkpoints'] + ['k12w_manifest.json', 'dataset-metadata.json'])
    merged = {'checkpoints': man11['checkpoints'] + manw['checkpoints'],
              'sources': dict(man11['sources'], **{n: {k: v for k, v in s.items() if k != 'sha256_full'} for n, s in srcs.items()})}
    json.dump(merged, open(f'{out_w}/merged_manifest.json', 'w'), indent=1)
    with open(f'{out_w}/UPLOAD_SHA256SUMS.txt', 'w', newline='\n') as fh:
        for f in sorted(os.listdir(upw)):
            fh.write(f'{fsha(f"{upw}/{f}")}  {f}\n')
    rec_w = dict(built=time.strftime('%Y-%m-%d %H:%M:%S'), k12=True, dataset=f'eeshsaxena/{K12W}', upload=upw,
                 files=sorted(os.listdir(upw)), mb=round(sum(os.path.getsize(f'{upw}/{f}') for f in os.listdir(upw)) / 2**20, 1),
                 base_dataset=f'eeshsaxena/{K11W}', base_manifest_sha256=fsha(f'{up11}/fleet_manifest.json'),
                 k12w_manifest_sha256=fsha(f'{upw}/k12w_manifest.json'), merged_manifest_sha256=fsha(f'{out_w}/merged_manifest.json'),
                 picks=[{k: r.get(k) for k in ('fold', 'file', 'staged', 'sha256', 'epoch', 'epochs', 'cv', 'src', 'simulated',
                                                'train_labels', 'clean_names')} for r in dos],
                 simulated=sim, do_not_submit=nonfinal or a.test, members_json_sha256=fsha(a.members_json))
    json.dump(rec_w, open(f'{out_w}/K12W_BUILD.json', 'w'), indent=1)
    # ---------------- the kernels ----------------
    merge_body = open(MERGE_CELL, encoding='utf-8').read()
    compile(merge_body, 'k12_merge', 'exec')
    k12m = {'base_ds': K11W, 'add_ds': K12W, 'base_manifest': man11, 'add_manifest': manw, 'merged_manifest': merged}
    merge_src = f'K12M_CFG = {k12m!r}   # literal, written by k12/build_k12.py (pushed kernels get no env vars)\n' + merge_body
    compile(merge_src, 'k12_merge', 'exec')
    by_fold = {r['fold']: r['staged'] for r in dos}
    built = {}
    for base_dir, sfx in ((a.fn11, ''), (a.fn11_n70, '-n70')):
        fb = json.load(open(f'{base_dir}/FN_BUILD.json'))
        assert not fb.get('do_not_push') and fb.get('final'), f'{base_dir} is not a FINAL FN11 build'
        (slug11, k11), = fb['kernels'].items()
        nb = json.load(open(f'{k11["dir"]}/{k11["code_file"]}', encoding='utf-8'))
        assert fsha(f'{k11["dir"]}/{k11["code_file"]}') == k11['nb_sha256']
        meta11 = json.load(open(f'{k11["dir"]}/kernel-metadata.json', encoding='utf-8'))
        assert len(nb['cells']) == 4
        cfg, cmt, body = cfg_line(''.join(nb['cells'][2]['source']), 'K2_CFG = ')
        B = list(cfg['MEMBERS'])
        assert B == fb['members'] == man11['checkpoints'], 'FN11 MEMBERS != the k11w manifest'
        if a.variant == 'R':
            mem = [by_fold[int(n[1])] if n.startswith(f'f{n[1]}_s1043f75_raw_') else n for n in B]
            assert sum(n.split('_')[1] == 's1043f75oai' for n in mem) == len(dos) or a.test
        else:
            mem = B + [r['staged'] for r in dos]
        assert len(set(mem)) == len(mem)
        tta = {n: 'AB' for n in mem}
        cfg2 = dict(cfg, MEMBERS=mem, TTA=tta)
        assert set(cfg2) == set(cfg) and all(cfg2[k] == cfg[k] for k in cfg if k not in ('MEMBERS', 'TTA'))
        body12 = gen['cells/k12_fleet_cell.py']
        assert mt.unpatch_k12(body12) == mt10.unpatch_cell(body, K11W), 'fleet bodies do not reverse to the same K5 cell'
        sets = [s for s in ('C', 'D', 'DO', 'E', 'E2') if s != 'D' or a.variant == 'A']
        fleet_src = (f'K2_CFG = {cfg2!r}   # literal, written by k12/build_k12.py = the FN11 K2_CFG with MEMBERS / TTA of '
                     f'{" + ".join(sets)} ({"REPLACE" if a.variant == "R" else "ADD"}); the fleet dir is the K12-MERGE dir\n' + body12)
        ncfg, ncmt, nrest = cfg_line(''.join(nb['cells'][3]['source']), 'K9N_CFG = ')
        ncfg2 = dict(ncfg, TAG='K12')
        nsrc = f'K9N_CFG = {ncfg2!r}   #{ncmt}\n' + nrest
        slug = 'rsna-sub-k12fn' + sfx
        md = ''.join(nb['cells'][0]['source'])
        lines = md.split('\n')
        assert lines[0].startswith(f'# {slug11}:') and lines[2].startswith('* **F**: our ')
        n = len(mem)
        cnt = {'C': 5, 'D': 5, 'DO': len(dos), 'E': 5, 'E2': 5}
        lines[0] = f'# {slug}:' + lines[0][len(f'# {slug11}:'):]
        lines[2] = MD_F.format(n=n, desc=' + '.join(f'{cnt[s]} x {SETDESC[s]}' for s in sets), a=K11W, b=K12W)
        cells = [nb_cell('markdown', '\n'.join(lines), nb['cells'][0]['metadata']), copy.deepcopy(nb['cells'][1]),
                 nb_cell('code', merge_src, {'k2': 'k12_merge'}), copy.deepcopy(nb['cells'][2]), copy.deepcopy(nb['cells'][3])]
        cells[3]['source'] = fleet_src.splitlines(keepends=True)
        cells[4]['source'] = nsrc.splitlines(keepends=True)
        nb2 = {k: copy.deepcopy(v) for k, v in nb.items() if k != 'cells'}
        nb2['cells'] = cells
        od = out + ('_n70' if sfx else '')
        kd = f'{od}/kernel/{slug}'
        os.makedirs(kd)
        code_file = slug.replace('-', '_') + '.ipynb'
        with open(f'{kd}/{code_file}', 'w', encoding='utf-8') as fh:
            json.dump(nb2, fh, indent=1, ensure_ascii=False)
        m = dict(meta11, id=f'eeshsaxena/{slug}', title=slug.replace('-', ' '), code_file=code_file,
                 dataset_sources=[f'eeshsaxena/{K11W}', f'eeshsaxena/{K12W}'] + [d for d in meta11['dataset_sources'] if d != f'eeshsaxena/{K11W}'])
        assert meta11['dataset_sources'][0] == f'eeshsaxena/{K11W}' and len(meta11['dataset_sources']) == 2
        with open(f'{kd}/kernel-metadata.json', 'w', encoding='utf-8') as fh:
            json.dump(m, fh, indent=2)
        txt = open(f'{kd}/{code_file}', encoding='utf-8').read()
        assert 'rsna-knee-cnxt-fleet\'' not in txt and 'eeshsaxena/rsna-knee-cnxt-fleet-k10' not in txt
        k = dict(k11, tag='K12', dir=kd, code_file=code_file, nb_sha256=fsha(f'{kd}/{code_file}'),
                 meta_sha256=fsha(f'{kd}/kernel-metadata.json'), n_cell_sha16=sha(nsrc)[:16], dataset_sources=m['dataset_sources'])
        info = dict(fb, built=time.strftime('%Y-%m-%d %H:%M:%S'), k12=True, variant=a.variant, kit='k12', final=not (a.test or nonfinal),
                    do_not_push=bool(a.test or nonfinal), simulated=sim, members=mem, tta=tta, k2_cfg=dict(fb['k2_cfg'], MEMBERS=mem, TTA=tta),
                    sets={s: cnt[s] for s in sets}, fleet_dataset=f'eeshsaxena/{K11W} + eeshsaxena/{K12W} (merged by the K12-MERGE cell)',
                    fleet_manifest=f'{out_w}/merged_manifest.json', fleet_manifest_sha256=fsha(f'{out_w}/merged_manifest.json'),
                    k12w_build=out_w, base_fn11_build=base_dir, base_fn11_nb_sha256=k11['nb_sha256'],
                    merge_cell_sha256=sha(merge_body), k12_fleet_cell_sha16=sha(body12)[:16], kernels={slug: k})
        assert info['k2_cfg'] == {kk: cfg2[kk] for kk in info['k2_cfg']}, 'FN_BUILD k2_cfg != the notebook K2_CFG'
        info.pop('k11w_build', None)
        json.dump(info, open(f'{od}/FN_BUILD.json', 'w'), indent=1)
        json.dump(dict(k8=True, k12=True, variant=a.variant, members=mem, tta=tta, sets=info['sets'],
                       do_not_submit=bool(a.test or nonfinal), interim=bool(mj.get('interim')), simulated=sim),
                  open(f'{od}/K12_BUILD.json', 'w'), indent=1)
        import k12_nbdiff
        dtxt, summ = k12_nbdiff.check(f'{kd}', f'{k11["dir"]}', variant=a.variant)
        open(f'{od}/NB_DIFF_vs_FN11.txt', 'w', encoding='utf-8').write(dtxt)
        if a.test or nonfinal:
            open(f'{od}/DO_NOT_PUSH_TEST.txt', 'w').write('TEST K12 BUILD (interim / simulated D-OAI or --test). NEVER push, NEVER submit.\n')
        if sim:
            open(f'{od}/DO_NOT_PUSH_SIMULATION.txt', 'w').write('SIMULATION: D-OAI folds simulated from set D weights. TEST ONLY.\n')
        built[slug] = dict(dir=kd, nb_sha256=k['nb_sha256'], members=n, diff=summ)
        print(f'{slug}: 5 cells (md, FN-IDS, K12-MERGE, fleet, N), variant {a.variant}, {n} members, W3 {k["W3"]}, TAG K12, '
              f'notebook sha256 {k["nb_sha256"][:16]}, datasets {m["dataset_sources"]} | NB diff vs {slug11}: {summ}')
    for d in (out_w,):
        if a.test or nonfinal:
            open(f'{d}/DO_NOT_PUSH_TEST.txt', 'w').write('TEST K12 dataset folder. NEVER create / version it.\n')
        if sim:
            open(f'{d}/DO_NOT_PUSH_SIMULATION.txt', 'w').write('SIMULATION: D-OAI folds simulated from set D weights. TEST ONLY.\n')
    tag = 'FINAL' if not (a.test or nonfinal) else ('SIMULATION TEST ONLY' if sim else 'TEST ONLY')
    print(f'\nK12 BUILD OK {tag} [variant {a.variant}]: dataset {upw} ({rec_w["mb"]} MB, {len(rec_w["files"])} files: '
          f'{len(dos)} DO ckpts + k12w_manifest.json + dataset-metadata.json) | kernels {list(built)} | merged manifest '
          f'{len(merged["checkpoints"])} checkpoints')
    return dict(w=rec_w, kernels=built)


if __name__ == '__main__':
    main()
