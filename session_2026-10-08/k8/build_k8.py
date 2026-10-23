#!/usr/bin/env python3
"""build_k8.py (2026-10-07, Claude; a COPY of build_k7.py with the K8 member sets + the KF twin). Local only: nothing is
pushed, versioned or submitted.

K8 = parent + [5 x TTR288 mix50 s1044 + 5 x TTR224 mix50 s1043 (set A, the K5 members)] (NOVEL_PLAN_10-07 sec 2 item 2):
  set A  f{0..4}_s1043_ttr_mix50       TTR224, seed 1043, raw best (folds 3-4 Kaggle-filled, f34 receipt)
  set C  f{0..4}_s1044_ttr288mix50     TTR288, seed 1044, raw best (box chain), shipped keys f{F}_s1044t288
  set B (s1044 TTR224, teammate) is DROPPED: a (1044, 224) member is refused.
Every member TTR (fleet_model_ttr, model_sha f141aeed136dfc01), every member A+B TTA, equal members (fleet = raw-prob mean,
each 1/10), parent W .50. Three kernels from ONE upload folder (one dataset version):
  K8-main  eeshsaxena/rsna-sub-k8-fleet  strict W .50 (variants .40 / .60), parent cell-28 map    K5 cell (byte for byte)
  K8-solo  eeshsaxena/rsna-sub-k8-solo   solo (submission.csv = the fleet alone)                  K5 cell (byte for byte)
  K8-KF    eeshsaxena/rsna-sub-k8-kf     strict W .50 with the KF flat parent K2_CFG['OUTER_FLAT'] = 0.60
                                         (cells/k2_fleet_cell_kf.py = K5 cell + gold_prep KF block, make_kf_cell.py)
K7-KEY (kept): every member whose ttr_size != 224 is re-keyed f{F}_s{S} -> f{F}_s{S}t{N} (file name only, bytes identical),
so the TTR288 s1044 members never clash with any 224 s1044 file. Member identity (seed, fold, ttr_size).

  python build_k8.py [--members_json k8/members.json] [--out DIR] [--overwrite] [--w_main 0.50] [--outer_flat 0.60]

Default --out: k8/build when A 5/5 and C 5/5 (FINAL); else k8/build_i<N> (INTERIM: members.json must carry
do_not_submit (k8_collect.sh K8_INTERIM=1); K5_BUILD.json then says interim + do_not_submit, the summary line says NEVER
SUBMIT, and k2_check.py --kaggle_commit refuses it). A FINAL build refuses a members.json marked do_not_submit.
"""
import argparse, glob, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
SP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
K2_META = f'{SP}/k2/build/kernel/rsna-sub-k2-fleet/kernel-metadata.json'      # the proven K2 kernel's sources
MAIN, SOLO, KF = 'rsna-sub-k8-fleet', 'rsna-sub-k8-solo', 'rsna-sub-k8-kf'
SETS = {(1043, 224): 'A', (1044, 288): 'C'}                      # (seed, ttr_size) -> set
SEEDS = (1043, 1044)
TTR_SHA = 'f141aeed136dfc01'
TRAIN_LABELS = 'labels_mix50_logit.csv'                            # settled: mix50 labels only
FROZEN = ('fleet_common.py', 'fleet_model.py', 'fleet_model_f2.py', 'fleet_model_ttr.py')   # model_sha / tensor contract
KF_CHANGED = ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Lateral OA', 'Fracture']        # the map labels != .60


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def strip_k5(cell):
    """The K5 cell minus its K5a block and K5a-marked lines (must give the K3 cell byte for byte)."""
    b, e = cell.index('# ---- K5a begin'), cell.index('# ---- K5a end\n') + len('# ---- K5a end\n')
    assert cell[e:e + 2] == '\n\n', 'K5a block must be followed by the 2 blank lines it inserted'
    t = cell[:b] + cell[e + 2:]
    return '\n'.join(l for l in t.split('\n') if not l.rstrip().endswith('# K5a'))


def member_key(r):
    """K7-KEY: the shipped key. TTR224 members keep f{F}_s{S} (byte-identical names to K5); others get a t{N} suffix."""
    k = f'f{r["fold"]}_s{r["seed"]}'
    return k if int(r['ttr_size']) == 224 else f'{k}t{int(r["ttr_size"])}'


def load_members(path, ckdir):
    m = json.load(open(path, encoding='utf-8'))
    assert not m.get('failed'), f'members.json lists failed folds {m["failed"]}: do not build'
    mem = sorted(m['members'], key=lambda r: (int(r['ttr_size']), r['seed'], r['fold']))     # A, then C
    assert mem, 'members.json has no member'
    ids = [(r['seed'], r['fold'], int(r['ttr_size'])) for r in mem]
    assert len(set(ids)) == len(ids), f'duplicate seed/fold/ttr_size members {ids} (one pick per run)'
    files = [r['file'] for r in mem]
    assert len(set(files)) == len(files), f'duplicate member files {files}'
    for r in mem:
        st = (r['seed'], int(r['ttr_size']))
        assert st != (1044, 224), f'{r["file"]}: set B (s1044 TTR224) is DROPPED from K8 (NOVEL_PLAN 10-07 sec 2 item 2)'
        assert st in SETS, f'{r["file"]}: (seed, ttr_size) {st} is not a K8 set (A = (1043, 224), C = (1044, 288))'
        assert r.get('train_labels') == TRAIN_LABELS, f'{r["file"]}: train_labels {r.get("train_labels")} (K8 = mix50 only)'
        assert r.get('model_mod') == 'fleet_model_ttr' and r.get('model_sha') == TTR_SHA, (r['file'], r.get('model_mod'), r.get('model_sha'))
        assert r.get('tag') == 'raw', f'{r["file"]}: tag {r.get("tag")} (K8 ships RAW best checkpoints; picker --tag raw)'
        p = f'{ckdir}/{r["file"]}'
        assert os.path.isfile(p), f'missing {p} (run k8_collect.sh)'
        assert r['file'].startswith(f'f{r["fold"]}_s{r["seed"]}_{r["tag"]}_ttr{int(r["ttr_size"])}-'), r['file']
        h = sha256(p)
        assert h == r['sha256'], f'{r["file"]}: laptop sha256 {h} != members.json {r["sha256"]}'
        r['path'] = p
        r['key'] = member_key(r)
        r['staged'] = r['key'] + r['file'][len(f'f{r["fold"]}_s{r["seed"]}'):]
        r['set'] = SETS[st]
    keys = [r['key'] for r in mem]
    assert len(set(keys)) == len(keys), f'duplicate shipped keys {keys}'
    nA = sorted(r['fold'] for r in mem if r['set'] == 'A')
    assert nA == [0, 1, 2, 3, 4], f'set A folds {nA}: K8 needs the 5 K5 members'
    on_disk = sorted(os.path.basename(p) for sd in SEEDS for p in glob.glob(f'{ckdir}/f*_s{sd}_*_best.pt'))
    assert on_disk == sorted(files), \
        f'k8/ckpt files {on_disk} != members.json {files} (stale file? re-run k8_collect.sh)'
    return m, mem


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--members_json', default=f'{HERE}/members.json')
    ap.add_argument('--ckpt_dir', default=f'{HERE}/ckpt')
    ap.add_argument('--tta', choices=('A', 'AB'), default='AB', help='TTA of every member (G-TTA: TTR PASS -> AB)')
    ap.add_argument('--store', default=f'{HERE}/_real/store')
    ap.add_argument('--out', default=None, help='default k8/build (A 5/5 + C 5/5) or k8/build_i<N> (interim)')
    ap.add_argument('--w_main', default='0.50')
    ap.add_argument('--w_variants', default='0.40,0.60')
    ap.add_argument('--outer_flat', default='0.60', help='KF twin: flat outer weight (parent_audit.md: .60, the pre-probe value)')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args(argv)
    mj, mem = load_members(a.members_json, a.ckpt_dir.replace('\\', '/'))
    names = [r['staged'] for r in mem]
    specs = [r['path'] + (f'={r["key"]}' if r['staged'] != r['file'] else '') for r in mem]
    tta = {n: a.tta for n in names}
    nA = sum(r['set'] == 'A' for r in mem); nC = sum(r['set'] == 'C' for r in mem)
    interim = not (nA == 5 and nC == 5)
    sim = any('/k8_test/sim/' in str(r.get('run', '')) for r in mem)   # collected from the SIMULATION root (test only)
    if interim:
        assert mj.get('do_not_submit') and mj.get('interim'), \
            f'set C {nC}/5: an INTERIM build needs a members.json marked do_not_submit (k8_collect.sh K8_INTERIM=1)'
    else:
        assert not mj.get('do_not_submit'), 'members.json is marked do_not_submit but holds A 5/5 + C 5/5: re-collect'
    if not a.out:
        a.out = f'{HERE}/build' + (f'_i{len(mem)}' if interim else '') + ('' if a.tta == 'AB' else '_A')
    a.out = a.out.replace(chr(92), '/')
    for f in FROZEN:
        assert sha256(f'{HERE}/code/{f}') == sha256(f'{SP}/k3/code/{f}'), f'{f} differs from K3 (model_sha contract)'
    assert sha256(f'{HERE}/code/fleet_worker.py') == sha256(f'{SP}/k5/code/fleet_worker.py'), \
        'k8/code/fleet_worker.py != the K5 worker (K8 must ship the worker that scored .946, byte for byte)'
    assert sha256(f'{HERE}/code/pack_fleet_ds.py') == sha256(f'{SP}/k7/code/pack_fleet_ds.py'), 'k8 pack tool != K7 pack tool'
    wsrc = open(f'{HERE}/code/fleet_worker.py', encoding='utf-8').read()
    assert 'K5-TTA' in wsrc and 'TTA_HALF = 0.5 - 1e-3' in wsrc, 'k8/code/fleet_worker.py is not the K5 TTA worker'
    tta_s = ','.join(f'{n}={m}' for n, m in tta.items())
    import build_k2 as bk
    import make_kf_cell
    common = ['--out', a.out]
    bk.main(sum((['--ckpt', s] for s in specs), []) +
            ['--store', a.store, '--slug', MAIN, '--main_mode', 'strict', '--members', ','.join(names),
             '--w_main', a.w_main, '--w_variants', a.w_variants, '--tta', tta_s] + common
            + (['--overwrite'] if a.overwrite else []))
    bk.main(['--kernel_only', '--slug', SOLO, '--main_mode', 'solo', '--members', ','.join(names),
             '--w_main', a.w_main, '--w_variants', a.w_variants] + common)
    summ = bk.main(['--kernel_only', '--slug', KF, '--main_mode', 'strict', '--members', ','.join(names),
                    '--w_main', a.w_main, '--w_variants', a.w_variants, '--outer_flat', a.outer_flat] + common)
    # ---- post checks
    flat = float(a.outer_flat)
    want = {MAIN: ('strict', float(a.w_main), None), SOLO: ('solo', None, None), KF: ('strict', float(a.w_main), flat)}
    ks = {k['slug']: k for k in summ['kernels']}
    assert set(ks) == set(want), sorted(ks)
    for slug, (mode, w, of) in want.items():
        cfg = ks[slug]['K2_CFG']
        assert cfg['MAIN_MODE'] == mode and cfg['MEMBERS'] == names, (slug, cfg)
        assert cfg.get('TTA') == tta, (slug, 'K2_CFG TTA (K5b)', cfg.get('TTA'), tta)
        if w is not None:
            assert abs(cfg['W_MAIN'] - w) < 1e-9, (slug, cfg['W_MAIN'], w)
        if of is None:                                                   # K8-KF: absent on main / solo
            assert 'OUTER_FLAT' not in cfg, (slug, 'OUTER_FLAT must be absent', cfg)
        else:
            assert cfg.get('OUTER_FLAT') == of and mode == 'strict', (slug, 'KF', cfg)
    up = summ['upload']
    assert up.get('manifest') == names, (up.get('manifest'), names)
    assert up.get('tta') == tta, (up.get('tta'), tta)
    n_probe = len({s for r in summ['members'] for s in r['decode_probe']})
    assert 'fleet_model_ttr.py' in up['files'] and len(up['files']) == 5 + len(names) + 2 + n_probe, up['files']
    assert sha256(f'{up["path"]}/fleet_worker.py') == sha256(f'{HERE}/code/fleet_worker.py'), 'shipped worker != k8/code'
    man = json.load(open(f'{up["path"]}/fleet_manifest.json', encoding='utf-8'))
    srcs = man.get('sources') or {}
    assert {k: (v or {}).get('tta') for k, v in srcs.items()} == tta, ('manifest tta', srcs)
    for r in mem:                                                       # K3c: the bytes the commit must report
        s = srcs.get(r['staged'])
        s16 = (s.get('sha256') if isinstance(s, dict) else s) or ''
        assert s16[:16] == r['sha256'][:16], (r['staged'], s, r['sha256'][:16])
        assert sha256(f'{up["path"]}/{r["staged"]}') == r['sha256'], f'{r["staged"]}: shipped bytes != collected {r["file"]}'
    wkeys = ['_'.join(n.split('_')[:2]) for n in man['checkpoints']]     # == fleet_worker.py fs_key
    assert len(wkeys) == len(set(wkeys)), f'the worker would refuse these names: duplicate keys {sorted(wkeys)}'
    by_staged = {x['staged']: x for x in mem}
    for r in summ['members']:
        x = by_staged[r['staged']]
        assert (int(r['fold']) == x['fold'] and int(r['seed']) == x['seed'] and int(r['ttr_size']) == int(x['ttr_size'])
                and r['epoch_1based'] == x['epoch'] and abs(float(r['cv']) - x['cv']) < 1e-12), (r, x)
    cell = open(f'{HERE}/cells/k2_fleet_cell.py', encoding='utf-8').read()
    k3cell = open(f'{SP}/k3/cells/k2_fleet_cell.py', encoding='utf-8').read()
    assert strip_k5(cell) == k3cell, 'k8 cell minus the K5a block != the K3 cell (only the K5a block may differ)'
    assert sha256(f'{HERE}/cells/k2_fleet_cell.py') == sha256(f'{SP}/k5/cells/k2_fleet_cell.py'), 'k8 cell != the K5 cell'
    kfcell = open(f'{HERE}/cells/k2_fleet_cell_kf.py', encoding='utf-8').read()
    assert kfcell == make_kf_cell.make() and make_kf_cell.strip_kf(kfcell) == cell, 'KF cell != K5 cell + the KF splice'
    k2m = json.load(open(K2_META, encoding='utf-8')) if os.path.isfile(K2_META) else None
    for slug, k in ks.items():
        nb = json.load(open(f'{k["out"]}/{k["code_file"]}', encoding='utf-8'))
        src = ''.join(nb['cells'][31]['source'])
        hdr, body = src.split('\n', 1)
        assert hdr.startswith('K2_CFG = ') and body == (kfcell if slug == KF else cell), \
            f'{slug}: cell 31 != {"the KF cell" if slug == KF else "the K5 cell"} + K2_CFG line'
        meta = json.load(open(f'{k["out"]}/kernel-metadata.json', encoding='utf-8'))
        if k2m:
            for key in ('dataset_sources', 'kernel_sources', 'competition_sources', 'model_sources', 'machine_shape',
                        'enable_gpu', 'enable_internet'):
                assert meta.get(key) == k2m.get(key), (slug, key, meta.get(key), k2m.get(key))
        assert meta['id'] == f'eeshsaxena/{slug}' and meta['is_private'] is True
        print(f'{slug}: {len(nb["cells"])} cells, K2_CFG {ks[slug]["K2_CFG"]}, '
              f'notebook sha256 {sha256(f"{k["out"]}/{k["code_file"]}")[:16]}, dataset_sources == K2: {bool(k2m)}')
    info = dict(built=time.strftime('%Y-%m-%d %H:%M:%S'), k8=True, interim=interim, simulated=sim,
                do_not_submit=interim or sim,
                n_members=len(names), members=names, sets=dict(A=nA, C=nC), folds=[r['fold'] for r in mem], tta=tta,
                expect_tta=tta_s, worker_sha16=sha256(f'{HERE}/code/fleet_worker.py')[:16],
                cell_sha16=hashlib.sha256(cell.encode()).hexdigest()[:16],
                kf_cell_sha16=hashlib.sha256(kfcell.encode()).hexdigest()[:16],
                outer_flat=flat, kf_changed_labels=KF_CHANGED,
                picks=[dict({k: r[k] for k in ('set', 'seed', 'fold', 'ttr_size', 'tag', 'epoch', 'cv', 'other_tag_cv', 'file',
                                               'staged', 'sha256', 'source')}, suffix=r.get('suffix'), oof=r.get('oof'))
                       for r in mem],
                upload=dict(path=up['path'], mb=up['mb'], files=len(up['files'])),
                kernels={s: dict(mode=want[s][0], w_main=ks[s]['K2_CFG']['W_MAIN'], outer_flat=want[s][2],
                                 dir=ks[s]['out'], nb_sha16=sha256(f'{ks[s]["out"]}/{ks[s]["code_file"]}')[:16])
                         for s in want},
                members_json_generated=mj.get('generated'), skipped=mj.get('skipped'), set_c=mj.get('set_c'))
    json.dump(info, open(f'{a.out}/K5_BUILD.json', 'w'), indent=1)
    # FIX 10-08: --overwrite wiped the hand-made build_sim/DO_NOT_PUSH_SIMULATION.txt; the build now writes its own marker
    for fn, cond, txt in (('DO_NOT_PUSH_SIMULATION.txt', sim, 'SIMULATION BUILD (members from the ~/k8_test/sim root).'),
                          ('DO_NOT_SUBMIT_INTERIM.txt', interim, f'INTERIM BUILD (set A {nA}/5, set C {nC}/5).')):
        if cond:                                                         # marker in the build dir, never inside upload/
            open(f'{a.out}/{fn}', 'w').write(txt + ' TEST ONLY. NEVER version the upload folder, NEVER push these '
                                             'kernels, NEVER submit. K5_BUILD.json: do_not_submit true.\n')
    keys8 =[f'{r["set"]}:{r["key"]}' for r in mem]
    tag = (f'INTERIM ({len(mem)} members: A {nA}/5, C {nC}/5: {keys8}) -- TEST ONLY, do_not_submit, NEVER SUBMIT'
           if interim else f'FINAL ({len(mem)} members: A 5/5, C 5/5: {keys8})')
    if sim:
        tag = 'SIMULATION (members from ~/k8_test/sim: TEST ONLY, NEVER VERSION / PUSH / SUBMIT) ' + tag
    print(f'\nK8 BUILD OK {tag}: upload {up["path"]} ({up["mb"]} MB, {len(up["files"])} files) | members {names} | '
          f'TTA {tta} | kernels {MAIN} (strict W {a.w_main}), {SOLO} (solo), {KF} (strict W {a.w_main}, OUTER_FLAT {flat})')
    print(f'expect_members: {",".join(names)}')
    print(f'expect_tta: {tta_s}')
    return info


if __name__ == '__main__':
    main()
