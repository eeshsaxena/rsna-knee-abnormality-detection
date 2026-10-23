#!/usr/bin/env python3
"""build_k9.py (2026-10-08, Claude; a COPY of build_k8.py with the K9 member sets). Local only: nothing is pushed, versioned
or submitted.

K9 = parent + [5 x TTR288 mix50 s1044 (set C) + 5 x TTR224 frac75 s1043 (set D)] (NOVEL_PLAN_10-07 sec 4 row 10-09):
  set C  f{0..4}_s1044_ttr288mix50   TTR288, seed 1044, labels_mix50_logit.csv,  raw best (box chain), key f{F}_s1044t288
  set D  f{0..4}_s1043_ttr_frac75    TTR224, seed 1043, labels_frac75_logit.csv, raw best (Kaggle swapiy f01 E17,
                                     kmrsuman f23 + f4x E~21, run receipts),       key f{F}_s1043f75
  set A (s1043 TTR224 mix50, the K5 members) is REPLACED by set D and refused here; set B (s1044 TTR224) is refused.
Every member TTR (fleet_model_ttr, model_sha f141aeed136dfc01), every member A+B TTA, equal members (fleet = raw-prob mean,
each 1/10), parent W .50. Three kernels from ONE upload folder (one dataset version of eeshsaxena/rsna-knee-cnxt-fleet):
  K9-main  eeshsaxena/rsna-sub-k9-fleet  strict W .50 (variants .40 / .60), parent cell-28 map    K5 cell (byte for byte)
  K9-solo  eeshsaxena/rsna-sub-k9-solo   solo (submission.csv = the fleet alone)                  K5 cell (byte for byte)
  K9-KF    eeshsaxena/rsna-sub-k9-kf     strict W .50 with the KF flat parent K2_CFG['OUTER_FLAT'] = 0.60
                                         (cells/k2_fleet_cell_kf.py == the K8 KF cell, make_kf_cell.py)
K9-KEY: member identity = (seed, fold, ttr_size, train_labels). The shipped key carries a set tag so no K9 key can equal a
K5/K7/K8 key: set C f{F}_s1044 -> f{F}_s1044t288 (the K7-KEY rule), set D f{F}_s1043 -> f{F}_s1043f75 (labels tag; a frac75
member would otherwise ship as f{F}_s1043_raw_ttr224-..., the same key as the K5 set A member of that fold). File name only:
the bytes are identical (pack_fleet_ds.py P-TAG; the worker's fold grouping uses field 1 'f{F}', its duplicate check fields
1-2).

  python build_k9.py [--members_json k9/members.json] [--out DIR] [--overwrite] [--w_main 0.50] [--outer_flat 0.60]

Default --out: k9/build when C 5/5 and D 5/5 (FINAL); else k9/build_i<N> (INTERIM: members.json must carry do_not_submit
(k9_collect.sh K9_INTERIM=1); K5_BUILD.json then says interim + do_not_submit, the summary line says NEVER SUBMIT, and
k2_check.py --expect_tta <K5_BUILD.json> refuses it). A FINAL build refuses a members.json marked do_not_submit. A build
from the simulation root (~/k9_test/sim) is SIMULATION + do_not_submit. Interim / simulation builds get a marker file
DO_NOT_PUSH_SIMULATION.txt / DO_NOT_SUBMIT_INTERIM.txt in the build dir (never inside upload/).
"""
import argparse, glob, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
SP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
K2_META = f'{SP}/k2/build/kernel/rsna-sub-k2-fleet/kernel-metadata.json'      # the proven K2 kernel's sources
MAIN, SOLO, KF = 'rsna-sub-k9-fleet', 'rsna-sub-k9-solo', 'rsna-sub-k9-kf'
MIX50, FRAC75 = 'labels_mix50_logit.csv', 'labels_frac75_logit.csv'
SETS = {(1044, 288, MIX50): 'C', (1043, 224, FRAC75): 'D'}         # (seed, ttr_size, train_labels) -> set
KEY_TAG = {'C': 't288', 'D': 'f75'}                                # K9-KEY: shipped key = f{F}_s{S}<tag>
REFUSED = {(1043, 224, MIX50): 'set A (the K5 mix50 s1043 members) is REPLACED by set D in K9',
           (1044, 224, MIX50): 'set B (s1044 TTR224) is DROPPED (NOVEL_PLAN 10-07 sec 2 item 2)'}
SEEDS = (1043, 1044)
TTR_SHA = 'f141aeed136dfc01'
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


def set_of(r):
    return SETS.get((int(r['seed']), int(r['ttr_size']), r.get('train_labels')))


def member_key(r):
    """K9-KEY: the shipped key f{F}_s{S}<set tag>: set C t288 (== the K7/K8 rule), set D f75 (labels tag). No K9 key equals
    a plain f{F}_s{S} key, so a frac75 member can never collide with a set A (K5) member of the same fold + seed."""
    return f'f{r["fold"]}_s{r["seed"]}{KEY_TAG[set_of(r)]}'


def load_members(path, ckdir):
    m = json.load(open(path, encoding='utf-8'))
    assert not m.get('failed'), f'members.json lists failed folds {m["failed"]}: do not build'
    mem = sorted(m['members'], key=lambda r: (int(r['ttr_size']), r['seed'], r['fold']))     # D (224), then C (288)
    assert mem, 'members.json has no member'
    ids = [(r['seed'], r['fold'], int(r['ttr_size']), r.get('train_labels')) for r in mem]
    assert len(set(ids)) == len(ids), f'duplicate seed/fold/ttr_size/train_labels members {ids} (one pick per run)'
    files = [r['file'] for r in mem]
    assert len(set(files)) == len(files), f'duplicate member files {files}'
    for r in mem:
        st = (int(r['seed']), int(r['ttr_size']), r.get('train_labels'))
        assert st not in REFUSED, f'{r["file"]}: {REFUSED.get(st)}'
        assert st in SETS, (f'{r["file"]}: (seed, ttr_size, train_labels) {st} is not a K9 set '
                            f'(C = (1044, 288, {MIX50}), D = (1043, 224, {FRAC75}))')
        assert r.get('set') in (None, SETS[st]), f'{r["file"]}: members.json says set {r.get("set")}, its identity is set {SETS[st]}'
        assert r.get('model_mod') == 'fleet_model_ttr' and r.get('model_sha') == TTR_SHA, (r['file'], r.get('model_mod'), r.get('model_sha'))
        assert r.get('tag') == 'raw', f'{r["file"]}: tag {r.get("tag")} (K9 ships RAW best checkpoints; picker --tag raw)'
        p = f'{ckdir}/{r["file"]}'
        assert os.path.isfile(p), f'missing {p} (run k9_collect.sh)'
        assert r['file'].startswith(f'f{r["fold"]}_s{r["seed"]}_{r["tag"]}_ttr{int(r["ttr_size"])}-'), r['file']
        h = sha256(p)
        assert h == r['sha256'], f'{r["file"]}: laptop sha256 {h} != members.json {r["sha256"]}'
        r['path'] = p
        r['key'] = member_key(r)
        r['staged'] = r['key'] + r['file'][len(f'f{r["fold"]}_s{r["seed"]}'):]
        r['set'] = SETS[st]
    keys = [r['key'] for r in mem]
    assert len(set(keys)) == len(keys), f'duplicate shipped keys {keys}'
    assert all(k.split('_')[1] not in ('s1043', 's1044') for k in keys), f'K9-KEY: an untagged key in {keys}'
    on_disk = sorted(os.path.basename(p) for sd in SEEDS for p in glob.glob(f'{ckdir}/f*_s{sd}_*_best.pt'))
    assert on_disk == sorted(files), \
        f'k9/ckpt files {on_disk} != members.json {files} (stale file? re-run k9_collect.sh)'
    return m, mem


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--members_json', default=f'{HERE}/members.json')
    ap.add_argument('--ckpt_dir', default=f'{HERE}/ckpt')
    ap.add_argument('--tta', choices=('A', 'AB'), default='AB', help='TTA of every member (G-TTA: TTR PASS -> AB)')
    ap.add_argument('--store', default=f'{HERE}/_real/store')
    ap.add_argument('--out', default=None, help='default k9/build (C 5/5 + D 5/5) or k9/build_i<N> (interim)')
    ap.add_argument('--w_main', default='0.50')
    ap.add_argument('--w_variants', default='0.40,0.60')
    ap.add_argument('--outer_flat', default='0.60', help='KF twin: flat outer weight (parent_audit.md: .60, the pre-probe value)')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args(argv)
    mj, mem = load_members(a.members_json, a.ckpt_dir.replace('\\', '/'))
    names = [r['staged'] for r in mem]
    specs = [r['path'] + (f'={r["key"]}' if r['staged'] != r['file'] else '') for r in mem]
    tta = {n: a.tta for n in names}
    nC = sum(r['set'] == 'C' for r in mem); nD = sum(r['set'] == 'D' for r in mem)
    fC = sorted(r['fold'] for r in mem if r['set'] == 'C'); fD = sorted(r['fold'] for r in mem if r['set'] == 'D')
    interim = not (fC == [0, 1, 2, 3, 4] and fD == [0, 1, 2, 3, 4])
    sim = any('/k9_test/sim/' in str(r.get('run', '')) for r in mem)   # collected from the SIMULATION root (test only)
    if interim:
        assert mj.get('do_not_submit') and mj.get('interim'), \
            f'set C {nC}/5, set D {nD}/5: an INTERIM build needs a members.json marked do_not_submit (k9_collect.sh K9_INTERIM=1)'
    else:
        assert not mj.get('do_not_submit'), 'members.json is marked do_not_submit but holds C 5/5 + D 5/5: re-collect'
    if not a.out:
        a.out = f'{HERE}/build' + (f'_i{len(mem)}' if interim else '') + ('' if a.tta == 'AB' else '_A')
    a.out = a.out.replace(chr(92), '/')
    for f in FROZEN:
        assert sha256(f'{HERE}/code/{f}') == sha256(f'{SP}/k3/code/{f}'), f'{f} differs from K3 (model_sha contract)'
    assert sha256(f'{HERE}/code/fleet_worker.py') == sha256(f'{SP}/k5/code/fleet_worker.py'), \
        'k9/code/fleet_worker.py != the K5 worker (K9 must ship the worker that scored .946, byte for byte)'
    assert sha256(f'{HERE}/code/pack_fleet_ds.py') == sha256(f'{SP}/k7/code/pack_fleet_ds.py'), 'k9 pack tool != K7 pack tool'
    wsrc = open(f'{HERE}/code/fleet_worker.py', encoding='utf-8').read()
    assert 'K5-TTA' in wsrc and 'TTA_HALF = 0.5 - 1e-3' in wsrc, 'k9/code/fleet_worker.py is not the K5 TTA worker'
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
    assert sha256(f'{up["path"]}/fleet_worker.py') == sha256(f'{HERE}/code/fleet_worker.py'), 'shipped worker != k9/code'
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
                and r['train_labels'] == x['train_labels'] and r['key'] == x['key']
                and r['epoch_1based'] == x['epoch'] and abs(float(r['cv']) - x['cv']) < 1e-12), (r, x)
    cell = open(f'{HERE}/cells/k2_fleet_cell.py', encoding='utf-8').read()
    k3cell = open(f'{SP}/k3/cells/k2_fleet_cell.py', encoding='utf-8').read()
    assert strip_k5(cell) == k3cell, 'k9 cell minus the K5a block != the K3 cell (only the K5a block may differ)'
    assert sha256(f'{HERE}/cells/k2_fleet_cell.py') == sha256(f'{SP}/k5/cells/k2_fleet_cell.py'), 'k9 cell != the K5 cell'
    assert sha256(f'{HERE}/cells/k2_fleet_cell_kf.py') == sha256(f'{SP}/k8/cells/k2_fleet_cell_kf.py'), 'k9 KF cell != K8 KF cell'
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
    info = dict(built=time.strftime('%Y-%m-%d %H:%M:%S'), k8=True, k9=True, interim=interim, simulated=sim,
                do_not_submit=interim or sim,
                n_members=len(names), members=names, sets=dict(C=nC, D=nD), folds=[r['fold'] for r in mem], tta=tta,
                expect_tta=tta_s, worker_sha16=sha256(f'{HERE}/code/fleet_worker.py')[:16],
                cell_sha16=hashlib.sha256(cell.encode()).hexdigest()[:16],
                kf_cell_sha16=hashlib.sha256(kfcell.encode()).hexdigest()[:16],
                outer_flat=flat, kf_changed_labels=KF_CHANGED,
                picks=[dict({k: r[k] for k in ('set', 'seed', 'fold', 'ttr_size', 'train_labels', 'tag', 'epoch', 'epochs',
                                               'cv', 'other_tag_cv', 'file', 'key', 'staged', 'sha256', 'source')},
                            suffix=r.get('suffix'), oof=r.get('oof'), kaggle_receipt=r.get('kaggle_receipt'))
                       for r in mem],
                upload=dict(path=up['path'], mb=up['mb'], files=len(up['files'])),
                kernels={s: dict(mode=want[s][0], w_main=ks[s]['K2_CFG']['W_MAIN'], outer_flat=want[s][2],
                                 dir=ks[s]['out'], nb_sha16=sha256(f'{ks[s]["out"]}/{ks[s]["code_file"]}')[:16])
                         for s in want},
                members_json_generated=mj.get('generated'), skipped=mj.get('skipped'), set_c=mj.get('set_c'),
                set_d=mj.get('set_d'))
    json.dump(info, open(f'{a.out}/K5_BUILD.json', 'w'), indent=1)
    for fn, cond, txt in (('DO_NOT_PUSH_SIMULATION.txt', sim, 'SIMULATION BUILD (members from the ~/k9_test/sim root).'),
                          ('DO_NOT_SUBMIT_INTERIM.txt', interim, f'INTERIM BUILD (set C {nC}/5, set D {nD}/5).')):
        if cond:                                                         # marker in the build dir, never inside upload/
            open(f'{a.out}/{fn}', 'w').write(txt + ' TEST ONLY. NEVER version the upload folder, NEVER push these '
                                             'kernels, NEVER submit. K5_BUILD.json: do_not_submit true.\n')
    keys9 = [f'{r["set"]}:{r["key"]}' for r in mem]
    tag = (f'INTERIM ({len(mem)} members: C {nC}/5, D {nD}/5: {keys9}) -- TEST ONLY, do_not_submit, NEVER SUBMIT'
           if interim else f'FINAL ({len(mem)} members: C 5/5, D 5/5: {keys9})')
    if sim:
        tag = 'SIMULATION (members from ~/k9_test/sim: TEST ONLY, NEVER VERSION / PUSH / SUBMIT) ' + tag
    print(f'\nK9 BUILD OK {tag}: upload {up["path"]} ({up["mb"]} MB, {len(up["files"])} files) | members {names} | '
          f'TTA {tta} | kernels {MAIN} (strict W {a.w_main}), {SOLO} (solo), {KF} (strict W {a.w_main}, OUTER_FLAT {flat})')
    print(f'expect_members: {",".join(names)}')
    print(f'expect_tta: {tta_s}')
    return info


if __name__ == '__main__':
    main()
