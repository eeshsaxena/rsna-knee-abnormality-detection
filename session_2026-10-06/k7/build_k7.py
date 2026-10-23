#!/usr/bin/env python3
"""build_k7.py (2026-10-06 evening, Claude; a COPY of build_k6.py, itself a copy of build_k5.py). Local only: nothing is
pushed, versioned or submitted.

K7 = mix50 labels only, every member TTR (fleet_model_ttr, model_sha f141aeed136dfc01), every member served A+B:
  set A  f{0..4}_s1043_ttr_mix50     TTR224, seed 1043 (the K5 members; folds 3-4 Kaggle-filled, f34 receipt)
  set B  f{0..4}_s1044_ttr_mix50     TTR224, seed 1044 (teammate swapiy's Kaggle runs, receipts kaggle_s1044_*)
  set C  f0_s1044_ttr288mix50        TTR288, seed 1044, fold 0 (optional; k7_collect.sh includes it only when its
                                     picked val cv >= .8876)
Fleet = raw-prob mean over ALL members (each member weight 1/N). Kernels from ONE upload folder:
  K7-main  eeshsaxena/rsna-sub-k7-fleet  MAIN_MODE strict, W_MAIN .50 (variants .40 / .60)  (W .50 = the settled default)
  K7-solo  eeshsaxena/rsna-sub-k7-solo   MAIN_MODE solo (submission.csv = raw-prob mean of the set)

K7-KEY (the set B / set C conflict): both carry f0_s1044, and the K5 worker refuses two members with the same
f{fold}_s{seed} file prefix (fleet_worker.py line 109-110: fs_key / 'duplicate fold/seed members (raw AND ema?)'). The
worker is NOT changed (byte-identical to the K5 worker that scored .946). Instead every member whose ttr_size != 224 is
re-keyed with the existing build_k2.py PATH=NEWKEY / pack_fleet_ds.py P-TAG mechanism: f0_s1044 -> f0_s1044t288, so the
TTR288 member ships as f0_s1044t288_raw_ttr288-320x24_cv<NNNN>_best.pt (bytes identical, only the file name changes).
The worker's fold grouping (field 1, 'f0') and its duplicate check (fields 1-2) stay valid: B f0 key 'f0_s1044', C f0
key 'f0_s1044t288'. Member identity here = (seed, fold, ttr_size); a true duplicate (same run twice, or raw AND ema of one
run) is still refused by this script, by build_k2.py (run key incl. ttr_size), by pack_fleet_ds.py P-RUN (identical
ckpt args, even when re-keyed) and by the worker (same key).

  python build_k7.py [--members_json k7/members.json] [--out DIR] [--overwrite] [--w_main 0.50]

Default --out: k7/build when sets A and B are complete (10 TTR224 members, + C if collected) = FINAL, else
k7/build_i<N> (INTERIM). K5_BUILD.json gains per pick 'staged' (the shipped file name), 'seed', 'ttr_size', 'suffix'; the
stage / launch scripts map staged -> collected file through it.
"""
import argparse, glob, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
SP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
K2_META = f'{SP}/k2/build/kernel/rsna-sub-k2-fleet/kernel-metadata.json'      # the proven K2 kernel's sources
MAIN, SOLO = 'rsna-sub-k7-fleet', 'rsna-sub-k7-solo'
SET_AB = {(1043, 224): 'A', (1044, 224): 'B'}                    # (seed, ttr_size) -> set; set C = (1044, 288)
SET_C = (1044, 288)
SEEDS = (1043, 1044)
TTR_SHA = 'f141aeed136dfc01'
TRAIN_LABELS = 'labels_mix50_logit.csv'                            # settled: mix50 labels only
FROZEN = ('fleet_common.py', 'fleet_model.py', 'fleet_model_f2.py', 'fleet_model_ttr.py')   # model_sha / tensor contract


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
    """K7-KEY: the shipped key. TTR224 members keep f{F}_s{S} (byte-identical names to K5/K6); others get a t{N} suffix."""
    k = f'f{r["fold"]}_s{r["seed"]}'
    return k if int(r['ttr_size']) == 224 else f'{k}t{int(r["ttr_size"])}'


def load_members(path, ckdir):
    m = json.load(open(path, encoding='utf-8'))
    assert not m.get('failed'), f'members.json lists failed folds {m["failed"]}: do not build'
    mem = sorted(m['members'], key=lambda r: (int(r['ttr_size']), r['seed'], r['fold']))     # A, B, then C
    assert mem, 'members.json has no member'
    ids = [(r['seed'], r['fold'], int(r['ttr_size'])) for r in mem]
    assert len(set(ids)) == len(ids), f'duplicate seed/fold/ttr_size members {ids} (one pick per run)'
    files = [r['file'] for r in mem]
    assert len(set(files)) == len(files), f'duplicate member files {files}'
    for r in mem:
        st = (r['seed'], int(r['ttr_size']))
        assert st in SET_AB or st == SET_C, f'{r["file"]}: (seed, ttr_size) {st} is not a K7 set'
        assert r.get('train_labels') == TRAIN_LABELS, f'{r["file"]}: train_labels {r.get("train_labels")} (K7 = mix50 only)'
        assert r.get('model_mod') == 'fleet_model_ttr' and r.get('model_sha') == TTR_SHA, (r['file'], r.get('model_mod'), r.get('model_sha'))
        p = f'{ckdir}/{r["file"]}'
        assert os.path.isfile(p), f'missing {p} (run k7_collect.sh)'
        assert r['file'].startswith(f'f{r["fold"]}_s{r["seed"]}_{r["tag"]}_ttr{int(r["ttr_size"])}-'), r['file']
        h = sha256(p)
        assert h == r['sha256'], f'{r["file"]}: laptop sha256 {h} != members.json {r["sha256"]}'
        r['path'] = p
        r['key'] = member_key(r)
        r['staged'] = r['key'] + r['file'][len(f'f{r["fold"]}_s{r["seed"]}'):]
        r['set'] = SET_AB.get(st, 'C')
    keys = [r['key'] for r in mem]
    assert len(set(keys)) == len(keys), f'duplicate shipped keys {keys}'
    on_disk = sorted(os.path.basename(p) for sd in SEEDS for p in glob.glob(f'{ckdir}/f*_s{sd}_*_best.pt'))
    assert on_disk == sorted(files), \
        f'k7/ckpt files {on_disk} != members.json {files} (stale file? re-run k7_collect.sh)'
    return m, mem


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--members_json', default=f'{HERE}/members.json')
    ap.add_argument('--ckpt_dir', default=f'{HERE}/ckpt')
    ap.add_argument('--tta', choices=('A', 'AB'), default='AB', help='TTA of every member (G-TTA: TTR PASS -> AB)')
    ap.add_argument('--store', default=f'{HERE}/_real/store')
    ap.add_argument('--out', default=None, help='default k7/build (A + B complete) or k7/build_i<N> (interim)')
    ap.add_argument('--w_main', default='0.50')
    ap.add_argument('--w_variants', default='0.40,0.60')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args(argv)
    mj, mem = load_members(a.members_json, a.ckpt_dir.replace('\\', '/'))
    names = [r['staged'] for r in mem]
    specs = [r['path'] + (f'={r["key"]}' if r['staged'] != r['file'] else '') for r in mem]
    tta = {n: a.tta for n in names}
    nA = sum(r['set'] == 'A' for r in mem); nB = sum(r['set'] == 'B' for r in mem); nC = sum(r['set'] == 'C' for r in mem)
    interim = not (nA == 5 and nB == 5)
    if not a.out:
        a.out = f'{HERE}/build' + (f'_i{len(mem)}' if interim else '') + ('' if a.tta == 'AB' else '_A')
    a.out = a.out.replace(chr(92), '/')
    for f in FROZEN:
        assert sha256(f'{HERE}/code/{f}') == sha256(f'{SP}/k3/code/{f}'), f'{f} differs from K3 (model_sha contract)'
    assert sha256(f'{HERE}/code/fleet_worker.py') == sha256(f'{SP}/k5/code/fleet_worker.py'), \
        'k7/code/fleet_worker.py != the K5 worker (K7 must ship the worker that scored .946, byte for byte)'
    wsrc = open(f'{HERE}/code/fleet_worker.py', encoding='utf-8').read()
    assert 'K5-TTA' in wsrc and 'TTA_HALF = 0.5 - 1e-3' in wsrc, 'k7/code/fleet_worker.py is not the K5 TTA worker'
    tta_s = ','.join(f'{n}={m}' for n, m in tta.items())
    import build_k2 as bk
    common = ['--out', a.out]
    bk.main(sum((['--ckpt', s] for s in specs), []) +
            ['--store', a.store, '--slug', MAIN, '--main_mode', 'strict', '--members', ','.join(names),
             '--w_main', a.w_main, '--w_variants', a.w_variants, '--tta', tta_s] + common
            + (['--overwrite'] if a.overwrite else []))
    summ = bk.main(['--kernel_only', '--slug', SOLO, '--main_mode', 'solo', '--members', ','.join(names),
                    '--w_main', a.w_main, '--w_variants', a.w_variants] + common)
    # ---- post checks
    want = {MAIN: ('strict', float(a.w_main)), SOLO: ('solo', None)}
    ks = {k['slug']: k for k in summ['kernels']}
    assert set(ks) == set(want), sorted(ks)
    for slug, (mode, w) in want.items():
        cfg = ks[slug]['K2_CFG']
        assert cfg['MAIN_MODE'] == mode and cfg['MEMBERS'] == names, (slug, cfg)
        assert cfg.get('TTA') == tta, (slug, 'K2_CFG TTA (K5b)', cfg.get('TTA'), tta)
        if w is not None:
            assert abs(cfg['W_MAIN'] - w) < 1e-9, (slug, cfg['W_MAIN'], w)
    up = summ['upload']
    assert up.get('manifest') == names, (up.get('manifest'), names)
    assert up.get('tta') == tta, (up.get('tta'), tta)
    n_probe = len({s for r in summ['members'] for s in r['decode_probe']})
    assert 'fleet_model_ttr.py' in up['files'] and len(up['files']) == 5 + len(names) + 2 + n_probe, up['files']
    assert sha256(f'{up["path"]}/fleet_worker.py') == sha256(f'{HERE}/code/fleet_worker.py'), 'shipped worker != k7/code'
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
    assert strip_k5(cell) == k3cell, 'k7 cell minus the K5a block != the K3 cell (only the K5a block may differ)'
    assert sha256(f'{HERE}/cells/k2_fleet_cell.py') == sha256(f'{SP}/k5/cells/k2_fleet_cell.py'), 'k7 cell != the K5 cell'
    k2m = json.load(open(K2_META, encoding='utf-8')) if os.path.isfile(K2_META) else None
    for slug, k in ks.items():
        nb = json.load(open(f'{k["out"]}/{k["code_file"]}', encoding='utf-8'))
        src = ''.join(nb['cells'][31]['source'])
        hdr, body = src.split('\n', 1)
        assert hdr.startswith('K2_CFG = ') and body == cell, f'{slug}: cell 31 != cells/k2_fleet_cell.py + K2_CFG line'
        meta = json.load(open(f'{k["out"]}/kernel-metadata.json', encoding='utf-8'))
        if k2m:
            for key in ('dataset_sources', 'kernel_sources', 'competition_sources', 'model_sources', 'machine_shape',
                        'enable_gpu', 'enable_internet'):
                assert meta.get(key) == k2m.get(key), (slug, key, meta.get(key), k2m.get(key))
        assert meta['id'] == f'eeshsaxena/{slug}' and meta['is_private'] is True
        print(f'{slug}: {len(nb["cells"])} cells, K2_CFG {ks[slug]["K2_CFG"]}, '
              f'notebook sha256 {sha256(f"{k["out"]}/{k["code_file"]}")[:16]}, dataset_sources == K2: {bool(k2m)}')
    info = dict(built=time.strftime('%Y-%m-%d %H:%M:%S'), interim=interim, n_members=len(names), members=names,
                sets=dict(A=nA, B=nB, C=nC), folds=[r['fold'] for r in mem], tta=tta, expect_tta=tta_s,
                worker_sha16=sha256(f'{HERE}/code/fleet_worker.py')[:16],
                cell_sha16=hashlib.sha256(cell.encode()).hexdigest()[:16],
                picks=[dict({k: r[k] for k in ('set', 'seed', 'fold', 'ttr_size', 'tag', 'epoch', 'cv', 'other_tag_cv', 'file',
                                               'staged', 'sha256', 'source')}, suffix=r.get('suffix'), oof=r.get('oof'))
                       for r in mem],
                upload=dict(path=up['path'], mb=up['mb'], files=len(up['files'])),
                kernels={s: dict(mode=want[s][0], w_main=ks[s]['K2_CFG']['W_MAIN'], dir=ks[s]['out'],
                                 nb_sha16=sha256(f'{ks[s]["out"]}/{ks[s]["code_file"]}')[:16]) for s in want},
                members_json_generated=mj.get('generated'), skipped=mj.get('skipped'), set_c=mj.get('set_c'))
    json.dump(info, open(f'{a.out}/K5_BUILD.json', 'w'), indent=1)
    keys7 = [f'{r["set"]}:{r["key"]}' for r in mem]
    tag = (f'INTERIM ({len(mem)} members: A {nA}/5, B {nB}/5, C {nC}: {keys7})' if interim
           else f'FINAL ({len(mem)} members: A 5/5, B 5/5, C {nC}: {keys7})')
    print(f'\nK7 BUILD OK {tag}: upload {up["path"]} ({up["mb"]} MB, {len(up["files"])} files) | members {names} | '
          f'TTA {tta} | kernels {MAIN} (strict W {a.w_main}), {SOLO} (solo)')
    print(f'expect_members: {",".join(names)}')
    print(f'expect_tta: {tta_s}')
    return info


if __name__ == '__main__':
    main()
