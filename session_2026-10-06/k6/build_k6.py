#!/usr/bin/env python3
"""build_k6.py (2026-10-06, Claude; a COPY of build_k5.py): K6 = the K5 set (5-fold TTR mix50, f{0..4}_s1043_ttr_mix50) PLUS the
5-fold TTR v4 set (f{0..4}_s1042_ttr, pure llm_labels_v4_blend targets), all 10 served A+B, raw-prob mean over the 10 members
(each set weight .5), blended into the parent at W .50 (K5 reads: W .4-.6 flat, W .5 optimal). Only member handling
(seed+fold keys, 10 = FINAL) and slugs differ from build_k5.py. Original K5 docstring follows.

build_k5.py: the K5 kernels = the K4 set (5-fold TTR mix50, f{0..4}_s1043_ttr_mix50) served with per-member TTA
(TTR members A+B, M1 pass A only), from ONE upload folder (= one new version of eeshsaxena/rsna-knee-cnxt-fleet), one
command. Wraps build_k2.py (K5 copy: --tta); local only, nothing is pushed, versioned or submitted.

  python build_k5.py [--members_json k5/members.json] [--with_m1] [--tta_ttr AB|A] [--tta_m1 A|AB] [--no_w50]
                     [--w_main 0.20] [--out DIR] [--overwrite]

Default --out: k5/build for the FINAL 5-fold set, k5/build_i<N> for an INTERIM N-fold set (N < 5), + _m1 with --with_m1
(+ _ttrA when --tta_ttr A), so a later build never overwrites a kit whose commit still has to be gated.

  K5-main  eeshsaxena/rsna-sub-k5-fleet  MAIN_MODE strict, W_MAIN .20 (+ w30 / w50 / fleetsolo diagnostics), MEMBERS = set
  K5-solo  eeshsaxena/rsna-sub-k5-solo   MAIN_MODE solo   (submission.csv = raw-prob mean of the set),     MEMBERS = set
  K5-w50   eeshsaxena/rsna-sub-k5-w50    MAIN_MODE strict, W_MAIN .50 (+ w20 / w35),  kernel_only,         MEMBERS = set

TTA (BUILD.md B6, G-TTA gate ~/jump2/tta/run1): every TTR mix50 fold member is served A+B (--tta_ttr AB, default: J1
PASSED the gate, A+B - A = +.0020 val P 1.00, gold58 +.0015); M1 (F1 mix50 f0_s42, --with_m1) is served pass A only
(--tta_m1 A, default: it FAILED the gate, gold58 -.0013). The map is written into the manifest (sources[name]['tta'])
and K5_BUILD.json ('tta', 'expect_tta' = the k2_check.py --expect_tta string).
Members = every checkpoint in k5/members.json (k5_collect.sh), each verified: file in k5/ckpt, sha256 == members.json,
no other f*_s1043_*_best.pt in k5/ckpt, folds distinct. 1-4 members = INTERIM (printed loudly); 5 = FINAL.
Post checks: cell 31 of every notebook == cells/k2_fleet_cell.py + its K2_CFG line; that cell == the K3 cell plus ONLY
the K5a block (stripping it gives the K3 cell byte for byte); the shipped worker == k5/code/fleet_worker.py (K5-TTA);
fleet_common / fleet_model* == K3 (model_sha contract); manifest checkpoints / sha256[:16] / tta == the plan; kernel
metadata == the proven K2 kernel's.
"""
import argparse, glob, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
SP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
M1_NAME = 'f0_s42_raw_cnxt320x24_cv8810_best.pt'
M1_SHA16 = 'dc7e1495404e4e7a'
M1_DEFAULT = f'{SP}/k3/ckpt/{M1_NAME}'                     # read only (k3 is never edited)
K2_META = f'{SP}/k2/build/kernel/rsna-sub-k2-fleet/kernel-metadata.json'      # the proven K2 kernel's sources
MAIN, SOLO, W50 = 'rsna-sub-k6-fleet', 'rsna-sub-k6-solo', 'rsna-sub-k6-w50'
N_FINAL = 10                                                       # K6: 5 mix50 (s1043) + 5 v4 (s1042) folds
SEEDS = (1042, 1043)
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


def load_members(path, ckdir):
    m = json.load(open(path, encoding='utf-8'))
    assert not m.get('failed'), f'members.json lists failed folds {m["failed"]}: do not build'
    mem = sorted(m['members'], key=lambda r: (r['seed'], r['fold']))                       # K6: (seed, fold) keys
    assert mem, 'members.json has no member'
    keys = [(r['seed'], r['fold']) for r in mem]
    assert len(set(keys)) == len(keys), f'duplicate seed/fold members {keys}'
    assert {r['seed'] for r in mem} <= set(SEEDS), keys
    for r in mem:
        p = f'{ckdir}/{r["file"]}'
        assert os.path.isfile(p), f'missing {p} (run k5_collect.sh)'
        assert r['file'].startswith(f'f{r["fold"]}_s{r["seed"]}_{r["tag"]}_ttr224-'), r['file']
        h = sha256(p)
        assert h == r['sha256'], f'{r["file"]}: laptop sha256 {h} != members.json {r["sha256"]}'
        r['path'] = p
    on_disk = sorted(os.path.basename(p) for sd in SEEDS for p in glob.glob(f'{ckdir}/f*_s{sd}_*_best.pt'))
    assert on_disk == sorted(r['file'] for r in mem), \
        f'k5/ckpt TTR mix50 files {on_disk} != members.json {[r["file"] for r in mem]} (stale file? re-run k5_collect.sh)'
    return m, mem


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--members_json', default=f'{HERE}/members.json')
    ap.add_argument('--ckpt_dir', default=f'{HERE}/ckpt')
    ap.add_argument('--with_m1', action='store_true', help='also ship M1 (F1 mix50 f0_s42 raw) in every kernel')
    ap.add_argument('--m1', default=M1_DEFAULT)
    ap.add_argument('--tta_ttr', choices=('A', 'AB'), default='AB', help='TTA of every TTR mix50 member (G-TTA: PASS -> AB)')
    ap.add_argument('--tta_m1', choices=('A', 'AB'), default='A', help='TTA of M1 (G-TTA: FAIL -> A)')
    ap.add_argument('--store', default=f'{HERE}/_real/store')
    ap.add_argument('--out', default=None, help='default k5/build (5 folds) or k5/build_i<N> (interim), + _m1 / _ttrA')
    ap.add_argument('--w_main', default='0.50')
    ap.add_argument('--w_variants', default='0.40,0.60')
    ap.add_argument('--w50', default='0.50')
    ap.add_argument('--w50_variants', default='0.20,0.35')
    ap.add_argument('--no_w50', action='store_true', help='skip the optional K5-w50 kernel')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args(argv)
    mj, mem = load_members(a.members_json, a.ckpt_dir.replace('\\', '/'))
    names = [r['file'] for r in mem]
    specs = [r['path'] for r in mem]
    tta = {n: a.tta_ttr for n in names}
    if a.with_m1:
        assert os.path.isfile(a.m1) and os.path.basename(a.m1) == M1_NAME, a.m1
        assert sha256(a.m1)[:16] == M1_SHA16, f'M1 bytes differ from the K2 member ({M1_SHA16})'
        specs = [a.m1] + specs
        names = [M1_NAME] + names
        tta[M1_NAME] = a.tta_m1
    tta = {n: tta[n] for n in names}                                    # manifest order
    interim = len(mem) < N_FINAL
    if not a.out:
        a.out = (f'{HERE}/build' + (f'_i{len(mem)}' if interim else '') + ('_m1' if a.with_m1 else '')
                 + ('' if a.tta_ttr == 'AB' else '_ttrA') + ('' if a.tta_m1 == 'A' else '_m1AB'))
    a.out = a.out.replace(chr(92), '/')
    # the K5 code: frozen model / contract files == K3 byte for byte, worker = the K5 TTA worker
    for f in FROZEN:
        assert sha256(f'{HERE}/code/{f}') == sha256(f'{SP}/k3/code/{f}'), f'{f} differs from K3 (model_sha contract)'
    wsrc = open(f'{HERE}/code/fleet_worker.py', encoding='utf-8').read()
    assert 'K5-TTA' in wsrc and 'TTA_HALF = 0.5 - 1e-3' in wsrc, 'k5/code/fleet_worker.py is not the K5 TTA worker'
    tta_s = ','.join(f'{n}={m}' for n, m in tta.items())
    import build_k2 as bk
    common = ['--out', a.out]
    bk.main(sum((['--ckpt', p] for p in specs), []) +
            ['--store', a.store, '--slug', MAIN, '--main_mode', 'strict', '--members', ','.join(names),
             '--w_main', a.w_main, '--w_variants', a.w_variants, '--tta', tta_s] + common
            + (['--overwrite'] if a.overwrite else []))
    summ = bk.main(['--kernel_only', '--slug', SOLO, '--main_mode', 'solo', '--members', ','.join(names),
                    '--w_main', a.w_main, '--w_variants', a.w_variants] + common)
    if not a.no_w50:
        summ = bk.main(['--kernel_only', '--slug', W50, '--main_mode', 'strict', '--members', ','.join(names),
                        '--w_main', a.w50, '--w_variants', a.w50_variants] + common)
    # ---- post checks
    want = {MAIN: ('strict', float(a.w_main)), SOLO: ('solo', None)}
    if not a.no_w50:
        want[W50] = ('strict', float(a.w50))
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
    assert sha256(f'{up["path"]}/fleet_worker.py') == sha256(f'{HERE}/code/fleet_worker.py'), 'shipped worker != k5/code'
    man = json.load(open(f'{up["path"]}/fleet_manifest.json', encoding='utf-8'))
    srcs = man.get('sources') or {}
    assert {k: (v or {}).get('tta') for k, v in srcs.items()} == tta, ('manifest tta', srcs)
    for r in mem:                                                       # K3c: the bytes the commit must report
        s = srcs.get(r['file'])
        s16 = (s.get('sha256') if isinstance(s, dict) else s) or ''
        assert s16[:16] == r['sha256'][:16], (r['file'], s, r['sha256'][:16])
    for r in summ['members']:
        if r['staged'] in [x['file'] for x in mem]:
            x = [y for y in mem if y['file'] == r['staged']][0]
            assert r['fold'] == x['fold'] and r['epoch_1based'] == x['epoch'] and abs(float(r['cv']) - x['cv']) < 1e-12, (r, x)
    cell = open(f'{HERE}/cells/k2_fleet_cell.py', encoding='utf-8').read()
    k3cell = open(f'{SP}/k3/cells/k2_fleet_cell.py', encoding='utf-8').read()
    assert strip_k5(cell) == k3cell, 'k5 cell minus the K5a block != the K3 cell (only the K5a block may differ)'
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
                with_m1=a.with_m1, folds=[r['fold'] for r in mem], tta=tta, expect_tta=tta_s,
                worker_sha16=sha256(f'{HERE}/code/fleet_worker.py')[:16],
                cell_sha16=hashlib.sha256(cell.encode()).hexdigest()[:16],
                picks=[{k: r[k] for k in ('fold', 'tag', 'epoch', 'cv', 'other_tag_cv', 'file', 'sha256')} for r in mem],
                upload=dict(path=up['path'], mb=up['mb'], files=len(up['files'])),
                kernels={s: dict(mode=want[s][0], w_main=ks[s]['K2_CFG']['W_MAIN'], dir=ks[s]['out'],
                                 nb_sha16=sha256(f'{ks[s]["out"]}/{ks[s]["code_file"]}')[:16]) for s in want},
                members_json_generated=mj.get('generated'), skipped=mj.get('skipped'))
    json.dump(info, open(f'{a.out}/K5_BUILD.json', 'w'), indent=1)
    keys6 = [f's{r["seed"]}f{r["fold"]}' for r in mem]
    tag = (f'INTERIM ({len(mem)}/{N_FINAL} members: {keys6})' if interim else f'FINAL ({N_FINAL}/{N_FINAL} members: {keys6})')
    print(f'\nK6 BUILD OK {tag}: upload {up["path"]} ({up["mb"]} MB, {len(up["files"])} files) | members {names} | '
          f'TTA {tta} | kernels {MAIN} (strict W {a.w_main}), {SOLO} (solo)' + ('' if a.no_w50 else f', {W50} (strict W {a.w50})'))
    print(f'expect_members: {",".join(names)}')
    print(f'expect_tta: {tta_s}')
    return info


if __name__ == '__main__':
    main()
