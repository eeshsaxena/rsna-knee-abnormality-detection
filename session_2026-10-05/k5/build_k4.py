#!/usr/bin/env python3
"""build_k4.py: the K4 kernels for the 5-fold TTR mix50 set from ONE upload folder (= one new version of
eeshsaxena/rsna-knee-cnxt-fleet), one command. Wraps build_k2.py (K3 kit, unchanged); local only, nothing is pushed,
versioned or submitted.

  python build_k4.py [--members_json k4/members.json] [--with_m1] [--no_w50] [--w_main 0.20] [--out DIR] [--overwrite]

Default --out: k4/build for the FINAL 5-fold set, k4/build_i<N> for an INTERIM N-fold set (N < 5), k4/build_*_m1 with
--with_m1, so a later build never overwrites an interim kit whose commit still has to be gated against its manifest.

  K4-main  eeshsaxena/rsna-sub-k4-fleet  MAIN_MODE strict, W_MAIN .20 (+ w30 / w50 / fleetsolo diagnostics), MEMBERS = set
  K4-solo  eeshsaxena/rsna-sub-k4-solo   MAIN_MODE solo   (submission.csv = raw-prob mean of the set),     MEMBERS = set
  K4-w50   eeshsaxena/rsna-sub-k4-w50    MAIN_MODE strict, W_MAIN .50 (+ w20 / w35),  kernel_only,         MEMBERS = set

Members = every checkpoint in k4/members.json (written by k4_collect.sh: one tag per FINISHED fold run of
f{0..4}_s1043_ttr_mix50), each verified here: file in k4/ckpt, sha256 == members.json, no other f*_s1043_*_best.pt in
k4/ckpt (a stale pick can never ship), folds distinct. 1-4 members = an INTERIM build (printed loudly); 5 = final.
--with_m1 adds M1 (f0_s42_raw_cnxt320x24_cv8810_best.pt, the K2 family-1 mix50 member, from k3/ckpt, sha256 pinned
dc7e1495...) to the upload AND to every kernel's member list. Default OFF: K4 measures the TTR mix50 set alone.
Every kernel names its members explicitly, so a commit that mounts any other dataset version (e.g. pushed before the
new version is ready) fails loud (strict / solo: no submission.csv) instead of silently running another member set.
Post checks: cell 31 of every notebook == cells/k2_fleet_cell.py + its own K2_CFG line, dataset_sources and the other
kernel metadata == the proven K2 kernel's, manifest checkpoints == the set, manifest sources sha256[:16] == members.json.
"""
import argparse, glob, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
SP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
M1_NAME = 'f0_s42_raw_cnxt320x24_cv8810_best.pt'
M1_SHA16 = 'dc7e1495404e4e7a'
M1_DEFAULT = f'{SP}/k3/ckpt/{M1_NAME}'                     # read only (k3 is never edited)
K2_META = f'{SP}/k2/build/kernel/rsna-sub-k2-fleet/kernel-metadata.json'      # the proven K2 kernel's sources
MAIN, SOLO, W50 = 'rsna-sub-k4-fleet', 'rsna-sub-k4-solo', 'rsna-sub-k4-w50'
N_FINAL = 5


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def load_members(path, ckdir):
    m = json.load(open(path, encoding='utf-8'))
    assert not m.get('failed'), f'members.json lists failed folds {m["failed"]}: do not build'
    mem = sorted(m['members'], key=lambda r: r['fold'])
    assert mem, 'members.json has no member'
    folds = [r['fold'] for r in mem]
    assert len(set(folds)) == len(folds), f'duplicate folds {folds}'
    for r in mem:
        p = f'{ckdir}/{r["file"]}'
        assert os.path.isfile(p), f'missing {p} (run k4_collect.sh)'
        assert r['file'].startswith(f'f{r["fold"]}_s1043_{r["tag"]}_ttr224-'), r['file']
        h = sha256(p)
        assert h == r['sha256'], f'{r["file"]}: laptop sha256 {h} != members.json {r["sha256"]}'
        r['path'] = p
    on_disk = sorted(os.path.basename(p) for p in glob.glob(f'{ckdir}/f*_s1043_*_best.pt'))
    assert on_disk == sorted(r['file'] for r in mem), \
        f'k4/ckpt TTR mix50 files {on_disk} != members.json {[r["file"] for r in mem]} (stale file? re-run k4_collect.sh)'
    return m, mem


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--members_json', default=f'{HERE}/members.json')
    ap.add_argument('--ckpt_dir', default=f'{HERE}/ckpt')
    ap.add_argument('--with_m1', action='store_true', help='also ship M1 (F1 mix50 f0_s42 raw) in every kernel')
    ap.add_argument('--m1', default=M1_DEFAULT)
    ap.add_argument('--store', default=f'{HERE}/_real/store')
    ap.add_argument('--out', default=None, help='default k4/build (5 folds) or k4/build_i<N> (interim), + _m1')
    ap.add_argument('--w_main', default='0.20')
    ap.add_argument('--w_variants', default='0.30,0.50')
    ap.add_argument('--w50', default='0.50')
    ap.add_argument('--w50_variants', default='0.20,0.35')
    ap.add_argument('--no_w50', action='store_true', help='skip the optional K4-w50 kernel')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args(argv)
    mj, mem = load_members(a.members_json, a.ckpt_dir.replace('\\', '/'))
    names = [r['file'] for r in mem]
    specs = [r['path'] for r in mem]
    if a.with_m1:
        assert os.path.isfile(a.m1) and os.path.basename(a.m1) == M1_NAME, a.m1
        assert sha256(a.m1)[:16] == M1_SHA16, f'M1 bytes differ from the K2 member ({M1_SHA16})'
        specs = [a.m1] + specs
        names = [M1_NAME] + names
    interim = len(mem) < N_FINAL
    if not a.out:
        a.out = f'{HERE}/build' + (f'_i{len(mem)}' if interim else '') + ('_m1' if a.with_m1 else '')
    a.out = a.out.replace(chr(92), '/')
    import build_k2 as bk
    common = ['--out', a.out]
    bk.main(sum((['--ckpt', p] for p in specs), []) +
            ['--store', a.store, '--slug', MAIN, '--main_mode', 'strict', '--members', ','.join(names),
             '--w_main', a.w_main, '--w_variants', a.w_variants] + common + (['--overwrite'] if a.overwrite else []))
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
        if w is not None:
            assert abs(cfg['W_MAIN'] - w) < 1e-9, (slug, cfg['W_MAIN'], w)
    up = summ['upload']
    assert up.get('manifest') == names, (up.get('manifest'), names)
    n_probe = len({s for r in summ['members'] for s in r['decode_probe']})
    assert 'fleet_model_ttr.py' in up['files'] and len(up['files']) == 5 + len(names) + 2 + n_probe, up['files']
    man = json.load(open(f'{up["path"]}/fleet_manifest.json', encoding='utf-8'))
    srcs = man.get('sources') or {}
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
    assert cell == k3cell, 'k4/cells/k2_fleet_cell.py differs from the K3 cell (K4 ships the K3 cell unchanged)'
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
        print(f'{slug}: {len(nb["cells"])} cells, K2_CFG {hdr[len("K2_CFG = "):hdr.index("}") + 1]}, '
              f'notebook sha256 {sha256(f"{k["out"]}/{k["code_file"]}")[:16]}, dataset_sources == K2: {bool(k2m)}')
    info = dict(built=time.strftime('%Y-%m-%d %H:%M:%S'), interim=interim, n_members=len(names), members=names,
                with_m1=a.with_m1, folds=[r['fold'] for r in mem],
                picks=[{k: r[k] for k in ('fold', 'tag', 'epoch', 'cv', 'other_tag_cv', 'file', 'sha256')} for r in mem],
                upload=dict(path=up['path'], mb=up['mb'], files=len(up['files'])),
                kernels={s: dict(mode=want[s][0], w_main=ks[s]['K2_CFG']['W_MAIN'], dir=ks[s]['out'],
                                 nb_sha16=sha256(f'{ks[s]["out"]}/{ks[s]["code_file"]}')[:16]) for s in want},
                members_json_generated=mj.get('generated'), skipped=mj.get('skipped'))
    json.dump(info, open(f'{a.out}/K4_BUILD.json', 'w'), indent=1)
    tag = (f'INTERIM ({len(mem)}/{N_FINAL} folds: {[r["fold"] for r in mem]})' if interim else f'FINAL ({N_FINAL}/{N_FINAL} folds)')
    print(f'\nK4 BUILD OK {tag}: upload {up["path"]} ({up["mb"]} MB, {len(up["files"])} files) | members {names} | '
          f'kernels {MAIN} (strict W {a.w_main}), {SOLO} (solo)' + ('' if a.no_w50 else f', {W50} (strict W {a.w50})'))
    print(f'expect_members: {",".join(names)}')
    return info


if __name__ == '__main__':
    main()
