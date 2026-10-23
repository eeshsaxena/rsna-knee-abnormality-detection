#!/usr/bin/env python3
"""build_k2.py (K3 kit, scratchpad/k3): from chosen fleet checkpoints (copied from the box) to (1) the upload folder of
dataset eeshsaxena/rsna-knee-cnxt-fleet and (2) the kernel folder(s). Local only: nothing is pushed, created or versioned.

K3 changes (the K2 behaviour of every old flag is unchanged):
  - TTR members (ckpt args model_mod 'fleet_model_ttr', bigjump trainer): model_sha checked against the SHIPPED
    k3/code/fleet_model_ttr.py + fleet_model.py (f141aeed136dfc01), ttr_size buffer == args; fleet_model_ttr.py and
    the bigjump worker (W-TTR) ship in the upload folder; the bigjump pack tool writes the manifest (+ 'sources')
  - --main_mode strict (P+F at W_MAIN, submission.csv DELETED on any failure), --members NAME[,NAME] (member filter
    written into the kernel's K2_CFG; names must be in the manifest), --slug as before
  - --kernel_only: add ONE more kernel to an existing --out build (same upload folder, re-verified byte for byte), so
    two kernels with different modes / filters share one dataset version

  python build_k2.py --ckpt C:/.../fleet_ckpts/f0_s42_ema_cnxt320x24_cv8721_best.pt --store C:/.../fleet_parity_src
        [--ckpt PATH2=NEWKEY ...] [--w_main 0.30 --w_variants 0.20,0.40] [--combined] [--out k2/build]

Steps (each one asserts; any failure stops the build with nothing half-written in --out):
  1. preflight every member: contract == k2/code/fleet_common.CONTRACT, model_sha == k2/code fleet_model(_f2), amp fp16
     (never bf16), one tag per run (raw AND ema of one run refused), no two members with the same fold+seed unless
     re-keyed explicitly (PATH=NEWKEY renames the file key, e.g. f0_s42 -> f0_s42v4), parity npz present in --store
  2. stage the members under OUT/members/ (byte copies; a re-key changes only the file name)
  3. k2/code/pack_fleet_ds.py -> OUT/upload/rsna-knee-cnxt-fleet (code + members + fleet_manifest.json + flat
     parity_<sid>.npz + dataset-metadata.json id eeshsaxena/rsna-knee-cnxt-fleet), then re-verify that folder
  4. make_k2_kernel.py -> OUT/kernel/<slug> (fleet-only K2) and, with --combined, OUT/kernel/<slug_combined>
  5. OUT/BUILD_SUMMARY.json + the exact kaggle commands to run by hand (printed, never executed)
K5 (scratchpad/k5): --tta NAME=MODE[,NAME=MODE] (staged file names, MODE A or AB; every member must be named) ->
  pack_fleet_ds.py --tta writes sources[NAME]['tta']; verify_upload checks the manifest's tta == the request and that the
  shipped worker is the K5 (K5-TTA) worker; the map is kept in BUILD_SUMMARY.json ('tta') and re-verified by
  --kernel_only. Without --tta: no 'tta' key in the manifest (every member pass A only, as before).
"""
import argparse, glob, hashlib, json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
CODE = f'{HERE}/code'
sys.path.insert(0, HERE)
DS_ID, DS_TITLE = 'eeshsaxena/rsna-knee-cnxt-fleet', 'rsna-knee-cnxt-fleet'
SHIP = ['fleet_common.py', 'fleet_model.py', 'fleet_model_f2.py', 'fleet_model_ttr.py', 'fleet_worker.py']
MODS = ('fleet_model', 'fleet_model_f2', 'fleet_model_ttr')


def sha(path, n=16):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()[:n]


def model_sha(mod):
    """== k2/code/fleet_worker.py _model_sha (line-ending agnostic source sha)."""
    import inspect
    sys.path.insert(0, CODE)
    import fleet_model as fm
    src = chr(10).join(inspect.getsource(fm).splitlines())
    if mod == 'fleet_model_f2':
        import fleet_model_f2 as f2
        src = chr(10).join(inspect.getsource(f2).splitlines()) + chr(10) + src
    elif mod == 'fleet_model_ttr':                                       # == bigjump worker W-TTR
        import fleet_model_ttr as tt
        src = chr(10).join(inspect.getsource(tt).splitlines()) + chr(10) + src
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def preflight(specs, store):
    import torch
    sys.path.insert(0, CODE)
    import fleet_common as fc
    rows, keys, runs, fs = [], set(), set(), set()
    for path, newkey in specs:
        assert os.path.isfile(path), f'missing checkpoint {path}'
        base = os.path.basename(path)
        parts = base.split('_')
        assert len(parts) >= 4 and parts[0].startswith('f') and parts[1].startswith('s') and parts[2] in ('raw', 'ema'), \
            f'{base}: expected f<fold>_s<seed>_<raw|ema>_..._best.pt'
        ck = torch.load(path, map_location='cpu', weights_only=False)
        a = ck['args']
        mod = a.get('model_mod') or 'fleet_model'
        assert a.get('contract') == fc.CONTRACT, (base, 'contract differs from k2/code/fleet_common.py')
        assert a.get('amp') == 'fp16' and not a.get('bf16', False), (base, 'amp', a.get('amp'), 'bf16', a.get('bf16'))
        assert mod in MODS, (base, 'unknown model_mod', mod)
        ms = model_sha(mod)
        assert a.get('model_sha') in (None, ms), (base, f'{mod} source changed since training', a.get('model_sha'), ms)
        if mod == 'fleet_model_ttr':                                     # W-TTR / TTR-6, as the worker asserts
            assert a.get('model_sha') == ms, (base, 'TTR member without model_sha', a.get('model_sha'))
            tsz = ck['model'].get('ttr_size_t')
            assert tsz is not None and int(tsz) == int(a['ttr_size']), (base, 'ttr_size buffer != args', tsz, a.get('ttr_size'))
        # a run = its training args (raw and ema of one run are saved with the SAME args dict); both label variants
        # name their files f0_s42_<tag>_..., so neither the folder nor the file prefix identifies a run
        run = (int(a['fold']), int(a['seed']), a.get('train_labels'), a.get('backbone'), mod)
        assert run not in runs, f'{base}: raw AND ema of one run {run} (one tag per run, FLEET_REVIEW W1)'
        runs.add(run)
        fsd = (int(a['fold']), int(a['seed']))
        key = newkey or '_'.join(parts[:2])
        assert key not in keys, f'{base}: duplicate member key {key} (pass PATH=NEWKEY to re-key a second label variant)'
        if fsd in fs and not newkey:
            raise AssertionError(f'{base}: fold {fsd[0]} seed {fsd[1]} already shipped (ENDGAME D1: the loser never ships; '
                                 f're-key with PATH=NEWKEY only if both variants are meant to ship)')
        keys.add(key); fs.add(fsd)
        probe = sorted(a.get('decode_probe', {}))
        miss = [s for s in probe if not os.path.isfile(f'{store}/{s}.npz')]
        assert not miss, f'{base}: parity npz missing in --store for {miss} (scp them from ~/fleet_store)'
        newbase = base if not newkey else newkey + base[len('_'.join(parts[:2])):]
        assert newbase.split('_')[0].startswith('f'), newbase
        rows.append(dict(path=path, staged=newbase, key=key, tag=parts[2], fold=a['fold'], seed=a['seed'],
                         epoch_1based=int(ck.get('epoch', -2)) + 1, cv=ck.get('cv'), train_labels=a.get('train_labels'),
                         sel_labels=a.get('sel_labels'), backbone=a.get('backbone'), model_mod=mod, model_sha=a.get('model_sha'),
                         ttr_size=a.get('ttr_size'), sha16=sha(path),
                         decode_probe=probe, mb=round(os.path.getsize(path) / 2**20, 1)))
        del ck
    return rows


def verify_upload(up, rows, tta=None):
    files = sorted(os.listdir(up))
    assert not [f for f in files if os.path.isdir(f'{up}/{f}')], 'upload folder must be flat'
    for f in SHIP:
        assert sha(f'{up}/{f}', 64) == sha(f'{CODE}/{f}', 64), f'{f} differs from k2/code'
    wsrc = open(f'{up}/fleet_worker.py', encoding='utf-8').read()
    assert 'K2b' in wsrc, 'not the K2 (microbench) worker'
    assert 'W-TTR' in wsrc and "'fleet_model_ttr'" in wsrc, 'not the bigjump (W-TTR) worker'
    man = json.load(open(f'{up}/fleet_manifest.json'))
    assert man['checkpoints'] == [r['staged'] for r in rows], (man, [r['staged'] for r in rows])
    got = {k: (v or {}).get('tta') for k, v in (man.get('sources') or {}).items()}              # K5
    if tta:
        assert got == tta, f'manifest tta {got} != requested {tta}'
        assert 'K5-TTA' in wsrc and 'TTA_HALF = 0.5 - 1e-3' in wsrc, 'TTA requested but the shipped worker is not the K5 worker'
    else:
        assert not any(got.values()), f'manifest carries tta {got} but none was requested'
    for r in rows:
        assert sha(f'{up}/{r["staged"]}', 64) == sha(r['path'], 64), f'{r["staged"]} bytes differ from {r["path"]}'
    probe = sorted({s for r in rows for s in r['decode_probe']})
    for s in probe:
        assert os.path.isfile(f'{up}/parity_{s}.npz'), s
    meta = json.load(open(f'{up}/dataset-metadata.json'))
    assert meta['id'] == DS_ID and meta['title'] == DS_TITLE and meta.get('licenses'), meta
    expect = (set(SHIP) | {r['staged'] for r in rows} | {f'parity_{s}.npz' for s in probe}
              | {'fleet_manifest.json', 'dataset-metadata.json'})
    extra = set(files) - expect
    assert not extra, f'unexpected files in the upload folder: {sorted(extra)}'
    assert not (expect - set(files)), f'missing: {sorted(expect - set(files))}'
    return dict(files=files, mb=round(sum(os.path.getsize(f'{up}/{f}') for f in files) / 2**20, 1), tta=tta or None,
                fleet_common_sha16=sha(f'{up}/fleet_common.py'), worker_sha16=sha(f'{up}/fleet_worker.py'),
                manifest=man['checkpoints'])


def parse_members(spec, names):
    """--members NAME[,NAME]: checkpoint FILE names that must all be in the manifest (None = all members)."""
    if not spec:
        return None
    want = [m.strip() for m in spec.split(',') if m.strip()]
    assert want and len(set(want)) == len(want), f'--members {spec!r}: empty or duplicate'
    unknown = [m for m in want if m not in names]
    assert not unknown, f'--members {unknown} not in the manifest {names}'
    return want


def parse_tta(spec, names):
    """K5 --tta NAME=MODE[,..]: every member named once, MODE A or AB (None = no tta key in the manifest)."""
    if not spec:
        return None
    out = {}
    for x in [y.strip() for y in spec.split(',') if y.strip()]:
        n, _, m = x.rpartition('=')
        assert n and m in ('A', 'AB') and n not in out, f'--tta entry {x!r}: want NAME=A or NAME=AB, each name once'
        out[n] = m
    assert sorted(out) == sorted(names), f'--tta names {sorted(out)} != the members {sorted(names)}'
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', default=None, help='PATH or PATH=NEWKEY (repeatable)')
    ap.add_argument('--store', default=None, help='dir with <sid>.npz for every decode_probe study (from ~/fleet_store)')
    ap.add_argument('--out', default=f'{HERE}/build')
    ap.add_argument('--slug', default='rsna-sub-k2-fleet')
    ap.add_argument('--slug_combined', default='rsna-sub-k2-fleetcrop')
    ap.add_argument('--w_main', type=float, default=0.30)
    ap.add_argument('--main_mode', choices=('blend', 'solo', 'strict'), default='blend')
    ap.add_argument('--members', default=None, help='K3: comma list of manifest checkpoint FILE names to run (default all)')
    ap.add_argument('--kernel_only', action='store_true',
                    help='K3: build ONE more kernel (--slug, --main_mode, --w_main, --members) against the existing --out upload')
    ap.add_argument('--w_variants', default='0.20,0.40')
    ap.add_argument('--wall_h', type=float, default=8.6)
    ap.add_argument('--min_start_min', type=float, default=20.0)
    ap.add_argument('--combined', action='store_true', help='also build the fleet + crop kernel')
    ap.add_argument('--c_w', type=float, default=0.15)
    ap.add_argument('--cf_wc', type=float, default=0.15)
    ap.add_argument('--cf_wf', type=float, default=0.25)
    ap.add_argument('--crop_min_start_min', type=float, default=10.0)
    ap.add_argument('--crop_cell_sha', default='049033d6002cf9fc')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--tta', default=None, help='K5: NAME=A|AB[,NAME=A|AB] for EVERY member (staged file names)')
    a = ap.parse_args(argv)
    if a.kernel_only:
        return kernel_only(a)
    assert a.ckpt and a.store, '--ckpt and --store are required (except with --kernel_only)'
    specs = []
    for c in a.ckpt:
        p, _, k = c.rpartition('=') if '=' in os.path.basename(c) else (c, '', '')
        specs.append((p, k or None))
    out = a.out.replace('\\', '/')
    if os.path.exists(out) and os.listdir(out):
        assert a.overwrite, f'{out} exists and is not empty (pass --overwrite)'
        shutil.rmtree(out)
    t0 = time.time()
    rows = preflight(specs, a.store)
    want = parse_members(a.members, [r['staged'] for r in rows])        # K3: refuse a bad filter before any copy
    tta = parse_tta(a.tta, [r['staged'] for r in rows])                  # K5: refuse a bad map before any copy
    for r in rows:
        print(f'member {r["staged"]}: fold {r["fold"]} seed {r["seed"]} tag {r["tag"]} epoch {r["epoch_1based"]} '
              f'cv {r["cv"]} | train {r["train_labels"]} sel {r["sel_labels"]} | {r["model_mod"]} {r["backbone"]} '
              f'sha {r["model_sha"]} | {r["mb"]} MB', flush=True)
    os.makedirs(f'{out}/members')
    staged = []
    for r in rows:
        dst = f'{out}/members/{r["staged"]}'
        shutil.copy2(r['path'], dst); staged.append(dst)
    up = f'{out}/upload/{DS_TITLE}'
    cmd = [sys.executable, '-W', 'ignore', f'{CODE}/pack_fleet_ds.py', '--out', up, '--store', a.store,
           '--id', DS_ID, '--title', DS_TITLE] + (
        ['--tta', ','.join(tta[r['staged']] for r in rows)] if tta else []) + staged                    # K5
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.rstrip()); assert r.returncode == 0, r.stderr[-2000:]
    upinfo = verify_upload(up, rows, tta)
    print(f'upload folder OK: {up} ({upinfo["mb"]} MB, {len(upinfo["files"])} files, fleet_common {upinfo["fleet_common_sha16"]}, '
          f'worker {upinfo["worker_sha16"]})', flush=True)
    import make_k2_kernel as mk
    assert want is None or set(want) <= set(upinfo['manifest']), (want, upinfo['manifest'])
    common = ['--w_main', str(a.w_main), '--w_variants', a.w_variants, '--wall_h', str(a.wall_h),
              '--min_start_min', str(a.min_start_min), '--main_mode', a.main_mode] + \
             (['--members', ','.join(want)] if want else []) + \
             (['--tta', ','.join(f'{n}={m}' for n, m in tta.items())] if tta else [])          # K5b
    kinfo = [mk.main(['--out', f'{out}/kernel/{a.slug}', '--slug', a.slug] + common)]
    if a.combined:
        kinfo.append(mk.main(['--out', f'{out}/kernel/{a.slug_combined}', '--slug', a.slug_combined, '--combined',
                              '--c_w', str(a.c_w), '--cf_wc', str(a.cf_wc), '--cf_wf', str(a.cf_wf),
                              '--crop_min_start_min', str(a.crop_min_start_min), '--crop_cell_sha', a.crop_cell_sha] + common))
    summary = dict(built=time.strftime('%Y-%m-%d %H:%M:%S'), seconds=round(time.time() - t0, 1), members=rows,
                   upload=dict(path=up, **upinfo), kernels=kinfo)
    json.dump(summary, open(f'{out}/BUILD_SUMMARY.json', 'w'), indent=1, default=str)
    print_commands(up, kinfo, out)
    return summary


def print_commands(up, kinfo, out):
    print('\n# ---- by hand, in this order (nothing above was pushed) ----')
    print(f'kaggle datasets version -p "{up}" -m "<msg>"   # EXISTING dataset {DS_ID}: the new version REPLACES the whole '
          f'file set, so this folder must hold every file the kernels need (verified above)')
    print(f'kaggle datasets status {DS_ID}              # wait for "ready" before any push')
    for k in kinfo:
        print(f'kaggle kernels push -p "{k["out"]}"         # {k["id"]}: {len(k["files"])} files {k["files"]}')
        print(f'kaggle kernels status {k["id"]} ; kaggle kernels output {k["id"]} -p "{k["out"]}/out"')
    print(f'# summary: {out}/BUILD_SUMMARY.json')


def kernel_only(a):
    """One more kernel against an existing build: the upload folder is re-verified (code == k3/code, every member's
    bytes == its source checkpoint, manifest == the build's members) before the kernel is written."""
    out = a.out.replace(chr(92), '/')
    summ_p = f'{out}/BUILD_SUMMARY.json'
    assert os.path.isfile(summ_p), f'{summ_p} missing: run the full build first'
    summary = json.load(open(summ_p))
    rows = summary['members']
    up = summary['upload']['path']
    upinfo = verify_upload(up, rows, summary['upload'].get('tta'))      # K5
    kdir = f'{out}/kernel/{a.slug}'
    assert a.slug not in [k['slug'] for k in summary['kernels']], f'{a.slug} already built in {out}'
    assert not os.path.exists(kdir), f'{kdir} exists'
    want = parse_members(a.members, upinfo['manifest'])
    import make_k2_kernel as mk
    args = ['--out', kdir, '--slug', a.slug, '--w_main', str(a.w_main), '--w_variants', a.w_variants,
            '--wall_h', str(a.wall_h), '--min_start_min', str(a.min_start_min), '--main_mode', a.main_mode] + \
           (['--members', ','.join(want)] if want else []) + \
           (['--tta', ','.join(f'{n}={m}' for n, m in upinfo['tta'].items())] if upinfo.get('tta') else [])   # K5b
    k = mk.main(args)
    summary['kernels'].append(k)
    summary.setdefault('kernel_only_added', []).append(dict(slug=a.slug, at=time.strftime('%Y-%m-%d %H:%M:%S')))
    json.dump(summary, open(summ_p, 'w'), indent=1, default=str)
    print(f'kernel_only: {k["id"]} built against {up} (manifest {upinfo["manifest"]}, member filter {want})')
    print_commands(up, summary['kernels'], out)
    return summary


if __name__ == '__main__':
    main()
