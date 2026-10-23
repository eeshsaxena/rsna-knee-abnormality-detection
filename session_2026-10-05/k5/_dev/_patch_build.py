import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
p = 'build_k2.py'
s = open(p, encoding='utf-8').read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''"""build_k2.py: from chosen fleet checkpoints (copied from the box) to (1) the upload folder of the NEW dataset
eeshsaxena/rsna-knee-cnxt-fleet and (2) the K2 kernel folder(s). Local only: nothing is pushed, created or versioned.
''', '''"""build_k2.py (K3 kit, scratchpad/k3): from chosen fleet checkpoints (copied from the box) to (1) the upload folder of
dataset eeshsaxena/rsna-knee-cnxt-fleet and (2) the kernel folder(s). Local only: nothing is pushed, created or versioned.

K3 changes (the K2 behaviour of every old flag is unchanged):
  - TTR members (ckpt args model_mod 'fleet_model_ttr', bigjump trainer): model_sha checked against the SHIPPED
    k3/code/fleet_model_ttr.py + fleet_model.py (f141aeed136dfc01), ttr_size buffer == args; fleet_model_ttr.py and
    the bigjump worker (W-TTR) ship in the upload folder; the bigjump pack tool writes the manifest (+ 'sources')
  - --main_mode strict (P+F at W_MAIN, submission.csv DELETED on any failure), --members NAME[,NAME] (member filter
    written into the kernel's K2_CFG; names must be in the manifest), --slug as before
  - --kernel_only: add ONE more kernel to an existing --out build (same upload folder, re-verified byte for byte), so
    two kernels with different modes / filters share one dataset version
''')
rep("SHIP = ['fleet_common.py', 'fleet_model.py', 'fleet_model_f2.py', 'fleet_worker.py']",
    "SHIP = ['fleet_common.py', 'fleet_model.py', 'fleet_model_f2.py', 'fleet_model_ttr.py', 'fleet_worker.py']\n"
    "MODS = ('fleet_model', 'fleet_model_f2', 'fleet_model_ttr')")
rep('''    if mod == 'fleet_model_f2':
        import fleet_model_f2 as f2
        src = chr(10).join(inspect.getsource(f2).splitlines()) + chr(10) + src
    return hashlib.sha256(src.encode()).hexdigest()[:16]''', '''    if mod == 'fleet_model_f2':
        import fleet_model_f2 as f2
        src = chr(10).join(inspect.getsource(f2).splitlines()) + chr(10) + src
    elif mod == 'fleet_model_ttr':                                       # == bigjump worker W-TTR
        import fleet_model_ttr as tt
        src = chr(10).join(inspect.getsource(tt).splitlines()) + chr(10) + src
    return hashlib.sha256(src.encode()).hexdigest()[:16]''')
rep('''        assert mod in ('fleet_model', 'fleet_model_f2'), (base, mod)
        ms = model_sha(mod)
        assert a.get('model_sha') in (None, ms), (base, f'{mod} source changed since training', a.get('model_sha'), ms)''',
    '''        assert mod in MODS, (base, 'unknown model_mod', mod)
        ms = model_sha(mod)
        assert a.get('model_sha') in (None, ms), (base, f'{mod} source changed since training', a.get('model_sha'), ms)
        if mod == 'fleet_model_ttr':                                     # W-TTR / TTR-6, as the worker asserts
            assert a.get('model_sha') == ms, (base, 'TTR member without model_sha', a.get('model_sha'))
            tsz = ck['model'].get('ttr_size_t')
            assert tsz is not None and int(tsz) == int(a['ttr_size']), (base, 'ttr_size buffer != args', tsz, a.get('ttr_size'))''')
rep('''                         sel_labels=a.get('sel_labels'), backbone=a.get('backbone'), model_mod=mod, model_sha=a.get('model_sha'),''',
    '''                         sel_labels=a.get('sel_labels'), backbone=a.get('backbone'), model_mod=mod, model_sha=a.get('model_sha'),
                         ttr_size=a.get('ttr_size'), sha16=sha(path),''')
rep('''    assert 'K2b' in open(f'{up}/fleet_worker.py', encoding='utf-8').read(), 'not the K2 (microbench) worker'
    man = json.load(open(f'{up}/fleet_manifest.json'))''', '''    wsrc = open(f'{up}/fleet_worker.py', encoding='utf-8').read()
    assert 'K2b' in wsrc, 'not the K2 (microbench) worker'
    assert 'W-TTR' in wsrc and "'fleet_model_ttr'" in wsrc, 'not the bigjump (W-TTR) worker'
    man = json.load(open(f'{up}/fleet_manifest.json'))''')
rep('''    return dict(files=files, mb=round(sum(os.path.getsize(f'{up}/{f}') for f in files) / 2**20, 1),
                fleet_common_sha16=sha(f'{up}/fleet_common.py'), worker_sha16=sha(f'{up}/fleet_worker.py'))''',
    '''    return dict(files=files, mb=round(sum(os.path.getsize(f'{up}/{f}') for f in files) / 2**20, 1),
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
    return want''')
rep('''    ap.add_argument('--ckpt', action='append', required=True, help='PATH or PATH=NEWKEY (repeatable)')
    ap.add_argument('--store', required=True, help='dir with <sid>.npz for every decode_probe study (from ~/fleet_store)')''',
    '''    ap.add_argument('--ckpt', action='append', default=None, help='PATH or PATH=NEWKEY (repeatable)')
    ap.add_argument('--store', default=None, help='dir with <sid>.npz for every decode_probe study (from ~/fleet_store)')''')
rep('''    ap.add_argument('--main_mode', choices=('blend', 'solo'), default='blend')''',
    '''    ap.add_argument('--main_mode', choices=('blend', 'solo', 'strict'), default='blend')
    ap.add_argument('--members', default=None, help='K3: comma list of manifest checkpoint FILE names to run (default all)')
    ap.add_argument('--kernel_only', action='store_true',
                    help='K3: build ONE more kernel (--slug, --main_mode, --w_main, --members) against the existing --out upload')''')
rep('''    a = ap.parse_args(argv)
    specs = []''', '''    a = ap.parse_args(argv)
    if a.kernel_only:
        return kernel_only(a)
    assert a.ckpt and a.store, '--ckpt and --store are required (except with --kernel_only)'
    specs = []''')
rep('''    import make_k2_kernel as mk
    common = ['--w_main', str(a.w_main), '--w_variants', a.w_variants, '--wall_h', str(a.wall_h),
              '--min_start_min', str(a.min_start_min), '--main_mode', a.main_mode]''',
    '''    import make_k2_kernel as mk
    want = parse_members(a.members, upinfo['manifest'])
    common = ['--w_main', str(a.w_main), '--w_variants', a.w_variants, '--wall_h', str(a.wall_h),
              '--min_start_min', str(a.min_start_min), '--main_mode', a.main_mode] + \\
             (['--members', ','.join(want)] if want else [])''')
rep('''    print('\\n# ---- by hand, in this order (nothing above was pushed) ----')
    print(f'kaggle datasets create -p "{up}"            # NEW dataset {DS_ID} (private); later: kaggle datasets version -p ... -m "..."')
    print(f'kaggle datasets status {DS_ID}              # wait for "ready" before any push')''',
    '''    print_commands(up, kinfo, out)
    return summary


def print_commands(up, kinfo, out):
    print('\\n# ---- by hand, in this order (nothing above was pushed) ----')
    print(f'kaggle datasets version -p "{up}" -m "<msg>"   # EXISTING dataset {DS_ID}: the new version REPLACES the whole '
          f'file set, so this folder must hold every file the kernels need (verified above)')
    print(f'kaggle datasets status {DS_ID}              # wait for "ready" before any push')''')
rep('''        print(f'kaggle kernels status {k["id"]} ; kaggle kernels output {k["id"]} -p "{k["out"]}/out"')
    print(f'# summary: {out}/BUILD_SUMMARY.json')
    return summary
''', '''        print(f'kaggle kernels status {k["id"]} ; kaggle kernels output {k["id"]} -p "{k["out"]}/out"')
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
    upinfo = verify_upload(up, rows)
    kdir = f'{out}/kernel/{a.slug}'
    assert a.slug not in [k['slug'] for k in summary['kernels']], f'{a.slug} already built in {out}'
    assert not os.path.exists(kdir), f'{kdir} exists'
    want = parse_members(a.members, upinfo['manifest'])
    import make_k2_kernel as mk
    args = ['--out', kdir, '--slug', a.slug, '--w_main', str(a.w_main), '--w_variants', a.w_variants,
            '--wall_h', str(a.wall_h), '--min_start_min', str(a.min_start_min), '--main_mode', a.main_mode] + \\
           (['--members', ','.join(want)] if want else [])
    k = mk.main(args)
    summary['kernels'].append(k)
    summary.setdefault('kernel_only_added', []).append(dict(slug=a.slug, at=time.strftime('%Y-%m-%d %H:%M:%S')))
    json.dump(summary, open(summ_p, 'w'), indent=1, default=str)
    print(f'kernel_only: {k["id"]} built against {up} (manifest {upinfo["manifest"]}, member filter {want})')
    print_commands(up, summary['kernels'], out)
    return summary
''')
open(p, 'w', encoding='utf-8', newline='').write(s)
print('patched')
