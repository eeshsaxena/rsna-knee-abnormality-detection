# test_ao_1015.py (session_2026-10-15/v224f/tests, 10-15, Claude subagent): builder tests for the AO bases of build_v224fs.py
# (--base flip | flrm; RESULT.md "AO 10-15"). Laptop CPU, offline, no Kaggle, no box. Every build runs the builder as a subprocess
# into a temp dir (--s-manifest = the stand-in S as a manifest file, exactly test_fix1015's), except the stand-in regression block.
#   python tests/test_ao_1015.py
# A  refusals: AO + --flip-budget-h, AO + an FF start budget, AO + an unreviewed guard, AO + w_f .25, no manifest / stand-in.
# B  the main-session builds (--base flip / flrm --s-manifest): pushable kernel dir, metadata, BUILD json, and proofs made HERE,
#    independent of the builder: base cells == the Kaggle-pulled base notebook (cell dicts) and the file minus the appended cells ==
#    the base file bytes; ENV-SNAP / FN-IDS / K12-MERGE / CLOCK / ENV-RESTORE / fleet body == the commit-tested V224-F20 kernel;
#    K2 literal == V224-F20's but MEMBERS / TTA / MIN_START_MIN; S-MERGE body == the FFSR stand-in's; HF-AO == the FFSR stand-in HF-S
#    with exactly 2 edits; SAVE-AO == the FFSR SAVE but header / tag / budget knobs / ao_base; no time-aware driver.
# C  guard 6.5 build (the default before fix aofix 10-15 F1): MIN_START_MIN 108 in K2 + SAVE; guard 6.0: 138.
#    (fix aofix 10-15 F1: the default AO guard is 7.0 h / MIN_START_MIN 78, checked in B; was 6.5 / 108 with 7.0 tested in C)
# D  f / ff unchanged: the 4 stand-ins rebuilt == the pre-change shas (all 20 files); the --s-manifest ff and f builds of the new
#    builder == those of build_v224fs.py.pre_ao_1015, byte for byte (notebook, metadata, BUILD json); the pre-change builder refuses
#    --base flip (argparse).
import os, sys, json, glob, shutil, subprocess, tempfile, re, hashlib, difflib

T = os.path.dirname(os.path.abspath(__file__)); V = os.path.dirname(T); ROOT = os.path.normpath(os.path.join(V, '..', '..'))
BUILD = os.path.join(V, 'build_v224fs.py'); BUILD_PRE = BUILD + '.pre_ao_1015'
F20 = os.path.join(V, 'kernel_v224f20', 'rsna_sub_v224f20.ipynb')
FFSR = os.path.join(T, 'standin_v224ffsr20', 'rsna_sub_v224ffsr20.ipynb')
BASES = {'flip': (os.path.join(ROOT, 'session_2026-10-15', 'v224rm', 'pulled_v224flip', 'rsna-sub-v224flip.ipynb'),
                  os.path.join(ROOT, 'session_2026-10-14', 'v224', 'kernel_v224flip', 'rsna_sub_v224flip.ipynb'), 15),
         'flrm': (os.path.join(ROOT, 'session_2026-10-15', 'v224rm', 'pulled_v224flrm', 'rsna-sub-v224flrm.ipynb'),
                  os.path.join(ROOT, 'session_2026-10-15', 'v224rm', 'kernel_v224flrm', 'rsna_sub_v224flrm.ipynb'), 17)}
STANDIN_SHA = {   # tests/standin_* before the AO change (MEASURED 10-15, RESULT.md "AO 10-15")
    'standin_v224fs20/BUILD_V224FS.json': '2595d095aa182ffb7f8fe9c795a66fd0c072011112fc6a8bdd2d2fe8e91e68b7',
    'standin_v224fs20/DO_NOT_PUSH.txt': '30ead26d45e22d0abb40e23dd7ed20898f1e7a8a5da2e560f2e6be3401776600',
    'standin_v224fs20/celldiff_v224fs20.txt': 'a7e13ca04477295d0fa18dbed09f84e776e0b03402c1badf48b108b22c966508',
    'standin_v224fs20/kernel-metadata.STANDIN_NOT_PUSHABLE.json': 'c6d04773e80cf0ee68fb8fba799e0817be12510754bceea67d22d746444b0abc',
    'standin_v224fs20/rsna_sub_v224fs20.ipynb': 'a6055779f776551f4137ab3ba08bad22e558d20e215c04a5fdcbea2af74c0009',
    'standin_v224fsr20/BUILD_V224FS.json': '2acb0e0f36f4cfdd4df9b625126b1469274c74b183b4de91ed78211a7ec8b071',
    'standin_v224fsr20/DO_NOT_PUSH.txt': '30ead26d45e22d0abb40e23dd7ed20898f1e7a8a5da2e560f2e6be3401776600',
    'standin_v224fsr20/celldiff_v224fsr20.txt': '5be732611b8373134d364127ee204d00204dead059687ed82cd97cbfe7db0628',
    'standin_v224fsr20/kernel-metadata.STANDIN_NOT_PUSHABLE.json': '40b0e01eded1838a340674c398b758570c0190b78d90b9edf5c5080c06b64186',
    'standin_v224fsr20/rsna_sub_v224fsr20.ipynb': 'acb97bf821d9e0fd31c1f827ecfa3221f3da78e5c7e1d63fda8553cae3eb0912',
    'standin_v224ffs20/BUILD_V224FS.json': '5b90a34002796035be5a1481539b27874beedb671ebf9c34d1add81fcc7fb7be',
    'standin_v224ffs20/DO_NOT_PUSH.txt': '30ead26d45e22d0abb40e23dd7ed20898f1e7a8a5da2e560f2e6be3401776600',
    'standin_v224ffs20/celldiff_v224ffs20.txt': 'b8659e3e1062b75ecc6670af281c88deffee428ace4e5c5e12b6c004d82fb643',
    'standin_v224ffs20/kernel-metadata.STANDIN_NOT_PUSHABLE.json': '9f8463d5fb86b74595f98f0a4ed8cb8efdc6bd68fe966822a66d73b4e750881f',
    'standin_v224ffs20/rsna_sub_v224ffs20.ipynb': '190d0c70321fbd7fb72667d17599c38e0bdffbd3c7502a8b32e9078438e80e95',
    'standin_v224ffsr20/BUILD_V224FS.json': 'ddadd6759163b21e10d176c05b57c03f9e20a32ead9c524f3fc913e84d698f70',
    'standin_v224ffsr20/DO_NOT_PUSH.txt': '30ead26d45e22d0abb40e23dd7ed20898f1e7a8a5da2e560f2e6be3401776600',
    'standin_v224ffsr20/celldiff_v224ffsr20.txt': 'c2c2a0a6a60517eb380060f2b4410966f1f8d6570cd8196bb7d9130b8a463bee',
    'standin_v224ffsr20/kernel-metadata.STANDIN_NOT_PUSHABLE.json': 'c583d6db8b93c846e9df0b62097d6a264c17668939f49c5afe733e743d5fbff4',
    'standin_v224ffsr20/rsna_sub_v224ffsr20.ipynb': '4f4c0cf7f9a05cb87a6fcdf714de90bf7ece1c4b967d6160a37f75e2c12db215'}
DEF_G, DEF_M = 7.0, 78.0          # the builder's default AO start guard (fix aofix 10-15 F1; was 6.5 h / 108)
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
dumps = lambda nb: json.dumps(nb, ensure_ascii=False, separators=(',', ':'))
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print(f'  {"OK  " if ok else "FAIL"} {name} {info}', flush=True)


def run(args):
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True, encoding='utf-8')
    return r.returncode, r.stdout, r.stderr


TMP = tempfile.mkdtemp(prefix='ao1015_')
FNB = json.load(open(F20, encoding='utf-8')); FC = [src(c) for c in FNB['cells']]
k = [c for c in FC if c.startswith('K12M_CFG = {')][0].split('\n', 1)[0]
K12M = eval(k[len('K12M_CFG = '):k.index('   # literal')], {})
add = K12M['add_manifest']
smap = {n: re.sub(r'^f(\d)_s1043f75oai_', r'f\1_s1047r7tb_', n) for n in add['checkpoints']}
MAN = os.path.join(TMP, 'r7tb_manifest.json')
json.dump({'checkpoints': [smap[n] for n in add['checkpoints']],
           'sources': {smap[n]: dict(add['sources'][n], original=smap[n]) for n in add['checkpoints']}}, open(MAN, 'w'), indent=1)
print(f'TEST_AO_1015 | tmp {TMP}', flush=True)


def build(tag, extra, builder=BUILD, manifest=True):
    out = os.path.join(TMP, tag)
    os.makedirs(out)
    rc, so, se = run([builder, '--wf', '0.20', '--out', out] + (['--s-manifest', MAN] if manifest else []) + extra)
    kd = glob.glob(os.path.join(out, 'kernel_*'))
    return rc, so, se, out, kd


# ---- A refusals -----------------------------------------------------------------------------------------------------------------
for nm, extra, man in (('flip + --flip-budget-h 4.9', ['--base', 'flip', '--flip-budget-h', '4.9'], True),
                       ('flip + FF start budget 5.1', ['--base', 'flip', '--start-budget-h', '5.1'], True),
                       ('flrm + unreviewed guard 7.5', ['--base', 'flrm', '--start-budget-h', '7.5'], True),
                       ('flip + w_f .25', ['--base', 'flip', '--wf', '0.25'], True),
                       ('flrm without --s-manifest / --standin', ['--base', 'flrm'], False)):
    rc, so, se, out, kd = build('ref_' + re.sub(r'\W+', '_', nm), extra, manifest=man)
    chk('A refused: ' + nm, rc != 0 and not kd and 'AssertionError' in se, (rc, se.strip().splitlines()[-1][:120] if se.strip() else ''))

# ---- B the main-session AO builds ----------------------------------------------------------------------------------------------
FF = json.load(open(FFSR, encoding='utf-8')); FFC = [src(c) for c in FF['cells']]
HF_FF = [c for c in FFC if c.startswith('# [eeshsaxena V224-F-S HF')][0]
SAVE_FF = [c for c in FFC if c.startswith('# [eeshsaxena V224-F-S SAVE')][0]
SM_FF = [c for c in FFC if c.startswith('S_CFG = {')][0]
for base in ('flip', 'flrm'):
    pulled, local, n0 = BASES[base]
    tag = 'v224%ss20' % base
    rc, so, se, out, kd = build('B_' + base, ['--base', base])
    ok = rc == 0 and [os.path.basename(d) for d in kd] == ['kernel_' + tag]
    chk('B %s: the main-session build succeeds -> kernel_%s' % (base, tag), ok, (rc, se.strip()[-300:]))
    if not ok:
        continue
    K = kd[0]
    files = sorted(os.listdir(K))
    chk('B %s: pushable kernel dir = notebook + kernel-metadata.json + BUILD json + celldiff (no stand-in marker)' % base,
        files == sorted(['rsna_sub_%s.ipynb' % tag, 'kernel-metadata.json', 'BUILD_V224FS.json', 'celldiff_%s.txt' % tag]), files)
    meta = json.load(open(os.path.join(K, 'kernel-metadata.json'))); bmeta = json.load(open(os.path.join(os.path.dirname(local), 'kernel-metadata.json')))
    chk('B %s: metadata = the base kernel\'s (docker 2757e0c7, T4, internet off, private) + k11w, k14w, r7tb; id rsna-sub-%s' % (base, tag),
        meta['id'] == 'eeshsaxena/rsna-sub-' + tag and meta['code_file'] == 'rsna_sub_%s.ipynb' % tag and meta['is_private'] is True
        and meta['enable_internet'] is False and meta['enable_gpu'] is True and meta['docker_image'] == bmeta['docker_image']
        and '2757e0c7' in meta['docker_image'] and meta['machine_shape'] == 'NvidiaTeslaT4'
        and meta['dataset_sources'] == bmeta['dataset_sources'] + ['eeshsaxena/rsna-knee-cnxt-fleet-k11w', 'eeshsaxena/rsna-knee-cnxt-fleet-k14w',
                                                                   'eeshsaxena/rsna-knee-cnxt-fleet-r7tb'], meta['dataset_sources'])
    Bj = json.load(open(os.path.join(K, 'BUILD_V224FS.json')))
    chk('B %s: BUILD json: base, REPLACE, 5 S, s_share 1, w_f .2, guard %.1f / MIN_START_MIN %.0f (default), flip budget None, not stand-in'
        % (base, DEF_G, DEF_M),
        Bj['base'] == base and Bj['tag'] == tag and Bj['replace'] is True and Bj['k14'] == [] and len(Bj['s']) == 5 and Bj['s_share'] == 1.0
        and Bj['w_f'] == 0.2 and Bj['start_budget_h'] == DEF_G and Bj['min_start_min'] == DEF_M and Bj['flip_budget_h'] is None
        and Bj['standin'] is False and Bj['base_sha256'] == sha(local) and Bj['notebook_sha256'] == sha(os.path.join(K, 'rsna_sub_%s.ipynb' % tag)),
        {k_: Bj.get(k_) for k_ in ('base', 'tag', 'start_budget_h', 'min_start_min')})
    nb = json.load(open(os.path.join(K, 'rsna_sub_%s.ipynb' % tag), encoding='utf-8')); C = [src(c) for c in nb['cells']]
    pb = json.load(open(pulled, encoding='utf-8'))
    chk('B %s: cells 0-%d == the Kaggle-pulled base notebook\'s cells (type, source, metadata, outputs)' % (base, n0 - 1),
        len(nb['cells']) == n0 + 9 and nb['cells'][:n0] == pb['cells'] and len(pb['cells']) == n0)
    bk = dict(nb); bk['cells'] = nb['cells'][:n0]
    chk('B %s: the notebook minus cells %d-%d re-serialises to the base FILE byte for byte' % (base, n0, n0 + 8),
        dumps(bk).encode('utf-8') == open(local, 'rb').read() == open(pulled, 'rb').read())
    ap = C[n0:]
    chk('B %s: ENV-SNAP / FN-IDS / K12-MERGE / CLOCK / ENV-RESTORE == the commit-tested V224-F20 kernel cells 15-18 / 20' % base,
        [ap[1], ap[2], ap[3], ap[5], ap[7]] == [FC[15], FC[16], FC[17], FC[18], FC[20]])
    k2n, k2o = ap[6].split('\n', 1), FC[19].split('\n', 1)
    cn = eval(k2n[0][len('K2_CFG = '):k2n[0].index('   # literal')], {}); co = eval(k2o[0][len('K2_CFG = '):k2o[0].index('   # literal')], {})
    chk('B %s: fleet body == V224-F20 kernel cell 19 body; K2 literal == V224-F20\'s except MEMBERS / TTA (5 S) and MIN_START_MIN 168 -> %.0f'
        % (base, DEF_M),
        k2n[1] == k2o[1] and [x for x in co if co[x] != cn[x]] == ['MIN_START_MIN', 'MEMBERS', 'TTA'] and list(cn) == list(co)
        and cn['MEMBERS'] == Bj['s'] and cn['MIN_START_MIN'] == DEF_M and co['MIN_START_MIN'] == 168.0, [x for x in co if co[x] != cn[x]])
    s_n = eval(ap[4].split('\n', 1)[0][len('S_CFG = '):ap[4].split('\n', 1)[0].index('   # literal')], {})
    s_f = eval(SM_FF.split('\n', 1)[0][len('S_CFG = '):SM_FF.split('\n', 1)[0].index('   # literal')], {})
    for n_ in s_f['s_manifest']['sources'].values():                 # the stand-in manifest also records standin_of
        n_.pop('standin_of', None)
    chk('B %s: S-MERGE body == the FFSR stand-in S-MERGE body; S_CFG literal == the stand-in\'s (same S manifest, minus standin_of)' % base,
        ap[4].split('\n', 1)[1] == SM_FF.split('\n', 1)[1] and s_n == s_f, [k_ for k_ in s_n if s_n[k_] != s_f.get(k_)])
    hf_a = ap[8].replace("HF_TAG = '%s'" % tag, "HF_TAG = 'v224ffsr20'")
    ops = [o for o in difflib.SequenceMatcher(None, HF_FF.splitlines(), hf_a.splitlines(), autojunk=False).get_opcodes() if o[0] != 'equal']
    chg = [(o[0], HF_FF.splitlines()[o[1]:o[2]], hf_a.splitlines()[o[3]:o[4]]) for o in ops]
    chk('B %s: HF-AO == the FFSR stand-in HF-S except exactly 2 edits: the header line and the _hf_blank line -> the AO rule (+ comments, '
        'fired_before_hf)' % base, len(chg) == 2 and chg[0][1][0].startswith('# [eeshsaxena V224-F-S HF BLEND') and chg[0][2][0].startswith('# [eeshsaxena V224-AO-S HF BLEND')
        and chg[1][1] == ["_hf_blank = ['Fracture'] if 'r224->Fracture' in (globals().get('_canary') or {}).get('fired', []) else []"]
        and chg[1][2][-1] == "_hf_blank = ['Fracture'] if any(str(_x).endswith('->Fracture') for _x in _hf_fired) else []"
        and any("globals().get('_gcr')" in ln for ln in chg[1][2]), [(c[0], len(c[1]), len(c[2])) for c in chg])
    ops = [o for o in difflib.SequenceMatcher(None, SAVE_FF.splitlines(), ap[0].splitlines(), autojunk=False).get_opcodes() if o[0] != 'equal']
    new_lines = [ln for o in ops for ln in ap[0].splitlines()[o[3]:o[4]]]
    chk('B %s: SAVE-AO == the FFSR SAVE except the header (names the base), tag, start_budget_h %.1f / min_start_min %.0f and ao_base'
        % (base, DEF_G, DEF_M),
        len(new_lines) == 4 and 'V224-AO-S SAVE' in new_lines[0] and ('V224-FLIP-RM' if base == 'flrm' else 'V224-FLIP (') in new_lines[1]
        and "'tag': '%s'" % tag in new_lines[2] and "'start_budget_h': %r" % DEF_G in new_lines[2] and "'min_start_min': %r" % DEF_M in new_lines[2]
        and "'ao_base': '%s'" % base in new_lines[3] and "'s_share': 1.0" in new_lines[3], new_lines)
    bad = [m for m in ('compact_infer_ta', 'FLIP_TA_BUDGET_H', 'V224FF_FLIP_BUDGET_H', "'--ta-") if any(m in c for c in C)]
    chk('B %s: no time-aware flip driver anywhere; the V224-FLIP cell (FLIP_TTA_FORCED_ON) is present once' % base,
        not bad and sum('FLIP_TTA_FORCED_ON = True' in c for c in C) == 1, bad)

# ---- C other reviewed guards -----------------------------------------------------------------------------------------------------
for g, ms in ((6.5, 108.0), (6.0, 138.0)):
    rc, so, se, out, kd = build('C_%s' % g, ['--base', 'flip', '--start-budget-h', repr(g)])
    ok = rc == 0 and len(kd) == 1
    if ok:
        nb = json.load(open(glob.glob(os.path.join(kd[0], '*.ipynb'))[0], encoding='utf-8')); C = [src(c) for c in nb['cells']]
        k2 = C[15 + 6].split('\n', 1)[0]; cn = eval(k2[len('K2_CFG = '):k2.index('   # literal')], {})
        ok = cn['MIN_START_MIN'] == ms and ("'start_budget_h': %r, 'wall_h': 8.3, 'min_start_min': %r" % (g, ms)) in C[15]
    chk('C flip guard %.1f h builds with MIN_START_MIN %.0f in K2 and SAVE' % (g, ms), ok, (rc, se.strip()[-200:]))

# ---- D f / ff unchanged ---------------------------------------------------------------------------------------------------------
for extra in (['--base', 'f'], ['--base', 'f', '--replace'], ['--base', 'ff'], ['--base', 'ff', '--replace']):
    rc, so, se = run([BUILD, '--wf', '0.20', '--standin'] + extra)
    chk('D stand-in rebuild %s' % ' '.join(extra), rc == 0, se.strip()[-200:])
cur = {p: sha(os.path.join(T, p)) for p in STANDIN_SHA}
chk('D the 4 f / ff stand-ins (20 files) == their pre-change shas', cur == STANDIN_SHA, [p for p in STANDIN_SHA if cur[p] != STANDIN_SHA[p]])
for extra in (['--base', 'ff'], ['--base', 'ff', '--replace'], ['--base', 'f'], ['--base', 'ff', '--start-budget-h', '5.0', '--flip-budget-h', '4.4']):
    nm = re.sub(r'\W+', '_', ' '.join(extra))
    rcn, _, sen, on, kn = build('Dn_' + nm, extra)
    rco, _, seo, oo, ko = build('Do_' + nm, extra, builder=BUILD_PRE)
    same = rcn == rco == 0 and len(kn) == len(ko) == 1 and sorted(os.listdir(kn[0])) == sorted(os.listdir(ko[0])) and all(
        open(os.path.join(kn[0], f), 'rb').read() == open(os.path.join(ko[0], f), 'rb').read() for f in os.listdir(kn[0]))
    chk('D --s-manifest %s: new builder == build_v224fs.py.pre_ao_1015, every file byte for byte' % ' '.join(extra), same,
        (rcn, rco, sorted(os.listdir(kn[0])) if kn else None))
rc, so, se, out, kd = build('D_pre_flip', ['--base', 'flip'], builder=BUILD_PRE)
chk('D the pre-change builder refuses --base flip (argparse choices f / ff)', rc == 2 and not kd and 'invalid choice' in se, se.strip()[-120:])
shutil.rmtree(TMP, ignore_errors=True)
print(f'TEST_AO_1015 {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}', flush=True)
sys.exit(0 if all(R) else 1)
