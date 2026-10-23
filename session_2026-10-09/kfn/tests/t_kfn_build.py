#!/usr/bin/env python3
"""t_kfn_build.py (laptop, light CPU; kfn 2026-10-09, Claude): build + gate-logic tests that need no model.
  T1  FN N cell == make_fn_cell.make() == the K9N N cell + the FN-P patches only (every other line unchanged); compiles
  T2  kernels: FN = md | FN-IDS | fleet (K9-main cell 31 body byte for byte, metadata kept, K2_CFG = K9-main + RHO_MIN_N
      1e9) | N (K9N_CFG MODE FN, P/F/N 0/.40/.60 + embedded child == the K9N child); N-only = md | FN-IDS | N (MODE NONLY,
      0/0/1); metadata exactly competition + fleet ds (FN) + N weights ds, private, T4, GPU, no internet, no docker pin,
      no kernel / model source; no parent source NAME as a path anywhere (notebook text, decoded child); the fleet worker
      code in the live upload reads only the competition paths and its own FLEET_DIR
  T3  a SIMULATION K9 build is refused without --test; --test marks DO_NOT_PUSH and fn_check fails such a build
  T4  --compare: identical rebuild passes, a one-byte change fails
  T5  rank arithmetic (cell == fn_check == pandas, random + tie-heavy), fail-loud file names for FN and N-only
  T6  FN-IDS cell on a fake /kaggle tree: clean (3 files, placeholder 0.5 in test.csv order, RSNA_COMP_ROOT, T0) and 5
      failures (no test_series, duplicate ids, empty test.csv, sample_submission column order, sample_submission ids):
      none of the 3 files left (a stale submission.csv is removed too), no exception escapes the cell
  T7  N-cell gate logic with a FAKE child (FN on a synthetic strict fleet + the placeholder parent; N-only on the
      placeholder alone): clean paths pass fn_check (--skip_fleet for FN); negatives fail loud with submission.csv ABSENT:
      P weight != 0, NONLY triple != [0,0,1], RHO_REF wrong, placeholder missing, FN parent != placeholder, fleet not
      clean, NONLY with fleet files present, mirror == plain, ids, label order, fallback (n=3), csv formula, child weights
      sha, child rc 1; K9N mode (no MODE key) of the same cell still passes k9n_check on its K9N fixture
  T8  --expect_n_ranks log949 through fn_check (FN and N-only): author ranks PASS, one swapped label FAILS
  T9  Spearman gate at n = 60 (>= RHO_MIN_N): FN with F = N + noise passes, F with reversed columns FAILS (fleet~N);
      N-only passes (N~its own csv == 1)
  python tests/t_kfn_build.py
"""
import base64, glob, hashlib, json, os, re, shutil, sys, tempfile, time, traceback
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
KFN = os.path.dirname(HERE)
SP = os.path.dirname(KFN)
sys.path.insert(0, KFN); sys.path.insert(0, f'{KFN}/n949')
import make_child, make_fn_cell, build_kfn, fn_check, k9n_check, k2_check      # noqa: E402

T = k2_check.TARGETS
R = []
WDS = 'nartaa/rsna-knee-publication-swa-weights-20261007'


def ok(name, cond, detail=''):
    R.append((name, bool(cond), str(detail)[:300]))
    print(f'{"PASS" if cond else "FAIL"}  {name}  {"" if cond else str(detail)[:300]}', flush=True)


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def t1():
    src = make_fn_cell.make()
    ok('T1 cells/fn_n949_cell.py == make_fn_cell.make()', open(make_fn_cell.OUT, encoding='utf-8', newline='').read() == src)
    base = open(make_fn_cell.SRC, encoding='utf-8').read()
    ok('T1 the K9N base cell == k9n/cells/k9n_n949_cell.py (sha16 bd87efb8fa89bbac, the K9N_BUILD n_cell_sha16)',
       sha(base)[:16] == 'bd87efb8fa89bbac' and base == open(f'{SP}/k9n/cells/k9n_n949_cell.py', encoding='utf-8').read())
    # exact reverse: undo FN-P4 (the S2 block) then every FN-P patch in reverse order -> the K9N cell byte for byte
    blk = base[base.index(make_fn_cell.S2_START):base.index(make_fn_cell.S2_END) + len(make_fn_cell.S2_END)]
    new_blk = make_fn_cell.s2_new(blk)
    rev, good = src, src.count(new_blk) == 1
    rev = rev.replace(new_blk, blk)
    for name, old, new in reversed(make_fn_cell.P):
        good = good and rev.count(new) == 1
        rev = rev.replace(new, old)
    ok(f'T1 FN cell minus the {len(make_fn_cell.P) + 1} FN-P patches (each found exactly once) == the K9N cell byte for byte',
       good and rev == base)
    ind_ok = all(l == '' or l.startswith('    ') for l in blk[len(make_fn_cell.S2_START):].splitlines())
    ok('T1 FN-P4 only re-indents the K9N S2 block by 4 spaces under else (+ a NONLY branch before it)', ind_ok
       and all(('    ' + l if l.strip() else l) in new_blk for l in blk[len(make_fn_cell.S2_START):].splitlines()))


def t2(build):
    kb = json.load(open(f'{build}/FN_BUILD.json'))
    k9 = kb['k9_build']
    meta9 = json.load(open(f'{k9}/kernel/rsna-sub-k9-fleet/kernel-metadata.json'))
    nb9 = json.load(open(f'{k9}/kernel/rsna-sub-k9-fleet/{meta9["code_file"]}', encoding='utf-8'))
    c31 = nb9['cells'][31]
    h31, b31 = ''.join(c31['source']).split('\n', 1)
    cfg9 = eval(h31[len('K2_CFG = '):].split('   #', 1)[0])
    ids_src = open(build_kfn.IDS_CELL, encoding='utf-8').read()
    child = make_child.make()
    psl = build_kfn.parent_slugs(meta9)
    ok('T2 the parent-source list covers the K9-main parent (14 datasets + 2 kernels + 1 model)', len(psl) == 17, psl)
    for slug, mode in (('rsna-sub-k9fn', 'FN'), ('rsna-sub-nonly', 'NONLY')):
        kd = f'{build}/kernel/{slug}'
        m = json.load(open(f'{kd}/kernel-metadata.json'))
        nb = json.load(open(f'{kd}/{m["code_file"]}', encoding='utf-8'))
        cells = nb['cells']
        ncells = 4 if mode == 'FN' else 3
        ok(f'T2 {slug}: {ncells} cells (md, FN-IDS{", fleet" if mode == "FN" else ""}, N)', len(cells) == ncells
           and cells[0]['cell_type'] == 'markdown' and all(c['cell_type'] == 'code' for c in cells[1:]))
        ok(f'T2 {slug}: FN-IDS cell == cells/fn_ids_cell.py', ''.join(cells[1]['source']) == ids_src)
        if mode == 'FN':
            h, b = ''.join(cells[2]['source']).split('\n', 1)
            cfg = eval(h[len('K2_CFG = '):].split('   #', 1)[0])
            ok('T2 FN fleet cell body == K9-main cell 31 body byte for byte (K5 cell e37f0604997cd40f), metadata kept',
               b == b31 and sha(b)[:16] == 'e37f0604997cd40f' and cells[2]['metadata'] == c31['metadata'])
            want = dict(cfg9); want['RHO_MIN_N'] = 10 ** 9
            ok('T2 FN fleet K2_CFG == K9-main K2_CFG (strict, W .50, 10 members, TTA AB) + RHO_MIN_N 1e9 only', cfg == want
               and cfg['MAIN_MODE'] == 'strict' and cfg['W_MAIN'] == 0.5 and len(cfg['MEMBERS']) == 10
               and set(cfg['TTA'].values()) == {'AB'}, cfg)
        src = ''.join(cells[-1]['source'])
        l1, l2, body = src.split('\n', 2)
        ncfg = eval(l1[len('K9N_CFG = '):].split('   #', 1)[0])
        ch = base64.b64decode(l2.split("'")[1])
        w = [0.0, 0.4, 0.6] if mode == 'FN' else [0.0, 0.0, 1.0]
        ok(f'T2 {slug}: N cfg MODE {mode}, W3 {w}, RHO_REF {"F" if mode == "FN" else "NONE"}', ncfg['MODE'] == mode
           and ncfg['W3'] == w and ncfg['RHO_REF'] == ('F' if mode == 'FN' else 'NONE') and ncfg['TAG'] == mode
           and all(v[0] == 0.0 for v in ncfg['W3_VARIANTS']), ncfg)
        ok(f'T2 {slug}: embedded child == CHILD_SHA256 == make_child == the K9N kit child (8d567a63...)',
           sha(ch) == ncfg['CHILD_SHA256'] == sha(child) == json.load(open(f'{SP}/k9n/build/K9N_BUILD.json'))['child_sha256'])
        ok(f'T2 {slug}: N cell body == cells/fn_n949_cell.py; weights / source sha == make_child',
           body == make_fn_cell.make() and ncfg['WEIGHTS_SHA256'] == make_child.WEIGHTS_SHA256
           and ncfg['SRC_SHA256'] == make_child.SRC_SHA256 and ncfg['WEIGHTS_DS'] == WDS)
        ds = ['eeshsaxena/rsna-knee-cnxt-fleet', WDS] if mode == 'FN' else [WDS]
        ok(f'T2 {slug}: metadata = competition + {ds}; no kernel / model source; private, T4, GPU, no internet, no docker',
           m['dataset_sources'] == ds and m['competition_sources'] == ['rsna-knee-abnormality-detection']
           and m['kernel_sources'] == [] and m['model_sources'] == [] and m['is_private'] is True
           and m['machine_shape'] == 'NvidiaTeslaT4' and m['enable_gpu'] and not m['enable_internet']
           and 'docker_image' not in m and m['id'] == f'eeshsaxena/{slug}' and m['title'] == slug.replace('-', ' '), m)
        txt = open(f'{kd}/{m["code_file"]}', encoding='utf-8').read() + '\n' + ch.decode()
        hits = [p for p in psl if '/' + p.split('/')[1] in txt or p in txt]
        ok(f'T2 {slug}: no parent source referenced as a path / slug (notebook text + decoded child)', not hits, hits)
        paths = sorted(set(re.findall(r"/kaggle/input/[A-Za-z0-9_\-./{}\[\]\"']+", txt)))
        allowed = ('/kaggle/input/competitions/rsna-knee-abnormality-detection', '/kaggle/input/rsna-knee-abnormality-detection',
                   '/kaggle/input/datasets/eeshsaxena/rsna-knee-cnxt-fleet', '/kaggle/input/rsna-knee-cnxt-fleet',
                   '/kaggle/input/{K9N', '/kaggle/input/datasets/{K9N', "/kaggle/input'", '/kaggle/input"', '/kaggle/input/*/',
                   '/kaggle/input/raptor-knee-arms/', '/kaggle/input/raptor-cnn336/', '/kaggle/input/.../rsna-knee-cnxt-fleet')
        other = [p for p in paths if not p.startswith(allowed) and p != '/kaggle/input/competitions/']
        ok(f'T2 {slug}: every /kaggle/input path is the competition, our fleet, the N weights mount or the child\'s '
           f'unreachable verbatim weight fallbacks (K9N-P4 returns the pinned path first)', not other, other)
        try:
            import nbformat
            nbformat.validate(nbformat.read(f'{kd}/{m["code_file"]}', as_version=4)); nbv = True
        except Exception as e:
            nbv = repr(e)
        ok(f'T2 {slug}: nbformat v4 schema valid (nbformat.validate)', nbv is True, nbv)
        ok(f'T2 {slug}: notebook sha256 == FN_BUILD.json', sha(open(f'{kd}/{m["code_file"]}', 'rb').read()) == kb['kernels'][slug]['nb_sha256'])
    up = f'{k9}/upload/rsna-knee-cnxt-fleet'
    wtxt = ''.join(open(p, encoding='utf-8').read() for p in sorted(glob.glob(f'{up}/fleet_*.py')))
    wpaths = sorted(set(re.findall(r"/kaggle/input/[A-Za-z0-9_\-./]+", wtxt)))
    ok('T2 fleet worker code (live upload) reads only the competition paths under /kaggle/input',
       all(p.rstrip('/').endswith('rsna-knee-abnormality-detection') for p in wpaths) and wpaths, wpaths)


def t3(tmp):
    sim = f'{SP}/k9/build_sim'
    try:
        build_kfn.main(['--k9_build', sim, '--out', f'{tmp}/b_sim'])
        ok('T3 a SIMULATION K9 build is refused without --test', False)
    except AssertionError as e:
        ok('T3 a SIMULATION K9 build is refused without --test', 'not a FINAL K9 build' in str(e), e)
    info = build_kfn.main(['--k9_build', sim, '--out', f'{tmp}/b_sim', '--test'])
    ok('T3 --test build: DO_NOT_PUSH_TEST.txt + do_not_push', os.path.isfile(f'{tmp}/b_sim/DO_NOT_PUSH_TEST.txt') and info['do_not_push'])
    return f'{tmp}/b_sim'


def t4(build, tmp):
    k9 = json.load(open(f'{build}/FN_BUILD.json'))['k9_build']
    build_kfn.main(['--k9_build', k9, '--out', f'{tmp}/b_re', '--compare', build])
    ok('T4 rebuild from the same K9 build is byte-identical (--compare)', True)
    p = f'{tmp}/b_re/kernel/rsna-sub-nonly/kernel-metadata.json'
    open(p, 'a').write(' ')
    try:
        build_kfn.main(['--k9_build', k9, '--out', f'{tmp}/b_re2', '--compare', f'{tmp}/b_re'])
        ok('T4 --compare catches a one-byte change', False)
    except AssertionError as e:
        ok('T4 --compare catches a one-byte change', 'differs' in str(e), e)


def n_src(build, slug, work, inp, cfg_edit=None, child_src=None):
    kb = json.load(open(f'{build}/FN_BUILD.json'))
    kd = kb['kernels'][slug]
    nb = json.load(open(f'{kd["dir"]}/{kd["code_file"]}', encoding='utf-8'))
    src = ''.join(nb['cells'][-1]['source'])
    l1, l2, body = src.split('\n', 2)
    cfg = eval(l1[len('K9N_CFG = '):].split('   #', 1)[0])
    if cfg_edit:
        cfg.update(cfg_edit)
        for k, v in list(cfg.items()):
            if v is None:
                cfg.pop(k)
    if child_src is not None:
        l2 = f"_K9N_CHILD_B64 = '{base64.b64encode(child_src.encode()).decode()}'"
        cfg['CHILD_SHA256'] = sha(child_src)
    s = f'K9N_CFG = {cfg!r}\n{l2}\n{body}'.replace('/kaggle/working', work).replace('/kaggle/input', inp)
    return s, cfg, kb


def t5(build):
    tmp = tempfile.mkdtemp().replace(chr(92), '/')
    for slug, names in (('rsna-sub-k9fn', ['submission_n949.csv', 'submission_pfn_004060.csv', 'submission_pfn_003070.csv',
                                           'submission_pfn_005050.csv']),
                        ('rsna-sub-nonly', ['submission_n949.csv', 'submission_pfn_0000100.csv'])):
        s, _, _ = n_src(build, slug, f'{tmp}/w', f'{tmp}/i')
        g = {}
        exec(compile(s.replace('_k9n_receipt = _k9n_entry()', ''), 'ncell', 'exec'), g)
        ok(f'T5 {slug}: fail-loud set {names}', g['_n_loud_names']() == names, g['_n_loud_names']())
        rng = np.random.default_rng(0)
        for name, M in (('random', rng.uniform(0, 1, (57, 12))), ('tie-heavy', rng.integers(0, 4, (57, 12)).astype(float))):
            a1 = g['_n_rank'](M); a2 = fn_check.avg_rank_pct(M); a3 = pd.DataFrame(M).rank(method='average', pct=True).to_numpy()
            ok(f'T5 {slug} rank {name}: cell == fn_check == pandas', np.array_equal(a1, a2) and np.array_equal(a1, a3))
    shutil.rmtree(tmp, ignore_errors=True)


def ids_tree(tmp, case):
    """A fake /kaggle tree for the FN-IDS cell; returns (src, work, ids)."""
    inp, work = f'{tmp}/input', f'{tmp}/working'
    comp = f'{inp}/competitions/rsna-knee-abnormality-detection'
    os.makedirs(f'{comp}/test_series'); os.makedirs(work)
    ids = [f'1.2.826.0.1.{i}' for i in (5, 3, 9, 1)]
    t = ids[:]
    if case == 'dup':
        t = ids + [ids[0]]
    if case == 'empty':
        t = []
    pd.DataFrame({'StudyInstanceUID': t}).to_csv(f'{comp}/test.csv', index=False)
    pd.DataFrame({'StudyInstanceUID': ids, 'SeriesInstanceUID': ids}).to_csv(f'{comp}/test_series.csv', index=False)
    cols = list(T)
    if case == 'ss_cols':
        cols = cols[::-1]
    ss = pd.DataFrame(0.5, index=range(len(ids)), columns=cols)
    ss.insert(0, 'StudyInstanceUID', ids if case != 'ss_ids' else ids[:-1] + ['1.2.3.x'])
    ss.to_csv(f'{comp}/sample_submission.csv', index=False)
    if case == 'no_series':
        os.rmdir(f'{comp}/test_series')
    open(f'{work}/submission.csv', 'w').write('stale\n')                 # a stale file must never survive
    src = open(build_kfn.IDS_CELL, encoding='utf-8').read().replace('/kaggle/working', work).replace('/kaggle/input', inp)
    return src, work, ids, comp


def t6():
    for case in ('clean', 'no_series', 'dup', 'empty', 'ss_cols', 'ss_ids'):
        tmp = tempfile.mkdtemp(prefix='kfn_t6_').replace(chr(92), '/')
        src, work, ids, comp = ids_tree(tmp, case)
        old = os.environ.pop('RSNA_COMP_ROOT', None)
        g = {}
        try:
            exec(compile(src, 'fn_ids', 'exec'), g)
            esc = None
        except BaseException as e:
            esc = e
        files = [f for f in ('submission.csv', '_fn_ids.csv', '_fn_ids.json') if os.path.exists(f'{work}/{f}')]
        if case == 'clean':
            ph = pd.read_csv(f'{work}/_fn_ids.csv', dtype={'StudyInstanceUID': str})
            rj = json.load(open(f'{work}/_fn_ids.json'))
            ok('T6 FN-IDS clean: 3 files, placeholder = test.csv ids in order x 12 targets all 0.5, submission.csv == _fn_ids.csv, '
               'RSNA_COMP_ROOT + T0 set', esc is None and len(files) == 3 and ph['StudyInstanceUID'].tolist() == ids
               and ph.columns.tolist() == ['StudyInstanceUID', *T] and (ph[T].to_numpy() == 0.5).all()
               and open(f'{work}/submission.csv', 'rb').read() == open(f'{work}/_fn_ids.csv', 'rb').read()
               and os.environ.get('RSNA_COMP_ROOT') == comp and isinstance(g.get('T0'), float)
               and rj['n'] == 4 and rj['file_sha256'] == sha(open(f'{work}/_fn_ids.csv', 'rb').read()), (files, esc))
        else:
            ok(f'T6 FN-IDS {case}: fail loud, none of the 3 files (stale submission.csv removed), no exception escapes',
               esc is None and not files and g.get('_fn_ids_rec') is None, (files, esc))
        if old is not None:
            os.environ['RSNA_COMP_ROOT'] = old
        else:
            os.environ.pop('RSNA_COMP_ROOT', None)
        shutil.rmtree(tmp, ignore_errors=True)


FAKE_CHILD = r'''
import os, sys, json, numpy as np, pandas as pd
LAB = %(lab)r
mode = %(mode)r
w = os.environ['N949_WEIGHTS']; assert os.path.isfile(w)
ids = pd.read_csv(%(test)r, dtype=str)['StudyInstanceUID'].tolist()
n = len(ids); rng = np.random.default_rng(3)
pl = rng.uniform(0.05, 0.95, (n, 12)).astype(np.float32); mi = (pl + np.float32(0.01)).astype(np.float32)
if %(pfile)r: pl = np.load(%(pfile)r).astype(np.float32); mi = (pl + np.float32(0.01)).astype(np.float32)
if mode == 'mirror_equal': mi = pl.copy()
P = (np.float32(0.5) * (pl + mi)).astype(np.float32)
if mode == 'pq_mismatch': P[0, 0] += np.float32(1e-3)
lab = list(LAB)
if mode == 'label_order': lab = lab[::-1]
fb = [ids[0]] if mode == 'fallback' else []
if mode == 'ids': ids = ids[:-1] + ['1.2.3.fake']
sha = [%(wsha)r]
if mode == 'child_sha': sha = ['0' * 64]
x = np.clip(P, 0, 1); rk = x.argsort(0).argsort(0).astype(np.float64) / max(1, n - 1)
if mode == 'csv_formula': rk = rk[::-1]
sub = pd.DataFrame(rk.astype(np.float32), columns=LAB); sub.insert(0, 'StudyInstanceUID', ids)
sub.to_csv(os.environ['N949_OUT'], index=False)
meta = dict(n=n, views=n, fallback=fb, empty=[], weights_sha256=sha, torch='x', timm='y', ngpu=1, python='3', seconds=0.1,
            k_eval=94, input_res=320, img=384, maxs=96, fastread='study_lut', labels=lab)
np.savez(os.environ['N949_NPZ'], ids=np.array(ids), P=P, P_plain=pl, P_mirror=mi, ranks=rk, gpu_s=np.full(n, 0.1),
         meta=np.array(json.dumps(meta)))
if mode == 'rc1': sys.exit(1)
'''


def run_n(build, slug, mode='clean', cfg_edit=None, n=3, ids=None, fleet='clean', parent='placeholder', fleet_F=None,
          pfile=None, extra=None):
    """Run the N cell of SLUG on a temp tree with a fake child. Returns (tmp, work, st, cfg, kb, ids, escaped)."""
    tmp = tempfile.mkdtemp(prefix='kfn_t7_').replace(chr(92), '/')
    work, inp = f'{tmp}/w', f'{tmp}/i'
    os.makedirs(work); os.makedirs(f'{inp}/datasets/{WDS}')
    wf = f'{inp}/datasets/{WDS}/{make_child.WEIGHTS_FILE}'
    open(wf, 'wb').write(b'fake weights for the laptop gate test\n')
    wsha = sha(open(wf, 'rb').read())
    ids = ids or [f'1.2.826.0.{i}' for i in range(n)]
    pd.DataFrame({'StudyInstanceUID': ids}).to_csv(f'{tmp}/test.csv', index=False)
    child = FAKE_CHILD % dict(lab=T, mode=mode, test=f'{tmp}/test.csv', wsha=wsha, pfile=pfile)
    ce = {'WEIGHTS_SHA256': wsha}; ce.update(cfg_edit or {})
    s, cfg, kb = n_src(build, slug, work, inp, ce, child)
    # FN-IDS placeholder in test.csv order (+ submission.csv copy, as the FN-IDS cell writes it)
    ph = pd.DataFrame(np.full((len(ids), 12), 0.5), columns=T); ph.insert(0, 'StudyInstanceUID', ids)
    if parent != 'missing':
        ph.to_csv(f'{work}/_fn_ids.csv', index=False)
        shutil.copyfile(f'{work}/_fn_ids.csv', f'{work}/submission.csv')
        json.dump(dict(n=len(ids), root='t7', file_sha256=sha(open(f'{work}/_fn_ids.csv', 'rb').read())),
                  open(f'{work}/_fn_ids.json', 'w'))
    if slug == 'rsna-sub-k9fn' or extra == 'fleet_files':
        # the strict fleet stage's outputs on the placeholder parent, as the K5 cell writes them
        par_p = f'{work}/_k2_parent.csv'
        if parent == 'real':
            pr = ph.copy(); pr[T] = np.random.default_rng(9).uniform(0, 1, (len(ids), 12)); pr.to_csv(par_p, index=False)
        elif parent != 'missing':
            shutil.copyfile(f'{work}/_fn_ids.csv', par_p)
        rng = np.random.default_rng(1)
        F = fleet_F if fleet_F is not None else rng.uniform(0, 1, (len(ids), 12))
        files = {}
        fs = pd.DataFrame(F, columns=T); fs.insert(0, 'StudyInstanceUID', ids)
        fs.to_csv(f'{work}/submission_fleetsolo.csv', index=False)
        files['submission_fleetsolo.csv'] = sha(open(f'{work}/submission_fleetsolo.csv', 'rb').read())
        rk = lambda M: pd.DataFrame(np.asarray(M, np.float64)).rank(method='average', pct=True).to_numpy()
        for w in (0.5, 0.4, 0.6):
            d = pd.DataFrame(rk((1 - w) * rk(ph[T].to_numpy()) + w * rk(F)), columns=T); d.insert(0, 'StudyInstanceUID', ids)
            d.to_csv(f'{work}/{k2_check.wname(w)}', index=False)
            files[k2_check.wname(w)] = sha(open(f'{work}/{k2_check.wname(w)}', 'rb').read())
        if fleet == 'clean':
            shutil.copyfile(f'{work}/submission_w50.csv', f'{work}/submission.csv')
            files['submission.csv'] = files['submission_w50.csv']
        else:
            os.remove(f'{work}/submission.csv')
        json.dump(dict(clean=fleet == 'clean', why='t7 stub', files=files if fleet == 'clean' else {}, main_mode='strict',
                       stage_min=1.0), open(f'{work}/_k2_stage_fleet.json', 'w'))
    g = {'T0': time.time() - 1.0 * 3600}
    import torch
    _c = (torch.cuda.is_available, torch.cuda.device_count, torch.cuda.empty_cache)
    torch.cuda.is_available = lambda: True; torch.cuda.device_count = lambda: 1; torch.cuda.empty_cache = lambda: None
    esc = None
    try:
        exec(compile(s, 'ncell', 'exec'), g)
    except BaseException as e:
        esc = e
    finally:
        torch.cuda.is_available, torch.cuda.device_count, torch.cuda.empty_cache = _c
    st = json.load(open(f'{work}/_k9n_stage.json')) if os.path.isfile(f'{work}/_k9n_stage.json') else {}
    kb2 = dict(kb); kb2['weights_sha256'] = wsha; kb2['child_sha256'] = cfg['CHILD_SHA256']
    kb2['kernels'] = {k: dict(v) for k, v in kb['kernels'].items()}
    if cfg_edit and 'W3' in cfg_edit:
        kb2['kernels'][slug]['W3'] = cfg['W3']
    json.dump(kb2, open(f'{tmp}/kb.json', 'w'))
    return tmp, work, st, cfg, kb2, ids, esc


def fail_loud(work, st):
    return (st.get('clean') is False and not os.path.exists(f'{work}/submission.csv')
            and not any(os.path.exists(f) for f in glob.glob(f'{work}/submission_pfn_*.csv'))
            and not os.path.exists(f'{work}/submission_n949.csv'))


def t7(build):
    for slug, kern in (('rsna-sub-k9fn', 'fn'), ('rsna-sub-nonly', 'nonly')):
        tmp, work, st, cfg, kb2, ids, esc = run_n(build, slug)
        ok(f'T7 {kern} clean (fake child): stage clean, submission.csv written, no exception', st.get('clean') and esc is None
           and os.path.isfile(f'{work}/submission.csv'), (st.get('why'), esc))
        argv = ['--work', work, '--fn_build', f'{tmp}/kb.json', '--kernel', kern, '--expect_n', 'clean', '--json', f'{tmp}/c.json']
        if kern == 'fn':
            argv += ['--skip_fleet']
        rc = fn_check.main(argv)
        ok(f'T7 {kern} clean: fn_check PASS (every file == independent recompute, P leg removed == P at weight 0)', rc == 0,
           [c for c in json.load(open(f'{tmp}/c.json'))['checks'] if not c[1]][:3])
        shutil.rmtree(tmp, ignore_errors=True)
    cases = [
        ('fn', 'P weight != 0', dict(cfg_edit={'W3': [0.1, 0.3, 0.6]}), 'P weight 0'),
        ('fn', 'RHO_REF P in FN', dict(cfg_edit={'RHO_REF': 'P'}), 'RHO_REF'),
        ('fn', 'MODE unknown', dict(cfg_edit={'MODE': 'XYZ'}), 'unknown'),
        ('fn', 'placeholder missing', dict(parent='missing'), 'placeholder'),
        ('fn', 'parent != placeholder (a real parent)', dict(parent='real'), 'placeholder bytes'),
        ('fn', 'fleet stage not clean', dict(fleet='fail'), 'fleet stage NOT clean'),
        ('nonly', 'NONLY triple != [0,0,1]', dict(cfg_edit={'W3': [0.0, 0.4, 0.6]}), '[0, 0, 1]'),
        ('nonly', 'NONLY with fleet files present', dict(extra='fleet_files'), 'fleet stage output exists'),
        ('nonly', 'NONLY placeholder missing', dict(parent='missing'), 'placeholder'),
    ]
    for mode in ('pq_mismatch', 'mirror_equal', 'ids', 'label_order', 'fallback', 'csv_formula', 'child_sha', 'rc1'):
        cases.append(('fn', f'child {mode}', dict(mode=mode), ''))
        cases.append(('nonly', f'child {mode}', dict(mode=mode), ''))
    for kern, name, kw, why in cases:
        slug = 'rsna-sub-k9fn' if kern == 'fn' else 'rsna-sub-nonly'
        tmp, work, st, cfg, kb2, ids, esc = run_n(build, slug, **kw)
        ok(f'T7 {kern} NEG {name}: fail loud (submission.csv + N files ABSENT){", why names it" if why else ""}',
           esc is None and fail_loud(work, st) and (why in str(st.get('why'))), (st.get('why'), esc))
        shutil.rmtree(tmp, ignore_errors=True)
    # K9N mode of the same cell (no MODE key) on the K9N fixture: behaviour unchanged (k9n_check passes)
    tmp, work, st, cfg, kb2, ids, esc = run_n(build, 'rsna-sub-k9fn', cfg_edit={'MODE': None, 'RHO_REF': None, 'TAG': 'K9N-a',
                                                                             'W3': [0.1, 0.3, 0.6],
                                                                             'W3_VARIANTS': [[0.3, 0.3, 0.4], [0.4, 0.3, 0.3]]},
                                          parent='real')
    kk = json.load(open(f'{tmp}/kb.json'))
    kk['kernels'] = {'rsna-sub-k9n-a': {'W3': [0.1, 0.3, 0.6], 'W3_VARIANTS': [[0.3, 0.3, 0.4], [0.4, 0.3, 0.3]]}}
    json.dump(kk, open(f'{tmp}/kb9.json', 'w'))
    rc = k9n_check.main(['--work', work, '--k9n_build', f'{tmp}/kb9.json', '--tag', 'a', '--expect_n', 'clean', '--skip_fleet'])
    ok('T7 K9N mode of the FN cell (no MODE key, a real parent, P/F/N .10/.30/.60): clean and k9n_check PASS', st.get('clean')
       and rc == 0 and cfg.get('MODE') is None, st.get('why'))
    shutil.rmtree(tmp, ignore_errors=True)


def t8(build):
    ref = k9n_check.log949_ranks()
    ids = sorted(ref)
    for kern, slug in (('fn', 'rsna-sub-k9fn'), ('nonly', 'rsna-sub-nonly')):
        for flip in (False, True):
            tmp0 = tempfile.mkdtemp().replace(chr(92), '/')
            Rm = np.array([ref[s] for s in ids], np.float64)
            Pm = 0.2 + 0.6 * Rm - 0.005
            if flip:
                Pm[0, 3], Pm[1, 3] = Pm[1, 3], Pm[0, 3]
            np.save(f'{tmp0}/p.npy', Pm)
            tmp, work, st, cfg, kb2, _, esc = run_n(build, slug, ids=ids, pfile=f'{tmp0}/p.npy')
            argv = ['--work', work, '--fn_build', f'{tmp}/kb.json', '--kernel', kern, '--expect_n', 'clean',
                    '--expect_n_ranks', 'log949', '--json', f'{tmp}/c.json'] + (['--skip_fleet'] if kern == 'fn' else [])
            rc = fn_check.main(argv)
            line = [c for c in json.load(open(f'{tmp}/c.json'))['checks'] if 'public-study ranks' in c[0]][0]
            if flip:
                ok(f'T8 {kern}: one swapped label -> the log949 check FAILS naming the label + gap', rc == 1 and not line[1]
                   and 'Lateral Meniscus' in line[2] and 'gap' in line[2], line)
            else:
                ok(f'T8 {kern}: author ranks reproduced -> --expect_n_ranks log949 PASS', rc == 0 and line[1], line)
            shutil.rmtree(tmp, ignore_errors=True); shutil.rmtree(tmp0, ignore_errors=True)


def t9(build):
    n = 60
    ids = [f'1.2.826.0.9.{i}' for i in range(n)]
    tmp0 = tempfile.mkdtemp().replace(chr(92), '/')
    Pm = np.random.default_rng(4).uniform(0.05, 0.9, (n, 12))
    np.save(f'{tmp0}/p.npy', Pm)
    good_F = np.clip(Pm + np.random.default_rng(5).normal(0, 0.05, (n, 12)), 0, 1)
    for name, F, want in (('F = N + noise', good_F, True), ('F columns reversed (label-order bug)', good_F[:, ::-1].copy(), False)):
        tmp, work, st, cfg, kb2, _, esc = run_n(build, 'rsna-sub-k9fn', ids=ids, pfile=f'{tmp0}/p.npy', fleet_F=F)
        if want:
            rc = fn_check.main(['--work', work, '--fn_build', f'{tmp}/kb.json', '--kernel', 'fn', '--expect_n', 'clean', '--skip_fleet'])
            ok(f'T9 FN n={n} {name}: clean, fleet~N Spearman gate passes, fn_check PASS', st.get('clean') and rc == 0, st.get('why'))
        else:
            ok(f'T9 FN n={n} {name}: fleet~N Spearman gate FAILS loud', fail_loud(work, st) and 'fleet~N Spearman' in str(st.get('why')),
               st.get('why'))
        shutil.rmtree(tmp, ignore_errors=True)
    tmp, work, st, cfg, kb2, _, esc = run_n(build, 'rsna-sub-nonly', ids=ids, pfile=f'{tmp0}/p.npy')
    rec = json.load(open(f'{work}/n949_receipt.json'))
    rc = fn_check.main(['--work', work, '--fn_build', f'{tmp}/kb.json', '--kernel', 'nonly', '--expect_n', 'clean'])
    ok(f'T9 N-only n={n}: N~own-csv Spearman == 1 (gate on), fn_check PASS', st.get('clean') and rc == 0
       and abs(rec['rho_mean'] - 1) < 1e-9 and rec['rho_ref'] == 'N-own-csv', (st.get('why'), rec.get('rho_mean')))
    shutil.rmtree(tmp, ignore_errors=True); shutil.rmtree(tmp0, ignore_errors=True)


def main():
    t0 = time.time()
    build = f'{KFN}/build'
    tmp = tempfile.mkdtemp(prefix='kfn_tb_').replace(chr(92), '/')
    try:
        t1(); t2(build); sim = t3(tmp)
        rc = fn_check.main(['--work', tmp, '--fn_build', f'{sim}/FN_BUILD.json', '--kernel', 'nonly', '--skip_ids'])
        ok('T3 fn_check FAILS on a do_not_push (test) build', rc == 1)
        t4(build, tmp); t5(build); t6(); t7(build); t8(build); t9(build)
    except Exception:
        traceback.print_exc(); ok('no unexpected exception', False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(g for _, g, _ in R)
    print(f'TKFNBUILD {"PASS" if n_ok == len(R) else "FAIL"} {n_ok}/{len(R)} ({time.time() - t0:.0f} s)')
    return 0 if n_ok == len(R) else 1


if __name__ == '__main__':
    sys.exit(main())
