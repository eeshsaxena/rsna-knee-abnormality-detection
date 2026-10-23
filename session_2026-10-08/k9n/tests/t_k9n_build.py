#!/usr/bin/env python3
"""t_k9n_build.py (laptop, light CPU; K9N 2026-10-08): build + gate-logic tests that need no model.
  T1  child == make_child.make(); 11 patches each matched once; child compiles; pinned public-source sha
  T2  K9N kernels: cells 0-31 == K9-main byte for byte, cell 32 = K9N_CFG + embedded child (sha) + the N cell body,
      metadata = K9-main + the weights dataset, private, T4, no internet, no docker pin; W3 a/b + variants
  T3  build refusals: a SIMULATION K9 build is refused without --test; --test writes DO_NOT_PUSH_TEST.txt and
      K9N_BUILD.json do_not_push, and k9n_check flags such a build
  T4  --compare: identical rebuild passes, a one-byte change fails
  T5  rank arithmetic: the cell's _n_rank == k9n_check.avg_rank_pct == pandas average pct rank (random + tie-heavy),
      file names (_n_tname) and the fail-loud set
  T7  --expect_n_ranks log949 (commit parity gate): exact author ranks PASS, one swapped label FAILS with the gap
  T6  N-cell gate logic with a FAKE child (tiny python script, test-only CHILD_SHA256 / WEIGHTS_SHA256 edits, synthetic
      fleet outputs, laptop temp dir): clean path writes every file and k9n_check --skip_fleet passes on it; then each
      corruption must fail loud with submission.csv ABSENT: P != (plain+mirror)/2, mirror == plain, ids mismatch, label
      order, fallback on a 3-study run, child csv != formula, child weights sha wrong, child rc 1
  python tests/t_k9n_build.py
"""
import base64, hashlib, json, os, shutil, subprocess, sys, tempfile, time, traceback
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
K9N = os.path.dirname(HERE)
SP = os.path.dirname(K9N)
sys.path.insert(0, K9N); sys.path.insert(0, f'{K9N}/n949')
import make_child, build_k9n, k9n_check, k2_check                     # noqa: E402

PY = sys.executable
T = k2_check.TARGETS
R = []


def ok(name, cond, detail=''):
    R.append((name, bool(cond), str(detail)[:300]))
    print(f'{"PASS" if cond else "FAIL"}  {name}  {"" if cond else str(detail)[:300]}', flush=True)


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def t1():
    src = make_child.make()
    ok('T1 child on disk == make_child.make()', open(make_child.OUT, encoding='utf-8', newline='').read() == src)
    ok('T1 11 K9N-P patches, each anchor found exactly once in the public cell 4', len(make_child.P) == 11)
    nb = json.load(open(make_child.NB, encoding='utf-8'))
    ok('T1 public cell 4 sha256 == the author-stated source sha (bcd864cb...)',
       sha(''.join(nb['cells'][4]['source'])) == make_child.SRC_SHA256 == 'bcd864cbf349882e876959b49cb348cbb89b4c62fc204d4e4261049c5e3a79c5')
    # outside the patched lines the child equals the public source
    pub = ''.join(nb['cells'][4]['source']).splitlines()
    ch = src.splitlines()
    import difflib
    added = [l for l in difflib.ndiff(pub, ch) if l.startswith('+ ')]
    removed = [l for l in difflib.ndiff(pub, ch) if l.startswith('- ')]
    ok('T1 child removes only the 2 replaced public lines (fastread path, output path)', len(removed) == 2, removed)
    ok('T1 every added line is a K9N comment / marked K9N-P line or inside _k9n_dump',
       all(('K9N' in l or l.strip() in ('+', '') or l.startswith('+ #') or l.startswith('+     ') or l.startswith('+ def _k9n_dump')
            or l.startswith('+ \n')) for l in added), [l for l in added if 'K9N' not in l][:3])


def t2(build):
    kb = json.load(open(f'{build}/K9N_BUILD.json'))
    k9 = kb['k9_build']
    meta9 = json.load(open(f'{k9}/kernel/rsna-sub-k9-fleet/kernel-metadata.json'))
    nb9 = json.load(open(f'{k9}/kernel/rsna-sub-k9-fleet/{meta9["code_file"]}', encoding='utf-8'))
    for tag, w in (('a', [0.1, 0.3, 0.6]), ('b', [0.3, 0.3, 0.4])):
        slug = f'rsna-sub-k9n-{tag}'
        kd = f'{build}/kernel/{slug}'
        m = json.load(open(f'{kd}/kernel-metadata.json'))
        nb = json.load(open(f'{kd}/{m["code_file"]}', encoding='utf-8'))
        ok(f'T2 {slug}: 33 cells, 0-31 == K9-main', len(nb['cells']) == 33 and all(nb['cells'][i] == nb9['cells'][i] for i in range(32)))
        src = ''.join(nb['cells'][32]['source'])
        l1, l2, body = src.split('\n', 2)
        cfg = eval(l1[len('K9N_CFG = '):].split('   #', 1)[0])
        b64 = l2.split("'")[1]
        child = base64.b64decode(b64)
        ok(f'T2 {slug}: W3 {w}, tag K9N-{tag}, variants = the other triple + .40/.30/.30',
           cfg['W3'] == w and cfg['TAG'] == f'K9N-{tag}' and len(cfg['W3_VARIANTS']) == 2 and [0.4, 0.3, 0.3] in cfg['W3_VARIANTS'], cfg)
        ok(f'T2 {slug}: embedded child sha256 == CHILD_SHA256 == make_child', sha(child) == cfg['CHILD_SHA256'] == sha(make_child.make()))
        ok(f'T2 {slug}: cell body == cells/k9n_n949_cell.py', body == open(build_k9n.N_CELL, encoding='utf-8').read())
        ok(f'T2 {slug}: metadata = K9-main + weights dataset; private, T4, GPU, no internet, no docker pin',
           m['dataset_sources'] == meta9['dataset_sources'] + ['nartaa/rsna-knee-publication-swa-weights-20261007']
           and m['is_private'] is True and m['machine_shape'] == 'NvidiaTeslaT4' and m['enable_gpu'] and not m['enable_internet']
           and 'docker_image' not in m and m['id'] == f'eeshsaxena/{slug}', m)
        ok(f'T2 {slug}: notebook sha256 == K9N_BUILD.json', sha(open(f'{kd}/{m["code_file"]}', 'rb').read()) == kb['kernels'][slug]['nb_sha256'])


def t3(tmp):
    sim = f'{SP}/k9/build_sim'
    try:
        build_k9n.main(['--k9_build', sim, '--out', f'{tmp}/b_sim'])
        ok('T3 a SIMULATION K9 build is refused without --test', False)
    except AssertionError as e:
        ok('T3 a SIMULATION K9 build is refused without --test', 'not a FINAL K9 build' in str(e), e)
    info = build_k9n.main(['--k9_build', sim, '--out', f'{tmp}/b_sim', '--test'])
    ok('T3 --test build: DO_NOT_PUSH_TEST.txt + do_not_push', os.path.isfile(f'{tmp}/b_sim/DO_NOT_PUSH_TEST.txt') and info['do_not_push'])
    return f'{tmp}/b_sim'


def t4(build, tmp):
    k9 = json.load(open(f'{build}/K9N_BUILD.json'))['k9_build']
    build_k9n.main(['--k9_build', k9, '--out', f'{tmp}/b_re', '--compare', build])
    ok('T4 rebuild from the same K9 build is byte-identical (--compare)', True)
    p = f'{tmp}/b_re/kernel/rsna-sub-k9n-b/kernel-metadata.json'
    open(p, 'a').write(' ')
    try:
        build_k9n.main(['--k9_build', k9, '--out', f'{tmp}/b_re2', '--compare', f'{tmp}/b_re'])
        ok('T4 --compare catches a one-byte change', False)
    except AssertionError as e:
        ok('T4 --compare catches a one-byte change', 'differs' in str(e), e)


def cell_ns(work, inp, build, tag='a', cfg_edit=None, child_src=None):
    kb = json.load(open(f'{build}/K9N_BUILD.json'))
    kd = kb['kernels'][f'rsna-sub-k9n-{tag}']
    nb = json.load(open(f'{kd["dir"]}/{kd["code_file"]}', encoding='utf-8'))
    src = ''.join(nb['cells'][32]['source'])
    l1, l2, body = src.split('\n', 2)
    cfg = eval(l1[len('K9N_CFG = '):].split('   #', 1)[0])
    if cfg_edit:
        cfg.update(cfg_edit)
    if child_src is not None:
        l2 = f"_K9N_CHILD_B64 = '{base64.b64encode(child_src.encode()).decode()}'"
        cfg['CHILD_SHA256'] = sha(child_src)
    s = f'K9N_CFG = {cfg!r}\n{l2}\n{body}'.replace('/kaggle/working', work).replace('/kaggle/input', inp)
    return s, cfg


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


def t5(build):
    tmp = tempfile.mkdtemp().replace(chr(92), '/')
    s, _ = cell_ns(f'{tmp}/w', f'{tmp}/i', build)
    s = s.replace('_k9n_receipt = _k9n_entry()', '')
    g = {}
    exec(compile(s, 'cell32', 'exec'), g)
    rng = np.random.default_rng(0)
    for name, M in (('random', rng.uniform(0, 1, (57, 12))), ('tie-heavy', rng.integers(0, 4, (57, 12)).astype(float))):
        a1 = g['_n_rank'](M); a2 = k9n_check.avg_rank_pct(M); a3 = pd.DataFrame(M).rank(method='average', pct=True).to_numpy()
        ok(f'T5 rank {name}: cell _n_rank == k9n_check.avg_rank_pct == pandas', np.array_equal(a1, a2) and np.allclose(a1, a3, atol=0, rtol=0))
    ok('T5 file names', g['_n_tname']([0.1, 0.3, 0.6]) == 'submission_pfn_103060.csv' == k9n_check.tname([0.1, 0.3, 0.6]))
    ok('T5 fail-loud set (K9N-a)', g['_n_loud_names']() == ['submission_n949.csv', 'submission_pfn_103060.csv',
                                                          'submission_pfn_303040.csv', 'submission_pfn_403030.csv'], g['_n_loud_names']())
    shutil.rmtree(tmp, ignore_errors=True)


def t6(build):
    kb = json.load(open(f'{build}/K9N_BUILD.json'))
    for mode in ('clean', 'pq_mismatch', 'mirror_equal', 'ids', 'label_order', 'fallback', 'csv_formula', 'child_sha', 'rc1'):
        tmp = tempfile.mkdtemp(prefix='k9n_t6_').replace(chr(92), '/')
        work, inp = f'{tmp}/w', f'{tmp}/i'
        os.makedirs(work); os.makedirs(f'{inp}/datasets/nartaa/rsna-knee-publication-swa-weights-20261007')
        wf = f'{inp}/datasets/nartaa/rsna-knee-publication-swa-weights-20261007/{make_child.WEIGHTS_FILE}'
        open(wf, 'wb').write(b'fake weights for the laptop gate test\n')
        wsha = sha(open(wf, 'rb').read())
        ids = [f'1.2.826.0.{i}' for i in range(3)]
        pd.DataFrame({'StudyInstanceUID': ids}).to_csv(f'{tmp}/test.csv', index=False)
        child = FAKE_CHILD % dict(lab=T, mode=mode, test=f'{tmp}/test.csv', wsha=wsha, pfile=None)
        s, cfg = cell_ns(work, inp, build, 'a', {'WEIGHTS_SHA256': wsha}, child)
        # synthetic strict fleet outputs
        rng = np.random.default_rng(1)
        par = pd.DataFrame(rng.uniform(0, 1, (3, 12)), columns=T); par.insert(0, 'StudyInstanceUID', ids[::-1])
        par.to_csv(f'{work}/_k2_parent.csv', index=False)
        files = {}
        F = pd.DataFrame(rng.uniform(0, 1, (3, 12)), columns=T); F.insert(0, 'StudyInstanceUID', ids[::-1])
        F.to_csv(f'{work}/submission_fleetsolo.csv', index=False); files['submission_fleetsolo.csv'] = sha(open(f'{work}/submission_fleetsolo.csv', 'rb').read())
        par.to_csv(f'{work}/submission_w50.csv', index=False); par.to_csv(f'{work}/submission.csv', index=False)
        files['submission_w50.csv'] = files['submission.csv'] = sha(open(f'{work}/submission.csv', 'rb').read())
        json.dump(dict(clean=True, why='t6 stub', files=files, main_mode='strict', stage_min=1.0), open(f'{work}/_k2_stage_fleet.json', 'w'))
        g = {'T0': time.time() - 7 * 3600}
        import torch
        _c = (torch.cuda.is_available, torch.cuda.device_count, torch.cuda.empty_cache)
        torch.cuda.is_available = lambda: True; torch.cuda.device_count = lambda: 1; torch.cuda.empty_cache = lambda: None
        try:
            exec(compile(s, 'cell32', 'exec'), g)
        except BaseException as e:
            ok(f'T6 {mode}: no exception escapes the cell', False, repr(e))
        finally:
            torch.cuda.is_available, torch.cuda.device_count, torch.cuda.empty_cache = _c
        st = json.load(open(f'{work}/_k9n_stage.json'))
        if mode == 'clean':
            ok('T6 clean (fake child): stage clean, submission.csv written', st['clean'] and os.path.isfile(f'{work}/submission.csv'), st['why'])
            kb2 = dict(kb); kb2['weights_sha256'] = wsha; kb2['child_sha256'] = cfg['CHILD_SHA256']
            json.dump(kb2, open(f'{tmp}/kb.json', 'w'))
            rc = k9n_check.main(['--work', work, '--k9n_build', f'{tmp}/kb.json', '--tag', 'a', '--expect_n', 'clean', '--skip_fleet'])
            ok('T6 clean (fake child): k9n_check --skip_fleet PASS (every file == independent recompute)', rc == 0)
        else:
            ok(f'T6 {mode}: fail loud, submission.csv ABSENT, no N file', not st['clean'] and not os.path.exists(f'{work}/submission.csv')
               and not any(os.path.exists(f'{work}/{f}') for f in ('submission_n949.csv', 'submission_pfn_103060.csv')), st['why'])
        shutil.rmtree(tmp, ignore_errors=True)


def t7(build):
    """--expect_n_ranks log949: the 3 public ids with probabilities that reproduce the author's ranks -> PASS; one label
    of one study moved -> FAIL with the flipped label and the probability gap in the detail."""
    kb = json.load(open(f'{build}/K9N_BUILD.json'))
    ref = k9n_check.log949_ranks()
    ok('T7 log949.txt parsed: 3 public studies x 12 ranks in {0, .5, 1}', len(ref) == 3 and all(
        len(v) == 12 and set(v) <= {0.0, 0.5, 1.0} for v in ref.values()), ref)
    for flip in (False, True):
        tmp = tempfile.mkdtemp(prefix='k9n_t7_').replace(chr(92), '/')
        work, inp = f'{tmp}/w', f'{tmp}/i'
        os.makedirs(work); os.makedirs(f'{inp}/datasets/nartaa/rsna-knee-publication-swa-weights-20261007')
        wf = f'{inp}/datasets/nartaa/rsna-knee-publication-swa-weights-20261007/{make_child.WEIGHTS_FILE}'
        open(wf, 'wb').write(b'fake weights for the laptop gate test' + bytes([10]))
        wsha = sha(open(wf, 'rb').read())
        ids = sorted(ref)
        pd.DataFrame({'StudyInstanceUID': ids}).to_csv(f'{tmp}/test.csv', index=False)
        R = np.array([ref[s] for s in ids], np.float64)
        Pm = 0.2 + 0.6 * R - 0.005                                  # mean of plain and plain + .01 == 0.2 + 0.6 R
        if flip:
            Pm[0, 3], Pm[1, 3] = Pm[1, 3], Pm[0, 3]
        np.save(f'{tmp}/p.npy', Pm)
        child = FAKE_CHILD % dict(lab=T, mode='clean', test=f'{tmp}/test.csv', wsha=wsha, pfile=f'{tmp}/p.npy')
        s, cfg = cell_ns(work, inp, build, 'a', {'WEIGHTS_SHA256': wsha}, child)
        rng = np.random.default_rng(1)
        par = pd.DataFrame(rng.uniform(0, 1, (3, 12)), columns=T); par.insert(0, 'StudyInstanceUID', ids[::-1])
        par.to_csv(f'{work}/_k2_parent.csv', index=False)
        files = {}
        F = pd.DataFrame(rng.uniform(0, 1, (3, 12)), columns=T); F.insert(0, 'StudyInstanceUID', ids[::-1])
        F.to_csv(f'{work}/submission_fleetsolo.csv', index=False); files['submission_fleetsolo.csv'] = sha(open(f'{work}/submission_fleetsolo.csv', 'rb').read())
        par.to_csv(f'{work}/submission_w50.csv', index=False); par.to_csv(f'{work}/submission.csv', index=False)
        files['submission_w50.csv'] = files['submission.csv'] = sha(open(f'{work}/submission.csv', 'rb').read())
        json.dump(dict(clean=True, why='t7 stub', files=files, main_mode='strict', stage_min=1.0), open(f'{work}/_k2_stage_fleet.json', 'w'))
        g = {'T0': time.time() - 7 * 3600}
        import torch
        _c = (torch.cuda.is_available, torch.cuda.device_count, torch.cuda.empty_cache)
        torch.cuda.is_available = lambda: True; torch.cuda.device_count = lambda: 1; torch.cuda.empty_cache = lambda: None
        try:
            exec(compile(s, 'cell32', 'exec'), g)
        finally:
            torch.cuda.is_available, torch.cuda.device_count, torch.cuda.empty_cache = _c
        kb2 = dict(kb); kb2['weights_sha256'] = wsha; kb2['child_sha256'] = cfg['CHILD_SHA256']
        json.dump(kb2, open(f'{tmp}/kb.json', 'w'))
        rc = k9n_check.main(['--work', work, '--k9n_build', f'{tmp}/kb.json', '--tag', 'a', '--expect_n', 'clean',
                             '--skip_fleet', '--expect_n_ranks', 'log949', '--json', f'{tmp}/c.json'])
        cj = json.load(open(f'{tmp}/c.json'))
        line = [c for c in cj['checks'] if 'public-study ranks' in c[0]][0]
        if flip:
            ok('T7 one swapped label -> the log949 rank check FAILS naming the label and the gap',
               rc == 1 and not line[1] and 'Lateral Meniscus' in line[2] and 'gap' in line[2], line)
        else:
            ok('T7 probabilities reproducing the author ranks -> --expect_n_ranks log949 PASS', rc == 0 and line[1], line)
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    t0 = time.time()
    build = f'{K9N}/build'
    tmp = tempfile.mkdtemp(prefix='k9n_tb_').replace(chr(92), '/')
    try:
        t1(); t2(build); sim = t3(tmp)
        rc = k9n_check.main(['--work', tmp, '--k9n_build', f'{sim}/K9N_BUILD.json', '--tag', 'a', '--skip_fleet'])
        ok('T3 k9n_check FAILS on a do_not_push (test) build', rc == 1)
        t4(build, tmp); t5(build); t6(build); t7(build)
    except Exception:
        traceback.print_exc(); ok('no unexpected exception', False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(g for _, g, _ in R)
    print(f'TK9NBUILD {"PASS" if n_ok == len(R) else "FAIL"} {n_ok}/{len(R)} ({time.time() - t0:.0f} s)')
    return 0 if n_ok == len(R) else 1


if __name__ == '__main__':
    sys.exit(main())
