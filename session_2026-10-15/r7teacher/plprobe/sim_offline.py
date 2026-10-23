#!/usr/bin/env python
# sim_offline.py (2026-10-14): offline execution of the build_pl.py cells on REAL commit outputs, before any Kaggle commit.
#   H-tests: her stage-3 cell (from the built notebook) + PL-H + CANARY executed on the V224-exact commit csvs
#            (session_2026-10-14/v224/out_v224_v1): ident must give V224-exact's submission.csv byte for byte; oa65 / tmulti
#            must change exactly the mapped columns and match an independent numpy recompute; a forced r224 failure must
#            still blank Fracture (canary after PL-H) and keep the PL columns.
#   F-test:  PL-F executed on the HFR7-RR commit output (session_2026-10-13/hr7rr/out_hfr7rr_v1: submission_H.csv as H,
#            submission_fleetsolo.csv as F) with f = .2 everywhere must give HFR7-RR's submission.csv (the HF cell) exactly;
#            F missing -> Synovitis 0.5 canary; r224 not used -> Fracture 0.5.
# usage: python sim_offline.py SCRATCH_DIR      (prints SIM_OFFLINE PASS n/m)
import os, sys, json, shutil, io, contextlib, time, hashlib, subprocess
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V = ROOT / 'session_2026-10-14' / 'v224'
OUT_EXACT = V / 'out_v224_v1'
HFRR = ROOT / 'session_2026-10-13' / 'hr7rr' / 'out_hfr7rr_v1'
SCR = Path(sys.argv[1])
SCR.mkdir(parents=True, exist_ok=True)
R = []
A224 = 0.35


def chk(name, ok, info=''):
    R.append(bool(ok))
    print(f'  {"OK  " if ok else "FAIL"} {name} {info}', flush=True)


def build(mapfile, tag, extra=()):
    o = SCR / ('b_' + tag)
    r = subprocess.run([sys.executable, str(HERE / 'build_pl.py'), '--map', str(mapfile), '--out', str(o), *extra],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    d = [p for p in o.iterdir() if p.is_dir()][0]
    nb = json.load(open([p for p in d.iterdir() if p.suffix == '.ipynb'][0], encoding='utf-8'))
    return [c['source'] if isinstance(c['source'], str) else ''.join(c['source']) for c in nb['cells']]


def avg_rank_pct(v):
    return pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values


def recompute(work, shares, use224=True):
    ids = pd.read_csv(ROOT / '_data' / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
    labs = list(pd.read_csv(ROOT / '_data' / 'sample_submission.csv', nrows=1).columns)[1:]
    rd = lambda n: pd.read_csv(work / n, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, labs]
    raw, ours = rd('raptor_raw.csv'), rd('ours_sub.csv')
    r224 = rd('raptor224_sub.csv') if use224 else None
    out = np.zeros((len(ids), len(labs)))
    for j, c in enumerate(labs):
        rr = avg_rank_pct(raw[c].values)
        if r224 is not None:
            rr = (1 - A224) * rr + A224 * avg_rank_pct(r224[c].values)
        s = shares.get(c, 0.5)
        out[:, j] = s * rr + (1 - s) * avg_rank_pct(ours[c].values)
    return pd.DataFrame(out, columns=labs, index=ids)


def run_h(cells, tag, r224_ok=True):
    st3 = [s for s in cells if s.startswith('# Stage 3: per-finding rank blend')][0]
    plh = [s for s in cells if s.startswith('# [eeshsaxena PL-H')][0]
    can = [s for s in cells if s.startswith('# [eeshsaxena CANARY')][0]
    i3, ip, ic = cells.index(st3), cells.index(plh), cells.index(can)
    assert i3 + 1 == ip and ip + 1 == ic, (i3, ip, ic)
    sim = SCR / ('h_' + tag)
    if sim.exists():
        shutil.rmtree(sim)
    work, data, pred = sim / 'out', sim / 'data', sim / 'pred'
    for d in (work, data, pred / 'worker0'):
        d.mkdir(parents=True)
    for f in ('ours_sub.csv', 'raptor_raw.csv', 'raptor_sub.csv', 'raptor224_sub.csv'):
        shutil.copy(OUT_EXACT / f, work / f)
    if not r224_ok:
        os.remove(work / 'raptor224_sub.csv')
    shutil.copy(ROOT / '_data' / 'test.csv', data / 'test.csv')
    shutil.copy(ROOT / '_data' / 'sample_submission.csv', data / 'sample_submission.csv')
    shutil.copy(work / 'ours_sub.csv', pred / 'worker0' / 'predictions.csv')
    g = {'DATA': data, 'WORK': work, 'pred': pred, 'R224_OK': r224_ok, 'RAPTOR_SECONDS': 50.0, 'NB_T0': time.time(),
         'json': json, 'time': time, 'display': lambda *a, **k: None}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for s in (st3, plh, can):
            exec(compile(s, 'cell', 'exec'), g)
    return work, buf.getvalue(), g


print('SIM_OFFLINE', SCR, flush=True)
labs = list(pd.read_csv(ROOT / '_data' / 'sample_submission.csv', nrows=1).columns)[1:]
exact_bytes = open(OUT_EXACT / 'submission.csv', 'rb').read()

# 1. identity map == V224-exact commit, byte for byte
cells = build(HERE / 'maps' / 'ident.json', 'ident')
work, log, g = run_h(cells, 'ident')
chk('ident: PL_H_OK, nothing changed', 'PL_H_OK' in log and '"changed_s": {}' in log, log.split('PL_H_OK')[-1][:90].strip())
chk('ident: CANARY_CLEAN after PL-H', log.find('CANARY_CLEAN') > log.find('PL_H_OK') > log.find('NOTEBOOK_COMPLETE') >= 0)
got = open(work / 'submission.csv', 'rb').read()
chk('ident: submission.csv == V224-exact commit submission.csv (bytes, CRLF-normalised)',
    got.replace(b'\r\n', b'\n') == exact_bytes.replace(b'\r\n', b'\n'), hashlib.sha256(got).hexdigest()[:16])

# 2. oa65 and tmulti: only the mapped columns change; they match the independent recompute
for mf, tag in (('oa65.json', 'oa65'), ('t_multi.json', 'tmulti')):
    m = json.load(open(HERE / 'maps' / mf, encoding='utf-8'))
    cells = build(HERE / 'maps' / mf, tag)
    res = json.load(open([p for p in (SCR / ('b_' + tag)).iterdir() if p.is_dir()][0] / 'pl_map_resolved.json'))
    shares = {c: v['s'] for c, v in res['resolved'].items()}
    work, log, g = run_h(cells, tag)
    sub = pd.read_csv(work / 'submission.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    ex = pd.read_csv(OUT_EXACT / 'submission.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    rc = recompute(work, shares)
    d = float(np.abs(sub[labs].values - rc.loc[sub.index, labs].values).max())
    chk(f'{tag}: submission == independent recompute (per-label shares)', d <= 1e-6, 'max diff %.3g' % d)
    moved = [c for c in labs if not np.array_equal(sub[c].values, ex[c].values)]
    want = [c for c in labs if shares[c] != 0.5]
    chk(f'{tag}: changed columns are a subset of the mapped ones, untouched columns bit-identical to V224-exact',
        set(moved) <= set(want), {'moved': moved, 'mapped': want})
    chk(f'{tag}: at least one mapped column moved on these 3 rows', len(moved) > 0, moved)
    pr = json.load(open(work / 'pl_receipt.json'))
    chk(f'{tag}: pl_receipt changed_s == the map', pr['changed_s'] == {c: shares[c] for c in want} and pr['r224_used'] is True)

# 3. forced r224 failure on oa65: canary after PL-H blanks Fracture; OA columns still follow the PL shares (no r224)
cells = build(HERE / 'maps' / 'oa65.json', 'oa65c')
work, log, g = run_h(cells, 'oa65_ctl', r224_ok=False)
sub = pd.read_csv(work / 'submission.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
rc = recompute(work, {'Medial OA': .65, 'Lateral OA': .65, 'PF OA': .65}, use224=False)
keep = [c for c in labs if c != 'Fracture']
d = float(np.abs(sub[keep].values - rc.loc[sub.index, keep].values).max())
chk('control: CANARY_FIRED after PL_H_OK', log.find('CANARY_FIRED') > log.find('PL_H_OK') >= 0)
chk('control: Fracture == 0.5 exactly', bool((sub['Fracture'] == 0.5).all()))
chk('control: other 11 columns == recompute without r224 (OA at .65)', d <= 1e-6, 'max diff %.3g' % d)

# 4. self-check really guards: a tampered stage-3 `out` must make PL-H raise
cells = build(HERE / 'maps' / 'ident.json', 'identx')
st3 = [s for s in cells if s.startswith('# Stage 3: per-finding rank blend')][0]
bad = [s.replace("sub = pd.DataFrame(out.astype(np.float32), columns=labels)",
                 "out[0, 4] += 1e-12\nsub = pd.DataFrame(out.astype(np.float32), columns=labels)") if s == st3 else s for s in cells]
try:
    run_h(bad, 'tamper')
    chk('self-check raises on a 1e-12 change of her out', False)
except RuntimeError as e:
    chk('self-check raises on a 1e-12 change of her out', 'PL_SELF_CHECK_FAIL' in str(e), str(e)[:80])

# 5. PL-F on the HFR7-RR commit output: f .2 everywhere == the HF cell's submission.csv
fake = SCR / 'fake_fleet_base.ipynb'
nbv = json.load(open(V / 'kernel_v224' / 'rsna_sub_v224.ipynb', encoding='utf-8'))
nbv['cells'].append({'cell_type': 'code', 'metadata': {}, 'source': "# fake fleet stage (offline test only): writes /kaggle/working/submission_fleetsolo.csv\n", 'outputs': [], 'execution_count': None})
json.dump(nbv, open(fake, 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
cells = build(HERE / 'maps' / 't_f20.json', 'f20', ['--base-nb', str(fake), '--base-meta', str(V / 'kernel_v224' / 'kernel-metadata.json')])
chk('f20 build: PL-F appended last, PL-H after stage 3', cells[-1].startswith('# [eeshsaxena PL-F') and
    cells[cells.index([s for s in cells if s.startswith('# Stage 3')][0]) + 1].startswith('# [eeshsaxena PL-H'))
plf = cells[-1]
res = json.load(open([p for p in (SCR / 'b_f20').iterdir() if p.is_dir()][0] / 'pl_map_resolved.json'))


def run_f(tag, fleet=True, r224_used=True):
    w = SCR / ('f_' + tag)
    if w.exists():
        shutil.rmtree(w)
    w.mkdir(parents=True)
    shutil.copy(HFRR / 'submission_H.csv', w / 'submission_PLH.csv')
    if fleet:
        shutil.copy(HFRR / 'submission_fleetsolo.csv', w / 'submission_fleetsolo.csv')
    json.dump({'name': 'tf20', 'map_sha': res['map_sha'], 'map': res['resolved'], 'r224_used': r224_used}, open(w / 'pl_receipt.json', 'w'))
    code = plf.replace("'/kaggle/working'", repr(str(w).replace('\\', '/')))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, 'plf', 'exec'), {})
    return w, buf.getvalue()


w, log = run_f('ok')
a = pd.read_csv(w / 'submission.csv', dtype={'StudyInstanceUID': str})
b = pd.read_csv(HFRR / 'submission.csv', dtype={'StudyInstanceUID': str})
chk('PL-F f .2 == HFR7-RR HF cell output (values exactly equal)', a.equals(b) and 'PL_F_DONE' in log and '"fired": []' in log)
chk('PL-F f .2 == HFR7-RR submission.csv bytes (CRLF-normalised)',
    open(w / 'submission.csv', 'rb').read().replace(b'\r\n', b'\n') == open(HFRR / 'submission.csv', 'rb').read().replace(b'\r\n', b'\n'))
w, log = run_f('nofleet', fleet=False)
a = pd.read_csv(w / 'submission.csv', dtype={'StudyInstanceUID': str})
h = pd.read_csv(HFRR / 'submission_H.csv', dtype={'StudyInstanceUID': str})
others = [c for c in labs if c != 'Synovitis']
chk('PL-F fleet missing: Synovitis 0.5, other 11 == H', bool((a['Synovitis'] == 0.5).all()) and a[others].equals(h[others])
    and 'F->Synovitis' in log)
w, log = run_f('no224', r224_used=False)
a = pd.read_csv(w / 'submission.csv', dtype={'StudyInstanceUID': str})
chk('PL-F r224 not used: Fracture 0.5', bool((a['Fracture'] == 0.5).all()) and 'r224->Fracture' in log)

# 6. PL-F (f .2) on the V224-F commit output (session_2026-10-15/v224f/out_v224f20_v1, built by the parallel V224-F agent):
#    H = their submission_V224.csv, F = their submission_fleetsolo.csv -> must equal their HF output (submission.csv)
V224F_OUT = ROOT / 'session_2026-10-15' / 'v224f' / 'out_v224f20_v1'
if (V224F_OUT / 'submission_V224.csv').is_file() and (V224F_OUT / 'submission_fleetsolo.csv').is_file():
    w = SCR / 'f_v224f'
    if w.exists():
        shutil.rmtree(w)
    w.mkdir(parents=True)
    shutil.copy(V224F_OUT / 'submission_V224.csv', w / 'submission_PLH.csv')
    shutil.copy(V224F_OUT / 'submission_fleetsolo.csv', w / 'submission_fleetsolo.csv')
    json.dump({'name': 'tf20', 'map_sha': res['map_sha'], 'map': res['resolved'], 'r224_used': True}, open(w / 'pl_receipt.json', 'w'))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(plf.replace("'/kaggle/working'", repr(w.as_posix())), 'plf', 'exec'), {})
    a = pd.read_csv(w / 'submission.csv', dtype={'StudyInstanceUID': str})
    b = pd.read_csv(V224F_OUT / 'submission.csv', dtype={'StudyInstanceUID': str})
    d = float(np.abs(a[labs].values - b[labs].values).max())
    chk('PL-F f .2 on the V224-F commit == V224-F HF output', a['StudyInstanceUID'].equals(b['StudyInstanceUID']) and d == 0.0, 'max diff %.3g' % d)
else:
    print('  info V224-F commit output not present: test 6 skipped')

n_ok = sum(R)
print(f'SIM_OFFLINE {"PASS" if all(R) else "FAIL"} {n_ok}/{len(R)}')
sys.exit(0 if all(R) else 1)
