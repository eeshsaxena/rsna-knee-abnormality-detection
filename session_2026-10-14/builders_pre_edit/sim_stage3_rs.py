# sim_stage3_rs.py (laptop CPU, synthetic csvs): proves the HR7-RS stage-3 edit.
#   A  RS OK      -> submission == an independent recompute (r384 .45 / r224 .25 / ours .30, renormalised per row)
#   B  RS failed  -> submission byte-identical to the PARENT stage 3 (HR7-RR v2) on the same inputs
#   C  RS OK but our raw has a NaN -> RAPTOR_SEEDS_FALLBACK, byte-identical to the parent
#   D  RS OK, r224 failed (RR fallback) -> mix of r384 .45 / ours .30 only (renormalised), == recompute
# usage: python sim_stage3_rs.py KERNEL_DIR/rsna_sub_hr7rs.ipynb PARENT.ipynb
import sys, json, os, tempfile, shutil, time, hashlib
import numpy as np, pandas as pd
from pathlib import Path
nb, parent = sys.argv[1], sys.argv[2]
src = lambda c: ''.join(c['source'])
s3 = [src(c) for c in json.load(open(nb, encoding='utf-8'))['cells'] if src(c).startswith('# Stage 3: per-finding rank blend')][0]
p3 = [src(c) for c in json.load(open(parent, encoding='utf-8'))['cells'] if src(c).startswith('# Stage 3: per-finding rank blend')][0]
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
rng = np.random.default_rng(0); n = 40; ids = ['1.2.%d' % i for i in range(n)]


def setup(root, bad_ours=False):
    rng = np.random.default_rng(0)                    # same data for the new and the parent run of a case
    D = Path(root) / 'data'; W = Path(root) / 'work'; P = W / 'pred'; D.mkdir(); W.mkdir(); (P / 'worker0').mkdir(parents=True)
    pd.DataFrame({'StudyInstanceUID': ids}).to_csv(D / 'test.csv', index=False)
    pd.DataFrame(columns=['StudyInstanceUID'] + LAB).to_csv(D / 'sample_submission.csv', index=False)
    def wr(name, a):
        df = pd.DataFrame(a, columns=LAB); df.insert(0, 'StudyInstanceUID', ids); df.to_csv(W / name, index=False)
    base = rng.random((n, 12))
    r384 = np.clip(base + .1 * rng.random((n, 12)), 0, 1); r384[3] = .5          # study 3: r384 failed
    r224 = np.clip(base + .2 * rng.random((n, 12)), 0, 1); r224[5] = .5          # study 5: r224 failed
    our3 = np.clip(base + .3 * rng.random((n, 12)), 0, 1); our3[7] = .5          # study 7: our seeds failed
    if bad_ours: our3[9, 2] = np.nan
    fl = rng.random((n, 12))
    rk = lambda a: pd.DataFrame(a).rank(pct=True).values
    wr('raptor_raw.csv', r384); wr('raptor_sub.csv', rk(r384)); wr('raptor224_raw.csv', r224); wr('raptor224_sub.csv', rk(r224))
    wr('raptor_ours_raw.csv', our3); wr('raptor_ours_sub.csv', rk(np.nan_to_num(our3, nan=.5))); wr('ours_sub.csv', fl)
    df = pd.DataFrame(fl, columns=LAB); df.insert(0, 'StudyInstanceUID', ids); df.to_csv(P / 'worker0' / 'predictions.csv', index=False)
    return D, W, P, r384, r224, our3, fl


def run(code, D, W, P, rr_ok, rs_ok):
    g = dict(DATA=D, WORK=W, pred=P, NB_T0=time.time(), RAPTOR_SECONDS=1.0, RR_OK=rr_ok, RR_REASON='', RR_SECONDS=1.0,
             RAPTOR_SEEDS_OK=rs_ok, RS_REASON='' if rs_ok else 'sim: forced', RS_SECONDS=1.0, display=lambda *a: None, json=json, time=time)
    exec(compile(code, 'stage3', 'exec'), g)
    return (W / 'submission.csv').read_bytes(), json.loads((W / 'blend_receipt.json').read_text())


def recompute(r384, r224, our3, fl, use224):
    pct = lambda v, ok: (lambda r: (r.__setitem__(ok, pd.Series(v[ok]).rank(method='average', pct=True).values) or r))(np.full(len(v), np.nan))
    rf = (r384 == .5).all(1); f2 = (r224 == .5).all(1); f3 = (our3 == .5).all(1)
    out = np.zeros((n, 12))
    for j in range(12):
        p1, p3_ = pct(r384[:, j], ~rf), pct(our3[:, j], ~f3)
        num = .45 * p1; den = np.full(n, .45); num[~f3] += .30 * p3_[~f3]; den[~f3] += .30
        if use224:
            p2 = pct(r224[:, j], ~f2); num[~f2] += .25 * p2[~f2]; den[~f2] += .25
        mix = num / den
        rr, ro = pct(mix, ~rf), pct(fl[:, j], np.ones(n, bool))
        w = .5
        b = w * rr + (1 - w) * ro; b[rf] = ro[rf]; out[:, j] = b
    return out.astype(np.float32)


res = {}
for case, rr_ok, rs_ok, bad in (('A', True, True, False), ('B', True, False, False), ('C', True, True, True), ('D', False, True, False)):
    for code, tag in ((s3, 'new'), (p3, 'parent')):
        d = tempfile.mkdtemp()
        try:
            D, W, P, r384, r224, our3, fl = setup(d, bad)
            res[(case, tag)] = run(code, D, W, P, rr_ok, rs_ok) + ((r384, r224, our3, fl),)
        finally:
            shutil.rmtree(d)
ok = True
sub = lambda b: pd.read_csv(__import__('io').BytesIO(b))[LAB].values
A_new = sub(res[('A', 'new')][0]); A_rc = recompute(*res[('A', 'new')][2], use224=True)
d = np.abs(A_new - A_rc).max(); print(f'A RS OK: max |submission - recompute| {d:.2e} | receipt raptor_seeds used {res[("A", "new")][1]["raptor_seeds"]["used"]}'); ok &= d < 1e-6
for case in ('B', 'C'):
    same = res[(case, 'new')][0] == res[(case, 'parent')][0]
    print(f'{case}: byte-identical to the parent: {same} | raptor_seeds used {res[(case, "new")][1]["raptor_seeds"]["used"]} | reason {res[(case, "new")][1]["raptor_seeds"]["reason"][:70]}'); ok &= same
D_new = sub(res[('D', 'new')][0]); D_rc = recompute(*res[('D', 'new')][2], use224=False)
d = np.abs(D_new - D_rc).max(); print(f'D RR failed, RS OK: max |submission - recompute (r384 + ours)| {d:.2e}'); ok &= d < 1e-6
print('SIM_STAGE3_RS', 'PASS' if ok else 'FAIL')
