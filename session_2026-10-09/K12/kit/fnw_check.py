#!/usr/bin/env python3
"""fnw_check.py (2026-10-09, Claude; kfn kit, byte copy in k10/): the commit / A0 gate for a parent-free FN kernel with its
F/N weights PINNED on the command line. It runs fn_check.py UNCHANGED (imported; every check, message and exit code) with
the FN slug taken from the given FN_BUILD.json (exactly one MODE FN kernel, one of SLUGS; fn_check.SLUG['fn'] is the
constant 'rsna-sub-k9fn'), and adds the weight checks:
  W1  the build record's W3 == [0, F, N], its variants == the canonical set {0/.40/.60, 0/.30/.70, 0/.50/.50} minus the
      main triple, main_file == submission_pfn_00FFNN.csv
  W2  the N receipt's config W3 / W3_VARIANTS == the same (whenever the N stage wrote a receipt, clean or not)
  W3  (submission.csv present) submission.csv == the average-tie percentile rank of F rank(fleetsolo) + N rank(N), with
      N = the child npz P in placeholder order (fallback / empty rows -> column median), recomputed HERE with pandas from
      the command-line weights (independent of FN_BUILD.json and of fn_check's numpy ranks), rows in placeholder order
  W4  (--expect_differs_from F2,N2; TEST ONLY) the studies can tell the weights apart: the F2/N2 blend differs from
      submission.csv on >= 1 label (on 2 studies, or on 3 with no strong F~N disagreement, the blends coincide)
  INFO per canonical triple: how many labels' ranks differ from the main blend on this run.
The W checks are appended to fn_check's --json ('pass' updated), so the box harness counts them. Last line:
FNWCHECK PASS|FAIL (fn_check n/m, weights k/l).
  python fnw_check.py --work DIR --fn_build .../FN_BUILD.json --kernel fn --expect_w3 0.30,0.70 [every fn_check option]
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
sys.path.insert(0, HERE)
import fn_check, k2_check                                              # noqa: E402

SLUGS = ('rsna-sub-k9fn', 'rsna-sub-k9fn-n70', 'rsna-sub-k10fn', 'rsna-sub-k10fn3', 'rsna-sub-k10fn3-n70') + ('rsna-sub-k11fn', 'rsna-sub-k11fn-n70', 'rsna-sub-k11fn-d2', 'rsna-sub-k11fn-d2-n70') + ('rsna-sub-k12fn', 'rsna-sub-k12fn-n70')   # K11 (k11/make_k11_tools.py) + K12 (k12/make_k12_tools.py)
CANON = ([0.0, 0.4, 0.6], [0.0, 0.3, 0.7], [0.0, 0.5, 0.5])
T = k2_check.TARGETS


def set_slug(fn_build):
    kb = json.load(open(fn_build))
    fn = [s for s, k in kb['kernels'].items() if k.get('mode') == 'FN']
    assert len(fn) == 1, f'{fn_build}: want exactly one MODE FN kernel, got {fn}'
    assert fn[0] in SLUGS, f'{fn[0]} is not a known FN kernel slug {SLUGS}'
    fn_check.SLUG['fn'] = fn[0]
    return fn[0], kb


def _pop(argv, flag):
    if flag not in argv:
        return None
    i = argv.index(flag)
    v = argv[i + 1]
    del argv[i:i + 2]
    return v


def _w(txt):
    f, n = [round(float(x), 4) for x in txt.split(',')]
    assert abs(f + n - 1) < 1e-9 and f > 0 and n > 0, txt
    return [0.0, f, n]


def _rk(M):
    return pd.DataFrame(np.asarray(M, np.float64)).rank(method='average', pct=True).to_numpy()


def weight_checks(work, slug, kb, W, differs):
    res, info = [], {}

    def ok(name, cond, detail=''):
        res.append((f'W: {name}', bool(cond), str(detail)[:300]))
    k = kb['kernels'][slug]
    var = [w for w in CANON if w != W]
    ok(f'W1 FN_BUILD {slug}: W3 == {W}, variants == {var}, main file {fn_check.tname(W)}',
       k['W3'] == W and k['W3_VARIANTS'] == var and k.get('main_file', fn_check.tname(W)) == fn_check.tname(W),
       f'{k["W3"]} {k["W3_VARIANTS"]} {k.get("main_file")}')
    rcp = f'{work}/n949_receipt.json'
    if os.path.isfile(rcp):
        cfg = json.load(open(rcp)).get('config') or {}
        ok(f'W2 N receipt config: W3 == {W}, variants == {var}', cfg.get('W3') == W and cfg.get('W3_VARIANTS') == var,
           f'{cfg.get("W3")} {cfg.get("W3_VARIANTS")}')
    sub = f'{work}/submission.csv'
    if os.path.isfile(sub):
        ph = pd.read_csv(f'{work}/_fn_ids.csv', dtype={'StudyInstanceUID': str})
        ids = ph['StudyInstanceUID'].tolist()
        fso = pd.read_csv(f'{work}/submission_fleetsolo.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        z = np.load(f'{work}/_n949_probs.npz', allow_pickle=False)
        meta = json.loads(str(z['meta']))
        bad = set(meta.get('fallback') or []) | set(meta.get('empty') or [])
        pos = {str(s): i for i, s in enumerate(z['ids'])}
        NV = np.asarray(z['P'], np.float64)[[pos[s] for s in ids]]
        badr = np.array([s in bad for s in ids])
        if badr.any():
            NV[badr] = np.median(NV[~badr], axis=0)[None]
        FV = fso.loc[ids, T].to_numpy(np.float64)
        blend = {tuple(w): _rk(w[1] * _rk(FV) + w[2] * _rk(NV)) for w in CANON + (W,)}
        got = pd.read_csv(sub, dtype={'StudyInstanceUID': str})
        g = got.columns.tolist() == ['StudyInstanceUID', *T] and got['StudyInstanceUID'].tolist() == ids
        d = float(np.abs(got[T].to_numpy(np.float64) - blend[tuple(W)]).max()) if g else float('inf')
        ok(f'W3 submission.csv == pandas rank blend F/N {W[1]}/{W[2]} of fleetsolo + N npz (CLI weights, P leg absent), '
           f'placeholder row order', g and d < 1e-9, f'max|diff| {d:.2e}')
        info['labels_differing_from_main'] = {fn_check.tname(w): int((np.abs(blend[tuple(w)] - blend[tuple(W)]) > 1e-12).any(0).sum())
                                              for w in CANON if w != W}
        info['n_studies'] = len(ids)
        if differs:
            nd = int((np.abs(blend[tuple(differs)] - got[T].to_numpy(np.float64)) > 1e-12).any(0).sum()) if g else 0
            ok(f'W4 (TEST) these {len(ids)} studies tell F/N {W[1]}/{W[2]} from {differs[1]}/{differs[2]}: labels differing >= 1',
               nd >= 1, f'{nd} label(s) differ')
            info['labels_differing_vs_expect_differs_from'] = nd
    return res, info


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ew = _pop(argv, '--expect_w3')
    df = _pop(argv, '--expect_differs_from')
    assert ew, '--expect_w3 F,N is required (e.g. 0.30,0.70)'
    assert argv[argv.index('--kernel') + 1] == 'fn', 'fnw_check gates FN kernels only (--kernel fn)'
    W, differs = _w(ew), (_w(df) if df else None)
    fb = argv[argv.index('--fn_build') + 1]
    work = argv[argv.index('--work') + 1]
    jp = argv[argv.index('--json') + 1] if '--json' in argv else None
    slug, kb = set_slug(fb)
    rc = fn_check.main(argv)
    res, info = weight_checks(work, slug, kb, W, differs)
    for name, good, detail in res:
        print(f'{"PASS" if good else "FAIL"}  {name}  {detail}')
    print('INFO-W', json.dumps(dict(slug=slug, expect_w3=W, **info)))
    wok = all(g for _, g, _ in res)
    allok = rc == 0 and wok
    nfc = None
    if jp:
        j = json.load(open(jp))
        nfc = (sum(bool(c[1]) for c in j['checks']), len(j['checks']))
        j['checks'] += [list(r) for r in res]
        j['pass'] = bool(j['pass'] and wok)
        j.setdefault('info', {})['weights'] = dict(slug=slug, expect_w3=W, **info)
        json.dump(j, open(jp, 'w'), indent=1, default=str)
    fc = f'{nfc[0]}/{nfc[1]}' if nfc else ('PASS' if rc == 0 else 'FAIL')
    print(f'FNWCHECK {"PASS" if allok else "FAIL"} (fn_check {fc}, weights {sum(g for _, g, _ in res)}/{len(res)}) '
          f'slug {slug} P/F/N {W}')
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
