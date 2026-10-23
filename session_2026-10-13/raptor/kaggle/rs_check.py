#!/usr/bin/env python
# rs_check.py (2026-10-13, Claude): commit gate for HR7-RS / HFR7-RS (raptor/kaggle/build_hr7rs.py) = rr_check.py (HR7-RR) plus
# the stage-1c checks (our Raptor seeds) and the 3-member Raptor_mix (r384 .45 / r224 .25 / ours .30, PREREG_RAPTOR_KERNELS.md).
# Base: rr_check.py (HR7-RR). Reads a fetched kernel output dir
# (kaggle kernels output eeshsaxena/rsna-sub-hr7rr -p OUT --page-size 200). Prints RR_CHECK PASS/FAIL n/m; exit 0 only on PASS.
#   usage: python rr_check.py OUT [--w384 0.65] [--hf-wf 0.2] [--sample PATH_TO_sample_submission.csv]
# Checks: both Raptor stages PASS with 0 fallbacks (r384 + r224), R7 stage PASS with 0 fallbacks, submission ids == sample,
# values finite in [0,1], and the stage-3 output == an independent recomputation of the blend from the saved csvs (max diff < 1e-6):
#   p(x) = average-tie rank / n per label; mix = w384 p(r384 raw) + (1 - w384) p(r224 raw); H = 0.5 p(mix) + 0.5 p(ours).
# HFR7-RR (--hf-wf W): the stage-3 output is submission_H.csv, and submission.csv must == (1 - W) p(H) + W p(fleetsolo),
# with H recomputed independently (h_check.py checks the same rule from the saved submission_H.csv).
import os, sys, json, glob, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--seeds', required=True, help='seeds.json from pack_seeds.py')
ap.add_argument('--w3', default='0.45,0.25,0.30')
ap.add_argument('--hf-wf', type=float, default=None)
ap.add_argument('--sample', default=os.path.normpath(os.path.join(HERE, '..', '..', '..', '_data', 'sample_submission.csv')))
A = ap.parse_args()
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print(f'  {"OK  " if ok else "FAIL"} {name} {info}', flush=True)


def P(name):
    return os.path.join(A.out, name)


def avg_rank_pct(v):
    # independent of the notebook's pandas rank: average 1-based rank of tied values, divided by n
    v = np.asarray(v, dtype=np.float64)
    order = np.argsort(v, kind='mergesort')
    sv = v[order]
    r = np.empty(len(v), dtype=np.float64)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r / len(v)


# ---- log --------------------------------------------------------------------------------------------------------------
logs = [p for p in glob.glob(P('*.log')) if 'cufile' not in os.path.basename(p)]
text = ''
if logs:
    try:
        text = '\n'.join(e.get('data', '') for e in json.load(open(logs[0], encoding='utf-8')))
    except Exception:
        text = open(logs[0], encoding='utf-8', errors='replace').read()
chk('kernel log present', bool(text), [os.path.basename(p) for p in logs[:1]])
chk('no Traceback in log', 'Traceback (most recent call last)' not in text)
chk('stage 1 r384: RAPTOR_STAGE_DONE', 'RAPTOR_STAGE_DONE' in text)
chk('stage 1 r384: checkpoint sha 7e5315da printed', '[weights] sha256 7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef' in text)
chk('stage 1b r224: RR_STAGE_DONE', 'RR_STAGE_DONE' in text)
chk('stage 1b r224: ran the c08_r224 checkpoint', "[blend] weighted rank-mean w={'raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt'" in text)
chk('stage 3: RR_MIX_OK', 'RR_MIX_OK' in text)
chk('no RR_FALLBACK', 'RR_FALLBACK' not in text)
chk('no BLEND_FALLBACK', 'BLEND_FALLBACK' not in text)
fb = [l for l in text.splitlines() if ' FALLBACK (' in l or ('[pre]' in l and 'FAILED' in l)]
chk('Raptor stages: 0 per-study FALLBACK / prep FAILED lines', not fb, fb[:3])
chk('NOTEBOOK_COMPLETE', 'NOTEBOOK_COMPLETE' in text)
SEEDS = json.load(open(A.seeds))
chk('stage 1c: RS_STAGE_DONE', 'RS_STAGE_DONE' in text)
chk('stage 3: RAPTOR_SEEDS_MIX_OK, no RAPTOR_SEEDS_FALLBACK', 'RAPTOR_SEEDS_MIX_OK' in text and 'RAPTOR_SEEDS_FALLBACK' not in text)
for _s in SEEDS:
    chk('stage 1c: seed sha printed ' + _s['file'], ('[weights] sha256 ' + _s['sha256']) in text)

# ---- files ------------------------------------------------------------------------------------------------------------
need = (['submission_H.csv', 'submission_fleetsolo.csv', 'hf_receipt.json'] if A.hf_wf is not None else []) + ['submission.csv', 'raptor_raw.csv', 'raptor_sub.csv', 'raptor224_raw.csv', 'raptor224_sub.csv', 'ours_sub.csv',
        'blend_receipt.json', 'submission_receipt.json', 'raptor_ours_raw.csv', 'raptor_ours_sub.csv']
miss = [f for f in need if not os.path.isfile(P(f))]
chk('all output files present', not miss, miss)
if miss:
    print(f'RS_CHECK FAIL {sum(R)}/{len(R)}')
    sys.exit(1)

# ---- R7 stage ---------------------------------------------------------------------------------------------------------
sr = json.load(open(P('submission_receipt.json'), encoding='utf-8'))
chk('R7 stage status PASS, 0 fallback studies, scored == rows',
    sr.get('status') == 'PASS' and sr.get('fallback_studies') == 0 and sr.get('scored') == sr.get('rows'),
    (sr.get('status'), sr.get('fallback_studies'), sr.get('scored'), sr.get('rows')))
chk('OURS_STAGE_COMPLETE with status PASS', 'OURS_STAGE_COMPLETE {"status": "PASS"' in text)

# ---- blend receipt ----------------------------------------------------------------------------------------------------
b = json.load(open(P('blend_receipt.json'), encoding='utf-8'))
mx = b.get('raptor_mix') or {}
chk('receipt: raptor_failed 0, ours_failed 0, both_failed 0',
    b.get('raptor_failed') == 0 and b.get('ours_failed') == 0 and b.get('both_failed') == 0,
    (b.get('raptor_failed'), b.get('ours_failed'), b.get('both_failed')))
chk('receipt: raptor_mix used, r224_failed 0, stage_ok', mx.get('used') is True and mx.get('r224_failed') == 0 and mx.get('stage_ok') is True,
    {k: mx.get(k) for k in ('used', 'r224_failed', 'stage_ok', 'reason', 'r224_seconds')})
W384, W224, W3O = [float(x) for x in A.w3.split(',')]
rs = b.get('raptor_seeds') or {}
chk('receipt: raptor_seeds used, ours_failed 0, with_r224, weights', rs.get('used') is True and rs.get('ours_failed') == 0 and rs.get('with_r224') is True
    and rs.get('weights') == {'r384': W384, 'r224': W224, 'ours': W3O}, {k: rs.get(k) for k in ('used', 'ours_failed', 'with_r224', 'weights', 'seconds', 'reason')})
chk('receipt: weights R7 .5 / Raptor .5 (per-class .5), parent mix r384 .65 / r224 .35 recorded',
    b.get('weights', {}).get('raptor') == 0.5 and b.get('weights', {}).get('ours') == 0.5
    and all(v == 0.5 for v in b.get('weights', {}).get('raptor_per_class', {}).values())
    and mx.get('weights') == {'r384': 0.65, 'r224': 0.35}, mx.get('weights'))

# ---- submission vs sample ---------------------------------------------------------------------------------------------
S = pd.read_csv(P('submission.csv'), dtype={'StudyInstanceUID': str})
labels = list(S.columns[1:])
SH = pd.read_csv(P('submission_H.csv'), dtype={'StudyInstanceUID': str}) if A.hf_wf is not None else S   # stage-3 output
if A.hf_wf is not None:
    chk('submission_H ids/columns == submission ids/columns', SH.columns.tolist() == S.columns.tolist()
        and SH['StudyInstanceUID'].tolist() == S['StudyInstanceUID'].tolist())
    hr = json.load(open(P('hf_receipt.json'), encoding='utf-8'))
    chk(f'hf_receipt mode HF, w_f == {A.hf_wf}', hr.get('mode') == 'HF' and abs(float(hr.get('w_f', -1)) - A.hf_wf) < 1e-12,
        (hr.get('mode'), hr.get('w_f')))
    chk('HF_BLEND_OK, no HF_FALLBACK', 'HF_BLEND_OK' in text and 'HF_FALLBACK' not in text)
if os.path.isfile(A.sample):
    ss = pd.read_csv(A.sample, dtype={'StudyInstanceUID': str})
    chk('columns == sample columns', list(S.columns) == list(ss.columns))
    chk('ids == sample ids (same order)', S['StudyInstanceUID'].tolist() == ss['StudyInstanceUID'].tolist(), (len(S), len(ss)))
else:
    chk('sample_submission.csv found', False, A.sample)
v = S[labels].values.astype(np.float64)
chk('values finite in [0,1]', np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all())
ids = S['StudyInstanceUID'].tolist()


def load(name):
    d = pd.read_csv(P(name), dtype={'StudyInstanceUID': str})
    assert sorted(d['StudyInstanceUID']) == sorted(ids), name + ' ids differ'
    return d.set_index('StudyInstanceUID').loc[ids, labels].values.astype(np.float64)


r384, r224, ours = load('raptor_raw.csv'), load('raptor224_raw.csv'), load('ours_sub.csv')
s384, s224 = load('raptor_sub.csv'), load('raptor224_sub.csv')
r3, s3 = load('raptor_ours_raw.csv'), load('raptor_ours_sub.csv')
chk('our seeds raw: 0 failed rows (all-0.5), finite', not (r3 == 0.5).all(1).any() and np.isfinite(r3).all())
chk('r384 raw: 0 failed rows (all-0.5)', not (r384 == 0.5).all(1).any(), int((r384 == 0.5).all(1).sum()))
chk('r224 raw: 0 failed rows (all-0.5)', not (r224 == 0.5).all(1).any(), int((r224 == 0.5).all(1).sum()))
chk('raw csvs finite', np.isfinite(r384).all() and np.isfinite(r224).all() and np.isfinite(ours).all())
# each Raptor's raw dump must rank like its own stage submission (ordinal rankpct, so compare orderings with ties allowed)
def same_order(raw, sub):
    for j in range(raw.shape[1]):
        a, c = raw[:, j], sub[:, j]
        for x in range(len(a)):
            for y in range(len(a)):
                if a[x] < a[y] and not c[x] <= c[y]:
                    return False
    return True
chk('r384 raw ranks == r384 stage sub ranks', same_order(r384, s384))
chk('r224 raw ranks == r224 stage sub ranks', same_order(r224, s224))
chk('our seeds raw ranks == their stage sub ranks', same_order(r3, s3))

# ---- independent recomputation ----------------------------------------------------------------------------------------
vH = SH[labels].values.astype(np.float64)
chk('stage-3 output finite in [0,1]', np.isfinite(vH).all() and (vH >= 0).all() and (vH <= 1).all())
exp = np.zeros_like(vH)
for j in range(len(labels)):
    mix = (W384 * avg_rank_pct(r384[:, j]) + W224 * avg_rank_pct(r224[:, j]) + W3O * avg_rank_pct(r3[:, j])) / (W384 + W224 + W3O)
    exp[:, j] = 0.5 * avg_rank_pct(mix) + 0.5 * avg_rank_pct(ours[:, j])
d = float(np.abs(exp - vH).max())
chk('stage-3 output (%s) == independent recompute (max diff < 1e-6)' % ('submission_H.csv' if A.hf_wf is not None else 'submission.csv'),
    d < 1e-6, 'max diff %.3g' % d)
if A.hf_wf is not None:
    fl = load('submission_fleetsolo.csv')
    expF = np.stack([(1 - A.hf_wf) * avg_rank_pct(exp[:, j]) + A.hf_wf * avg_rank_pct(fl[:, j]) for j in range(len(labels))], 1)
    dF = float(np.abs(expF - v).max())
    chk('submission.csv == (1 - wf) p(H recomputed) + wf p(fleet) (max diff < 1e-6)', dF < 1e-6, 'max diff %.3g' % dF)
# informative: how far HR7-RR is from the HR7 rule (r384 alone) on these rows
hr7 = np.stack([0.5 * avg_rank_pct(r384[:, j]) + 0.5 * avg_rank_pct(ours[:, j]) for j in range(len(labels))], 1)
print(f'  info: rows {len(ids)}; stage-3 cells differing from the HR7 rule (r384 alone): {int((np.abs(hr7 - vH) > 1e-6).sum())} of {vH.size}; '
      f'max |HR7rule - H| {float(np.abs(hr7 - vH).max()):.4f}')
if mx.get('spearman_r384_vs_r224'):
    print('  info: spearman r384 vs r224 per label', mx['spearman_r384_vs_r224'])
print(f'RS_CHECK {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}')
sys.exit(0 if all(R) else 1)
