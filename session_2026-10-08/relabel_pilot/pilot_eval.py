#!/usr/bin/env python3
"""pilot_eval.py (gold_prep/relabel, 2026-10-06): the PRE-REGISTERED evaluation of the multimodal relabel pilot.

Inputs: results/soft_text_<model>.csv and results/soft_img_<model>.csv from relabel_call.py (same model, same prompt
sha), gold labels (original_gold58 rows of gold74_eval_set.csv == the 58 expert rows of train.csv), and for context the
training label files labels_mix50_logit.csv (what the fleet trains on) and llm_labels_v4_blend.csv (text only).
Score per finding = the model's probability / 100. AUC per label on the gold58 studies parsed OK in BOTH arms.

GATE (fixed 2026-10-06 before any API call; constants below, mirrored in prereg_v1.json):
  V  validity : both arms parsed for >= 55 of the 58 gold58 studies (paired set), same prompt sha, same model.
                A study whose answer in either arm came from a server-side refusal fallback model (served_model !=
                model_requested) is dropped from the paired set like a failed parse (added by the 10-06 verify).
  G1 primary  : AUC_img(Effusion) - AUC_text(Effusion) >= +0.040.
  G2 no harm  : for each of Synovitis, Fracture, Contusion, PF OA, Lateral OA: AUC_img - AUC_text >= -0.020
                (n = 58 with 11-27 positives: one paired SE is ~.02-.04, so "not lower" is read with a .02 tolerance),
                AND the mean of those 5 deltas >= 0.
  PASS = V and G1 and G2  ->  images help the VLM read the under-reported labels; the pipeline works.
  S  value test (decides whether a scale-up batch is worth paying for): blend = sigmoid(0.5 logit(mix50) +
     0.5 logit(clip(p_img, .02, .98))) on the 6 targets; mean over the 6 of AUC(blend) - AUC(mix50) >= +0.010. Context: mix50 already
     reads Effusion .984 on gold58 (its image teacher), so the headroom is Synovitis .823, Lateral OA .843, Fracture
     .843, PF OA .888, Contusion .906 (labels.md 10-05, MEASURED).
  Paired bootstrap (2,000 resamples of the gold58 studies, seed 0) 90% intervals are printed for every delta; they are
  information, not part of the gate.

  python pilot_eval.py [--model opus-5-5] [--text results/soft_text_opus-5-5.csv --img results/soft_img_opus-5-5.csv]

GATE AMENDMENT v2 (registered 2026-10-08 before any live call; prereg_v2.json; NOVEL_PLAN_10-07 item 5, Eesh OK 10-08):
  python pilot_eval.py --prereg prereg_v2.json     (without --prereg the v1 gate above runs unchanged)
  V  validity : unchanged (thresholds read from the prereg json).
  G1 sign     : AUC_img(Effusion) - AUC_text(Effusion) >= 0 (v1: a hard +0.040 veto, ~1 SE; P(false veto) ~.14).
  S  decision : unchanged, mean6 AUC(blend) - AUC(mix50) >= +0.010 decides the spend.
  secondary   : mean over the 5 non-Effusion targets of AUC_img - AUC_text, with paired-bootstrap SE and 90% CI.
                v1 G2 per-label tolerances are printed as information only.
  DECISION v2 : INVALID if not V; GO (scale-up worth paying for; spend still needs Eesh's separate OK) if G1 sign and S;
                else NO-GO. The v1 verdict is printed next to it for transparency.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
H = os.path.expanduser
TARGETS = ['Effusion', 'Synovitis', 'Fracture', 'Contusion', 'PF OA', 'Lateral OA']
OTHERS = TARGETS[1:]
G1_MIN, G2_TOL, G2_MEAN_MIN, S_MIN, N_MIN = 0.040, -0.020, 0.0, 0.010, 55


def first_existing(*c):
    for x in c:
        if x and os.path.exists(H(x)):
            return H(x)
    return H(c[0])


ap = argparse.ArgumentParser()
ap.add_argument('--model', default='opus-5-5', help='file tag used by relabel_call.py (model id without "claude-")')
ap.add_argument('--text', default=None)
ap.add_argument('--img', default=None)
ap.add_argument('--gold', default=None)
ap.add_argument('--mix50', default=None)
ap.add_argument('--v4', default=None)
ap.add_argument('--boot', type=int, default=2000)
ap.add_argument('--out', default=None)
ap.add_argument('--prereg', default=None, help='prereg json with gate_version 2 -> the amended gate (default: v1 gate)')
A = ap.parse_args()
PR2, PR2_SHA = None, None
if A.prereg:
    import hashlib
    _raw = open(H(A.prereg), 'rb').read()
    PR2_SHA = hashlib.sha256(_raw).hexdigest()
    PR2 = json.loads(_raw)
    assert PR2.get('gate_version') == 2, f'{A.prereg} is not a gate_version 2 prereg'
    _th = PR2['gate']['thresholds']
    N_MIN = int(_th['n_min_paired'])
    S_MIN = float(_th['s_min'])
    G1_SIGN_MIN = float(_th['g1_sign_min'])
    sys.path.insert(0, HERE)
    import prompt_v1 as _P
    assert PR2['prompt_sha256'] == _P.prompt_sha(), 'prompt_v1.py sha differs from the v2 prereg'
RD = os.path.join(HERE, 'results')
TXT = A.text or os.path.join(RD, f'soft_text_{A.model}.csv')
IMG = A.img or os.path.join(RD, f'soft_img_{A.model}.csv')
DS = '~/Downloads/rsna-knee/session_2026-10-05/kaggle_train/ds/rsna-knee-fleet-train'
GOLD = first_existing(A.gold, '~/rsna_train/gold74_eval_set.csv', f'{DS}/gold74_eval_set.csv')
MIX = first_existing(A.mix50, '~/rsna_train/labels_mix50_logit.csv', f'{DS}/labels_mix50_logit.csv')
V4 = first_existing(A.v4, '~/rsna_train/llm_labels_v4_blend.csv', f'{DS}/llm_labels_v4_blend.csv')


def rd(p):
    return pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')


gold = rd(GOLD)
g58 = gold[gold.source == 'original_gold58']
t, im = rd(TXT), rd(IMG)
for nm, d in (('text', t), ('img', im)):
    assert d.arm.eq(nm).all(), f'{nm} file has rows of another arm'
sha = set(t.prompt_sha.dropna()) | set(im.prompt_sha.dropna())
mods = set(t.model_requested.dropna()) | set(im.model_requested.dropna())
effs = (set(t['effort'].dropna()) | set(im['effort'].dropna())) if 'effort' in t and 'effort' in im else set()
both = [s for s in g58.index if s in t.index and s in im.index and t.at[s, 'status'] == 'ok' and im.at[s, 'status'] == 'ok']
served = pd.concat([t.loc[both, 'served_model'], im.loc[both, 'served_model']]).value_counts().to_dict()


def same_model(d, s):
    """False when a sync row was answered by a server-side refusal fallback model (served != requested). The gate
    compares ONE model across the arms, so such a study is treated like a failed parse (verify 10-06)."""
    sm, rq = d.at[s, 'served_model'], d.at[s, 'model_requested']
    return isinstance(sm, str) and isinstance(rq, str) and sm.startswith(rq)


ok = [s for s in both if same_model(t, s) and same_model(im, s)]
n_fb = len(both) - len(ok)
valid = len(ok) >= N_MIN and len(sha) == 1 and len(mods) == 1 and len(effs) <= 1
print(f'paired gold58 studies parsed OK in both arms: {len(ok)}/58 ({n_fb} more dropped: answered by a fallback model) '
      f'| prompt sha {sha} | model {mods} | effort {effs or "-"} | served {served}')
print(f'status text {t.status.value_counts().to_dict()} | img {im.status.value_counts().to_dict()}')

Y = (g58.loc[ok, TARGETS] >= 0.5).to_numpy()
mix = rd(MIX).loc[ok, TARGETS].to_numpy(float)
v4 = rd(V4).loc[ok, TARGETS].to_numpy(float)
PT, PI = t.loc[ok, TARGETS].to_numpy(float), im.loc[ok, TARGETS].to_numpy(float)
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
BL = 1 / (1 + np.exp(-(0.5 * lg(mix) + 0.5 * lg(np.clip(PI, 0.02, 0.98)))))   # a VLM 0 / 100 counts as 2% / 98%


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan


def aucs(P, idx=None):
    idx = np.arange(len(Y)) if idx is None else idx
    return np.array([auc(Y[idx, j], P[idx, j]) for j in range(len(TARGETS))])


base = {k: aucs(P) for k, P in (('text', PT), ('img', PI), ('mix50', mix), ('v4', v4), ('blend', BL))}
rng = np.random.default_rng(0)
bt = {'img-text': [], 'blend-mix50': []}
for _ in range(A.boot):
    i = rng.integers(0, len(Y), len(Y))
    bt['img-text'].append(aucs(PI, i) - aucs(PT, i)); bt['blend-mix50'].append(aucs(BL, i) - aucs(mix, i))
ci = {k: (np.nanpercentile(np.array(v), 5, 0), np.nanpercentile(np.array(v), 95, 0)) for k, v in bt.items()}
# v2 summaries over the same resamples: per-label SE, the 5-target non-Effusion img-text mean and S (mean6 blend-mix50)
B_it, B_bm = np.array(bt['img-text']), np.array(bt['blend-mix50'])
se = {'img-text': np.nanstd(B_it, 0, ddof=1), 'blend-mix50': np.nanstd(B_bm, 0, ddof=1)}
B_m5, B_s = np.nanmean(B_it[:, 1:], 1), np.nanmean(B_bm, 1)
summ = {'mean5_img_text': (np.nanstd(B_m5, ddof=1), np.nanpercentile(B_m5, 5), np.nanpercentile(B_m5, 95)),
        'S_mean6_blend_mix50': (np.nanstd(B_s, ddof=1), np.nanpercentile(B_s, 5), np.nanpercentile(B_s, 95))}

d_it = base['img'] - base['text']; d_bm = base['blend'] - base['mix50']
print(f'\n{"label":<11}{"n+":>4}{"text":>7}{"img":>7}{"img-text":>10}{"90% CI":>17}{"mix50":>8}{"v4":>7}{"blend":>7}'
      f'{"bl-mix":>8}{"90% CI":>17}')
for j, lab in enumerate(TARGETS):
    # no same-quote nested f-strings: they need Python >= 3.12 and the box rsna env is 3.11 (verify 10-06)
    ci_it = '[{:+.3f},{:+.3f}]'.format(ci['img-text'][0][j], ci['img-text'][1][j])
    ci_bm = '[{:+.3f},{:+.3f}]'.format(ci['blend-mix50'][0][j], ci['blend-mix50'][1][j])
    print(f'{lab:<11}{int(Y[:, j].sum()):>4}{base["text"][j]:>7.3f}{base["img"][j]:>7.3f}{d_it[j]:>+10.3f}'
          f'{ci_it:>17}{base["mix50"][j]:>8.3f}{base["v4"][j]:>7.3f}'
          f'{base["blend"][j]:>7.3f}{d_bm[j]:>+8.3f}{ci_bm:>17}')
print(f'{"mean6":<11}{"":>4}{np.nanmean(base["text"]):>7.3f}{np.nanmean(base["img"]):>7.3f}{np.nanmean(d_it):>+10.3f}'
      f'{"":>17}{np.nanmean(base["mix50"]):>8.3f}{np.nanmean(base["v4"]):>7.3f}{np.nanmean(base["blend"]):>7.3f}'
      f'{np.nanmean(d_bm):>+8.3f}')

g1 = d_it[0] >= G1_MIN
d_oth = d_it[1:]
g2 = bool(np.all(d_oth >= G2_TOL) and np.nanmean(d_oth) >= G2_MEAN_MIN)
s_ok = np.nanmean(d_bm) >= S_MIN
verdict = 'PASS' if (valid and g1 and g2) else ('INVALID' if not valid else 'FAIL')
print(f'\nV  validity  {"ok" if valid else "NOT OK"} ({len(ok)} >= {N_MIN} paired; required: 1 sha, 1 model, 1 effort; '
      f'got {len(sha)}, {len(mods)}, {len(effs)} (0 = not recorded))')
print(f'G1 Effusion img-text {d_it[0]:+.3f} >= {G1_MIN:+.3f}: {"pass" if g1 else "fail"}')
print(f'G2 others img-text {dict(zip(OTHERS, np.round(d_oth, 3)))} each >= {G2_TOL:+.3f} and mean '
      f'{np.nanmean(d_oth):+.3f} >= 0: {"pass" if g2 else "fail"}')
print(f'GATE: {verdict}')
print(f'S  value test: mean6 AUC(blend) - AUC(mix50) {np.nanmean(d_bm):+.4f} >= {S_MIN:+.3f}: '
      f'{"NO, the VLM labels do not add to mix50 on gold58" if not s_ok else "YES, scale-up worth paying for" if verdict == "PASS" else "yes, but the GATE is " + verdict + ", so no scale-up (RUNBOOK: PASS and S)"}')
v2 = None
if PR2 is not None:
    m_ok = mods == {PR2['model']}
    valid2 = valid and m_ok
    g1s = bool(d_it[0] >= G1_SIGN_MIN)
    s2 = bool(np.nanmean(d_bm) >= S_MIN)
    v2verdict = 'INVALID' if not valid2 else ('GO' if (g1s and s2) else 'NO-GO')
    m5 = float(np.nanmean(d_it[1:]))
    print(f'\n=== GATE v2 (prereg {os.path.basename(A.prereg)} sha256 {PR2_SHA[:16]}..., written {PR2.get("written")}) ===')
    print(f'V  validity  {"ok" if valid2 else "NOT OK"} ({len(ok)} >= {N_MIN} paired; 1 sha, 1 model = {PR2["model"]}: '
          f'{"yes" if m_ok else "NO, got " + str(mods)})')
    print(f'G1 sign      Effusion img-text {d_it[0]:+.4f} (SE {se["img-text"][0]:.4f}, 90% CI '
          f'[{ci["img-text"][0][0]:+.4f},{ci["img-text"][1][0]:+.4f}]) >= {G1_SIGN_MIN:+.3f}: {"pass" if g1s else "fail"}')
    print(f'S  decision  mean6 AUC(blend) - AUC(mix50) {np.nanmean(d_bm):+.4f} (SE {summ["S_mean6_blend_mix50"][0]:.4f}, '
          f'90% CI [{summ["S_mean6_blend_mix50"][1]:+.4f},{summ["S_mean6_blend_mix50"][2]:+.4f}]) >= {S_MIN:+.3f}: '
          f'{"pass" if s2 else "fail"}')
    print(f'secondary    5-target non-Effusion mean img-text {m5:+.4f} (SE {summ["mean5_img_text"][0]:.4f}, 90% CI '
          f'[{summ["mean5_img_text"][1]:+.4f},{summ["mean5_img_text"][2]:+.4f}])')
    print('info         v1 G2 per-label tolerance (>= {:+.3f}): {}'.format(
        G2_TOL, {lab: ('ok' if d >= G2_TOL else 'below') for lab, d in zip(OTHERS, d_it[1:])}))
    print('per-label SE img-text  {}'.format(dict(zip(TARGETS, np.round(se['img-text'], 4).tolist()))))
    print('per-label SE blend-mix {}'.format(dict(zip(TARGETS, np.round(se['blend-mix50'], 4).tolist()))))
    print(f'DECISION v2: {v2verdict}'
          + (' (scale-up worth paying for; the spend still needs Eesh\'s separate OK)' if v2verdict == 'GO' else '')
          + f'   [v1 gate for reference: {verdict}]')
    v2 = dict(prereg=os.path.basename(A.prereg), prereg_sha256=PR2_SHA, verdict=v2verdict, valid=bool(valid2),
              g1_sign=g1s, s=s2, S=round(float(np.nanmean(d_bm)), 5), S_se=round(float(summ['S_mean6_blend_mix50'][0]), 5),
              S_ci90=[round(float(x), 5) for x in summ['S_mean6_blend_mix50'][1:]],
              g1_delta=round(float(d_it[0]), 5), g1_se=round(float(se['img-text'][0]), 5),
              g1_ci90=[round(float(ci['img-text'][0][0]), 5), round(float(ci['img-text'][1][0]), 5)],
              mean5_img_text=round(m5, 5), mean5_se=round(float(summ['mean5_img_text'][0]), 5),
              mean5_ci90=[round(float(x), 5) for x in summ['mean5_img_text'][1:]],
              se_img_text=dict(zip(TARGETS, np.round(se['img-text'], 5).tolist())),
              se_blend_mix50=dict(zip(TARGETS, np.round(se['blend-mix50'], 5).tolist())),
              ci90_img_text={lab: [round(float(ci['img-text'][0][j]), 5), round(float(ci['img-text'][1][j]), 5)]
                             for j, lab in enumerate(TARGETS)},
              ci90_blend_mix50={lab: [round(float(ci['blend-mix50'][0][j]), 5), round(float(ci['blend-mix50'][1][j]), 5)]
                                for j, lab in enumerate(TARGETS)},
              v1_g2_info=bool(g2))
out = A.out or os.path.join(RD, f'pilot_eval_{A.model}{"_v2" if PR2 is not None else ""}.json')
json.dump(dict(n_paired=len(ok), n_dropped_fallback=n_fb, served=served, verdict=verdict, g1=bool(g1), g2=g2, s=bool(s_ok),
               auc={k: dict(zip(TARGETS, np.round(v, 4).tolist())) for k, v in base.items()},
               d_img_text=dict(zip(TARGETS, np.round(d_it, 4).tolist())),
               d_blend_mix50=dict(zip(TARGETS, np.round(d_bm, 4).tolist())), gate_v2=v2), open(out, 'w'), indent=1)
print(f'-> {out}')
