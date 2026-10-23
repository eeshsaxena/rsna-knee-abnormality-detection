# student_gate.py (r7teacher/labels, 2026-10-14): the PRE-REGISTERED student gate of PREREG_TEACHER.md sec 5 (box, CPU).
# Student S = a 5-fold TTR set trained on a teacher label table (labels_r7t_b.csv / _a.csv). References: the D family
# (same TTR224 recipe on frac75: D = s1043, D2 = s1044), the K14 fleet F (C + DT + E + E2, 20 members, the F slot of our
# LB kernels), the R7 teacher file (its gold rows = 4-fold mean, its pool rows = held-out fold model) and nartaa Raptor N
# (r384 SWA, serving geometry) on gold58.
#   G0 integrity : 5/5 folds closed; raw-best ckpt + OOF per fold; OOF ids == folds5 val ids minus gold74; ckpt args =
#                  (fold, seed, 21 ep, ttr 224, train_labels == the table's file name); table md5 == the build receipt
#   G1 gold58    : AUC58(S) - mean(AUC58(D), AUC58(D2)) >= -0.005        (expert labels; 5-member equal mean, pass A)
#   G2 uptake    : rho(S, R7) - rho_ref >= 0.10 x (1 - rho_ref), rho_ref = mean(rho(D, R7), rho(D2, R7)) (the student closes
#                  >= 10 % of the D family's Spearman gap to R7) AND OOF AUC(S vs v4) >= 0.85   (pool rows; sanity)
#   G3 diversity : rho(S, R7) <= rho(D, D2)    (the student stays less similar to R7 than two seeds of our own recipe)
#   G4 blend     : on gold58, base = .5 rk(R7) + .5 rk(N); X_split = .8 rk(base) + .2 rk(.5 F + .5 S) vs
#                  X_F = .8 rk(base) + .2 rk(F): AUC58(X_split) - AUC58(X_F) >= -0.002
#   rho = mean per-label Spearman on the non-gold pool rows (OOF), rk = per-label percentile rank.
#   VERDICT PASS iff G0-G4 all pass. Everything else printed is report only.
# PREREG_TEACHER Amendment A1 (10-14, before any full R7-OOF byte existed; PLAN_TEACHER sec 1):
#   FX4  checkpoint selection on the student's OWN table: the queue lines pass --eval_labels = the table, so G0 requires
#        ckpt args sel_labels == train_labels == the table's file name (was: sel_labels == llm_labels_v4_blend.csv).
#   FX2b the pool-row R7 ruler of G2 / G3 / G3' is the raw fold logit z|<label> (what M(R7) is built from since A1; Effusion /
#        Baker's p| are the all-row stackers' outputs) when the teacher file has z| columns; G4 (the LB proxy) keeps p| (the
#        deployed R7 output). Rank-based legs, so z| and sigmoid(z|) give the same rho on the other 10 labels.
#   FX5  G3' diversity ceiling (gates the LB entry): d = rho(S, R7) - rho_ref <= 0.5 x (1 - rho_ref) (the student may close at
#        most half of the D family's Spearman gap to R7; G2's floor is 0.10 x the same gap, so the two legs are consistent).
#        Reported: d, the gap share d / (1 - rho_ref), rho(S, OAI-free fleet OOF mean), rho(S, K14 OOF mean C+DT+E+E2).
#   VERDICT (A1) PASS iff G0, G1, G2, G3, G3', G4 all pass.
# usage: CUDA_VISIBLE_DEVICES= NTHR=6 taskset -c 34-39 nice -n 19 python student_gate.py --student f{F}_s1047_ttr_r7tb
#        --table ~/r7t_labels/real/labels_r7t_b.csv --receipt ~/r7t_labels/real/teacher_labels_receipt.json
#        --teacher ~/r7t_labels/real/R7OOF.csv [--infer] [--standin] [--out ~/r7t_labels/gate]
import argparse, glob, hashlib, json, os, re, subprocess, sys, time
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr

H = os.path.expanduser('~')
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
G1_BAR, G2_GAP, G2_FLOOR, G4_BAR = -0.005, 0.10, 0.85, -0.002          # PREREG_TEACHER sec 6 (frozen)
G3P_CEIL = 0.5                                                          # A1 FX5 (GUESS threshold, frozen)
REFS = {'D': 's1043_ttr_frac75', 'D2': 's1044_ttr_frac75'}
GOLD_CACHES = [f'{H}/novel/k9gold/preds', f'{H}/novel/k8gold/preds', f'{H}/novel/gold74_preds', f'{H}/k10_test/gold/preds',
               f'{H}/k11_test/gold/preds', f'{H}/k14_test/gold/preds', f'{H}/r7t_labels/gold_preds']
ap = argparse.ArgumentParser()
ap.add_argument('--student', required=True)          # run-dir template with {F}, e.g. f{F}_s1047_ttr_r7tb
ap.add_argument('--table', required=True); ap.add_argument('--receipt', required=True); ap.add_argument('--teacher', required=True)
ap.add_argument('--n_pred', default=f'{H}/raptor_store/run_s42/serve/wts__raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt.npy')
ap.add_argument('--k14_members', default=f'{H}/k14_test/members.json')
ap.add_argument('--runs', default=f'{H}/fleet_runs'); ap.add_argument('--folds', default=f'{H}/rsna_train/folds5.csv')
ap.add_argument('--gold', default=f'{H}/rsna_train/gold74_eval_set.csv'); ap.add_argument('--v4', default=f'{H}/rsna_train/llm_labels_v4_blend.csv')
ap.add_argument('--infer', action='store_true'); ap.add_argument('--pred_dir', default=f'{H}/r7t_labels/gold_preds')
ap.add_argument('--standin', action='store_true')    # dry run: an existing set plays the student (G0 label checks relaxed)
ap.add_argument('--out', default=f'{H}/r7t_labels/gate'); ap.add_argument('--boot', type=int, default=2000)
ap.add_argument('--fleet_oof', default=f'{H}/r7t_labels/fleet/oof_fleetOF_mean.csv')          # A1 FX5 reports
ap.add_argument('--prereg', default=f'{H}/r7t_labels/PREREG_TEACHER.md')                      # A1: code freeze check (REAL)
ap.add_argument('--k14_oof', nargs='*', default=[f'{H}/r7t_labels/fleet/oof_C.csv', f'{H}/r7t_labels/sim/oof_DT.csv',
                                                 f'{H}/r7t_labels/fleet/oof_E.csv', f'{H}/r7t_labels/fleet/oof_E2.csv'])
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True); os.makedirs(A.pred_dir, exist_ok=True)
L = []


def say(s=''):
    print(s, flush=True); L.append(s)


md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
folds = rd(A.folds)['fold']; gold = rd(A.gold); v4 = rd(A.v4)[T]
g58 = [s for s in gold.index if gold.loc[s, 'source'] == 'original_gold58']
gset = set(gold.index); pool = [s for s in folds.index if folds[s] >= 0 and s not in gset]
rec = json.load(open(A.receipt)); te = rd(A.teacher)
if not all(c in te.columns for c in T) and all('p|' + c in te.columns for c in T):      # job-1 layout (p|<label>)
    te = te.rename(columns={'p|' + c: c for c in T})
if all('z|' + c in te.columns for c in T):                                             # A1 FX2b: pool-row ruler = raw logits
    tz = te[['z|' + c for c in T]].astype(float); tz.columns = T; RULER = 'z|'
else:
    tz = te[T].astype(float); RULER = 'p|'
SIM = A.standin or not rec.get('real', False) or str(te.get('teacher_tag', pd.Series(['R7OOF'])).iloc[0]).startswith('SIM')
say(f'student_gate {time.strftime("%Y-%m-%d %H:%M:%S")} | student {A.student} | table {A.table} | teacher {A.teacher} | '
    f'{"SIMULATION (TEST ONLY)" if SIM else "REAL"}')


def run_dir(tpl, F):
    return f'{A.runs}/{tpl.format(F=F)}'


def seed_of(tpl):
    return int(re.search(r'_s(\d+)_', tpl.format(F=0)).group(1))


def members(tpl):
    out = []
    for F in range(5):
        d = run_dir(tpl, F); s = seed_of(tpl)
        ck = glob.glob(f'{d}/f{F}_s{s}_raw_*_best.pt'); oof = f'{d}/oof_f{F}_s{s}_raw.csv'
        out.append(dict(F=F, dir=d, ckpt=ck[0] if len(ck) == 1 else None, oof=oof if os.path.isfile(oof) else None, log=d + '.log'))
    return out


def gold_pred(run):
    return next((f'{d}/{run}__raw__A.csv' for d in GOLD_CACHES if os.path.isfile(f'{d}/{run}__raw__A.csv')), None)


# ---------------- G0 integrity ----------------
S = members(A.student); g0, why = True, []
tab_md5 = md5(A.table); want_md5 = None
for k, v in rec.items():
    if k.startswith('table_') and isinstance(v, dict) and os.path.basename(v.get('path', '')) == os.path.basename(A.table):
        want_md5 = v['md5']
if not A.standin:
    if want_md5 != tab_md5:
        g0 = False; why.append(f'table md5 {tab_md5} != receipt {want_md5}')
for m in S:
    tail = open(m['log']).read().strip().splitlines()[-1] if os.path.isfile(m['log']) else ''
    if not re.match(r'^\{"raw": [^,]+, "ema": [^}]+\}$', tail):
        g0 = False; why.append(f'f{m["F"]} no closing line')
    if not m['ckpt'] or not m['oof']:
        g0 = False; why.append(f'f{m["F"]} missing ckpt/oof'); continue
    o = rd(m['oof'])
    if sorted(o.index) != sorted(s for s in pool if folds[s] == m['F']):
        g0 = False; why.append(f'f{m["F"]} OOF ids != folds5 val ids')
    if not A.standin:
        import torch
        a = torch.load(m['ckpt'], map_location='cpu', weights_only=False)['args']
        ok = (a.get('fold') == m['F'] and a.get('seed') == seed_of(A.student) and a.get('epochs') == 21 and a.get('ttr_size') == 224
              and a.get('train_labels') == os.path.basename(A.table) and a.get('sel_labels') == os.path.basename(A.table))   # A1 FX4
        if not ok:
            g0 = False; why.append(f'f{m["F"]} ckpt args {dict((k, a.get(k)) for k in ("fold", "seed", "epochs", "ttr_size", "train_labels", "sel_labels"))}')
if not A.standin:                                                       # A1: this gate's md5 must be listed in the PREREG
    _me = md5(os.path.abspath(__file__)); _txt = open(A.prereg, encoding='utf-8').read() if os.path.isfile(A.prereg) else ''
    if not ('Amendment A1' in _txt and _me in _txt):
        g0 = False; why.append(f'student_gate.py md5 {_me} not in the PREREG A1 code freeze')
say(f'G0 integrity: {"PASS" if g0 else "FAIL " + "; ".join(why)}')

# ---------------- gold predictions (pass A, CPU) ----------------
need = []
for tpl in [A.student] + [f'f{{F}}_{t}' for t in REFS.values()]:
    for m in members(tpl):
        run = os.path.basename(m['dir'])
        if gold_pred(run) is None and m['ckpt']:
            need.append(m['ckpt'])
if need and A.infer:
    say(f'infer: {len(need)} checkpoint(s) -> gold_infer5.py (CPU, pass A)')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NTHR=os.environ.get('NTHR', '6'))
    rr = subprocess.run([sys.executable, '-W', 'ignore', f'{H}/novel/gold_infer5.py', '--passes', 'A', '--out', A.pred_dir] + need,
                        env=env, capture_output=True, text=True)
    assert rr.returncode == 0, rr.stderr[-2000:]
    need = [c for c in need if gold_pred(os.path.basename(os.path.dirname(c))) is None]
if need:
    say(f'MISSING gold predictions for {len(need)} checkpoint(s): re-run with --infer'); sys.exit(3)


def gold_mean(tpl):
    P = [rd(gold_pred(os.path.basename(m['dir']))).loc[g58, T].to_numpy(np.float64) for m in members(tpl)]
    return np.mean(P, axis=0)


def oof_all(tpl):
    return pd.concat([rd(m['oof'])[T] for m in members(tpl)]).loc[pool]


Y = gold.loc[g58, T].to_numpy(np.float64)


def auc58(P, ii=None):
    from sklearn.metrics import roc_auc_score
    ii = np.arange(len(Y)) if ii is None else ii; a = []
    for j in range(12):
        y = Y[ii, j]; ok = np.isfinite(y)
        if ok.sum() and 0 < (y[ok] >= .5).mean() < 1:
            a.append(roc_auc_score(y[ok] >= .5, P[ii, j][ok]))
    return float(np.mean(a))


rk = lambda P: np.stack([rankdata(P[:, j]) / len(P) for j in range(P.shape[1])], 1)
sp = lambda X, Z: float(np.mean([spearmanr(X[:, j], Z[:, j]).correlation for j in range(12)]))
gS = gold_mean(A.student); gR = {k: gold_mean(f'f{{F}}_{t}') for k, t in REFS.items()}
a58 = {'S': auc58(gS), **{k: auc58(v) for k, v in gR.items()}}
d1 = a58['S'] - np.mean([a58[k] for k in REFS])
g1 = d1 >= G1_BAR
say(f'G1 gold58: S {a58["S"]:.4f} | ' + ' | '.join(f'{k} {a58[k]:.4f}' for k in REFS) + f' | S - mean(D fam) {d1:+.4f} (bar >= {G1_BAR}) -> {"PASS" if g1 else "FAIL"}')

# ---------------- G2 / G3 on the pool OOF ----------------
oS = oof_all(A.student); oR = {k: oof_all(f'f{{F}}_{t}') for k, t in REFS.items()}
pr = [s for s in pool if s in tz.index]
X7 = tz.loc[pr, T].to_numpy(np.float64)                                   # A1 FX2b: z| when present
say(f"pool-row R7 ruler for G2 / G3 / G3': {RULER} ({len(pr)} rows)")
rho = {'S': sp(oS.loc[pr].values, X7), **{k: sp(v.loc[pr].values, X7) for k, v in oR.items()}}
rho_dd = sp(oR['D'].loc[pr].values, oR['D2'].loc[pr].values)
from sklearn.metrics import roc_auc_score
Yv = v4.loc[pool].values >= .5
auc_v4 = {k: float(np.mean([roc_auc_score(Yv[:, j], v.loc[pool].values[:, j]) for j in range(12)])) for k, v in (('S', oS), *oR.items())}
rho_ref = float(np.mean([rho[k] for k in REFS])); d2 = rho['S'] - rho_ref; bar2 = G2_GAP * (1 - rho_ref)
g2 = d2 >= bar2 and auc_v4['S'] >= G2_FLOOR
say(f'G2 uptake: rho(S,R7) {rho["S"]:.4f} | ' + ' | '.join(f'rho({k},R7) {rho[k]:.4f}' for k in REFS) + f' | delta {d2:+.4f} (bar >= +{bar2:.4f} = {G2_GAP} x (1 - {rho_ref:.4f}))'
    f' | OOF AUC vs v4: ' + ' '.join(f'{k} {v:.4f}' for k, v in auc_v4.items()) + f' (S floor {G2_FLOOR}) -> {"PASS" if g2 else "FAIL"}')
g3 = rho['S'] <= rho_dd
say(f'G3 diversity: rho(S,R7) {rho["S"]:.4f} <= rho(D,D2) {rho_dd:.4f} -> {"PASS" if g3 else "FAIL"}')
ceil3 = G3P_CEIL * (1 - rho_ref); g3p = d2 <= ceil3                       # A1 FX5
fl_rho = {}
if os.path.isfile(A.fleet_oof):
    fo = rd(A.fleet_oof)[T]
    fl_rho['fleetOF'] = {k: round(sp(v.loc[pr].values, fo.loc[pr].values), 4) for k, v in (('S', oS), *oR.items())}
if A.k14_oof and all(os.path.isfile(p) for p in A.k14_oof):
    ko = sum(rd(p)[T].loc[pool].to_numpy(np.float64) for p in A.k14_oof) / len(A.k14_oof)
    ko = pd.DataFrame(ko, index=pool, columns=T)
    fl_rho['K14'] = {k: round(sp(v.loc[pr].values, ko.loc[pr].values), 4) for k, v in (('S', oS), *oR.items())}
say(f"G3' diversity ceiling: d = rho(S,R7) - rho_ref {d2:+.4f} <= {G3P_CEIL} x (1 - {rho_ref:.4f}) = {ceil3:.4f} "
    f"(gap share {d2 / max(1e-9, 1 - rho_ref):+.3f}; G2 floor {bar2:.4f}) -> {'PASS' if g3p else 'FAIL (skip the LB entry)'}")
say('   report: rho vs fleet OOF (pool rows): ' + ' | '.join(f'{n}: ' + ' '.join(f'{k} {v}' for k, v in r.items()) for n, r in fl_rho.items()))

# ---------------- G4 gold58 blend proxy ----------------
sys.path.insert(0, f'{H}/k14_test/k14/box')
import k14_gate as kg  # noqa: E402 (k14 helpers: members.json + gold pred paths of the 20 K14 members)
M = json.load(open(A.k14_members))['members']
Fm = [kg.rd(kg.pred_path(r, f'{H}/k14_test/gold/preds')).loc[g58, T].to_numpy(np.float64) for r in M if r['set'] in kg.FLEETS['R']]
assert len(Fm) == 20, len(Fm)
gF = np.mean(Fm, axis=0)
N = np.load(A.n_pred); assert N.shape == (58, 12)
assert all(s in te.index for s in g58), 'teacher file lacks gold58 rows (the R7-OOF run must score the 58 gold studies)'
g7 = te.loc[g58, T].to_numpy(np.float64)
base = .5 * rk(g7) + .5 * rk(N)
X_F = .8 * rk(base) + .2 * rk(gF)
X_sp = .8 * rk(base) + .2 * rk(.5 * gF + .5 * gS)
X_rp = .8 * rk(base) + .2 * rk(gS)
rng = np.random.default_rng(61); Bs = [rng.integers(0, 58, 58) for _ in range(A.boot)]
res4 = {}
for nm, X in (('split', X_sp), ('replace', X_rp)):
    d = auc58(X) - auc58(X_F); bb = np.array([auc58(X, b) - auc58(X_F, b) for b in Bs])
    res4[nm] = dict(auc=auc58(X), d=d, se=float(np.nanstd(bb)), p_gt0=float(np.nanmean(bb > 0)))
g4 = res4['split']['d'] >= G4_BAR
say(f'G4 blend (gold58): base(R7 .5 + N .5) {auc58(base):.4f} | +F .2 {auc58(X_F):.4f} | +split(F,S) .2 {res4["split"]["auc"]:.4f} '
    f'd {res4["split"]["d"]:+.4f} (SE {res4["split"]["se"]:.4f}, P>0 {res4["split"]["p_gt0"]:.2f}; bar >= {G4_BAR}) -> {"PASS" if g4 else "FAIL"}')
say(f'   report: +replace(S) .2 {res4["replace"]["auc"]:.4f} d {res4["replace"]["d"]:+.4f} (SE {res4["replace"]["se"]:.4f}) | '
    f'R7 alone {auc58(g7):.4f} | N alone {auc58(N):.4f} | F alone {auc58(gF):.4f} | S alone {a58["S"]:.4f}')
pl = {}
for j, c in enumerate(T):
    y = Y[:, j]; ok = np.isfinite(y)
    if 0 < (y[ok] >= .5).mean() < 1:
        from sklearn.metrics import roc_auc_score as ra
        pl[c] = round(float(ra(y[ok] >= .5, gS[ok, j]) - np.mean([ra(y[ok] >= .5, gR[k][ok, j]) for k in REFS])), 4)
say(f'   report: per-label gold58 S - mean(D fam): {pl}')
say(f'   report: rho gold58 S~R7 {sp(gS, g7):.4f} | S~F {sp(gS, gF):.4f} | S~N {sp(gS, N):.4f} | F~R7 {sp(gF, g7):.4f}')
ok = g0 and g1 and g2 and g3 and g3p and g4
tag = 'SIMULATION (TEST ONLY)' if SIM else 'REAL'
say(f'R7T_STUDENT_GATE {tag} A1 | G0 {"P" if g0 else "F"} G1 {"P" if g1 else "F"} G2 {"P" if g2 else "F"} G3 {"P" if g3 else "F"} '
    + ("G3' " + ("P" if g3p else "F")) + f' G4 {"P" if g4 else "F"} | VERDICT {"PASS" if ok else "FAIL"}{" (not a real verdict)" if SIM else ""}')
p = f'{A.out}/student_gate_{"SIM_" if SIM else ""}{time.strftime("%Y%m%d_%H%M")}.txt'
open(p, 'w').write('\n'.join(L) + '\n' + json.dumps(dict(a58=a58, rho=rho, rho_dd=rho_dd, auc_v4=auc_v4, g4=res4, per_label=pl, g3p=dict(d=d2, ceil=ceil3, gap_share=d2 / max(1e-9, 1 - rho_ref)),
                                                       fleet_rho=fl_rho, ruler=RULER, amendment='A1',
                                                       verdict='PASS' if ok else 'FAIL', sim=SIM), indent=1) + '\n')
print(f'wrote {p}')
