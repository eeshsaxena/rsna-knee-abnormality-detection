#!/usr/bin/env python3
"""lab2_decomp.py (jump2 label track, 2026-10-05). CPU only, read-only inputs, writes ~/jump2/lab2_decomp.{txt,json}.
Q: how much of steven_v4's gold58 AUC deficit is READING error (fixable by a better report extractor) vs
REPORT-IMAGE disagreement (the report itself says something else than the radiologist image label; not fixable by any reader)?
Readers (5 independent LLM reads of the same report): steven_v4, steven_full, pilkwang verdict, lixin gpt56, jev_final2.
State per cell: U+ = all 5 readers say present, U- = none says present, C = contested (1..4 of 5).
U- is split into NEG (pilk verdict NO = report explicitly negates) and SIL (pilk UNK = report silent).
Pair attribution: every misordered (pos,neg) pair of a scorer is charged to the state pair (state_pos, state_neg).
Oracle bounds: contested cells resolved perfectly (to gold) while unanimous cells keep the scorer's value.
"""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
ID = 'StudyInstanceUID'; H = '/home/training'; RT = f'{H}/rsna_train'; LI = f'{H}/jump2/labels_in'; GP = f'{H}/jump2/gold74_preds'
tr = pd.read_csv(f'{RT}/train.csv')
gold = tr[tr[L].notna().any(axis=1)].set_index(ID)[L].astype(int); g = gold.index; assert len(g) == 58
def rd(p): return pd.read_csv(p).drop_duplicates(ID).set_index(ID).reindex(g)
v4 = rd(f'{RT}/llm_labels_v4_blend.csv')[L].astype(float)
sf = rd(f'{LI}/steven_full.csv')[L].astype(float)
pkr = rd(f'{RT}/report_labels_v2.csv')
pkv = pd.DataFrame({l: pkr[l + '__verdict'].fillna('UNK').values for l in L}, index=g)
pks = pkv.replace({'YES': 1.0, 'UNK': 0.5, 'NO': 0.0}).astype(float)
pkval = pkr[L].astype(float).fillna(pkr[L].astype(float).median())
lxr = rd(f'{LI}/lixin_gpt56_conf.csv')
lxb = lxr[L].astype(float)
lxs = pd.DataFrame({l: np.where(lxr[l] >= .5, .5 + .5 * lxr[l + '__conf'], .5 - .5 * lxr[l + '__conf']) for l in L}, index=g)
jv = rd(f'{LI}/jev_final2_scores.csv')[L].astype(float)
mix = rd(f'{RT}/labels_mix50_logit.csv')[L].astype(float)
gat = rd(f'{RT}/labels_gated75_logit.csv')[L].astype(float)
wd = rd(f'{RT}/teacher_train_probs.csv')[L].astype(float)
rp = rd(f'{LI}/T_raptor_proxy.csv')[L].astype(float)
own = {k: rd(f'{GP}/{f}')[L].astype(float) for k, f in
       [('own_F1mix50_f0', 'f0_s42__raw.csv'), ('own_TTRmix50_f0', 'f0_s1043_ttr_mix50__raw.csv'),
        ('own_TTRv4_f0', 'f0_s1042_ttr__ema.csv'), ('own_F1v4_f0', 'f0_s42_v4__raw.csv')]}

def aucs(S): return np.array([roc_auc_score(gold[l].values, S[l].values) for l in L])
out = {}; lines = []
def P(s=''): print(s); lines.append(s)

# A. per-label table
TAB = {'steven_v4': v4, 'steven_v4_bin(>.5)': (v4 > .5).astype(float), 'steven_full': sf,
       'pilk_verdict(Y1/U.5/N0)': pks, 'pilk_value': pkval, 'lixin_signedconf': lxs, 'lixin_bin': lxb,
       'jev_final2': jv, 'mix50_logit': mix, 'gated75_logit': gat, 'teacher_wd(img)': wd, 'raptor_proxy(img)': rp, **own}
P('A. gold58 AUC per label (MEASURED, label value as score vs train.csv gold image labels, n=58)')
P(f"{'scorer':26s} macro  | " + ' '.join(f'{l[:5]:>5s}' for l in L))
for k, S in TAB.items():
    a = aucs(S); out[k] = {'macro': float(a.mean()), 'per_label': [round(float(x), 4) for x in a]}
    P(f'{k:26s} {a.mean():.4f} | ' + ' '.join(f'{x:5.3f}' for x in a))
P(f"steven_v4 distinct values on gold58: {len(np.unique(v4.values.round(4)))}; levels per label median "
  f"{int(np.median([v4[l].round(4).nunique() for l in L]))}")

# B. reader states
votes = ((v4 > .5).astype(int) + (sf > .5).astype(int) + (pkv == 'YES').astype(int) + (lxb >= .5).astype(int)
         + (jv > .5).astype(int))
state = pd.DataFrame('C', index=g, columns=L)
state[votes == 5] = 'U+'
state[(votes == 0) & (pkv == 'NO')] = 'NEG'
state[(votes == 0) & (pkv != 'NO')] = 'SIL'
P('\nB. reader-state x gold counts per label (5 readers; U+ all say present; NEG none + pilk explicit NO; SIL none + pilk UNK; C contested)')
P(f"{'label':17s} " + ' '.join(f'{s:>9s}' for s in ['U+ g1/g0', 'C g1/g0', 'NEG g1/g0', 'SIL g1/g0']))
tot = {s: [0, 0] for s in ['U+', 'C', 'NEG', 'SIL']}; cnt = {}
for l in L:
    row = f'{l:17s} '
    for s in ['U+', 'C', 'NEG', 'SIL']:
        m = state[l] == s; a1 = int(gold[l][m].sum()); a0 = int(m.sum() - a1); tot[s][0] += a1; tot[s][1] += a0
        row += f'{a1:4d}/{a0:<4d} '; cnt[(l, s)] = (a1, a0)
    P(row)
P('TOTAL             ' + ' '.join(f'{tot[s][0]:4d}/{tot[s][1]:<4d}' for s in ['U+', 'C', 'NEG', 'SIL']))
n_cells = 58 * 12
gap_fp = tot['U+'][1]; gap_fn = tot['NEG'][0] + tot['SIL'][0]
P(f'unanimous-reader cells that disagree with gold (report-image gap, not reader-fixable): U+&g0 {gap_fp}, NEG&g1 {tot["NEG"][0]}, '
  f'SIL&g1 {tot["SIL"][0]} = {gap_fp + gap_fn}/{n_cells} cells; contested cells {tot["C"][0] + tot["C"][1]}')
out['states_total'] = tot

# C. pair attribution of the AUC deficit
def pair_attr(S):
    res = {}; deficit_total = 0.0; npairs_total = 0
    for l in L:
        y = gold[l].values; s = S[l].values; st = state[l].values
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
        for i in pos:
            for j in neg:
                w = 1.0 if s[i] < s[j] else (0.5 if s[i] == s[j] else 0.0)
                npairs_total += 1
                if w:
                    key = 'both_unanimous' if (st[i] != 'C' and st[j] != 'C') else 'involves_contested'
                    key2 = f'{st[i]}>{st[j]}'
                    res[key] = res.get(key, 0) + w / (len(pos) * len(neg)) / 12
                    res[key2] = res.get(key2, 0) + w / (len(pos) * len(neg)) / 12
                    deficit_total += w / (len(pos) * len(neg)) / 12
    return deficit_total, res
P('\nC. macro-AUC deficit (1 - AUC) attributed to (state of gold-positive cell > state of gold-negative cell)')
for k in ['steven_v4', 'pilk_verdict(Y1/U.5/N0)', 'mix50_logit', 'teacher_wd(img)', 'own_F1mix50_f0']:
    d, r = pair_attr(TAB[k])
    top = sorted([(kk, vv) for kk, vv in r.items() if '>' in kk], key=lambda t: -t[1])[:6]
    P(f'{k:26s} deficit {d:.4f} | both-unanimous {r.get("both_unanimous", 0):.4f} | involves-contested {r.get("involves_contested", 0):.4f} | top: '
      + ', '.join(f'{kk} {vv:.4f}' for kk, vv in top))
    out[f'deficit_{k}'] = {'total': d, **{kk: round(vv, 5) for kk, vv in r.items()}}

# D. oracle bounds for a better READER
def oracle(S, mode):
    O = S.copy()
    for l in L:
        m = state[l] == 'C'
        if mode == 'contested_to_gold':
            O.loc[m, l] = np.where(gold[l][m] == 1, 1.0 + 1e-3, -1e-3)  # perfect resolution of contested cells
        elif mode == 'contested_to_majority':
            O.loc[m, l] = np.where(votes[l][m] >= 3, 0.95, 0.05)
        elif mode == 'contested_to_votefrac':
            O.loc[m, l] = votes[l][m] / 5.0
    return O
P('\nD. reader oracles on gold58 (MODEL: bounds, contested cells only; unanimous cells keep the scorer value)')
for base in ['steven_v4', 'mix50_logit']:
    for mode in ['contested_to_gold', 'contested_to_majority', 'contested_to_votefrac']:
        a = aucs(oracle(TAB[base], mode)); out[f'oracle_{base}_{mode}'] = float(a.mean())
        P(f'{base:12s} {mode:24s} {a.mean():.4f} (base {out[base]["macro"]:.4f}) | ' + ' '.join(f'{x:5.3f}' for x in a))
vf = votes / 5.0
a = aucs(vf); P(f'vote fraction of 5 readers as score         {a.mean():.4f} | ' + ' '.join(f'{x:5.3f}' for x in a)); out['votefrac5'] = float(a.mean())
a = aucs(vf + 0.01 * v4.rank(pct=True)); P(f'vote fraction + v4 tiebreak                 {a.mean():.4f} | ' + ' '.join(f'{x:5.3f}' for x in a)); out['votefrac5_v4tb'] = float(a.mean())

# E. how the image teacher behaves inside each reader state (is the gap image-fixable?)
P('\nE. within-state AUC (pooled over labels, cells with both classes) : v4 vs mix50 vs teacher_wd vs own_F1mix50')
for s in ['U+', 'C', 'NEG', 'SIL']:
    ys, sc = [], {k: [] for k in ['steven_v4', 'mix50_logit', 'teacher_wd(img)', 'own_F1mix50_f0', 'raptor_proxy(img)']}
    per = {k: [] for k in sc}
    for l in L:
        m = (state[l] == s).values; y = gold[l].values[m]
        if 0 < y.sum() < len(y):
            for k in sc: per[k].append(roc_auc_score(y, TAB[k][l].values[m]))
    P(f'{s:4s} labels-with-both-classes {len(per["steven_v4"]):2d} | ' + ' '.join(f'{k} {np.mean(v):.3f}' for k, v in per.items() if v))

# F. error list for manual reading
rows = []
for l in L:
    for u in g:
        y = gold.loc[u, l]; s = v4.loc[u, l]
        if (s > .5) != (y == 1):
            rows.append(dict(uid=u, uid6=u[-6:], label=l, gold=int(y), v4=round(s, 3), votes=int(votes.loc[u, l]),
                             pilk=pkv.loc[u, l], state=state.loc[u, l], wd=round(wd.loc[u, l], 2), mix=round(mix.loc[u, l], 2)))
E = pd.DataFrame(rows); E.to_csv(f'{H}/jump2/lab2_v4err.csv', index=False)
P(f'\nF. steven_v4 thresholded at .5 disagrees with gold on {len(E)}/696 cells; by state ' + str(E.state.value_counts().to_dict())
  + '; by label ' + str(E.label.value_counts().to_dict()))
json.dump(out, open(f'{H}/jump2/lab2_decomp.json', 'w'), indent=1)
open(f'{H}/jump2/lab2_decomp.txt', 'w').write('\n'.join(lines) + '\n')
