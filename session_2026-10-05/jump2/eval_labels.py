#!/usr/bin/env python3
"""jump2 label lever, step 1: score every 4407x12 label set against gold58 (train.csv image labels).
CPU only, read-only on ~/rsna_train. Writes ~/jump2/label_eval.json + label_eval.txt.
Metric: AUC of the label value used as a score vs gold (per label, macro over 12).
Also: paired study-level bootstrap vs steven_v4 and vs mix50_logit, leak check (gold copied in),
and 'reading fidelity' on the 16 extra gold74 studies (report-read labels, NaN cells masked)."""
import os, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
def fauc(y, s):
    n1 = y.sum(); n0 = len(y) - n1
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

H = os.path.expanduser('~')
RT = f'{H}/rsna_train'; LI = f'{H}/jump2/labels_in'
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA",
     "Effusion","Synovitis","Baker's","Contusion","Fracture"]
ID = 'StudyInstanceUID'

tr = pd.read_csv(f'{RT}/train.csv')
gold = tr[tr[L].notna().any(axis=1)].set_index(ID)[L].astype(float)
g74 = pd.read_csv(f'{RT}/gold74_eval_set.csv').set_index(ID)
ex16 = g74[g74.source != 'original_gold58'][L].astype(float)
assert len(gold) == 58 and len(ex16) == 16

SRC = {  # name: (path, kind, note)
 'steven_v4':        (f'{RT}/llm_labels_v4_blend.csv', 'plain', 'public LLM labels (stevenleehans v4 blend); fleet eval labels'),
 'steven_full':      (f'{LI}/steven_full.csv', 'plain', 'stevenleehans llm_labels_full'),
 'steven_v2':        (f'{LI}/steven_v2.csv', 'plain', 'stevenleehans llm_labels_v2'),
 'pilkwang_v2':      (f'{RT}/report_labels_v2.csv', 'pilk', 'pilkwang report_labels_v2 (Opus-5 API; YES/NO/UNK verdicts)'),
 'lixin_gpt56':      (f'{RT}/labels_llm_gpt56sol.csv', 'plain', 'lixin gpt-5.6 binary'),
 'lixin_gpt56_conf': (f'{LI}/lixin_gpt56_conf.csv', 'conf', 'lixin gpt-5.6 binary x conf (signed)'),
 'yunus_v5':         (f'{LI}/yunus_v5.csv', 'plain', '4-source merged report_labels_v5'),
 'jev_final2':       (f'{LI}/jev_final2_scores.csv', 'plain', 'JEV 6-member text ensemble'),
 'jev_final2_bin':   (f'{RT}/FINAL2_labels_with_confidence.csv', 'plain', 'JEV binary'),
 'jev_hybrid':       (f'{LI}/jev_hybrid_scores.csv', 'plain', 'JEV hybrid = .95 hard-swap(steven/pilk, chosen ON gold58) + .05 text'),
 'qwen7b_box':       (f'{RT}/pseudo_labels_qwen7b.csv', 'plain', 'our Qwen2.5-7B zero-shot (gold rows overwritten by gold)'),
 'qwen_laptop':      (f'{LI}/qwen3b_laptop.csv', 'plain', 'our earlier Qwen (3B) zero-shot'),
 'pseudo_v1':        (f'{RT}/pseudo_labels_v1.csv', 'plain', 'our earliest rule/LLM pseudo labels'),
 'rulev0':           (f'{LI}/rulev0.csv', 'plain', 'our regex rules v0'),
 'refined':          (f'{LI}/refined_labels.csv', 'plain', 'teacher-refined labels'),
 'teacher_wd':       (f'{RT}/teacher_train_probs.csv', 'plain', 'IMAGE teacher: widedense Dread CoAtNet (parent family)'),
 'T_maxspan_v5':     (f'{LI}/T_maxspan_v5.csv', 'plain', 'IMAGE teacher: parent Raptor maxspan v5 arm'),
 'T_native384_v8':   (f'{LI}/T_native384_v8.csv', 'plain', 'IMAGE teacher: parent Raptor native384 v8 arm'),
 'T_finespacing_v9': (f'{LI}/T_finespacing_v9.csv', 'plain', 'IMAGE teacher: parent v9 arm'),
 'T_raptor_proxy':   (f'{LI}/T_raptor_proxy.csv', 'plain', 'IMAGE teacher: .75 v5 + .25 v8'),
 'mix50_logit':      (f'{RT}/labels_mix50_logit.csv', 'plain', '.5 steven_v4 + .5 logit-matched teacher_wd'),
 'mix50_ecdf':       (f'{LI}/mix50_ecdf.csv', 'plain', '.5 steven_v4 + .5 ECDF teacher'),
 'mixSA_logit':      (f'{RT}/labels_mixSA_logit.csv', 'plain', 'silence-aware: teacher .75 on silent cells, .5 else (logit)'),
 'mixSA_ecdf':       (f'{RT}/labels_mixSA_ecdf.csv', 'plain', 'silence-aware ECDF'),
 'mix50sil_ecdf':    (f'{LI}/mix50sil_ecdf.csv', 'plain', 'round3 mix50sil'),
 'gated75_logit':    (f'{RT}/labels_gated75_logit.csv', 'plain', 'teacher only on report-silent cells'),
}

def load(path, kind):
    df = pd.read_csv(path)
    df = df.drop_duplicates(ID).set_index(ID)
    if kind == 'conf':  # signed confidence: binary 1 -> conf, 0 -> 1-conf
        out = pd.DataFrame(index=df.index)
        for l in L:
            c = df[l + '__conf'].astype(float)
            out[l] = np.where(df[l].astype(float) >= 0.5, 0.5 + 0.5 * c, 0.5 - 0.5 * c)
        return out
    return df[L].astype(float)

def aucs(S, G):
    out = []
    for l in L:
        y = G[l].values; s = S[l].values; m = ~np.isnan(y) & ~np.isnan(s)
        y, s = y[m], s[m]
        out.append(np.nan if len(np.unique(y)) < 2 else roc_auc_score(y, s))
    return np.array(out)

rng = np.random.default_rng(0)
BOOT = rng.integers(0, 58, size=(2000, 58))
def boot_macro(S, G):
    Sv, Gv = S.values, G.values; res = np.empty(len(BOOT))
    for b, ix in enumerate(BOOT):
        a = []
        for j in range(12):
            y = Gv[ix, j]; s = Sv[ix, j]
            if len(np.unique(y)) < 2: continue
            a.append(fauc(y, s))
        res[b] = np.mean(a)
    return res

R = {}; SETS = {}
for name, (path, kind, note) in SRC.items():
    if not os.path.exists(path):
        print('MISSING', name, path); continue
    df = load(path, kind)
    cov = df.index.intersection(gold.index)
    if len(cov) < 56:
        R[name] = {'note': note, 'skip': f'only {len(cov)}/58 gold studies present'}; continue
    nmiss = 58 - len(cov)
    full = df.reindex(df.index.union(gold.index).union(ex16.index))
    df = full.fillna(df.median())  # missing studies -> column median (neutral), recorded below
    S = df.loc[gold.index]
    leak = float(np.mean(np.abs(S.values - gold.values) < 1e-6))
    a58 = aucs(S, gold)
    a16 = aucs(df.reindex(ex16.index), ex16) if ex16.index.isin(df.index).all() else np.full(12, np.nan)
    R[name] = {'note': note, 'n_rows': int(len(df)), 'gold58_macro': float(np.nanmean(a58)),
               'gold58_per_label': [round(float(x), 4) for x in a58], 'leak_exact_frac': round(leak, 3), 'n_gold_missing_filled': nmiss,
               'ex16_macro': float(np.nanmean(a16)) if not np.all(np.isnan(a16)) else None}
    SETS[name] = df

# rank-mean consensus of independent report reads (each column -> within-column percentile)
def rk(df): return df.rank(pct=True)
for cname, members in {
    'CONS_llm4': ['steven_full', 'pilkwang_v2', 'lixin_gpt56_conf', 'jev_final2'],
    'CONS_llm5': ['steven_v4', 'pilkwang_v2', 'lixin_gpt56_conf', 'jev_final2', 'yunus_v5'],
    'CONS_llm4+wd': ['steven_full', 'pilkwang_v2', 'lixin_gpt56_conf', 'jev_final2', 'teacher_wd'],
}.items():
    idx = gold.index.union(SETS['steven_v4'].index)
    M = sum(rk(SETS[m].reindex(SETS['steven_v4'].index)) for m in members) / len(members)
    SETS[cname] = M
    S = M.loc[gold.index]; a58 = aucs(S, gold); a16 = aucs(M.reindex(ex16.index), ex16)
    R[cname] = {'note': 'rank-mean of ' + '+'.join(members), 'gold58_macro': float(np.nanmean(a58)),
                'gold58_per_label': [round(float(x), 4) for x in a58], 'leak_exact_frac': 0.0,
                'ex16_macro': float(np.nanmean(a16))}

base = {k: boot_macro(SETS[k].loc[gold.index], gold) for k in ['steven_v4', 'mix50_logit']}
for name in R:
    if 'skip' in R[name] or name not in SETS: continue
    bm = boot_macro(SETS[name].loc[gold.index], gold)
    for k, v in base.items():
        d = bm - v
        R[name][f'd_vs_{k}'] = round(float(R[name]['gold58_macro'] - R[k]['gold58_macro']), 4)
        R[name][f'P_gt_{k}'] = round(float(np.mean(d > 0)), 3)
        R[name][f'd90_{k}'] = [round(float(np.quantile(d, .05)), 4), round(float(np.quantile(d, .95)), 4)]

json.dump(R, open(f'{H}/jump2/label_eval.json', 'w'), indent=1)
with open(f'{H}/jump2/label_eval.txt', 'w') as f:
    hdr = f"{'set':18s} {'g58':>6s} {'dV4':>7s} {'P>V4':>5s} {'dMix':>7s} {'P>Mx':>5s} {'ex16':>6s} {'leak':>5s} | " + ' '.join(f'{l[:5]:>5s}' for l in L)
    print(hdr); f.write(hdr + '\n')
    for name, r in sorted(R.items(), key=lambda kv: -kv[1].get('gold58_macro', 0)):
        if 'skip' in r:
            line = f"{name:18s} SKIP {r['skip']}"
        else:
            e = r['ex16_macro']; e = f'{e:.4f}' if e is not None else '  -   '
            line = (f"{name:18s} {r['gold58_macro']:.4f} {r.get('d_vs_steven_v4',0):+.4f} {r.get('P_gt_steven_v4',0):5.2f} "
                    f"{r.get('d_vs_mix50_logit',0):+.4f} {r.get('P_gt_mix50_logit',0):5.2f} {e} {r['leak_exact_frac']:5.2f} | "
                    + ' '.join(f'{x:5.3f}' for x in r['gold58_per_label']))
        print(line); f.write(line + '\n')
