# vlab1.py: adversarial re-check of the jump2 labels track (CPU only, read-only inputs, prints only).
import numpy as np, pandas as pd, re
from scipy.special import logit, expit
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
ID = "StudyInstanceUID"; H = "/home/training"; RT = f"{H}/rsna_train"; J = f"{H}/jump2"; LI = f"{J}/labels_in"; FR = f"{H}/fleet_runs"; GP = f"{J}/gold74_preds"
tr = pd.read_csv(f"{RT}/train.csv", dtype={ID: str})
gold = tr[tr["ACL"].notna()].drop_duplicates(ID).set_index(ID)[L].astype(int); g = gold.index
print("gold58 n", len(g), "| positives per label", gold.sum().tolist())
def rd(p): return pd.read_csv(p, dtype={ID: str}).drop_duplicates(ID).set_index(ID)
v4F = rd(f"{RT}/llm_labels_v4_blend.csv")[L].astype(float)
mixF = rd(f"{RT}/labels_mix50_logit.csv")[L].astype(float)
msaF = rd(f"{RT}/labels_mixSA_logit.csv")[L].astype(float)
wdF = rd(f"{RT}/teacher_train_probs.csv")[L].astype(float)
pk = rd(f"{RT}/report_labels_v2.csv")
own = {k: rd(f"{GP}/{f}")[L].astype(float) for k, f in [("TTRmix", "f0_s1043_ttr_mix50__raw.csv"), ("F1mix", "f0_s42__raw.csv"),
       ("TTRv4", "f0_s1042_ttr__ema.csv"), ("F1v4", "f0_s42_v4__raw.csv")]}
oofp = {"TTRmix": f"{FR}/f0_s1043_ttr_mix50/oof_f0_s1043_raw.csv", "F1mix": f"{FR}/f0_s42/oof_f0_s42_raw.csv",
        "TTRv4": f"{FR}/f0_s1042_ttr/oof_f0_s1042_ema.csv", "F1v4": f"{FR}/f0_s42_v4/oof_f0_s42_raw.csv"}
oof = {k: rd(p)[L].astype(float) for k, p in oofp.items()}
vid = sorted(set.intersection(*[set(v.index) for v in oof.values()]))
print("fold-0 val n", len(vid), "| overlap val & gold74:", len(set(vid) & set(pd.read_csv(f"{RT}/gold74_eval_set.csv", dtype={ID: str})[ID])))
def mac(S): return float(np.mean([roc_auc_score(gold[l], S.loc[g, l]) for l in L]))
Gv = gold.values; rng = np.random.default_rng(12345); B = rng.integers(0, 58, size=(2000, 58))
def boot(S):
    Sv = S.loc[g, L].values; out = np.full((len(B), 12), np.nan)
    for j in range(12):
        y = Gv[B, j]; s = Sv[B, j]; r = rankdata(s, axis=1); n1 = y.sum(1); n0 = 58 - n1
        ok = (n1 > 0) & (n0 > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            a = ((r * y).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
        out[ok, j] = a[ok]
    return np.nanmean(out, axis=1)
def unk(idx): return pd.DataFrame({l: pk.reindex(idx)[l + "__verdict"].fillna("UNK").values == "UNK" for l in L}, index=idx)
def lm(a, ref_mu, ref_sd, mu=None, sd=None):
    la = logit(np.clip(a, 1e-4, 1 - 1e-4)); mu = la.mean() if mu is None else mu; sd = la.std() if sd is None else sd
    return expit((la - mu) / sd * ref_sd + ref_mu)
def lstat(x): lx = logit(np.clip(np.asarray(x, float), 1e-4, 1 - 1e-4)); return lx.mean(), lx.std()

# ---------- 1. selfteach rebuild, two moment conventions ----------
TEACH = {"wd": wdF, **own}
def build(teachers, sa, moments):
    U = unk(g); out = pd.DataFrame(index=g)
    for l in L:
        ts = []
        for k in teachers:
            T = TEACH[k]
            if moments == "gold58":         # as lab2_selfteach.py: moments of the 58 eval rows themselves
                rmu, rsd = lstat(v4F.loc[g, l]); ts.append(lm(T.loc[g, l].values, rmu, rsd))
            else:                           # build-time convention: text moments over the table; teacher moments from its own train-time distribution
                if k == "wd": tmu, tsd = lstat(wdF[l]); rmu, rsd = lstat(v4F[l])
                else: tmu, tsd = lstat(oof[k].loc[vid, l]); rmu, rsd = lstat(v4F.loc[vid, l])
                ts.append(lm(T.loc[g, l].values, rmu, rsd, tmu, tsd))
        t = np.mean(ts, axis=0); ww = np.where(U[l].values, .75, .5) if sa else .5
        out[l] = (1 - ww) * v4F.loc[g, l].values + ww * t
    return out
VAR = [("v4+wd", ["wd"]), ("v4+TTRv4", ["TTRv4"]), ("v4+F1v4", ["F1v4"]), ("v4+TTRv4+F1v4", ["TTRv4", "F1v4"]),
       ("v4+TTRmix", ["TTRmix"]), ("v4+F1mix", ["F1mix"]), ("v4+TTRmix+F1mix", ["TTRmix", "F1mix"]),
       ("v4+wd+TTRv4", ["wd", "TTRv4"]), ("v4+wd+TTRmix", ["wd", "TTRmix"])]
base = {"v4": v4F, "mix50_file": mixF, "mixSA_file": msaF}
bb = {k: boot(S) for k, S in base.items()}
print("\n1. self-teach label sets on gold58 (P = paired bootstrap share > reference, 2000 study resamples)")
print(f"{'set':34s} {'mom=gold58':>10s} {'mom=table':>10s} | P>mixSA_file P>mix50_file P>v4 (table moments)")
for k in base: print(f"{k:34s} {mac(base[k]):10.4f}")
allb = {}
for nm, te in VAR:
    for sa in (False, True):
        S1 = build(te, sa, "gold58"); S2 = build(te, sa, "table"); b2 = boot(S2); b1 = boot(S1)
        tag = nm + (" SA" if sa else ""); allb[tag + " |g58mom"] = (mac(S1), b1); allb[tag + " |tabmom"] = (mac(S2), b2)
        print(f"{tag:34s} {mac(S1):10.4f} {mac(S2):10.4f} | {np.mean(b2 > bb['mixSA_file']):.2f} {np.mean(b2 > bb['mix50_file']):.2f} {np.mean(b2 > bb['v4']):.2f}"
              f" | g58mom: P>mixSA {np.mean(b1 > bb['mixSA_file']):.2f} d {mac(S1)-mac(msaF):+.4f} SE {np.std(b1-bb['mixSA_file']):.4f}")
# winner's curse: how often is the reported best the bootstrap winner among the 18 g58-moment variants
keys = [k for k in allb if k.endswith("|g58mom")]
M = np.column_stack([allb[k][1] for k in keys]); win = np.bincount(M.argmax(1), minlength=len(keys)) / len(M)
print("bootstrap win share among 18 variants (gold58 moments):", ", ".join(f"{keys[i].replace(' |g58mom','')} {win[i]:.2f}" for i in np.argsort(-win)[:5]))
best = M.max(1); print(f"bootstrap E[max of 18] - E[v4+wd+TTRmix SA] = {best.mean() - allb['v4+wd+TTRmix SA |g58mom'][1].mean():+.4f}")

# ---------- 2. teacher-averaging bias for a 5-ckpt gold check ----------
print("\n2. single-teacher vs averaged-teacher on gold58 (the 5-ckpt-mean gold check vs single-OOF train rows)")
for nm, te in [("v4+TTRmix SA", ["TTRmix"]), ("v4+F1mix SA", ["F1mix"]), ("v4+mean(TTRmix,F1mix) SA", ["TTRmix", "F1mix"])]:
    print(f"   {nm:30s} {mac(build(te, True, 'gold58')):.4f}")
avg_pred = (own["TTRmix"] + own["F1mix"]) / 2
print(f"   raw model gold58: TTRmix {mac(own['TTRmix']):.4f} F1mix {mac(own['F1mix']):.4f} mean-of-2 {mac(avg_pred):.4f}")

# ---------- 3. decomposition split ----------
sf = rd(f"{LI}/steven_full.csv")[L].astype(float).reindex(g); lx = rd(f"{LI}/lixin_gpt56_conf.csv").reindex(g); jv = rd(f"{LI}/jev_final2_scores.csv")[L].astype(float).reindex(g)
pkv = pd.DataFrame({l: pk.reindex(g)[l + "__verdict"].fillna("UNK").values for l in L}, index=g)
v4 = v4F.reindex(g)
votes = (v4 > .5).astype(int) + (sf > .5).astype(int) + (pkv == "YES").astype(int) + (lx[L].astype(float) >= .5).astype(int) + (jv > .5).astype(int)
state = pd.DataFrame("C", index=g, columns=L); state[votes == 5] = "U+"; state[(votes == 0) & (pkv == "NO")] = "NEG"; state[(votes == 0) & (pkv != "NO")] = "SIL"
def attr(S):
    res = {}
    for l in L:
        y = gold[l].values; s = S.loc[g, l].values; st = state[l].values; pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]; den = len(pos) * len(neg) * 12
        for i in pos:
            for j in neg:
                w = 1.0 if s[i] < s[j] else (0.5 if s[i] == s[j] else 0.0)
                if not w: continue
                a, b = st[i], st[j]
                if a == "C" or b == "C": k = "contested"
                elif a == b: k = f"same-state {a}"
                elif (a in ("NEG", "SIL") and b == "U+"): k = "cross: pos-cell report-negative/silent vs neg-cell report-positive"
                else: k = f"cross {a}>{b}"
                res[k] = res.get(k, 0) + w / den
    return res
r = attr(v4F); print("\n3. v4 deficit split:", {k: round(v, 4) for k, v in sorted(r.items(), key=lambda t: -t[1])}, "total", round(sum(r.values()), 4))
ties = 0
for l in L:
    s = v4F.loc[g, l].values; st = state[l].values; m = st == "SIL"; ties += len(np.unique(s[m])) if m.any() else 0
print("   v4 distinct values inside SIL cells per label:", [len(np.unique(v4F.loc[g, l].values[(state[l] == "SIL").values])) for l in L])

# ---------- 4. ex16 reading fidelity ----------
g74 = rd(f"{RT}/gold74_eval_set.csv"); ex = g74[g74.source != "original_gold58"]; E = ex[L].astype(float)
print("\n4. ex16: n", len(ex), "| sources", ex.source.value_counts().to_dict(), "| NaN cells", int(E.isna().sum().sum()), "of", E.size,
      "| positives per label", E.sum().astype(int).tolist())
q7 = rd(f"{RT}/pseudo_labels_qwen7b.csv")
q7c = [c for c in q7.columns if c in L]
print("   qwen7b columns", len(q7c), "| qwen7b gold58 rows == gold (leak check):", bool((q7.reindex(g)[L].astype(float).values == gold.values).all()))
def exauc(S):
    a = []
    for l in L:
        m = E[l].notna(); y = E[l][m]
        if y.nunique() == 2: a.append(roc_auc_score(y, S.reindex(ex.index)[l][m]))
    return np.mean(a), len(a)
pks = pd.DataFrame({l: pk[l + "__verdict"].map({"YES": 1.0, "UNK": .5, "NO": 0.0}) for l in L}, index=pk.index)
lxa = rd(f"{LI}/lixin_gpt56_conf.csv")
for nm, S in [("v4", v4F), ("v4 binarized", (v4F > .5).astype(float)), ("qwen7b binary", q7[L].astype(float)), ("pilk verdict", pks),
              ("lixin binary", lxa[L].astype(float)), ("mix50", mixF), ("teacher_wd", wdF)]:
    m_, n_ = exauc(S); print(f"   ex16 {nm:16s} {m_:.4f} ({n_} labels)")

# ---------- 5. reports ----------
lang = rd(f"{RT}/_reports_lang.csv"); rep = tr.drop_duplicates(ID).set_index(ID).Report.fillna("").astype(str)
print("\n5. reports n", len(rep), "| with micro sign", int(rep.str.contains("µ").sum()), "| greek", int((lang.lang == "el").sum()),
      "| greek with micro", int(rep.reindex(lang.index[lang.lang == "el"]).str.contains("µ").sum()),
      "| BILATERAL NOTE", int(rep.str.contains("BILATERAL NOTE").sum()), "| <80 chars", int((rep.str.len() < 80).sum()))
pat = r"shoulder|subscapular|supraspinat|infraspinat|rotator|glenoid|humer|рамо|раменн|субскапул|супраспин|ротатор|гленоид|хумер|омуз|ώμο|ωμοπλ|schulter|hombro|rame"
hits = rep[rep.str.lower().str.contains(pat, regex=True)]
print("   shoulder-term hits:", len(hits))
for u, t in hits.head(6).items(): print("     ", lang.lang.get(u, "?"), "|", re.sub(r"\s+", " ", t)[:220])
# effusion contested rate check
lxF = rd(f"{LI}/lixin_gpt56_conf.csv"); sfF = rd(f"{LI}/steven_full.csv")[L].astype(float); jvF = rd(f"{LI}/jev_final2_scores.csv")[L].astype(float)
ids = v4F.index
vt = (v4F > .5).astype(int) + (sfF.reindex(ids) > .5).astype(int) + (pk.reindex(ids)[[l + "__verdict" for l in L]].set_axis(L, axis=1) == "YES").astype(int) \
     + (lxF.reindex(ids)[L].astype(float) >= .5).astype(int) + (jvF.reindex(ids) > .5).astype(int)
print("   contested rate Effusion", round(float(((vt["Effusion"] > 0) & (vt["Effusion"] < 5)).mean()), 3))

# ---------- 6. parent-likeness ----------
rp = rd(f"{LI}/T_raptor_proxy.csv")[L].astype(float).reindex(vid)
def pl(S): return float(np.mean([spearmanr(S.reindex(vid)[l].values, rp[l].values).correlation for l in L]))
print("\n6. parent-likeness (mean per-label Spearman vs raptor proxy, fold-0 val):")
for nm, S in [("v4", v4F), ("mix50", mixF), ("teacher_wd (parent family)", wdF), ("own TTRv4 OOF", oof["TTRv4"]), ("own F1v4 OOF", oof["F1v4"]),
              ("own TTRmix OOF", oof["TTRmix"]), ("own F1mix OOF", oof["F1mix"]), ("mean(TTRv4,F1v4) OOF", (oof["TTRv4"] + oof["F1v4"]) / 2)]:
    print(f"   {nm:28s} {pl(S):.3f}")
# image-vs-image agreement that is not parent-specific: own v4 models vs each other
print("   own TTRv4 vs own F1v4 (two own image models, same labels):", round(float(np.mean([spearmanr(oof['TTRv4'].reindex(vid)[l], oof['F1v4'].reindex(vid)[l]).correlation for l in L])), 3))
print("   own TTRv4 OOF vs teacher_wd:", round(float(np.mean([spearmanr(oof['TTRv4'].reindex(vid)[l], wdF.reindex(vid)[l]).correlation for l in L])), 3))
print("   own TTRv4 OOF vs v4 labels:", round(float(np.mean([spearmanr(oof['TTRv4'].reindex(vid)[l], v4F.reindex(vid)[l]).correlation for l in L])), 3))
