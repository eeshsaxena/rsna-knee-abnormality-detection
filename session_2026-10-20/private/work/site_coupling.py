# site_coupling.py: how much does OA-compartment label coupling vary between sites (language = site proxy)?
# Competition data only (train reports + v4_blend report labels + gold74 expert labels). MEASURED / MODEL.
import numpy as np, pandas as pd
R = "C:/Users/eeshs/Downloads/rsna-knee/"
OA = ["Medial OA", "Lateral OA", "PF OA"]; PAIRS = [(0, 1), (0, 2), (1, 2)]
lang = pd.read_csv(R + "_data/_reports_lang.csv", usecols=["StudyInstanceUID", "lang"], dtype=str).set_index("StudyInstanceUID")
lab = pd.read_csv(R + "_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
df = lab.join(lang, how="inner"); Y = (df[OA] >= 0.5).astype(int)
def logor(y, a, b):
    t = pd.crosstab(y[:, a], y[:, b]).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values + 0.5
    return np.log(t[1, 1] * t[0, 0] / (t[0, 1] * t[1, 0])), np.sqrt((1 / t).sum())
g = pd.read_csv(R + "session_2026-10-18/next/g58/gold74_eval_set.csv"); g = g.dropna(subset=OA); yg = (g[OA].values >= 0.5).astype(int)
print("gold74 expert: n", len(g), "prev", yg.mean(0).round(3))
g_se = []
for a, b in PAIRS:
    l, s = logor(yg, a, b); g_se.append(s); print(f"  {OA[a]} x {OA[b]}: logOR {l:.2f} se {s:.2f}")
print("report labels (v4_blend >= .5), per language (site proxy):")
rows = []
for lg, d in df.groupby("lang"):
    y = Y.loc[d.index].values
    if len(d) < 80: continue
    out = [lg, len(d)] + list(y.mean(0).round(3))
    for a, b in PAIRS:
        l, s = logor(y, a, b); out += [l, s]
    rows.append(out)
cols = ["lang", "n", "pM", "pL", "pPF"] + [f"{p}_{k}" for p in ("ML", "MP", "LP") for k in ("lor", "se")]
T = pd.DataFrame(rows, columns=cols); print(T.round(2).to_string(index=False))
# DerSimonian-Laird between-site sd of the log OR, per pair; and prevalence heterogeneity
print("between-site heterogeneity (DerSimonian-Laird tau of log OR), in gold74-se units:")
for k, (p, (a, b)) in enumerate(zip(("ML", "MP", "LP"), PAIRS)):
    y = T[f"{p}_lor"].values; v = T[f"{p}_se"].values ** 2; wv = 1 / v
    mu = (wv * y).sum() / wv.sum(); Q = (wv * (y - mu) ** 2).sum(); dfree = len(y) - 1
    tau2 = max(0.0, (Q - dfree) / (wv.sum() - (wv ** 2).sum() / wv.sum())); tau = np.sqrt(tau2)
    print(f"  {p}: pooled logOR {mu:.2f}, Q {Q:.1f} on {dfree} df, tau {tau:.2f} = {tau / g_se[k]:.2f} gold74-se")
for c in ("pM", "pL", "pPF"):
    print(f"  prevalence {c}: site range {T[c].min():.3f}-{T[c].max():.3f}, sd {T[c].std():.3f}")
# public-vs-private difference of the site-mix coupling effect (MODEL): site effects u_s ~ N(0, tau^2);
# d_pub - d_priv = sum_s (wp_s - wq_s) u_s, wp / wq = site shares inside public / private.
def split_sd(shares, n_rep=20000, frac=0.30, rng=np.random.default_rng(0)):
    out = []
    for _ in range(n_rep):
        perm = rng.permutation(len(shares)); cum = np.cumsum(shares[perm]); k = np.searchsorted(cum, frac) + 1
        pub = np.zeros(len(shares), bool); pub[perm[:k]] = True
        wp = np.where(pub, shares, 0) / shares[pub].sum(); wq = np.where(~pub, shares, 0) / shares[~pub].sum()
        out.append(np.sqrt(((wp - wq) ** 2).sum()))
    return float(np.mean(out)), k
taus = []
for k, p in enumerate(("ML", "MP", "LP")):
    y = T[f"{p}_lor"].values; v = T[f"{p}_se"].values ** 2; wv = 1 / v; mu = (wv * y).sum() / wv.sum(); Q = (wv * (y - mu) ** 2).sum()
    taus.append(np.sqrt(max(0.0, (Q - (len(y) - 1)) / (wv.sum() - (wv ** 2).sum() / wv.sum()))) / g_se[k])
tau_u = float(np.mean(taus))
f9, k9 = split_sd(T.n.values / T.n.sum())
f22, k22 = split_sd(np.full(22, 1 / 22))
samp = np.sqrt(74 / 390 + 74 / 910)
print(f"mean tau over pairs = {tau_u:.2f} gold74-se (report-label proxy; expert labels from one central review should vary less)")
print(f"site split, 9 language groups (public ~{k9} groups): extra sd(d_pub - d_priv) = {f9:.2f} x tau = {f9 * tau_u:.2f} gold74-se")
print(f"site split, 22 equal sites (public ~{k22}): extra sd = {f22:.2f} x tau = {f22 * tau_u:.2f} gold74-se")
print(f"random split: coupling sampling sd public vs private (n 390 / 910) = {samp:.2f} gold74-se; private vs population {np.sqrt(74/910):.2f}")
print(f"total public-vs-private coupling sd: random {samp:.2f}; site split (22) {np.hypot(samp, f22 * tau_u):.2f}; site split (9) {np.hypot(samp, f9 * tau_u):.2f}")
