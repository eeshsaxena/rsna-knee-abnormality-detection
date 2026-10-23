# shake_mc.py: private-rank Monte Carlo with an explicit lineage-noise factor model and empirical-Bayes (NPMLE) shrinkage
# of the field. RESEARCH ONLY, local CPU, competition LB data only (no OAI).
#
# Scale: margin over the heliosli copy block, x1e-4 (copy block public .95802, picks.md / bpick.md ruler).
# Field (MEASURED structure, 17:10 UTC 10-20 snapshot): bands, exact-tie runs, pre-wave membership (10-14 05:27 snapshot).
# Model (MODEL / GUESS parameters, see shakeup.md):
#   public  p_i = t_i + w_i (l_i - l_fam) + e_i + b_i         (t = true margin vs family, l = lineage sampling noise on 390,
#   private q_i = t_i + w_i (l'_i - l'_fam) + w_i (g_i - g_fam) + e'_i + drift_i + bonus_i      l' on 910 = .655 l; g = site term)
#   w_i = share of the team's prediction that is NOT our public-notebook family; e = family-internal tweak noise; b = own selection.
#   t_i | p_i from an NPMLE prior fitted to the lineage teams (w >= .3); tweakers get a parametric prior (field_mc style).
#   l_fam (family public luck vs an average lineage) is ONE draw shared by every team (x w_i): the correlated part of the shake-up.
import sys, json, numpy as np
from scipy.stats import norm
rng0 = np.random.default_rng(2020)
W = "C:/Users/eeshs/Downloads/rsna-knee/"
NOW = W + "session_2026-10-20/private/work/api_lb_1020T_shake.json"
PRE = W + "session_2026-10-14/research960/lbprobe/api_lb_1014T0527.json"
ME = 16799346; COPY = 0.95802; US_PUB = (0.95991 - COPY) * 1e4    # pw30 unrounded (implied2 joint refit)

BASE = dict(sigP=22.0, dfam=0.0, ssite=0.0, beta=4.0, muA=13.0, sdA=7.5, muD=-4.0, sdD=8.5, sig_e=3.0, n_new=40, drift=1.0,
            bonus=0.7, p_ind_riser=0.3, p_ind_low=0.3, p_pre958=0.5, w_spread=0.15, N_final=6335, common=True, dfam_eb=False)
SCEN = {
    "base": {},
    "sigP15": dict(sigP=15.0), "sigP30": dict(sigP=30.0), "sigP10_fieldmc": dict(sigP=10.0),
    "famLucky": dict(dfam=+0.5), "famUnlucky": dict(dfam=-0.5), "famEB": dict(dfam_eb=True),
    "site10": dict(ssite=10.0), "site20": dict(ssite=20.0),
    "usSkeptic": dict(muA=10.0, sdA=8.0), "usOptim": dict(muA=16.0, sdA=7.0),
    "beta0": dict(beta=0.0), "beta8": dict(beta=8.0),
    "noCommon": dict(common=False), "newHi": dict(n_new=60, drift=2.0), "lowIndHi": dict(p_ind_low=0.6, p_pre958=0.8), "lowIndLo": dict(p_ind_low=0.1, p_pre958=0.2),
    "worst": dict(sigP=30.0, dfam=0.5, ssite=10.0, muA=10.0, sdA=8.0), "best": dict(sigP=15.0, dfam=-0.5, muA=16.0, sdA=7.0),
}

def load():
    now = json.load(open(NOW, encoding="utf-8")); pre = json.load(open(PRE, encoding="utf-8"))
    preb = {r["teamId"]: float(r["displayScore"]) for r in pre["lb"]}
    T = {t["teamId"]: t for t in now["teams"]}
    L = sorted(now["lb"], key=lambda r: r["rank"])
    runs = []; cur = [L[0]]
    for a, b in zip(L, L[1:]):
        if b["displayScore"] == a["displayScore"] and b["submissionId"] > a["submissionId"]: cur.append(b)
        else: runs.append(cur); cur = [b]
    runs.append(cur)
    rid = {}; rlen = {}
    for k, r in enumerate(runs):
        for x in r: rid[x["teamId"]] = k; rlen[x["teamId"]] = len(r)
    rows = []
    for r in L:
        v = float(r["displayScore"])
        if v < 0.930 or r["teamId"] == ME: continue
        rows.append(dict(tid=r["teamId"], rank=r["rank"], band=v, run=rid[r["teamId"]], rlen=rlen[r["teamId"]],
                         subs=T[r["teamId"]]["submissionCount"] or 0, pre=preb.get(r["teamId"], 0.0)))
    return rows, len(L)

def positions(rows, rng):
    # unrounded public margin (x1e-4) from band + rank order inside the band (MODEL)
    byb = {}
    for i, r in enumerate(rows): byb.setdefault(r["band"], []).append(i)
    p = np.zeros(len(rows))
    first_run = min(r["rank"] for r in rows if r["band"] == 0.958 and r["rlen"] >= 8)
    for b, ix in byb.items():
        ix = sorted(ix, key=lambda i: rows[i]["rank"]); n = len(ix)
        if b == 0.959:   # 3 teams above pw30 (u .91), the rest below
            up = [i for i in ix if rows[i]["rank"] < 141]; dn = [i for i in ix if rows[i]["rank"] > 141]
            for k, i in enumerate(up): p[i] = 0.91 + (1 - 0.91) * (len(up) - k - rng.uniform()) / len(up)
            for k, i in enumerate(dn): p[i] = 0.91 * (len(dn) - k - rng.uniform()) / len(dn)
        elif b == 0.958:
            wz = [i for i in ix if rows[i]["rank"] < first_run]; lo = [i for i in ix if rows[i]["rank"] >= first_run]
            top = wz[:30]; rest = wz[30:]      # ~30 teams above c4 (u .78), rest spread .02-.78 (CLAIMED rulers, MODEL)
            for k, i in enumerate(top): p[i] = 0.78 + 0.22 * (len(top) - k - rng.uniform()) / len(top)
            for k, i in enumerate(rest): p[i] = 0.02 + 0.76 * (len(rest) - k - rng.uniform()) / len(rest)
            for i in lo: p[i] = rng.uniform(0.0, 0.05)
        else:
            for k, i in enumerate(ix): p[i] = (n - k - rng.uniform()) / n
        for i in ix: p[i] = (b + p[i] * 1e-3 - COPY) * 1e4
    return p

def classes(rows, p, P, rng):
    n = len(rows); w = np.zeros(n); kind = np.empty(n, dtype=object)
    first_run = min(r["rank"] for r in rows if r["band"] == 0.958 and r["rlen"] >= 8)
    for i, r in enumerate(rows):
        if r["rlen"] >= 4 and r["band"] <= 0.958:
            kind[i] = "copy" if r["band"] == 0.958 else "copylow"; continue
        if r["pre"] >= 0.958 and r["band"] >= 0.959:
            kind[i] = "lin"; w[i] = rng.uniform(0.6, 1.0)        # strong before any public .958 notebook: own lineage
        elif r["pre"] >= 0.957 and r["band"] >= 0.959:
            kind[i] = "lin"; w[i] = rng.uniform(0.3, 0.9)        # own lineage, likely blended with the family since
        elif r["pre"] >= 0.957 and r["band"] == 0.958:
            if rng.uniform() < P["p_pre958"]: kind[i] = "lin"; w[i] = rng.uniform(0.3, 1.0)
            else: kind[i] = "tw"; w[i] = rng.uniform(0.0, 0.15)  # jumped .957 -> .958 with the copy wave: adopted the family
        elif r["band"] >= 0.959:
            if rng.uniform() < P["p_ind_riser"]: kind[i] = "lin"; w[i] = rng.uniform(0.6, 1.0)
            else: kind[i] = "lin"; w[i] = rng.uniform(0.3, 0.7)
        elif r["band"] == 0.958 and r["rank"] < first_run:
            if rng.uniform() < P["w_spread"]: kind[i] = "lin"; w[i] = rng.uniform(0.3, 1.0)
            else: kind[i] = "tw"; w[i] = rng.uniform(0.0, 0.15)
        elif r["band"] == 0.958:
            kind[i] = "tw"; w[i] = rng.uniform(0.0, 0.15)
        else:
            if rng.uniform() < P["p_ind_low"]: kind[i] = "lin"; w[i] = rng.uniform(0.6, 1.0)
            else: kind[i] = "tw"; w[i] = rng.uniform(0.0, 0.15)
    return w, kind

def npmle(x, s, grid, iters=400):
    # EM for a discrete prior on grid: x_i ~ N(theta_i, s_i^2), theta ~ g
    g = np.full(grid.size, 1.0 / grid.size)
    ll = norm.logpdf((x[:, None] - grid[None, :]) / s[:, None]) - np.log(s[:, None])
    Lk = np.exp(ll - ll.max(1, keepdims=True))      # per-row scaling: posteriors unaffected
    for _ in range(iters):
        m = Lk * g[None, :]; m /= m.sum(1, keepdims=True); g = m.mean(0)
    return g, Lk

def run(name, over, R=12, NS=2500):
    P = dict(BASE); P.update(over)
    sI = P["sigP"] / np.sqrt(2.0)          # per-lineage public sd (paired independent-vs-family = sigP)
    rows, N_now = load()
    silver = P["N_final"] // 20; bronze = P["N_final"] // 10; gold = 10 + P["N_final"] // 500
    TH = [gold, 50, 100, 141, 150, 211, silver, bronze]
    res_A = []; res_P = []; extra = []
    for rep in range(R):
        rng = np.random.default_rng(1000 * rep + 7)
        p = positions(rows, rng); w, kind = classes(rows, p, P, rng)
        subs = np.array([r["subs"] for r in rows]); run_id = np.array([r["run"] for r in rows])
        # projected arrivals: n_new W-zone tweakers become risers at .959-.961 by the deadline (MODEL, 12-25/day)
        lin = kind == "lin"; tw_w = np.where((kind == "tw") & (p > 0.5) & (p < 9.8))[0]
        riser_p = p[lin & (p >= 9.8) & (p < 39.8)]
        nn = min(P["n_new"], tw_w.size); pick = rng.choice(tw_w, nn, replace=False)
        p[pick] = rng.choice(riser_p, nn) + rng.uniform(-3, 3, nn); kind[pick] = "lin"; w[pick] = rng.uniform(0.3, 1.0, nn)
        lin = kind == "lin"
        # own-selection inflation b (MODEL): heavy submitters inflate more
        b = np.where(subs >= 50, P["beta"] * rng.uniform(0.5, 1.5, p.size), 0.5 * P["beta"] * rng.uniform(0, 1, p.size))
        su = np.where(p >= 9.8, 1.5, 2.0)                 # in-band position uncertainty
        s_lin = np.sqrt((w * sI) ** 2 + P["sig_e"] ** 2 + su ** 2)
        grid = np.arange(-320.0, 160.0, 1.0)
        g, Lk = npmle(p[lin] - b[lin], s_lin[lin], grid)
        post = Lk * g[None, :]; post /= post.sum(1, keepdims=True)
        pm = post @ grid; psd = np.sqrt(np.maximum(post @ grid ** 2 - pm ** 2, 0.0))
        PM = np.zeros(p.size); PSD = np.zeros(p.size); PM[lin] = pm; PSD[lin] = psd
        # tweakers: family tweaks, prior N(mu0 + 1, 2) around their base (W zone base 0; lower bands base = band floor area)
        twm = kind == "tw"
        base0 = np.where(p > -2, 0.0, p)                  # lower-band tweakers keep their level
        lam = 2.0 ** 2 / (2.0 ** 2 + P["sig_e"] ** 2)
        PM[twm] = (base0 + 1.0 + lam * (p - b - base0 - 1.0))[twm]; PSD[twm] = np.sqrt(lam) * P["sig_e"]
        # family luck prior: dfam in units of sI, or the EB value for a w=1 lineage sitting at the copy level
        if P["dfam_eb"]:
            Lk0 = norm.pdf((0.0 - grid) / np.sqrt(sI**2 + P["sig_e"]**2)) * g; Lk0 /= Lk0.sum(); dfam = 0.0 - (Lk0 @ grid)
        else:
            dfam = P["dfam"] * sI
        # copy clusters
        cpy = kind == "copy"; cl = np.unique(run_id[cpy]); cl_size = np.array([(run_id == c).sum() for c in cl])
        cpl = kind == "copylow"
        F = lin | twm
        mu_f = PM[F] + P["bonus"] + np.where(lin[F], rng.exponential(P["drift"], F.sum()), 0.0)
        sd_f = np.sqrt(PSD[F] ** 2 + (w[F] * 0.655 * sI) ** 2 + (w[F] * P["ssite"]) ** 2 + (0.655 * P["sig_e"]) ** 2)
        wf = w[F]
        low_mu = p[cpl]
        for k in range(NS // 500):
            n = 500; r2 = np.random.default_rng(rep * 100 + k + 99)
            lf = r2.normal(dfam, sI, (n, 1)); lpf = r2.normal(0, 0.655 * sI, (n, 1)); gf = r2.normal(0, P["ssite"] + 1e-12, (n, 1))
            common = (lf - lpf - gf) if P["common"] else 0.0
            y = mu_f[None, :] + sd_f[None, :] * r2.standard_normal((n, wf.size)) + wf[None, :] * common
            yl = low_mu[None, :] + 3.0 * r2.standard_normal((n, low_mu.size))
            dc = r2.normal(0, 1.0, (n, cl.size))
            mA = r2.normal(P["muA"], P["sdA"], n); mB = mA + r2.normal(P["muD"], P["sdD"], n); mP = np.maximum(mA, mB)
            for m, store in ((mA, res_A), (mP, res_P)):
                rk = 1 + (y > m[:, None]).sum(1) + (yl > m[:, None]).sum(1) + ((dc > m[:, None]) * cl_size[None, :]).sum(1)
                store.append(rk)
            if k == 0:
                # calibration: public vs private rank of the top non-copy field (Spearman, top-150 retention)
                pubv = p[F]; o = np.argsort(-pubv)[:300]
                for j in range(5):
                    priv = y[j][o]; pr = np.argsort(np.argsort(-priv)); pu = np.arange(300)
                    sp = np.corrcoef(pr, pu)[0, 1]; keep = (pr[:150] < 150).mean()
                    extra.append((sp, keep))
    A = np.concatenate(res_A); PR = np.concatenate(res_P)
    def summ(x):
        q = np.percentile(x, [5, 10, 25, 50, 75, 90, 95]).astype(int)
        d = {f"P(r<={t})": float((x <= t).mean()) for t in TH}
        d["quantiles p5/10/25/50/75/90/95"] = q.tolist(); d["mean_capped2000"] = float(np.minimum(x, 2000).mean())
        return d
    ex = np.array(extra)
    out = dict(name=name, params=P, sI=sI, cuts=dict(gold=gold, silver=silver, bronze=bronze),
               A=summ(A), pair=summ(PR), calib=dict(spearman_top300=float(ex[:, 0].mean()), top150_retained=float(ex[:, 1].mean())),
               n_lin=int(lin.sum()), n_tw=int(twm.sum()), n_copy=int(cpy.sum()), dfam_used=float(dfam))
    return out

if __name__ == "__main__":
    names = sys.argv[1:] or list(SCEN)
    allr = {}
    for nm in names:
        o = run(nm, SCEN[nm]); allr[nm] = o
        a, pr = o["A"], o["pair"]
        print(f"== {nm:14s} sI {o['sI']:.1f} dfam {o['dfam_used']:+.1f} | lin {o['n_lin']} tw {o['n_tw']} copy {o['n_copy']} | "
              f"calib spearman300 {o['calib']['spearman_top300']:.2f} top150 kept {o['calib']['top150_retained']:.2f}")
        for lab, d in (("A pw30", a), ("pair", pr)):
            print(f"   {lab:7s} " + " ".join(f"{k[2:-1]}:{v:.3f}" for k, v in d.items() if k.startswith("P(")) +
                  f" | q {d['quantiles p5/10/25/50/75/90/95']}")
        json.dump(allr, open("shake_mc_" + ("_".join(names) if len(names) < 4 else "all") + ".json", "w"), indent=1)
