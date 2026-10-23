# calib.py: shake-up intensity of the shake_mc field model, in the units of the 742327 forum study (CLAIMED there:
# teams at public ranks 75-100% of the bronze cut medalled 24% of the time, range 2-51%, 11 comps of 3,000-6,750 teams).
# Re-implements one field replicate of shake_mc.run and ranks EVERY team (field + copy clusters) on private. MODEL.
import sys, json, numpy as np, shake_mc as s
from scipy.stats import norm
def one(over, rep=0, NS=400):
    P = dict(s.BASE); P.update(over); sI = P["sigP"] / np.sqrt(2.0)
    rows, _ = s.load(); rng = np.random.default_rng(1000 * rep + 7)
    p = s.positions(rows, rng); w, kind = s.classes(rows, p, P, rng)
    subs = np.array([r["subs"] for r in rows]); run_id = np.array([r["run"] for r in rows]); prank = np.array([r["rank"] for r in rows])
    lin = kind == "lin"
    b = np.where(subs >= 50, P["beta"] * rng.uniform(0.5, 1.5, p.size), 0.5 * P["beta"] * rng.uniform(0, 1, p.size))
    su = np.where(p >= 9.8, 1.5, 2.0); s_lin = np.sqrt((w * sI) ** 2 + P["sig_e"] ** 2 + su ** 2)
    grid = np.arange(-320.0, 160.0, 1.0); g, Lk = s.npmle(p[lin] - b[lin], s_lin[lin], grid)
    post = Lk * g[None, :]; post /= post.sum(1, keepdims=True); pm = post @ grid; psd = np.sqrt(np.maximum(post @ grid ** 2 - pm ** 2, 0))
    PM = np.zeros(p.size); PSD = np.zeros(p.size); PM[lin] = pm; PSD[lin] = psd
    tw = kind == "tw"; base0 = np.where(p > -2, 0.0, p); lam = 4.0 / (4.0 + P["sig_e"] ** 2)
    PM[tw] = (base0 + 1.0 + lam * (p - b - base0 - 1.0))[tw]; PSD[tw] = np.sqrt(lam) * P["sig_e"]
    cpy = kind == "copy"; cpl = kind == "copylow"
    dfam = P["dfam"] * sI
    r2 = np.random.default_rng(55 + rep)
    sd = np.sqrt(PSD ** 2 + (w * 0.655 * sI) ** 2 + (w * P["ssite"]) ** 2 + (0.655 * P["sig_e"]) ** 2)
    out = {"silver": [], "bronze": [], "silver_out": [], "bronze_out": [], "top150kept": [], "first_rank": []}
    N = 6276; silver = N // 20; bronze = N // 10
    cl = np.unique(run_id[cpy])
    for k in range(NS):
        lf = r2.normal(dfam, sI); lpf = r2.normal(0, 0.655 * sI); gf = r2.normal(0, P["ssite"] + 1e-12)
        y = PM + sd * r2.standard_normal(p.size) + w * (lf - lpf - gf) + np.where(lin, P["bonus"], 0)
        dc = dict(zip(cl, r2.normal(0, 1.0, cl.size)))
        y = np.where(cpy, np.array([dc.get(c, 0.0) for c in run_id]), y)
        y = np.where(cpl, p + 3.0 * r2.standard_normal(p.size), y)
        # tie-break exact copies by public rank (earlier submission first), as Kaggle does
        order = np.lexsort((prank, -np.round(y, 6))); pr = np.empty(p.size, int); pr[order] = np.arange(1, p.size + 1)
        for lab, cut in (("silver", silver), ("bronze", bronze)):
            ins = (prank > 0.75 * cut) & (prank <= cut); outs = (prank > cut) & (prank <= 1.25 * cut)
            out[lab].append((pr[ins] <= cut).mean()); out[lab + "_out"].append((pr[outs] <= cut).mean())
        out["top150kept"].append((pr[prank <= 150] <= 150).mean())
        out["first_rank"].append(pr[prank == prank.min()][0])
    return {k: (float(np.mean(v)) if k != "first_rank" else [int(np.percentile(v, q)) for q in (10, 50, 90)]) for k, v in out.items()}
if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["base", "sigP10_fieldmc", "sigP30", "famLucky", "site20"]):
        r = one(s.SCEN[nm])
        print(f"{nm:14s} P(medal tier kept) just inside silver cut {r['silver']:.2f}, just outside {r['silver_out']:.2f} | "
              f"just inside bronze cut {r['bronze']:.2f}, just outside {r['bronze_out']:.2f} | top-150 kept {r['top150kept']:.2f} | "
              f"public #1 private rank p10/50/90 {r['first_rank']}")
