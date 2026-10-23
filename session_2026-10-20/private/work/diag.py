# diag.py: decomposition of who passes us in shake_mc (base unless argv), posterior shrink by public band. MODEL.
import sys, numpy as np, shake_mc as s
from scipy.stats import norm
over = eval(sys.argv[1]) if len(sys.argv) > 1 else {}
P = dict(s.BASE); P.update(over); sI = P["sigP"] / np.sqrt(2)
rows, _ = s.load(); rng = np.random.default_rng(7)
p = s.positions(rows, rng); w, kind = s.classes(rows, p, P, rng)
subs = np.array([r["subs"] for r in rows]); band = np.array([r["band"] for r in rows])
lin = kind == "lin"
b = np.where(subs >= 50, P["beta"] * rng.uniform(0.5, 1.5, p.size), 0.5 * P["beta"] * rng.uniform(0, 1, p.size))
su = np.where(p >= 9.8, 1.5, 2.0); s_lin = np.sqrt((w * sI) ** 2 + P["sig_e"] ** 2 + su ** 2)
grid = np.arange(-320.0, 160.0, 1.0); g, Lk = s.npmle(p[lin] - b[lin], s_lin[lin], grid)
post = Lk * g[None, :]; post /= post.sum(1, keepdims=True); pm = post @ grid; psd = np.sqrt(post @ grid**2 - pm**2)
print("sI", round(sI, 1), "lineage teams", lin.sum())
print("band  n_lin  mean_w  mean_pub  mean_post  shrink  post_sd   (margin over copy, e-4; us pub 18.9, us private base 13)")
for bb in sorted(set(band[lin]), reverse=True):
    m = band[lin] == bb
    print(f"{bb:.3f} {m.sum():5d} {w[lin][m].mean():6.2f} {p[lin][m].mean():8.1f} {pm[m].mean():9.1f} {pm[m].mean()-p[lin][m].mean():7.1f} {psd[m].mean():7.1f}")
# prior g summary
cdf = np.cumsum(g); q = [grid[np.searchsorted(cdf, x)] for x in (.1, .25, .5, .75, .9, .97, .99)]
print("NPMLE g quantiles .1/.25/.5/.75/.9/.97/.99:", q)
# expected passers of a fixed private margin m for us, by group, no common shock and with common shock integrated
for m_us in (8.0, 13.0, 18.0):
    sdp = np.sqrt(psd**2 + (w[lin]*0.655*sI)**2 + (0.655*P["sig_e"])**2)
    mu = pm + P["bonus"]
    grp = {">=.962": band[lin] >= 0.962, ".960-.961": (band[lin] >= 0.960) & (band[lin] < 0.962), ".959": band[lin] == 0.959,
           ".958": band[lin] == 0.958, "<=.957": band[lin] <= 0.957}
    out = []
    for k, mm in grp.items():
        e0 = norm.sf((m_us - mu[mm]) / sdp[mm]).sum()
        # with the common family shock (l_fam - l'_fam) ~ N(0, sI*sqrt(1+.43)) times w
        zs = np.linspace(-3, 3, 61); wz = norm.pdf(zs); wz /= wz.sum()
        cs = sI * np.sqrt(1 + 0.655**2)
        e1 = sum(wq * norm.sf((m_us - mu[mm] - w[lin][mm] * z * cs) / sdp[mm]).sum() for z, wq in zip(zs, wz))
        out.append(f"{k} {mm.sum()} teams: E[pass] {e0:.1f} (common-shock avg {e1:.1f})")
    print(f"us private margin {m_us}: " + " | ".join(out))
