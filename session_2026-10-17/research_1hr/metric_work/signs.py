# signs.py: MODEL. mu = true macro delta; gold58 point g ~ N(mu, sd58^2 + mis^2); public x ~ N(mu, sp^2); private z ~ N(mu, sv^2).
import json, sys, numpy as np
R = json.load(open(sys.argv[1])); rng = np.random.default_rng(7); M = 2_000_000
rows = [("wr45 - HW", "probe"), ("s615 - HW", "probe"), ("HW - clone(heliosli,noS6)", "pick"), ("R6pick(uniform+S6) - clone(heliosli,noS6)", "pick"), ("HW - R6pick(uniform+S6)", "pick")]
for key, kind in rows:
    r = R[key]; g, s58 = r["gold58_point"], r["sd_n58"]
    for sc in ("gold58", "report"):
        sp, sv = r[f"sd_pub_{sc}"], r[f"sd_priv_{sc}"]
        for prior, mis in (("flat", None), ("gold58", 0.001)):
            if prior == "flat":
                mu = rng.normal(0, 5e-4, M)                     # GUESS: tiny one-line edits, sign unknown
            else:
                v = s58**2 + mis**2; tau2 = 1e-3**2                # broad N(0, 1e-3^2) prior updated by gold58
                post_v = 1 / (1 / tau2 + 1 / v); post_m = post_v * g / v
                mu = rng.normal(post_m, np.sqrt(post_v), M)
            x = mu + rng.normal(0, sp, M); z = mu + rng.normal(0, sv, M); pos = x > 0
            print(f"{key:44s} {sc:6s} prior {prior:6s} | P(pub>0) {pos.mean():.2f} | P(priv>0) {(z>0).mean():.2f} | P(priv>0|pub>0) {(z[pos]>0).mean():.2f}"
                  f" | E[priv d|pub>0] {z[pos].mean():+.5f} | E[priv d|pub<=0] {z[~pos].mean():+.5f}")
