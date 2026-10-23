# cv_shrink_report.py: summarise cv_shrink_*.json (markdown tables) + MODEL extrapolation of the shrink curve to n_train = 58.
# MODEL: per label the A-half pick deviation d = delta* + eps (Var eps = s2 / n_train); held-out gain of w = c4 + lam d is
#   G(lam) = a lam - b lam^2 with a = 2 k delta*^2, b = k (delta*^2 + s2/n_train).  Fit a, b by least squares on the 9-point lam curve,
#   then rescale the noise part of b to another n_train: b_n = a/2 + (b - a/2) * n_fit / n.  lam*_n = a / (2 b_n).
import sys, json, numpy as np
LAMS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]; MAIN = [0.0, 0.25, 0.5, 0.75, 1.0]


def fit(curve):
    lam = np.array(LAMS); g = np.array([curve[str(l)]["mean"] for l in LAMS])
    X = np.stack([lam, -lam ** 2], 1); (a, b), *_ = np.linalg.lstsq(X, g, rcond=None); return a, b


def extrap(a, b, nfit, n):
    bn = b if b < a / 2 else a / 2 + (b - a / 2) * nfit / n      # b < a/2 = model violated (near-linear curve, boundary-capped picks): no rescale
    if a <= 0: return 0.0, 0.0, bn
    lam = min(1.0, a / (2 * bn)) if bn > 0 else 1.0
    return lam, a * lam - bn * lam ** 2, bn


for f in sys.argv[1:]:
    r = json.load(open(f)); nfit = r["ntrain"]
    print("\n## %s  (ours %s, base %s, train %d / test %d, %d reps)" % (f, r["ours"], r["base"], nfit, r["n"] - nfit, r["reps"]))
    print("| rule | mean abs(S) | " + " | ".join("lam %.3g" % l for l in MAIN) + " | MODEL lam*@%d (gain) | MODEL lam*@58 (gain) |" % nfit)
    print("|---|---|" + "---|" * (len(MAIN) + 2))
    for v, cur in r["cv"].items():
        a, b = fit(cur); l1, g1, _ = extrap(a, b, nfit, nfit); l2, g2, _ = extrap(a, b, nfit, 58)
        cells = ["%+.5f (P>0 %.2f)" % (cur[str(l)]["mean"], cur[str(l)]["p_gt0"]) if l > 0 else "0" for l in MAIN]
        print("| %s | %.2f | %s | %.2f (%+.5f) | %.2f (%+.5f) |" % (v, cur["1.0"]["mean_size"], " | ".join(cells), l1, g1, l2, g2))
    print("\nslate (LEAKY):", " ".join("%s %+.5f (P>0 %.2f)" % (k, s["mean"], s["p_gt0"]) for k, s in r["slate"].items()))
    print("per-label held-out macro contribution, ALL rule (label moved alone to its A-half pick):")
    print("| label | full pick | full dir-agree | nested dir-agree mean | " + " | ".join("lam %.3g" % l for l in MAIN[1:]) + " |")
    print("|---|---|---|---|" + "---|" * 4)
    for l in r["full"]["pick"]:
        print("| %s | %.2f | %s | %s | %s |" % (l, r["full"]["pick"][l], r["full"]["dir_agree"][l], r["nested_agree_mean"][l],
                                           " | ".join("%+.5f%s" % (r["cv"]["ALL"][str(x)]["per_label"][l] / 12, (" (P>0 %.2f)" % r["cv"]["ALL"][str(x)]["per_label_p_gt0"][l]) if "per_label_p_gt0" in r["cv"]["ALL"][str(x)] else "") for x in MAIN[1:])))
    print("TOP1 by label:", {k: (v["freq"], round(v["heldout_macro_l1"], 5), round(v["heldout_macro_l0.5"], 5)) for k, v in r["top1_by_label"].items()})
    print("picks_freq Fracture:", r["picks_freq"]["Fracture"], "| PF OA:", r["picks_freq"]["PF OA"])
    print("paired:", {k: (round(v["mean"], 5), round(v["p_gt0"], 2)) for k, v in r["paired"].items()})
    for v in ("ALL", "NEST_DIR75", "NEST_DIR60", "TOP1", "TOP2", "FULL_DIR75(leaky)"):
        print("full table %-18s" % v, {l: (t["changes"], round(t["insample_gain"], 5)) for l, t in r["full_tables"][v].items() if float(l) in (0.25, 0.5, 0.75, 1.0)})
