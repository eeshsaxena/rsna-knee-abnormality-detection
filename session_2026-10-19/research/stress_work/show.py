import json, sys
J = json.load(open(sys.argv[1]))
L = list(J["priv_counts_report"])
print("B", J["B"], "priv report", J["priv_counts_report"])
print("point macro:", {k: round(v, 5) for k, v in J["point_macro"].items()})
for k in ["HW", "a25", "c4", "c2max", "c3max", "fr100", "COPY"]:
    print("vs R6u", k, {l: v for l, v in J["point_per_label_vs_R6u"][k].items() if abs(v) > 1e-9})
for k in ["c2max", "c3max", "fr100", "c2", "pf10"]:
    print("vs c4", k, {l: v for l, v in J["point_per_label_vs_c4"][k].items() if abs(v) > 1e-9})
for s in ["S1", "S2", "S3"]:
    print("shift", s, {k: round(v, 5) for k, v in J["shift_macro"][s].items()})
for run, r in J["runs"].items():
    print("\n==", run)
    print(" mean", {k: round(v, 5) for k, v in r["mean"].items() if k in ("R6u", "HW", "a25", "c4", "c2max", "c3max", "fr100", "COPY")})
    print(" sd  ", {k: round(v, 5) for k, v in r["sd"].items() if k in ("R6u", "c4", "c2max")})
    pp = r["p_pair"]
    for a in ["c4", "c2max", "c3max", "fr100"]:
        print("  P(%s > x):" % a, {b: round(pp[a + ">" + b], 3) for b in ["R6u", "HW", "a25", "c4", "c2max", "c3max", "fr100"] if b != a})
    for key, gr in r["grid"].items():
        if key.split("|")[0] in ("c4", "c2max", "c3max", "fr100"):
            print("  %-12s hedge %.6f P(B>A) %.3f mean(B-A) %+.5f sd %.5f corrRaw %.3f corrMargin %.3f P(A>copy) %.3f P(max>copy) %.3f" % (
                key, gr["hedge"], gr["pB_gt_A"], gr["mean_BminusA"], gr["sd_BminusA"], gr["corr_raw"], gr["corr_margin_vs_copy"], gr["pA_gt_copy"], gr["pmax_gt_copy"]))
for nm, r in J["lb"].items():
    print("\n== LB", nm, "ess %.0f" % r["ess"])
    print(" terms", r["terms"])
    print(" prior steps", {k: [round(x, 5) for x in v] for k, v in r["prior_truth_steps"].items()})
    print(" post  steps", {k: [round(x, 5) for x in v] for k, v in r["posterior_truth_steps"].items()})
    for key, gr in r["grid"].items():
        if key.split("|")[0] in ("c4", "c2max", "c3max"):
            print("  %-12s hedge %.6f P(B>A) %.3f mean(B-A) %+.5f sd %.5f corrMargin %.3f P(A>copy) %.3f P(max>copy) %.3f" % (
                key, gr["hedge"], gr["pB_gt_A"], gr["mean_BminusA"], gr["sd_BminusA"], gr["corr_margin_vs_copy"], gr["pA_gt_copy"], gr["pmax_gt_copy"]))
for s, r in J["a15_rule"].items():
    print("\n== A15 rule", s, json.dumps(r, indent=None))
