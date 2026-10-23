# nested_r6_report.py (10-19, Claude): markdown tables from nested_r6_half.json + nested_r6_2to1.json.
import json, sys
H = json.load(open("nested_r6_half.json")); T = json.load(open("nested_r6_2to1.json"))
L = list(H["c4_full_per_label"].keys())
f = lambda s: "%+.5f (%.2f / %.2f)" % (s["mean"], s["p_gt0"], s["p_lt0"])
print("### Part 1: nested variants (held-out macro gain vs c4; P>0 / P<0)\n")
print("| variant | 29/29 (%d reps) | 39/19 (%d reps) |" % (H["reps"], T["reps"]))
print("|---|---|---|")
for v in H["nested"]:
    print("| %s | %s | %s |" % (v, f(H["nested"][v]), f(T["nested"][v])))
print("\n### Part 1b: per-label held-out macro contribution (29/29 | 39/19)\n")
VS = ["free", "shrink50", "down_only", "up_only", "step0.25", "dirpick0.25"]
print("| label | c4 | " + " | ".join(VS) + " | free P>0/P<0 (29/29) | free mean pick |")
print("|---|---|" + "---|" * (len(VS) + 2))
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
for l in L:
    cells = ["%+.5f / %+.5f" % (H["nested"][v]["per_label"][l], T["nested"][v]["per_label"][l]) for v in VS]
    print("| %s | %.2f | %s | %.2f / %.2f | %.2f |" % (l, C4.get(l, 0.5), " | ".join(cells), H["nested"]["free"]["per_label_p_gt0"][l],
                                                   H["nested"]["free"]["per_label_p_lt0"][l], H["nested"]["free"]["mean_pick"][l]))
print("\n### Part 2: direction stability (macro units; held-out = fixed candidate on B)\n")
print("| label | dir | w | step | dirA 29 / 39 | A prefers move (29) | held 29/29 (P>0/P<0) | held 39/19 (P>0/P<0) | held if dirA (29) | conc (29) | full-58 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for k, r in H["direction"].items():
    t = T["direction"][k]
    print("| %s | %s | %.3f | %.3f | %.2f / %.2f | %.2f | %+.5f (%.2f/%.2f) | %+.5f (%.2f/%.2f) | %s | %s | %+.5f |" % (
        r["label"], r["dir"], r["w"], r["step"], r["dir_agree"], t["dir_agree"], r["loc_pos"], r["held_mean"], r["held_p_gt0"], r["held_p_lt0"],
        t["held_mean"], t["held_p_gt0"], t["held_p_lt0"], "%+.5f" % r["held_if_dir"] if r["held_if_dir"] is not None else "-",
        r["concord"], r["full_gain"]))
print("\n### Part 3: fixed slate tables on B (LEAKY)\n")
print("| entry | 29/29 | 39/19 | full-58 | per-label (29/29) |")
print("|---|---|---|---|---|")
for k, s in H["slate"].items():
    print("| %s | %s | %s | %+.5f | %s |" % (k, f(s), f(T["slate"][k]), s["full_gain"], ", ".join("%s %+.5f" % (a, b) for a, b in s["per_label"].items())))
print("\n### full-58 in-sample per-label curves (macro units = per-label gain / 12 x 12? no: per-label AUC gain)\n")
for l in L:
    c = H["full_curve"][l]
    print("%-16s " % l + " ".join("%s:%+.3f" % (w[1:] if w.startswith("0") else w, v) for w, v in c.items()))
