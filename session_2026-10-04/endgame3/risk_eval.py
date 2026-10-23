# Reads risk_draws.npy (risk_sim.py) and scores pick policies, W-selection overfit, fallback contamination and
# read-conditional posteriors. All outputs are MODEL numbers (binormal two-reader + lens_rules field curves).
import numpy as np, itertools, glob, sys
from risk_sim import pmed, disp, SP
PRI = sys.argv[1] if len(sys.argv) > 1 else "cen"
FILES = sorted(glob.glob(SP + "/endgame3/risk_draws_%s*.npy" % PRI))
R = [r for f in FILES for r in np.load(f, allow_pickle=True)]; S = len(R)
print("prior set", PRI, "files", len(FILES))
CW = ["C25", "C20", "C15", "C35", "C45"]            # tie preference: pre-registered .25 first, then lower W
FW = ["F30", "F50"]; ALL = CW + FW + ["CF"]
PREF = {k: i for i, k in enumerate(["C25", "CF", "F30", "C20", "C15", "F50", "C35", "C45"])}
COPY = -1e-9                                         # exact parent copy: loses the timestamp tie
def D(r, k): return disp(r["p0"], r[k][0]) if k != "R" else disp(r["p0"], 0.0)
def V(r, k): return r[k][1] if k != "R" else COPY
def top(r, ks, n=1, avail=None):
    ks = [k for k in ks if avail is None or k in avail]
    return sorted(ks, key=lambda k: (-D(r, k), PREF.get(k, 99)))[:n]
def score(xs):
    xs = np.asarray(xs)
    return dict(E=xs.mean() * 1e4, P0=(xs > 0).mean(), P1=(xs > 0.001).mean(),
                Bb=pmed(xs, 0, 2).mean(), Bco=pmed(xs, 3, 2).mean(), Sb=pmed(xs, 0, 1).mean(), Sco=pmed(xs, 3, 1).mean())
def line(name, xs, extra=""):
    s = score(xs)
    print("%-44s E %+5.1f e-4 | P(>0) %.2f P(>+.001) %.2f | bronze %.2f/%.2f silver %.2f/%.2f %s" % (
        name, s["E"], s["P0"], s["P1"], s["Bb"], s["Bco"], s["Sb"], s["Sco"], extra))
print("draws", S, "| bronze/silver = base / creep+overfit field scenario\n")
print("== A. single candidates (no selection) ==")
line("R  pure repro (exact copy)", [COPY] * S)
for k in ALL + ["Csolo", "Fsolo"]:
    if k.endswith("solo"):
        print("%-44s public delta mean %+.4f (solo vs parent)" % (k, np.mean([r[k][0] for r in R]))); continue
    line(k, [V(r, k) for r in R], "| pub-disp>=.944 %.2f" % np.mean([D(r, k) >= 0.944 for r in R]))
print("\n== B. W selection inside ONE member (public overfit from many W variants) ==")
def wpick(r, rule):
    if rule == "fixed C25": return "C25"
    if rule == "max disp of 5 W": return top(r, CW)[0]
    if rule == "max disp of {C15,C25}": return top(r, ["C25", "C15"])[0]
    if rule == "max disp of {C25,C35}": return top(r, ["C25", "C35"])[0]
    if rule == "shrunk: other W only if >= C25+.002":
        b = top(r, CW)[0]; return b if D(r, b) >= D(r, "C25") + 0.002 - 1e-9 else "C25"
    if rule == "oracle best W (unattainable)": return max(CW, key=lambda k: r[k][1])
for rule in ["fixed C25", "max disp of {C15,C25}", "max disp of {C25,C35}", "max disp of 5 W",
             "shrunk: other W only if >= C25+.002", "oracle best W (unattainable)"]:
    ks = [wpick(r, rule) for r in R]
    opt = np.mean([(D(r, k) - 0.943 - r["p0"]) - r[k][1] for r, k in zip(R, ks)]) * 1e4
    hiW = np.mean([k in ("C35", "C45") for k in ks])
    line(rule, [V(r, k) for r, k in zip(R, ks)], "| optimism %+.1f e-4, picks W>=.35 %.0f%%" % (opt, 100 * hiW))
print("\n== C. final-2 portfolios (max of the two private deltas) ==")
def port(r, pol, avail=None):
    av = avail if avail is not None else set(ALL)
    if pol == "R + C25 (repro floor + crop)": return ["R", "C25"]
    if pol == "same member: top-2 crop W": return top(r, CW, 2, av)
    if pol == "pre-registered C25 + F30": return [k for k in ["C25", "F30"] if k in av]
    if pol == "best crop W + best fleet W": return top(r, CW, 1, av) + top(r, FW, 1, av)
    if pol == "top-2 display overall": return top(r, ALL, 2, av)
    if pol == "CF + best crop W": return [k for k in ["CF"] if k in av] + top(r, CW, 1, av)
    if pol == "CF + best fleet W": return [k for k in ["CF"] if k in av] + top(r, FW, 1, av)
    if pol == "rule: top overall, then best of OTHER family":
        a = top(r, ALL, 1, av)
        if not a: return []
        a = a[0]; fam = "C" if a.startswith("C") and a != "CF" else "F"
        other = FW if fam == "C" else CW
        return [a] + top(r, other, 1, av)
POLS = ["R + C25 (repro floor + crop)", "same member: top-2 crop W", "pre-registered C25 + F30",
        "best crop W + best fleet W", "top-2 display overall", "CF + best crop W", "CF + best fleet W",
        "rule: top overall, then best of OTHER family"]
for pol in POLS:
    xs = [max([V(r, k) for k in port(r, pol)] or [COPY]) for r in R]
    line(pol, xs)
print("\n== C2. correlation of private deltas (why 2 W of one member is a wasted pick) ==")
M = np.array([[r[k][1] for k in ["C15", "C25", "C45", "F30", "F50", "CF"]] for r in R])
print("        C15   C25   C45   F30   F50   CF")
for i, k in enumerate(["C15", "C25", "C45", "F30", "F50", "CF"]):
    print("%-5s" % k, " ".join("%5.2f" % v for v in np.corrcoef(M.T)[i]))
print("\n== D. silent fallback (parent bytes) vs fail-loud, policy 'best crop W + best fleet W' ==")
rng = np.random.default_rng(3)
for f in [0.0, 0.10, 0.25]:
    for mode in ["per-version (all W of a member)", "per-submission (independent)"]:
        xs_safe, xs_loud = [], []
        for r in R:
            if mode.startswith("per-version"):
                fbC, fbF = rng.random() < f, rng.random() < f
                fb = {k: (fbC if k.startswith("C") else fbF) for k in ALL}
                fb["CF"] = fbC or fbF
            else:
                fb = {k: rng.random() < f for k in ALL}
            # fail-safe: a fallback reads exactly the parent display and is an exact copy privately
            rr = dict(r);
            for k in ALL:
                if fb[k]: rr[k] = (0.0, COPY)
            xs_safe.append(max(V(rr, k) for k in port(rr, "best crop W + best fleet W")))
            av = {k for k in ALL if not fb[k]}
            ks = port(r, "best crop W + best fleet W", av)
            xs_loud.append(max([V(r, k) for k in ks] or [COPY]))
        line("f=%.2f %s fail-SAFE" % (f, mode[:12]), xs_safe)
        line("f=%.2f %s fail-LOUD" % (f, mode[:12]), xs_loud)
print("\n== E. what a C25 read means (old K=24 cell, member prior .912-.930, rho .78-.90) ==")
for d in [0.941, 0.942, 0.943, 0.944, 0.945]:
    sel = [r for r in R if (D(r, "C25") == d if d < 0.945 else D(r, "C25") >= 0.945)]
    if len(sel) < 20: print("C25 reads %.3f%s: n=%d (too few)" % (d, "+" if d >= 0.945 else "", len(sel))); continue
    line("C25 reads %.3f%s (n=%d)" % (d, "+" if d >= 0.945 else "", len(sel)), [r["C25"][1] for r in sel],
         "| rho %.2f m1 %.3f" % (np.mean([r["r1"] for r in sel]), np.mean([r["m1"] for r in sel])))
print("\n== E2. joint C15/C25 reads -> which W to keep ==")
for a, b in itertools.product([0.942, 0.943, 0.944], [0.942, 0.943, 0.944, 0.945]):
    sel = [r for r in R if D(r, "C15") == a and D(r, "C25") == b]
    if len(sel) < 25: continue
    e = {k: np.mean([r[k][1] for r in sel]) * 1e4 for k in CW}
    p1 = {k: np.mean([r[k][1] > 0.001 for r in sel]) for k in CW}
    print("C15 %.3f C25 %.3f n=%4d | E priv e-4: " % (a, b, len(sel)) + " ".join("%s %+.1f" % (k, e[k]) for k in ["C15", "C20", "C25", "C35", "C45"])
          + " | P(>+.001): " + " ".join("%s %.2f" % (k, p1[k]) for k in ["C15", "C25", "C35"]))
print("\n== F. fleet read G3 (F50) -> posterior ==")
for d in [0.942, 0.943, 0.944, 0.945]:
    sel = [r for r in R if (D(r, "F50") == d if d < 0.945 else D(r, "F50") >= 0.945)]
    if len(sel) < 20: continue
    line("F50 reads %.3f%s (n=%d)" % (d, "+" if d >= 0.945 else "", len(sel)), [r["F50"][1] for r in sel],
         "| F30 E %+.1f e-4 | m2 %.3f r2 %.2f" % (np.mean([r["F30"][1] for r in sel]) * 1e4, np.mean([r["m2"] for r in sel]), np.mean([r["r2"] for r in sel])))
print("\n== E3. solo read (fleetsolo-style diagnostic) + C25 read -> best W ==")
for lo, hi in [(0.905, 0.918), (0.918, 0.924), (0.924, 0.935)]:
    for b in [0.942, 0.943, 0.944, 0.945]:
        sel = [r for r in R if lo <= disp(r["p0"], r["Csolo"][0]) < hi and (D(r, "C25") == b if b < 0.945 else D(r, "C25") >= 0.945)]
        if len(sel) < 25: continue
        e = {k: np.mean([r[k][1] for r in sel]) * 1e4 for k in CW}
        best = max(e, key=e.get)
        print("solo [%.3f,%.3f) C25 %.3f%s n=%4d | E priv e-4: " % (lo, hi, b, "+" if b >= 0.945 else " ", len(sel))
              + " ".join("%s %+.1f" % (k, e[k]) for k in ["C15", "C20", "C25", "C35", "C45"]) + " | best " + best
              + " | rho %.2f" % np.mean([r["r1"] for r in sel]))
