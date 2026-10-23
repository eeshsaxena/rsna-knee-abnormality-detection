# picks_cov.py (10-19, think/picks): sampling-noise covariance of macro AUC for our LB entries, public (390) vs private (910).
# Predictions: REAL gold58 serve files (R6 leg, Raptor r384/r224 kernel-serve, S6fall n42gpu) blended exactly in the c4 kernel form
# (same as research/stress_work/final_picks_stress.py). Labels: gold58 (hand labels). Means on gold58 are NOT used (gold58 is a
# discredited screen); only the per-label placement structure is used to get the NOISE of differences (MODEL: population = the 58,
# per label resample n_pos positives / n_neg negatives with replacement, report-rate or gold58-rate counts).
import sys, json, numpy as np, pandas as pd
from scipy.stats import rankdata
D = "C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58"
B = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
HW = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
assert pd.read_csv(D + "/gold58_r224_ids.csv", dtype=str).StudyInstanceUID.tolist() == ids
R6 = pd.read_csv(D + "/R6_gold58_serve.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
R384 = np.load(D + "/gold58_r384_kernelserve_cpu.npy"); R224 = np.load(D + "/gold58_r224_kernelserve.npy")
R7 = pd.read_csv(D + "/R7OOF.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, ["p|" + c for c in L]].values   # R7 OOF stands in for served R7
S6 = pd.read_csv(D + "/n42gpu_gold58.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
pct = lambda v: rankdata(v) / len(v)
def final(table, A, ws6=0.2, wdef=0.5, wover=None, ours=None):
    OURS = R6 if ours is None else ours
    out = np.zeros((n, 12))
    for j, c in enumerate(L):
        rr = (1 - A) * pct(R384[:, j]) + A * pct(R224[:, j]); w = table.get(c, wdef if wover is None else wover)
        b = w * rr + (1 - w) * pct(OURS[:, j]); out[:, j] = (1 - ws6) * pct(b) + ws6 * pct(S6[:, j])
    return out
E = {
    "COPY": final(HW, 0.35, 0.0), "R6u": final({}, 0.35, 0.2), "HW": final(HW, 0.35, 0.2), "a25": final(HW, 0.25, 0.2),
    "c4": final(C4, 0.25, 0.2), "fr100": final({**C4, "Fracture": 1.0}, 0.25), "pf10": final({**C4, "PF OA": 0.1}, 0.25),
    "c2max": final({**C4, "Fracture": 1.0, "PF OA": 0.1}, 0.25), "c2": final({**C4, "Fracture": 0.9, "PF OA": 0.1}, 0.25),
    "c3max": final({**C4, "Fracture": 1.0, "PF OA": 0.1, "Lateral OA": 0.1}, 0.25),
    "fr65": final({**HW, "Fracture": 0.65}, 0.25), "a15": final(HW, 0.15), "c4a15": final(C4, 0.15), "pf50": final({**HW, "PF OA": 0.5}, 0.25),
    "hw3": final(HW, 0.35, 0.3), "s615": final(HW, 0.35, 0.15), "S6W": final({}, 0.35, 0.3),
    "wr45": final(HW, 0.35, 0.2, wover=0.45),
    "c4noS6": final(C4, 0.25, 0.0),
    "S6F": final({}, 0.35, 0.2, ours=R7), "FLIP": final({}, 0.35, 0.0, ours=R7),   # 57201874 (.957) and 57174021 (.956), R7 bundle          # hypothetical hedge entry (not built): c4 without the S6fall leg
    # different-lineage proxies (single legs): an outside team whose model is like one of these
    "leg_R6": np.column_stack([pct(R6[:, j]) for j in range(12)]), "leg_raptor": np.column_stack([pct(0.75 * pct(R384[:, j]) + 0.25 * pct(R224[:, j])) for j in range(12)]),
    "leg_S6": np.column_stack([pct(S6[:, j]) for j in range(12)]),
}
NAMES = list(E); K = len(NAMES); ix = {k: i for i, k in enumerate(NAMES)}
POS = [np.where(Y[:, j] == 1)[0] for j in range(12)]; NEG = [np.where(Y[:, j] == 0)[0] for j in range(12)]
CM = {(k, j): ((E[k][POS[j], j][:, None] > E[k][NEG[j], j][None, :]) + 0.5 * (E[k][POS[j], j][:, None] == E[k][NEG[j], j][None, :])).astype(float)
      for k in NAMES for j in range(12)}
REP = dict(zip(L, [.204, .147, .394, .151, .359, .257, .450, .593, .126, .247, .168, .068]))
def counts(N, regime):
    pos = np.array([round(N * REP[c]) for c in L]) if regime == "report" else np.round(N * Y.mean(0)).astype(int)
    return pos.astype(int), (N - pos).astype(int)
rng = np.random.default_rng(1019)
def draw(N, regime):
    npos, nneg = counts(N, regime); A = np.zeros((B, K, 12))
    for j in range(12):
        wp = rng.multinomial(npos[j], np.full(len(POS[j]), 1 / len(POS[j])), size=B).astype(float)
        wn = rng.multinomial(nneg[j], np.full(len(NEG[j]), 1 / len(NEG[j])), size=B).astype(float)
        for k in NAMES:
            A[:, ix[k], j] = ((wp @ CM[k, j]) * wn).sum(1) / (wp.sum(1) * wn.sum(1))
    return A
out = {"names": NAMES, "B": B}
for tag, N, reg in [("pub_report", 390, "report"), ("priv_report", 910, "report"), ("pub_g58", 390, "g58"), ("priv_g58", 910, "g58")]:
    A = draw(N, reg); M = A.mean(2)
    out[tag] = {"cov": np.cov(M.T).tolist(), "sd_abs": dict(zip(NAMES, M.std(0).round(6).tolist())),
                "per_label_sd_c4": dict(zip(L, A[:, ix["c4"], :].std(0).round(5).tolist()))}
    np.save(f"macro_{tag}.npy", M)
json.dump(out, open("picks_cov.json", "w"))
# print the key paired sds
for tag in ("pub_report", "priv_report", "pub_g58", "priv_g58"):
    C = np.array(out[tag]["cov"]); sd = lambda a, b: float(np.sqrt(C[ix[a], ix[a]] + C[ix[b], ix[b]] - 2 * C[ix[a], ix[b]]))
    print(f"== {tag}: abs sd c4 {np.sqrt(C[ix['c4'], ix['c4']]):.5f}")
    for a, b in [("c4", "COPY"), ("HW", "COPY"), ("R6u", "COPY"), ("a25", "COPY"), ("c4", "R6u"), ("c4", "HW"), ("c4", "a25"), ("HW", "R6u"), ("a25", "HW"),
                 ("fr100", "c4"), ("pf10", "c4"), ("c2max", "c4"), ("c2", "c4"), ("c3max", "c4"), ("fr65", "a25"), ("pf50", "a25"), ("a15", "HW"), ("c4a15", "c4"),
                 ("hw3", "HW"), ("s615", "HW"), ("wr45", "HW"), ("S6W", "R6u"), ("S6W", "c4"),
                 ("c4", "leg_R6"), ("c4", "leg_raptor"), ("c4", "leg_S6"), ("COPY", "leg_S6")]:
        print(f"   sd({a}-{b}) = {sd(a, b):.6f}")
