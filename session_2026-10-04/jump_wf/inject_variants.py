"""Step 3: fleet injection variants inside the parent graph vs the K2 global outer blend, on gold58.
OOS sub-stages (raptor_public, coat_family, raptor_coat_side) are scored honestly. The final is in-sample (DINO / Rad
fitted on these studies' gold labels), so for the final we report (i) movement vs the parent final (mean abs rank shift,
Spearman) and (ii) a PSEUDO-OOS final in which the transformer side is rebuilt out-of-fold: per study, only the DINO
members (4 seeds) and the Rad head whose fold held that study out (recovered from per-study loss; see inject_oof_probe)."""
import numpy as np, pandas as pd
from collections import Counter
from inject_common import *

D = load_all(); Y = D['Y']; ids = D['ids']; n = len(Y); F = D['fleet']
print('gold58 positives per label:', dict(zip(T, Y.sum(0).astype(int).tolist())))
boot = Boot(Y, B=2000, seed=0)


def loss(p):
    p = np.clip(np.asarray(p, np.float64), 1e-4, 1 - 1e-4)
    return -(Y * np.log(p) + (1 - Y) * np.log(1 - p)).mean(1)


# ---------------- pseudo-OOS transformer side ----------------
runs = sorted(set(D['dino_run']))
A = []
for r in runs:
    idx = [i for i, rr in enumerate(D['dino_run']) if rr == r]
    L = np.stack([loss(D['dino'][i]) for i in idx])
    A.append(np.array([D['dino_fold'][idx[k]] for k in np.argmax(L, 0)]))
A = np.stack(A)
dfold = np.array([Counter(A[:, s].tolist()).most_common(1)[0][0] for s in range(n)])
dino_oof = np.zeros((n, 12))
for s in range(n):
    mem = [i for i in range(len(D['dino_ids'])) if D['dino_fold'][i] == dfold[s]]
    dino_oof[s] = D['dino'][mem, s].astype(np.float64).mean(0)
Lrad = sum(np.stack([loss(h) for h in D[k]]) for k in ('rad_e10_raw', 'rad_e13_raw', 'rad_legacy_second_raw'))
rfold = np.argmax(Lrad, 0)
rad_oof = {k: D[k][rfold, np.arange(n)][None].astype(np.float32) for k in ('rad_e10_raw', 'rad_e13_raw', 'rad_legacy_second_raw')}
st1_oof = rank_pct(dino_oof)
st2_oof = stage2_a5(st1_oof, D['a5_rank_mean'], D['a5_ok'])
proto = protocol_features(D['tser'], ids)
st3_oof = stage3_rad(st2_oof, rad_oof['rad_e10_raw'], rad_oof['rad_e13_raw'], rad_oof['rad_legacy_second_raw'], D['cal'], proto)
print(f'pseudo-OOS transformer side: DINO-oof {mauc(Y, st1_oof):.4f} (in-sample stack {mauc(Y, D["st1"]):.4f}); '
      f'+A5 {mauc(Y, st2_oof):.4f}; Rad-oof e10 {mauc(Y, rad_oof["rad_e10_raw"][0]):.4f} e13 {mauc(Y, rad_oof["rad_e13_raw"][0]):.4f}; '
      f'T_oof {mauc(Y, st3_oof):.4f} (in-sample T {mauc(Y, D["st3"]):.4f})')
print(f'rho(T_oof, fleet) {rho_mean(st3_oof, F):.3f}  rho(T_in, fleet) {rho_mean(D["st3"], F):.3f}  '
      f'rho(T_oof, side) {rho_mean(st3_oof, D["side_csv"]):.3f}')

# ---------------- base graph ----------------
arms = list(D['raptor_raw']); cps = [D['coat_prob'][m] for m in COAT_MEMBERS]
pub0 = raptor_public(arms, RAPTOR_W); fam0 = coat_family(cps); side0 = hybrid(pub0, fam0)
fin0 = final_blend(D['st3'], side0); fin0_oof = final_blend(st3_oof, side0)
assert np.abs(pub0 - D['pub_csv']).max() < 1e-7 and (rank_pct(pub0) == rank_pct(D['pub_csv'])).all()
_nd = int((np.abs(fin0 - D['st4']) > 1e-9).sum())
print(f'chained base final vs stored st4: {_nd} of {fin0.size} cells differ (float tie-breaks of ~1e-16 in the side), '
      f'max {np.abs(fin0 - D["st4"]).max():.4f}; AUC chained {mauc(Y, fin0):.4f} stored {mauc(Y, D["st4"]):.4f}. '
      'All deltas below are vs the chained base, so the comparison is like for like.')
probblend0 = np.tensordot(RAPTOR_W / RAPTOR_W.sum(), np.stack([np.clip(a, 0, 1) for a in arms]), axes=(0, 0))
print(f'base gold58: pub {mauc(Y, pub0):.4f} fam {mauc(Y, fam0):.4f} side {mauc(Y, side0):.4f} final(in-sample) {mauc(Y, fin0):.4f} '
      f'final(pseudo-OOS) {mauc(Y, fin0_oof):.4f}')
sd = np.std(F, 0) / np.std(probblend0, 0)
print('fleet prob std / Raptor prob-blend std per label:', dict(zip(T, np.round(sd, 2))), 'mean', round(float(sd.mean()), 2))
print('rho(fleet, each Raptor arm):', [round(rho_mean(F, a), 3) for a in arms], ' rho(fleet, coat members):',
      [round(rho_mean(F, c), 3) for c in cps])
print('gold58 AUC Raptor arms:', [round(mauc(Y, a), 4) for a in arms], ' coat members:', [round(mauc(Y, c), 4) for c in cps])

rows = []


def add(group, name, pub=None, fam=None, side=None, fin=None, fin_oof=None, share=None):
    pub = pub0 if pub is None else pub; fam = fam0 if fam is None else fam
    if side is None:
        side = hybrid(pub, fam)
    if fin is None:
        fin = final_blend(D['st3'], side); fin_oof = final_blend(st3_oof, side)
    r = dict(group=group, variant=name, nominal_fleet_share_final=share,
             d_pub=mauc(Y, pub) - mauc(Y, pub0), d_fam=mauc(Y, fam) - mauc(Y, fam0), d_side=mauc(Y, side) - mauc(Y, side0))
    lo, hi, p = boot.delta(side, side0) if r['d_side'] != 0 else (0, 0, np.nan)
    r.update(side_ci90=f'[{lo:+.4f},{hi:+.4f}]', side_P=p)
    r['final_mars'] = float(np.abs(fin - fin0).mean())
    r['final_rho_old'] = rho_mean(fin, fin0)
    r['final_rho_fleet_gain'] = rho_mean(fin, F) - rho_mean(fin0, F)
    r['d_final_insample'] = mauc(Y, fin) - mauc(Y, fin0)
    r['d_final_pseudoOOS'] = mauc(Y, fin_oof) - mauc(Y, fin0_oof)
    lo, hi, p = boot.delta(fin_oof, fin0_oof)
    r.update(pOOS_ci90=f'[{lo:+.4f},{hi:+.4f}]', pOOS_P=p)
    rows.append(r)
    print(f"{group:3s} {name:34s} share~{share if share is not None else float('nan'):.3f} dPub {r['d_pub']:+.4f} dFam {r['d_fam']:+.4f} "
          f"dSide {r['d_side']:+.4f} {r['side_ci90']} | final MARS {r['final_mars']:.4f} rho {r['final_rho_old']:.4f} "
          f"dIn {r['d_final_insample']:+.4f} | pseudoOOS {r['d_final_pseudoOOS']:+.4f} {r['pOOS_ci90']} P>0 {r['pOOS_P']:.2f}", flush=True)


wl = np.mean([COATNET_W[t] for t in T])      # mean outer Raptor/CoAt weight (.6875)
# (a) fleet as a 5th Raptor arm, probability space (the parent's own arm arithmetic), fleet fraction f of the arm blend
for f in [.05, .10, .15, .20, .25, .33, .50]:
    add('a', f'5th arm prob f={f:.2f}', pub=raptor_public(arms, RAPTOR_W, F, f / (1 - f)), share=wl * .6 * f)
# (a-r) same, rank space (immune to the fleet's probability scale)
for f in [.10, .20, .25, .33, .50]:
    p_ = rank01((1 - f) * rank_pct(probblend0) + f * rank_pct(F)).astype(np.float32).astype(np.float64)
    add('ar', f'5th arm rank f={f:.2f}', pub=p_, share=wl * .6 * f)
# (b) fleet replaces one Raptor arm (takes its weight, probability space)
for k, a in enumerate(RAPTOR_ARMS):
    w = RAPTOR_W.copy(); wf = w[k]; w[k] = 0
    keep = [i for i in range(4) if i != k]
    add('b', f'replace {a} (w{wf:.2f})', pub=raptor_public([arms[i] for i in keep], w[keep], F, wf), share=wl * .6 * wf)
# (c) fleet into the CoAt family (prob mean), fraction f of the family mean; and replacing each member
for f in [.20, .33, .50]:
    add('c', f'coat add f={f:.2f}', fam=rank_pct((1 - f) * (sum(cps) / 4.) + f * F), share=wl * .4 * f)
for k, m in enumerate(COAT_MEMBERS):
    add('c', f'coat replace {m}', fam=coat_family([c for i, c in enumerate(cps) if i != k] + [F]), share=wl * .4 * .25)
# (c2) fleet at the Raptor x CoAt hybrid level (side not re-ranked, like the parent)
for f in [.10, .20, .30]:
    add('c2', f'side-level f={f:.2f}', side=(1 - f) * side0 + f * rank_pct(F), share=wl * f)
# (e) combined inner: 5th arm + coat add
for f1, f2 in [(.25, .20), (.33, .33)]:
    add('e', f'5th arm f={f1:.2f} + coat f={f2:.2f}', pub=raptor_public(arms, RAPTOR_W, F, f1 / (1 - f1)),
        fam=rank_pct((1 - f2) * (sum(cps) / 4.) + f2 * F), share=wl * (.6 * f1 + .4 * f2))
# (d) the K2 global outer blend
for Wg in [.05, .10, .15, .20, .30, .40]:
    add('d', f'global outer W={Wg:.2f}', fin=k2_global(fin0, F, Wg), fin_oof=k2_global(fin0_oof, F, Wg), share=Wg)

res = pd.DataFrame(rows)
res.to_csv('inject_variants_results.csv', index=False)
# equal-movement comparison: global W giving the same final mean-abs-rank-shift as each inner variant
g = res[res.group == 'd'].sort_values('final_mars')
res['global_W_same_MARS'] = np.interp(res.final_mars, g.final_mars, [float(v.split('=')[1]) for v in g.variant])
res.to_csv('inject_variants_results.csv', index=False)
print('\nwrote inject_variants_results.csv')
