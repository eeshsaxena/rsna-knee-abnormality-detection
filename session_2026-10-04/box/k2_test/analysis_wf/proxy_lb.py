"""proxy_lb.py : LB calibration of the binormal blend model + the K2 conditional table + own-ensemble projections.

Inputs (box, analysis_wf): proxy_fit_out.json (gold58 effective r of our members vs the OOS parent sides),
proxy_gold58_out.json (per-label profiles), proxy_oof867_out.json (member-to-member ensembles on fold-0 val).
LB anchors are typed in below from the session context / HANDOFF (public LB, displayed to 3 decimals).
Model: macro-level equal-variance binormal, two readers, latent-score blend at weight W (validated on gold58 for
global rank blends: proxy_gold58.log / proxy_fit.log). r here is always the EFFECTIVE (blend-fitted) correlation.
"""
import os, json, numpy as np
from scipy.stats import norm

H = os.path.expanduser
D = H('~/k2_test/analysis_wf/')
FIT = json.load(open(D + 'proxy_fit_out.json'))
G = json.load(open(D + 'proxy_gold58_out.json'))
O = json.load(open(D + 'proxy_oof867_out.json'))
out = {}


def dv(a): return np.sqrt(2) * norm.ppf(a)


def B(P, a, r, w):
    v = (1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * r
    return norm.cdf(((1 - w) * dv(P) + w * dv(a)) / np.sqrt(2 * v))


RG = np.linspace(0.40, 0.999, 1200)
WG = np.linspace(0, 1, 201)
P0 = 0.943
PG = np.linspace(0.9425, 0.9435, 5)   # true parent given the displayed .943


def feas(a, reads, PGr=PG, atol=0.0):
    ok = np.zeros_like(RG, bool)
    for P in PGr:
        for aa in ([a] if atol == 0 else np.linspace(a - atol, a + atol, 5)):
            g = np.ones_like(RG, bool)
            for w, d in reads: g &= np.abs(B(P, aa, RG, w) - d) <= 0.0005 + 1e-12
            ok |= g
    ls = RG[np.argmin(sum((B(PGr.mean(), a, RG, w) - d) ** 2 for w, d in reads))]
    lo, hi = (RG[ok].min(), RG[ok].max()) if ok.any() else (np.nan, np.nan)
    return float(lo), float(hi), float(ls)


def best(P, a, r):
    b = B(P, a, r, WG); i = int(np.argmax(b)); return float(WG[i]), float(b[i] - P)


print('=' * 100)
print('A. LB anchors -> implied effective r (macro binormal; displayed values +-.0005)')
anchors = {
    'C1 cropaug_f1 single, K24 served (.15->.942, .25->.941)': [(0.15, 0.942), (0.25, 0.941)],
    'C2 mixSA_f1 single (.25->.940)': [(0.25, 0.940)],
    'C3 K1 crop ens3, K8 (.25->.942)': [(0.25, 0.942)],
}
SOL = [0.905, 0.910, 0.915, 0.920, 0.925, 0.930, 0.935]
out['anchors'] = {}
for nm, reads in anchors.items():
    print(f'  {nm}')
    out['anchors'][nm] = {}
    for a in SOL:
        lo, hi, ls = feas(a, reads)
        out['anchors'][nm][a] = (lo, hi, ls)
        print(f'     assumed solo {a:.3f}: r in [{lo:.3f}, {hi:.3f}]  LS {ls:.3f}')
# cnx (solo known .894) on parent and on the fast base
print('  X cnx M448 fold0, solo .894 (MEASURED):')
x_par = feas(0.894, [(0.5, 0.930)], PG, atol=0.0005)
x_f2 = feas(0.894, [(0.4, 0.935), (0.5, 0.930)], np.linspace(0.9415, 0.9425, 5), atol=0.0005)
x_f3 = feas(0.894, [(0.4, 0.935), (0.5, 0.930)], PG, atol=0.0005)
print(f'     full parent W.50->.930: r in [{x_par[0]:.3f}, {x_par[1]:.3f}] LS {x_par[2]:.3f}')
print(f'     fast base (.942) W.40->.935, W.50->.930: r in [{x_f2[0]:.3f}, {x_f2[1]:.3f}] LS {x_f2[2]:.3f}  (if fast base were .943: [{x_f3[0]:.3f}, {x_f3[1]:.3f}])')
out['cnx'] = dict(parent=x_par, fast942=x_f2, fast943=x_f3)
# external (other teams, from notes; lower trust)
t_ = feas(0.927, [(0.10, 0.942), (0.20, 0.941)], np.linspace(0.9415, 0.9425, 5), atol=0.0005)
aa_ = feas(0.928, [(0.50, 0.945), (0.60, 0.945)], PG, atol=0.0005)
print(f'  external: Tian v09r .927 on a .942 fork (.10->.942, .20->.941): r in [{t_[0]:.3f}, {t_[1]:.3f}] LS {t_[2]:.3f} (gold58 Spearman .835 per notes)')
print(f'  external CLAIM: aastik fleet .928 on .943 (.5/.6 -> .945): r in [{aa_[0]:.3f}, {aa_[1]:.3f}] LS {aa_[2]:.3f}')
out['external'] = dict(tian=t_, aastik_claim=aa_)

print('\nB. effective parent strength (P_eff) from the cnx curvature (fast base, 2 reads, solo known):')
PP = np.arange(0.930, 0.9601, 0.00025); reg = []
for P in PP:
    for a in (0.8935, 0.894, 0.8945):
        g = (np.abs(B(P, a, RG, 0.4) - 0.935) <= 0.0005) & (np.abs(B(P, a, RG, 0.5) - 0.930) <= 0.0005)
        reg += [(P, r) for r in RG[g]]
reg = np.array(reg)
print(f'  feasible (P_eff, r): P_eff in [{reg[:, 0].min():.4f}, {reg[:, 0].max():.4f}], r in [{reg[:, 1].min():.3f}, {reg[:, 1].max():.3f}]'
      f'  (fast base displays .942)')
out['P_eff_cnx'] = dict(P=[float(reg[:, 0].min()), float(reg[:, 0].max())], r=[float(reg[:, 1].min()), float(reg[:, 1].max())])
# per-label heterogeneity check: profile shapes from gold58 (parent ~ raptor_coat_side shape, member ~ fleet shape)
T = list(G['arms']['fleet_f0_mix50']['per_label'])
prP = np.clip(np.array([G['arms']['raptor_coat_side_rank']['per_label'][t] for t in T]), 0.5, 0.995)
prM = np.clip(np.array([G['arms']['fleet_f0_mix50']['per_label'][t] for t in T]), 0.5, 0.995)


def shift(prof, target):
    lo, hi = -3.0, 3.0
    for _ in range(60):
        c = (lo + hi) / 2
        if norm.cdf((dv(prof) + c) / np.sqrt(2)).mean() < target: lo = c
        else: hi = c
    return dv(prof) + c


def Bpl(P, a, r, w):
    dP, dM = shift(prP, P), shift(prM, a)
    v = (1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * r
    return float(norm.cdf(((1 - w) * dP + w * dM) / np.sqrt(2 * v)).mean())


diffs = []
for a in (0.905, 0.915, 0.925, 0.935):
    for r in (0.85, 0.90, 0.95):
        for w in (0.15, 0.30, 0.50):
            diffs.append((a, r, w, Bpl(P0, a, r, w) - P0, float(B(P0, a, r, w) - P0)))
dd = np.array([d[3] - d[4] for d in diffs])
print(f'  per-label (gold58 shapes) minus macro model, delta over 36 cases: mean {dd.mean():+.4f}, max |.| {np.abs(dd).max():.4f}')
for d in diffs:
    if d[1] == 0.90 and d[2] == 0.30: print(f'     a {d[0]} r .90 W .30: per-label {d[3]:+.4f} macro {d[4]:+.4f}')
out['perlabel_vs_macro'] = dict(mean=float(dd.mean()), maxabs=float(np.abs(dd).max()))

print('\n' + '=' * 100)
print('C. transfer gold58 -> LB for the fleet (effective r vs the parent)')
refs = ['raptor_coat_side_rank', 'parent_final', 'raptor_public_rank', 'coat_family_rank']
gm = {m: {b: FIT['members'][m][b]['r_fit'] for b in refs} for m in FIT['members']}
crop_cases = {  # anchor, gold58 member, solo central, solo range (GUESS / MODEL)
    'C1': ('C1 cropaug_f1 single, K24 served (.15->.942, .25->.941)', 'crop_cropaug_f1', 0.920, (0.912, 0.926)),
    'C2': ('C2 mixSA_f1 single (.25->.940)', 'crop_mixSA_f1', 0.915, (0.907, 0.922)),
    'C3': ('C3 K1 crop ens3, K8 (.25->.942)', 'crop_K1_ens3', 0.925, (0.918, 0.931)),
}
rows = []
for key, (anm, gmem, sc, (s0, s1)) in crop_cases.items():
    reads = anchors[anm]
    lo_, hi_, ls_c = feas(sc, reads)
    rr = [feas(s, reads) for s in np.linspace(s0, s1, 7)]
    rlo = np.nanmin([x[0] for x in rr]); rhi = np.nanmax([x[1] for x in rr])
    for b in refs:
        g_c, g_f = gm[gmem][b], gm['fleet_f0_mix50'][b]
        for form in ('ratio', 'additive'):
            def tr_(rlb):
                if form == 'ratio': return 1 - (1 - g_f) * (1 - rlb) / (1 - g_c)
                return min(0.999, g_f + (rlb - g_c))
            rows.append(dict(anchor=key, ref=b, form=form, r_crop_lb=ls_c, r_crop_g58=g_c, r_fleet_g58=g_f,
                             fleet_c=tr_(ls_c), fleet_lo=tr_(rlo), fleet_hi=tr_(rhi)))
            print(f'  {key} solo {sc:.3f} [{s0:.3f},{s1:.3f}] r_LB {ls_c:.3f} [{rlo:.3f},{rhi:.3f}] | ref {b[:18]:18s} g58 crop {g_c:.3f} fleet {g_f:.3f}'
                  f' | {form:8s} -> fleet r_LB {tr_(ls_c):.3f} [{tr_(rlo):.3f},{tr_(rhi):.3f}]')
fc = np.array([r['fleet_c'] for r in rows])
flo = np.array([r['fleet_lo'] for r in rows]); fhi = np.array([r['fleet_hi'] for r in rows])
r_c = float(np.median(fc)); r_lo = float(np.percentile(np.minimum(flo, fhi), 10)); r_hi = float(min(0.999, np.percentile(np.maximum(flo, fhi), 90)))
r_lo_x = float(np.min(np.minimum(flo, fhi)))
print(f'  FLEET r_LB: median of central transfers {r_c:.3f}; 10-90% of transfer ranges [{r_lo:.3f}, {r_hi:.3f}]; extreme low {r_lo_x:.3f}')
print(f'  (no-transfer reference: fleet gold58 r_fit vs raptor_coat_side {gm["fleet_f0_mix50"]["raptor_coat_side_rank"]:.3f}, vs parent_final {gm["fleet_f0_mix50"]["parent_final"]:.3f})')
out['fleet_rLB'] = dict(central=r_c, lo=r_lo, hi=r_hi, extreme_lo=r_lo_x, rows=rows)

print('\n' + '=' * 100)
print('D. member-to-member effective r (ensemble model)')
ens_r = []
for e in O['ens_check']:
    a1, a2 = O['members'][e['a']], O['members'][e['b']]
    d_m = (dv(a1) + dv(a2)) / 2
    # Phi(d_m / sqrt(2 v)) = measured, v = (1 + r) / 2
    z = norm.ppf(e['measured']); v = (d_m / z) ** 2 / 2; r = 2 * v - 1
    lat = O['pairs'].get(f"{e['a']}|{e['b']}", O['pairs'].get(f"{e['b']}|{e['a']}", {})).get('rA', np.nan)
    ens_r.append((e['a'], e['b'], float(r), lat))
    print(f'  {e["a"]}+{e["b"]}: measured ens {e["measured"]:.4f} -> effective r_mm {r:.3f} (latent rA {lat:.3f})')
d_t = dv(0.927); z = norm.ppf(0.930); r_t = 2 * ((d_t / z) ** 2 / 2) - 1
print(f'  external: Tian 2 seeds .927 -> .930 => effective r_mm {r_t:.3f}')
out['r_mm'] = dict(pairs=ens_r, tian=float(r_t))
RMM = (0.93, 0.95, 0.97)


def ens(s, r_mp, k, r_mm):
    v = (1 + (k - 1) * r_mm) / k
    return float(norm.cdf(dv(s) / np.sqrt(2 * v))), float(min(0.999, r_mp / np.sqrt(v)))


print('\n' + '=' * 100)
print(f'E. K2 conditional table (parent .943 true; fleet r_LB prior central {r_c:.3f}, range [{r_lo:.3f}, {r_hi:.3f}])')
SROWS = [0.905, 0.915, 0.925, 0.935]
tabE = []
for s in SROWS:
    b_c = float(B(P0, s, r_c, 0.30)); b_lo = float(B(P0, s, r_hi, 0.30)); b_hi = float(B(P0, s, r_lo, 0.30))
    w_c, g_c = best(P0, s, r_c); w_o, g_o = best(P0, s, r_lo)
    e5 = ens(s, r_c, 5, 0.95); e10 = ens(s, r_c, 10, 0.95)
    g5 = best(P0, *e5); g10 = best(P0, *e10)
    e10o = ens(s, r_lo, 10, 0.93); g10o = best(P0, *e10o)
    tabE.append(dict(s=s, blend30=(b_c, b_lo, b_hi), wstar=(w_c, g_c), wstar_opt=(w_o, g_o), e5=e5, g5=g5, e10=e10, g10=g10, e10o=e10o, g10o=g10o))
    print(f'  solo {s:.3f}: BLEND W.30 pred {b_c:.4f} [{b_lo:.4f}, {b_hi:.4f}] | W* {w_c:.2f} lift {g_c:+.4f} (at r_lo: W* {w_o:.2f} {g_o:+.4f})'
          f' | k=5 (r_mm .95): solo {e5[0]:.4f} r {e5[1]:.3f} lift {g5[1]:+.4f} @W {g5[0]:.2f}'
          f' | k=10: solo {e10[0]:.4f} r {e10[1]:.3f} lift {g10[1]:+.4f} | k=10 optimistic (r_lo, r_mm .93): solo {e10o[0]:.4f} r {e10o[1]:.3f} lift {g10o[1]:+.4f}')
out['tabE'] = tabE

print('\n  inversion: implied fleet r_LB from the K2 BLEND W.30 read, given the K2 SOLO read (both +-.0005, parent .9425-.9435)')
reads = [0.936, 0.937, 0.938, 0.939, 0.940, 0.941, 0.942, 0.943, 0.944, 0.945]
inv = {}
print('   solo \\ read ' + ' '.join(f'{x:.3f}      ' for x in reads))
for s in SROWS:
    line = f'   {s:.3f}      '; inv[s] = {}
    for x in reads:
        lo, hi, ls = feas(s, [(0.30, x)], PG, atol=0.0005)
        inv[s][x] = (lo, hi)
        line += (f'[{lo:.2f},{hi:.2f}] ' if np.isfinite(lo) else '   --      ')
    print(line)
out['inversion'] = {str(k): {str(x): v for x, v in d.items()} for k, d in inv.items()}

print('\n' + '=' * 100)
print('F. what our ENSEMBLE must reach: min solo for a true lift of +.003/+.005/+.007 over .943 (optimal W)')
SG = np.arange(0.900, 0.9801, 0.0005)
need = {}
print('   r_eff   +.003 (W*)      +.005 (W*)      +.007 (W*)')
for r in (0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98):
    line = f'   {r:.2f}  '; need[r] = {}
    for tgt in (0.003, 0.005, 0.007):
        hit = None
        for s in SG:
            w, g = best(P0, s, r)
            if g >= tgt: hit = (float(s), w); break
        need[r][tgt] = hit
        line += f'  {hit[0]:.4f} ({hit[1]:.2f})  ' if hit else '     n/a        '
    print(line)
out['need'] = {str(k): {str(t): v for t, v in d.items()} for k, d in need.items()}

print('\n  ensemble build-up of the fleet member (effective r_mm .95; .93/.97 in brackets), member r_LB = prior central')
for s in SROWS:
    line = f'   member {s:.3f}:'
    for k in (1, 2, 3, 5, 10, 20):
        sk, rk = ens(s, r_c, k, 0.95); w, g = best(P0, sk, rk)
        sk3, _ = ens(s, r_c, k, 0.93); sk7, _ = ens(s, r_c, k, 0.97)
        line += f' | k{k}: {sk:.4f} [{sk3:.4f},{sk7:.4f}] r {rk:.3f} lift {g:+.4f}'
    print(line)

print('\n' + '=' * 100)
print('G. scenarios for +.007 (true .950): ensemble of k members, member solo s_m, member r_LB r_m, r_mm .95')
SC = [('K2-like members, as is', 0.920, r_c, 10), ('stronger recipe (+.008/member), same labels', 0.928, r_c, 10),
      ('stronger recipe + decorrelated labels (r -.06)', 0.928, r_c - 0.06, 10),
      ('upper tail: .935 members, r .85', 0.935, 0.85, 12), ('upper tail: .935 members, r .80', 0.935, 0.80, 12),
      ('best case: .940 members, r .80', 0.940, 0.80, 15)]
out['scen'] = []
for nm, sm, rm, k in SC:
    sk, rk = ens(sm, rm, k, 0.95); w, g = best(P0, sk, rk)
    out['scen'].append(dict(name=nm, s_m=sm, r_m=rm, k=k, s_k=sk, r_k=rk, W=w, lift=g))
    print(f'  {nm:48s}: member {sm:.3f} r {rm:.3f} x{k} -> ens solo {sk:.4f} r {rk:.3f} -> W* {w:.2f} lift {g:+.4f} (public ~{P0 + g:.4f})')
json.dump(out, open(D + 'proxy_lb_out.json', 'w'), indent=1, default=float)
print('wrote proxy_lb_out.json')
