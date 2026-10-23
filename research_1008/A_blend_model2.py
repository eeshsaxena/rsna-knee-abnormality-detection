"""A_blend_model2.py (2026-10-08 ~22:20 IST, research_1008 dimension A). MODEL only, laptop-light.

Same macro single-d binormal model as A_blend_model.py.  New MEASURED constraint added:
  N .70 + G .30 (flat, all labels) = .948  (prvsiyan the-bee-s-knees, bench row 56955785 and public row 56957785)
Question: what does one extra public leg X add on top of K9N-a (P .10 / F .30 / N .60)?
X candidates: G (.929), N2 = nartaa r224 (.945), H = hengck23 MaViT-288 (.935 single), M = renta0426 DINOv2-B (GUESS .925-.935).
"""
import numpy as np
from scipy.stats import norm
d = lambda a: np.sqrt(2) * norm.ppf(a)
def blend(s, R, w):
    w = np.asarray(w, float); dv = np.array([d(x) for x in s]); return norm.cdf((w @ dv) / np.sqrt(2 * (w @ R @ w)))
def b2(a, b, r, wb): return blend((a, b), np.array([[1, r], [r, 1]]), (1 - wb, wb))
inside = lambda x, t: (x >= t - .0005) & (x < t + .0005)
out = []
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

# 1. r_NG with the new flat .30 = .948 read
ok = []
for r in np.round(np.arange(.70, .999, .001), 3):
    hit = False
    for A in np.linspace(.9485, .94949, 5):
        for B in np.linspace(.9285, .92949, 5):
            if inside(b2(A, B, r, .22), .950) and inside(b2(A, B, r, .30), .948) and inside(b2(A, B, r, .145), .950):
                hit = True
    if hit: ok.append(r)
pr('r_NG consistent with {.145:.950, .22:.950, .30:.948}:', (ok[0], ok[-1]) if ok else 'NONE (model too smooth: .22->.950 and .30->.948 conflict)')
for r in (.80, .84, .87, .90, .93):
    pr(f'  r_NG {r}: N+G w .145 {b2(.949,.929,r,.145):.4f}  .22 {b2(.949,.929,r,.22):.4f}  .30 {b2(.949,.929,r,.30):.4f}')

# 2. Extra leg on top of K9N-a. Base R for P, F, N at calibrated points (B_blend_lb_model.md sec 3)
S = dict(P=.943, F=.944, N=.949)
rPF, rPN = .945, .945
pr('\nExtra leg X on top of K9N-a (P.10 F.30 N.60). Gain = best over w_X in [0,.30] of blend(K9N-a scaled by 1-w_X, X w_X) minus K9N-a.')
pr('Scan: AUC_X, r_FN, r_XN, r_XF (r_XP set = r_XF). MODEL, optimistic ~+.0005.')
pr('AUC_X  r_FN  r_XN  r_XF |  K9N-a  best_wX  best   gain')
rows = []
for ax in (.929, .935, .945):
    for rfn in (.88, .92):
        for rxn in (.80, .85, .90, .95):
            for rxf in (.80, .85, .90):
                R = np.array([[1, rPF, rPN, rxf], [rPF, 1, rfn, rxf], [rPN, rfn, 1, rxn], [rxf, rxf, rxn, 1]])
                if np.linalg.eigvalsh(R).min() < -1e-9: continue
                s = (S['P'], S['F'], S['N'], ax)
                base = blend(s, R, (.1, .3, .6, 0))
                best = max(((wx, blend(s, R, (.1*(1-wx), .3*(1-wx), .6*(1-wx), wx))) for wx in np.arange(0, .31, .02)), key=lambda t: t[1])
                rows.append((ax, rfn, rxn, rxf, base, best[0], best[1], best[1]-base))
for r in rows:
    if r[3] in (.85,) or r[2] in (.80, .95):
        pr(f'{r[0]:.3f}  {r[1]:.2f}  {r[2]:.2f}  {r[3]:.2f} | {r[4]:.4f}  {r[5]:.2f}  {r[6]:.4f}  {r[7]:+.4f}')
pr('\nSummary of gain by AUC_X (min / median / max over PSD scan):')
for ax in (.929, .935, .945):
    g = np.array([r[7] for r in rows if r[0] == ax]); pr(f'  AUC_X {ax}: {g.min():+.4f} / {np.median(g):+.4f} / {g.max():+.4f}  (n={len(g)})')
pr('Gain at fixed w_X .10 (no tuning), AUC_X .935, r_XN .85, r_XF .85, r_FN .90:')
R = np.array([[1, rPF, rPN, .85], [rPF, 1, .90, .85], [rPN, .90, 1, .85], [.85, .85, .85, 1]])
s = (S['P'], S['F'], S['N'], .935)
pr(f'  {blend(s, R, (.09, .27, .54, .10)) - blend(s, R, (.1, .3, .6, 0)):+.4f}')
open(__file__.replace('.py', '.out'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
