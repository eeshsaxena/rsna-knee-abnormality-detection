# skep_apick_gt.py (skeptic of apick; MODEL re-derivation): BMA-weighted pairwise P(swide private > alt), A alone, per weighting.
import json, numpy as np
exec(open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/lastday/bma.py', encoding='utf-8').read().split("out = open(")[0])
for wn in ('A22 priors', 'optimist', 'flat', 'levers', 'skeptic', 'EB-heavy'):
    wl, wm = LW[wn]
    post = {k: wl[k[0]] * wm[k[1]] * EV['%s|%s' % k]['ev'] for k in R}; s1 = sum(post.values()); post = {k: v / s1 for k, v in post.items()}
    gt = {a: sum(post[k] * R[k]['alone']['swide']['gt'][a] for k in R) for a in ('pw25', 'pw20', 'pw30', 'eoa20')}
    dE = {a: sum(post[k] * (R[k]['alone']['swide']['E'] - R[k]['alone'][a]['E']) for k in R) for a in ('pw25', 'pw20')}
    pneg = sum(post[k] for k in R if R[k]['decomp']['priv'][5] < 0)
    print('%-10s P(swide>pw25) %.2f  P(swide>pw20) %.2f  P(swide>pw30) %.2f  P(swide>eoa20) %.2f | alone dE vs pw25 %+.2f vs pw20 %+.2f | post weight on stack<0 worlds %.2f' % (
        wn, gt['pw25'], gt['pw20'], gt['pw30'], gt['eoa20'], dE['pw25'], dE['pw20'], pneg))
