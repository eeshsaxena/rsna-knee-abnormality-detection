import numpy as np
from scipy.stats import norm
def d(a): return np.sqrt(2) * norm.ppf(a)
def A(x): return norm.cdf(x / np.sqrt(2))
def blend(P, s, r, W):
    v = (1 - W) ** 2 + W ** 2 + 2 * W * (1 - W) * r
    return norm.cdf(((1 - W) * d(P) + W * d(s)) / np.sqrt(2 * v))
print('K2 W .30 read given solo .932, by r:', ' '.join(f'r{r:.2f}:{blend(.943,.932,r,.3):.4f}' for r in (.80,.85,.88,.90,.92,.94,.96)))
print('K2 W .30 read given solo .922 (pre-K2 prior centre), by r:', ' '.join(f'r{r:.2f}:{blend(.943,.922,r,.3):.4f}' for r in (.85,.90,.95)))
# r_mm insensitivity of the W .20 pick
for rmm in (.94, .955, .965, .975):
    for k in (5, 10):
        v = (1 + (k - 1) * rmm) / k; s = A(d(.9325) / np.sqrt(v)); r = .922 / np.sqrt(v)
        print(f'rmm {rmm} k{k}: fleet solo {s:.4f} r {r:.3f} | W.2 {blend(.943,s,r,.2):.5f} W.3 {blend(.943,s,r,.3):.5f} W.5 {blend(.943,s,r,.5):.5f}')
# member r if TTR mix50 member r is .925 instead of .922
v = (1 + 4 * .955) / 5; s = A(d(.9325) / np.sqrt(v)); r = .925 / np.sqrt(v); print('k5 rmm .955 member r .925:', round(s,4), round(r,3), round(blend(.943,s,r,.2),5))
# weaker TTR mix50 member (.929)
for ms in (.929, .9325, .936):
    v = (1 + 4 * .955) / 5; s = A(d(ms) / np.sqrt(v)); r = .925 / np.sqrt(v)
    print(f'member solo {ms}: fleet {s:.4f} r {r:.3f} W.2 {blend(.943,s,r,.2):.5f} W.1 {blend(.943,s,r,.1):.5f}')
