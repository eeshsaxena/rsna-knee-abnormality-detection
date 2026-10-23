# adversarial variant of sim_ta.py: per-study fixed cost share alpha (time = c*(alpha*mean_files + (1-alpha)*files)), report p95/max
import numpy as np
def worker(rng, n, plain_h, flip_h, pre_h, budget_h, aware, alpha, restart=250):
    files = np.maximum(20, rng.lognormal(np.log(220), 0.45, n)).astype(int)
    order = np.argsort(-files, kind='mergesort'); order = np.concatenate([order[::2], order[1::2]])
    f = files[order]; noise = rng.normal(1.0, 0.15, n).clip(0.6, 1.6)
    cost = alpha * f.mean() + (1 - alpha) * f
    rf = (flip_h * 3600 - restart) / cost.sum(); rp = (plain_h * 3600 - restart) / cost.sum()
    t = pre_h * 3600 + restart; tr = []
    for k in range(n):
        t += rf * cost[k] * noise[k]; tr.append(t)
        if not aware: continue
        kk = k + 1
        if kk >= n: break
        rem = f[kk:].sum()
        if kk >= 10:
            rate = (tr[-1] - tr[0]) / f[1:kk].sum(); work = 1.05 * rate * rem
            trig = t + work > budget_h * 3600 and work > 600
        else:
            trig = t > budget_h * 3600
        if trig:
            t += restart + (rp * cost[kk:] * noise[kk:]).sum(); return t / 3600, kk / n
    return t / 3600, 1.0
rng = np.random.default_rng(1)
for alpha in (0.0, 0.3, 0.5):
  for speed in (1.0, 1.15, 1.3, 1.45):
    for extra in (1.0, 1.4, 2.0):
        ends=[]; sh=[]
        for rep in range(200):
            w=[worker(rng,640,2.25*speed,(2.25+extra)*speed,0.45*speed,4.9,True,alpha) for _ in range(2)]
            ends.append(max(e for e,_ in w)); sh.append(np.mean([s for _,s in w]))
        ends=np.array(ends); fs=ends+0.30*speed+0.05
        print(f"alpha {alpha} x{speed} +{extra}: R7 end mean {ends.mean():.2f} p95 {np.percentile(ends,95):.2f} max {ends.max():.2f} | flip {100*np.mean(sh):.0f}% | P(fleet start>5.5) {np.mean(fs>5.5):.3f}")
