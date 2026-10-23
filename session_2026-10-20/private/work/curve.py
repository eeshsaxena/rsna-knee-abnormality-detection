# curve.py: P(private rank <= cut | our private margin m fixed), base field model. MODEL.
import json, numpy as np, shake_mc as s
res = {}
for m in (-2, 2, 5, 8, 11, 13, 16, 20, 25, 30):
    o = s.run("base", dict(muA=float(m), sdA=0.01, muD=-50.0, sdD=0.01), R=6, NS=1500)
    a = o["A"]; res[m] = a
    print(f"m {m:+3d}e-4 | P(silver r<=316) {a['P(r<=316)']:.3f} P(r<=150) {a['P(r<=150)']:.3f} P(r<=100) {a['P(r<=100)']:.3f} "
          f"P(bronze) {a['P(r<=633)']:.3f} | median r{a['quantiles p5/10/25/50/75/90/95'][3]} p10-p90 r{a['quantiles p5/10/25/50/75/90/95'][1]}-{a['quantiles p5/10/25/50/75/90/95'][5]}")
json.dump(res, open("curve_base.json", "w"), indent=1)
