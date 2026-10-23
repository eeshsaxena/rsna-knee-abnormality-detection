# ours_small_sh.py: split-half sign stability (300 splits) for the small-weight (.05) member additions on c4, gold58.
# Reuses the definitions of ours_more_g58.py (executes its source up to the results section; same argv).
import sys
src = open(sys.argv[4], encoding='utf-8').read().split('res = {"members"')[0]
sys.argv = sys.argv[:4]
exec(src)
for k, kw in [("R7oof end .05", dict(end=M["R7oof"], wend=0.05)), ("KC1 end .05", dict(end=M["KC1"], wend=0.05)),
              ("KC1+DT+Tb end .05", dict(end=M["KC1+DT+Tb"], wend=0.05)), ("K14DT end .05", dict(end=M["K14DT"], wend=0.05)),
              ("KC1 end .10", dict(end=M["KC1"], wend=0.10))]:
    a, h = split_half(kw, {})
    print("split-half %-20s sign agreement %.3f | held-out gain when half A says + %+.5f" % (k, a, h))
