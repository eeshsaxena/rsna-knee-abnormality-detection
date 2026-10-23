# wall_model.py OUT [OUT_REF_FLEET]: hidden-wall estimate for V224-F from a commit's fleet receipt (MODEL; inputs MEASURED).
# fleet hidden = fixed (commit stage minutes minus its 3-study loop) + studies_per_T4 x sum(ms/member/study over all served
# passes, the slower GPU) ; central uses the commit microbench, slow uses x1.15 (FN11 slow/central TTR288 532/466) and a
# x1.3 slow-session factor on everything. V224-exact part: 3.0 h central (2.4-3.6), from the 10-14 runtime review.
import sys, json
import numpy as np
out = sys.argv[1]
fr = json.load(open(out + '/fleet_receipt.json'))
bench = [b for b in fr.get('bench') or [] if b]
per_gpu = [sum(b.values()) / 1000.0 for b in bench]                  # s per study per GPU (all members, both passes)
loop_min = max(fr.get('worker_loop_min') or [0])
fixed_min = fr['stage_min'] - loop_min                                # model load + parity + setup, MEASURED on the commit
print('MEASURED commit: stage %.2f min, workers %.1f s, worker loop %s min, fixed ~%.2f min, microbench s/study/GPU %s, decode %.2f s/study/thread'
      % (fr['stage_min'], fr['workers_s'], fr.get('worker_loop_min'), fixed_min, [round(x, 2) for x in per_gpu],
         fr.get('decode_s_per_study_per_thread') or float('nan')))
if len(sys.argv) > 2:
    rf = json.load(open(sys.argv[2] + '/fleet_receipt.json'))
    rb = [sum(b.values()) / 1000.0 for b in rf.get('bench') or [] if b]
    print('REF (HFR7-RR commit, fleet before the R7 wheels): stage %.2f min, microbench s/study/GPU %s' % (rf['stage_min'], [round(x, 2) for x in rb]))
rows = []
for n_hidden in (1200, 1300):
    spg = int(np.ceil(n_hidden / 2))
    g = max(per_gpu)
    cen = (fixed_min * 60 + spg * g) / 3600
    slow = (fixed_min * 60 + spg * g * 1.15) / 3600
    rows.append((n_hidden, cen, slow))
    print('fleet hidden %d studies: central %.2f h, slow %.2f h, slow session x1.3 %.2f h' % (n_hidden, cen, slow, slow * 1.3))
f_c = max(r[1] for r in rows); f_s = max(r[2] for r in rows)
print('V224-F hidden wall: central %.1f h (V224 3.0 + fleet %.2f); range %.1f-%.1f h (V224 2.4-3.6 + fleet %.2f-%.2f); '
      'slow session x1.3 on the slow end %.1f h; fleet start budget 5.5 h, wall 8.3 h'
      % (3.0 + f_c, f_c, 2.4 + min(r[1] for r in rows), 3.6 + f_s, min(r[1] for r in rows), f_s, (3.6 + f_s) * 1.3))
