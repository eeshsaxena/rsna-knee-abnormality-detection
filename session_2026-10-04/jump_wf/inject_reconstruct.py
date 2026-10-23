"""Step 2: rebuild every parent stage on gold58 from the per-arm inputs and diff against the stored stage files."""
import numpy as np
from inject_common import *

D = load_all()
ids = D['ids']
rows = []


def cmp(name, mine, ref):
    d = float(np.abs(mine - ref).max())
    rk = float(np.abs(rank_pct(mine) - rank_pct(ref)).max())
    rows.append((name, d, rk))
    print(f'{name:58s} max|diff| {d:.3e}   max|rank diff| {rk:.3e}', flush=True)


s1 = stage1_dino(D['dino'])
cmp('stage1 DINO stack (c16 _combine, 20 equal-weight members)', s1, D['st1'])
s2 = stage2_a5(D['st1'], D['a5_rank_mean'], D['a5_ok'])
cmp('stage2 +A5 (c23/c24, A5_W .52) from stored st1', s2, D['st2'])
s2b = stage2_a5(s1, D['a5_rank_mean'], D['a5_ok'])
cmp('stage2 +A5 chained from rebuilt st1', s2b, D['st2'])
# a5_rank_mean itself: mean over 5 folds of rank01 on eligible studies
a5r = np.mean([rank01(f) for f in D['a5_raw']], axis=0)
cmp('a5_rank_mean (c23 _a5_rank_available) from speed_a5_raw', a5r, D['a5_rank_mean'])
proto = protocol_features(D['tser'], ids)
s3 = stage3_rad(D['st2'], D['rad_e10_raw'], D['rad_e13_raw'], D['rad_legacy_second_raw'], D['cal'], proto)
cmp('stage3 +Rad +calibrator (c26 _rad_main) from stored st2', s3, D['st3'])
s3b = stage3_rad(s2b, D['rad_e10_raw'], D['rad_e13_raw'], D['rad_legacy_second_raw'], D['cal'], proto)
cmp('stage3 chained from rebuilt st1->st2', s3b, D['st3'])
pub = raptor_public(D['raptor_raw'], RAPTOR_W)
cmp('raptor_public (c28 prob blend .6/.1/.1/.2 -> rank01, f32)', pub, D['pub_csv'])
fam = coat_family([D['coat_prob'][m] for m in COAT_MEMBERS])
cmp('coat_family (c28 rank of 4-member prob mean)', fam, D['fam_csv'])
side = hybrid(D['pub_csv'], D['fam_csv'])
cmp('raptor_coat_side (c28 .6 rank(pub) + .4 fam) from stored', side, D['side_csv'])
side_b = hybrid(pub, fam)
cmp('raptor_coat_side chained from rebuilt pub/fam', side_b, D['side_csv'])
fin = final_blend(D['st3'], D['side_csv'])
cmp('stage4 final (c28 per-label outer blend) from stored st3/side', fin, D['st4'])
fin_b = final_blend(s3b, side_b)
cmp('stage4 final FULLY chained from raw inputs', fin_b, D['st4'])
cmp('stage4 vs localLB_parent_preds.csv (= submission.csv)', D['st4'], D['parent_sub'])
print('\ngold58 macro AUC (MEASURED): ' + ', '.join(f'{k} {mauc(D["Y"], v):.4f}' for k, v in
      [('st1', D['st1']), ('st2', D['st2']), ('st3', D['st3']), ('pub', D['pub_csv']), ('fam', D['fam_csv']),
       ('side', D['side_csv']), ('final', D['st4']), ('fleet', D['fleet'])]))
