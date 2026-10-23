#!/bin/sh
# 10-19: cv_shrink with ours = R6 serve (sequential, laptop CPU, light numpy)
D=C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58
R=C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/research
O=$D/R6_gold58_serve.csv
cd $R
python cv_shrink_r6.py $D $O 2000 $R/cv_shrink_R6_half.json 0.5 40 20261018 c4 > cv_shrink_R6_half.log 2>&1
python cv_shrink_r6.py $D $O 1500 $R/cv_shrink_R6_2to1.json 0.6667 40 20261018 c4 > cv_shrink_R6_2to1.log 2>&1
python cv_shrink_r6.py $D $O 2000 $R/cv_shrink_R6_hwbase.json 0.5 40 20261018 hw > cv_shrink_R6_hwbase.log 2>&1
python cv_shrink_r6.py $D $O 1000 $R/cv_shrink_R6_half_seed2.json 0.5 40 7 c4 > cv_shrink_R6_half_seed2.log 2>&1
python cv_shrink_r6.py $D $O 2000 $R/cv_shrink_R6_half_bag.json 0.5 40 20261018 c4 30 > cv_shrink_R6_half_bag.log 2>&1
echo CHAIN_DONE > cv_chain.done
