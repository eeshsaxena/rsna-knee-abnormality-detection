#!/bin/sh
D=C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58
O=$D/R6_gold58_serve.csv
cd C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/research
python nested_r6.py $D $O 3000 0.5 0 nested_r6_half.json > nested_r6_half.log 2>&1
python nested_r6.py $D $O 2000 0.6667 1 nested_r6_2to1.json > nested_r6_2to1.log 2>&1
python nested_r6.py $D C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58/R7OOF.csv 2000 0.5 0 nested_r7_half.json > nested_r7_half.log 2>&1
echo DONE > nested_chain2.done
