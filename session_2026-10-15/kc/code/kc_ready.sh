#!/bin/bash
# ~/kc/kc_ready.sh -- KC_READY verdict writer (BOX_PLAN_1015 sec 3.3, trimmed per the adopted critique). See kc_ready.py.
#   bash ~/kc/kc_ready.sh --status                    what is still missing (writes nothing)
#   bash ~/kc/kc_ready.sh --a6_sha <sha256>           writes 'PASS kc_ready ...' only if every condition holds
#   bash ~/kc/kc_ready.sh --fail "<reason>"           main's explicit FAIL (the f0 line then #SKIPs; arm kc_gate_watch.sh
#                                                     or it writes the f0 FAIL itself so the AD lines #SKIP too)
exec env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 nice -n 19 $HOME/miniconda3/envs/rsna/bin/python $HOME/kc/code/kc_ready.py "$@"
