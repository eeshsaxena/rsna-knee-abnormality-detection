#!/bin/bash
# smoke_run.sh (box GPU, inside the pause window; the caller wraps it in `timeout 1800`): A = 800 study-steps channels_last,
# B = 300 study-steps without, then the serve-path check of A's SWA and N, then smoke_verdict.py.
set -u
PY=$HOME/miniconda3/envs/rsna/bin/python; D=$HOME/raptor_store/smoke; cd $HOME/raptor2
for x in A_cl1 B_cl0; do [ -e $D/$x ] && mv $D/$x $D/${x}_old_$(date +%m%d_%H%M%S); done
IDX="$HOME/raptor_store/mirror_index.csv $HOME/raptor_store/oai_index.csv"
[ -f $HOME/raptor_store/rsna_index.csv ] && IDX="$HOME/raptor_store/rsna_index.csv $HOME/raptor_store/oai_index.csv"
SIDX=$HOME/raptor_store/mirror_index.csv; [ -f $HOME/raptor_store/rsna_index.csv ] && SIDX=$HOME/raptor_store/rsna_index.csv
COMMON="--index $IDX --labels_rsna $HOME/raptor2/labels_raptor_rsna_blendjev_sol.csv --labels_oai $HOME/oai_build/oai_labels_raptorT1.csv --workers 6"
export OMP_NUM_THREADS=2
nice -n 19 taskset -c 8-15 $PY train_raptor.py $COMMON --out $D/A_cl1 --smoke_steps 800 --channels_last 1 > $D/A_cl1.out 2>&1
echo "A rc $?"
nice -n 19 taskset -c 8-15 $PY train_raptor.py $COMMON --out $D/B_cl0 --smoke_steps 300 --channels_last 0 > $D/B_cl0.out 2>&1
echo "B rc $?"
nice -n 19 taskset -c 8-15 $PY serve_check.py --index $SIDX --ck $D/A_cl1/swa.pt $HOME/kfn70_test/wts/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt --out $D/serve > $D/serve.log 2>&1
echo "serve rc $?"
$PY smoke_verdict.py
