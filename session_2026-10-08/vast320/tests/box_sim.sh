#!/bin/bash
# box_sim.sh: offline simulation of the instance-side lanes/status/manifest and the box-side verify + placeholders,
# run ON THE BOX in /tmp/vast320_sim with a fake trainer ("fakepy": writes fleet_train.py-shaped outputs in seconds,
# real torch ckpts with the expected args). No GPU, no rental, no change to ~/queue or ~/fleet_runs (asserted).
# Needs ~/vast320/{lane.sh,status.sh,box_ops.sh,argv_template.txt} (dryrun.sh copies them).
set -u
SIM=/tmp/vast320_sim; rm -rf $SIM; mkdir -p $SIM/inst/kit $SIM/boxruns $SIM/queue
PYB=$HOME/miniconda3/envs/rsna/bin/python; pass=0; fail=0
ok(){ echo "PASS $*"; pass=$((pass + 1)); }; no(){ echo "FAIL $*"; fail=$((fail + 1)); }
Q0=$(md5sum < ~/queue/queue.txt); FR0=$(ls ~/fleet_runs | md5sum)
cp ~/vast320/lane.sh ~/vast320/status.sh ~/vast320/argv_template.txt ~/vast320/kit.env $SIM/inst/kit/
cat > $SIM/fakepy <<'FK'
#!/bin/bash
shift; while [ $# -gt 0 ]; do case $1 in --out) OUT=$2;; --fold) F=$2;; esac; shift; done
mkdir -p $OUT/epochs
if [ "$F" = 3 ] && [ ! -e $OUT/../.f3_failed_once ]; then touch $OUT/../.f3_failed_once; echo "simulated early crash"; exit 1; fi
echo "labels 4407 | store has 4407 | held-out (fold<0 or --exclude) 74 | train 3466 | val 867 | all_data False | device cuda"
for e in $(seq 0 20); do
  echo "ep $e it 0/3466 loss 0.61 0.09s/it"; echo "ep $e TRAIN-ONLY 4.9 min (G0 timing)"
  echo "ep $e raw | gold74 0.8800 (12 labels) | gold58 0.9050 (12 labels) | val AUC vs llm_labels_v4_blend.csv 0.88$((10 + e)) (12 labels) | vs train labels 0.9100 | epoch 6.1 min"
  echo "ep $e ema | gold74 0.8700 (12 labels) | gold58 0.9000 (12 labels) | val AUC vs llm_labels_v4_blend.csv 0.87$((10 + e)) (12 labels) | vs train labels 0.9000 | epoch 6.3 min"
  if [ $e -ge 5 ]; then for t in raw ema; do echo "ep$e,$t" > $OUT/epochs/oof_f${F}_s1045_${t}_ep$((e + 1)).csv; echo x > $OUT/epochs/f${F}_s1045_${t}_ttr288-320x24_ep$((e + 1))_best.pt; done; fi
done
CUDA_VISIBLE_DEVICES= PYB_PLACEHOLDER - "$OUT" "$F" <<'PY'
import torch, sys
out, F = sys.argv[1], int(sys.argv[2])
for t in ('raw', 'ema'):
    torch.save({'model': {}, 'args': dict(fold=F, seed=1045, ttr_size=288, model_sha='f141aeed136dfc01',
                train_labels='labels_frac75_logit.csv', grad_ckpt=False), 'cv': 0.8909, 'epoch': 18},
               f'{out}/f{F}_s1045_{t}_ttr288-320x24_cv8909_best.pt')
    open(f'{out}/oof_f{F}_s1045_{t}.csv', 'w').write('StudyInstanceUID,ACL\n1,0.5\n')
PY
sleep 2; echo '{"raw": 0.8909, "ema": 0.8851}'
FK
sed -i "s#PYB_PLACEHOLDER#$PYB#" $SIM/fakepy; chmod +x $SIM/fakepy
NOW=$(date +%s)
printf 'FOLDS="2 3 4"\nDEADLINE_EPOCH=%s\nMODEL_FOLD_S=30\nPYI=%s\nEPOCHPT=0\n' $((NOW + 3600)) $SIM/fakepy > $SIM/inst/kit/plan.env
export RSNA_ROOT=$SIM/inst
mkdir -p $SIM/inst/fast_build/code2
echo "-- lanes: GPU0 'f2 f4', GPU1 'f3' (f3 crashes once early -> retried)"
bash $SIM/inst/kit/lane.sh 0 "2 4" & p0=$!
bash $SIM/inst/kit/lane.sh 1 "3" & p1=$!
wait $p0 $p1
ST=$(bash $SIM/inst/kit/status.sh); echo "$ST" | grep -E '^(FOLD|LANES|ALL)'
echo "$ST" | grep -q 'ALL_DONE=1' && ok "all 3 folds done, ALL_DONE=1" || no "ALL_DONE"
for F in 2 3 4; do echo "$ST" | grep -q "FOLD f$F state=done epochs=21/21" && ok "f$F done 21/21" || no "f$F state"; done
if grep -q 'f3: early failure, one retry' $SIM/inst/logs/lane_1.log && ls -d $SIM/inst/fleet_runs/f3_s1045_ttr288frac75_partial_* > /dev/null 2>&1
then ok "f3 early crash retried once, the partial run moved aside"; else no "f3 retry"; fi
if grep -q "LAUNCH f4 on GPU 0: .*fakepy fleet_train.py --model ttr --ttr_size 288 .* --fold 4 --store $SIM/inst/fleet_store .* --out $SIM/inst/fleet_runs/f4_s1045_ttr288frac75 --grad_ckpt 0" $SIM/inst/logs/lane_0.log
then ok "argv rendered from the template (fold 4, root substituted)"; else no "argv render"; fi
echo "-- idempotent re-run of lane 0"
bash $SIM/inst/kit/lane.sh 0 "2 4"
[ "$(grep -c 'already complete' $SIM/inst/logs/lane_0.log)" = 2 ] && ok "re-run skips completed folds" || no "re-run"
echo "-- deadline admission: fold 1, deadline in 10 s, estimate 600 s"
printf 'FOLDS="1"\nDEADLINE_EPOCH=%s\nMODEL_FOLD_S=600\nPYI=%s\nEPOCHPT=0\n' $(( $(date +%s) + 10 )) $SIM/fakepy > $SIM/inst/kit/plan.env
bash $SIM/inst/kit/lane.sh 2 "1"
[ -e $SIM/inst/status/f1.skipped ] && [ ! -e $SIM/inst/fleet_runs/f1_s1045_ttr288frac75.log ] && ok "a fold that cannot finish in time is skipped, not started" || no "deadline skip"
printf 'FOLDS="2 3 4"\nDEADLINE_EPOCH=%s\nMODEL_FOLD_S=30\nPYI=%s\nEPOCHPT=0\n' $((NOW + 3600)) $SIM/fakepy > $SIM/inst/kit/plan.env
echo "-- manifest (no epochs/*.pt) + box verify (emulated pull = tar copy of the same file set)"
MAN=$(bash $SIM/inst/kit/status.sh manifest 2)
echo "$MAN" | grep -q 'epochs/.*_best.pt' && no "manifest includes epoch ckpts" || ok "manifest excludes epochs/*.pt"
[ "$(echo "$MAN" | grep -c .)" = 37 ] && ok "manifest = 2 best ckpts + 2 oof + 32 epoch oof + log (37)" || no "manifest count $(echo "$MAN" | grep -c .)"
N=f2_s1045_ttr288frac75; mkdir -p $SIM/boxruns/$N
(cd $SIM/inst/fleet_runs && echo "$MAN" | awk '{print $2}' | tar -cf - -T -) | tar -C $SIM/boxruns -xf -
export VAST320_FR=$SIM/boxruns
V=$(echo "$MAN" | bash ~/vast320/box_ops.sh verify $N $N canonical 0000 SIM_GPU 0); echo "$V"
echo "$V" | grep -q "VERIFY_OK $N" && ok "verify OK on an intact copy" || no "verify intact"
[ -s $SIM/boxruns/vast320_${N}_receipt.json ] && "$PYB" -c "import json; json.load(open('$SIM/boxruns/vast320_${N}_receipt.json'))" && ok "receipt json written and valid" || no "receipt"
echo tamper >> $SIM/boxruns/$N/oof_f2_s1045_raw.csv
echo "$MAN" | bash ~/vast320/box_ops.sh verify $N $N canonical 0000 SIM_GPU 0 | grep -q VERIFY_FAIL && ok "verify FAILS on a tampered file" || no "tamper not caught"
echo "-- placeholders + release on a COPY of the queue (QR_* overrides; the real queue is never written)"
cp ~/queue/queue.txt $SIM/queue/queue.txt
export VAST320_Q=$SIM/queue/queue.txt QR_DIR=$SIM/queue QR_QUEUE=$SIM/queue/queue.txt QR_LOCK=$SIM/queue/queue.lock QR_BAK=$SIM/queue/bak QR_RUNNER=$HOME/queue/queue_runner.sh
rm -rf $SIM/boxruns/*
bash ~/vast320/box_ops.sh ph-create 2 3 4
c=0; for F in 2 3 4; do [ -d $SIM/boxruns/f${F}_s1045_ttr288frac75 ] && c=$((c + 1)); done
[ $c = 3 ] && ok "ph-create made 3 empty placeholder dirs" || no "ph-create ($c)"
bash ~/vast320/box_ops.sh ph-create 2 | grep -q 'already present' && ok "ph-create is idempotent" || no "ph-create twice"
sed -i -E 's/^(Q4 TTR288 s1045 f[234] frac75 )/#REFUSED 10-09 07:10 (out_dir or its .log exists) \1/' $SIM/queue/queue.txt
echo "partial" > $SIM/boxruns/f4_s1045_ttr288frac75/partial.txt
bash ~/vast320/box_ops.sh ph-release 3 4
[ ! -e $SIM/boxruns/f3_s1045_ttr288frac75 ] && ok "empty f3 placeholder removed" || no "f3 release"
ls -d $SIM/boxruns/f4_s1045_ttr288frac75_vastpartial_* > /dev/null 2>&1 && ok "f4 partial content moved aside, not deleted" || no "f4 aside"
first=$(grep -v '^#' $SIM/queue/queue.txt | grep -v '^[[:space:]]*$' | head -n 2 | sed -E 's/^(Q4 TTR288 s1045 f[0-9]).*/\1/' | tr '\n' ',')
[ "$first" = "Q4 TTR288 s1045 f3,Q4 TTR288 s1045 f4," ] && ok "released folds re-queued at the top, f3 then f4" || no "queue order: $first"
QR_OUT_ROOT=$HOME/fleet_runs bash ~/queue/queue_runner.sh --check $SIM/queue/queue.txt | tail -n 1 | grep -q ', 0 BAD' && ok "queue copy passes queue_runner --check" || no "queue check"
[ "$(md5sum < ~/queue/queue.txt)" = "$Q0" ] && ok "real ~/queue/queue.txt untouched" || no "REAL QUEUE CHANGED"
[ "$(ls ~/fleet_runs | md5sum)" = "$FR0" ] && ok "real ~/fleet_runs listing untouched" || no "REAL FLEET_RUNS CHANGED"
echo "== box_sim: $pass PASS, $fail FAIL"; rm -rf $SIM; [ $fail = 0 ]
