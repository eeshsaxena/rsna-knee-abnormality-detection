#!/bin/bash
# test_launch.sh (frac75 verify pass, 2026-10-07): CPU-only sandbox tests of ~/novel/frac75/launch_frac75.sh.
# Never touches the GPU, ~/fleet_runs or any running process. Everything lives in ~/novel/frac75_verify/sb:
#   fake HOME (argv_frac75.sh copy, a dummy label file, a dummy code2/fleet_train.py, fleet_runs/), a stub python that
#   records its argv and runs `sleep` under the argv0 "python fleet_trainSBX.py" (so the exact trainer pgrep can see it),
#   and a stub nvidia-smi driven by a state file. The launcher under test is a sed copy whose ONLY changes are the three
#   process patterns (auto_after_ttr288 -> auto_after_ttrSBX, queue_runner -> queue_runnerSBX, fleet_train[.]py in
#   TRAIN_PROC -> fleet_trainSBX[.]py) and, for one test, the freeze date: the real chain / trainer on this box would
#   otherwise make every check BUSY.
set -u
SRC=${1:-$HOME/novel/frac75/launch_frac75.sh}
V=$HOME/novel/frac75_verify; SB=$V/sb; rm -rf $SB; mkdir -p $SB/bin $SB/home/novel/frac75/out $SB/home/fast_build/code2 \
  $SB/home/fleet_runs $SB/home/miniconda3/envs/rsna/bin $SB/proc
cp $HOME/novel/frac75/argv_frac75.sh $SB/home/novel/frac75/
echo "id,Fracture" > $SB/home/novel/frac75/labels_frac75_logit.csv; echo "# dummy" > $SB/home/fast_build/code2/fleet_train.py
cat > $SB/home/miniconda3/envs/rsna/bin/python <<'EOF'
#!/bin/bash
echo "$*" >> "$SBX/calls.txt"
rc=$(cat "$SBX/rc" 2>/dev/null || echo 0); s=$(cat "$SBX/sleep" 2>/dev/null || echo 0)
if [ -f "$SBX/rc_gc0" ] && [[ "$*" == *"--grad_ckpt 0" ]]; then rc=$(cat "$SBX/rc_gc0"); fi
( exec -a "python fleet_trainSBX.py" sleep "$s" )
echo '{"raw": 0.5, "ema": 0.5}'
exit "$rc"
EOF
cat > $SB/bin/nvidia-smi <<'EOF'
#!/bin/bash
st=$(cat "$SBX/gpu" 2>/dev/null || echo free)
case $st in fail) exit 9;; python) echo "/x/miniconda3/envs/rsna/bin/python"; echo "/usr/libexec/gnome-remote-desktop-daemon";; *) echo "/usr/libexec/gnome-remote-desktop-daemon";; esac
EOF
chmod +x $SB/home/miniconda3/envs/rsna/bin/python $SB/bin/nvidia-smi
sed -e 's/auto_after_ttr288/auto_after_ttrSBX/g' -e 's/queue_runner/queue_runnerSBX/g' \
    -e "s/^TRAIN_PROC='\(.*\)fleet_train\[\.\]py\(.*\)$/TRAIN_PROC='\1fleet_trainSBX[.]py\2/" $SRC > $SB/launch.sh
diff <(sed -e 's/auto_after_ttr288/X/g' -e 's/queue_runner/Y/g' -e 's/fleet_train\[\.\]py( \.\*)?/Z/' $SRC) \
     <(sed -e 's/auto_after_ttrSBX/X/g' -e 's/queue_runnerSBX/Y/g' -e 's/fleet_trainSBX\[\.\]py( \.\*)?/Z/' $SB/launch.sh) > /dev/null \
  && echo "test copy == launcher except the 3 process patterns" || { echo "TEST COPY DIFFERS"; exit 1; }
grep -q "^TRAIN_PROC=" $SRC && { grep -q "fleet_trainSBX\[.\]py" $SB/launch.sh || { echo "TRAIN_PROC not renamed"; exit 1; }; }
export SBX=$SB
NP=0; NF=0
ok(){ if eval "$2"; then NP=$((NP+1)); echo "  PASS $1"; else NF=$((NF+1)); echo "  FAIL $1"; fi; }
L(){ env -u CUDA_VISIBLE_DEVICES HOME=$SB/home PATH=$SB/bin:$PATH "$@" bash $SB/launch.sh "${ARGS[@]}" > $SB/last.out 2>&1; echo $?; }
reset(){ rm -rf $SB/home/fleet_runs/* $SB/calls.txt $SB/rc $SB/rc_gc0 $SB/home/fleet_runs/PAUSE_CHAIN $SB/home/novel/frac75/out/*; echo 0 > $SB/sleep; echo free > $SB/gpu; }
R=$SB/home/fleet_runs

echo "T1 print mode (no LAUNCH)"; reset; ARGS=(probe); rc=$(L)
( HOME=$SB/home; source $SB/home/novel/frac75/argv_frac75.sh; frac75_argv 0; echo "cd ~/fast_build/code2 && $SB/home/miniconda3/envs/rsna/bin/python fleet_train.py ${FRAC75_ARGV[*]} > $FRAC75_OUT.log 2>&1" ) > $SB/want0.txt
ok "probe prints exactly the frac75_argv 0 command, rc 0, nothing run" '[ $rc = 0 ] && diff -q $SB/last.out $SB/want0.txt > /dev/null && [ ! -e $SB/calls.txt ]'
ARGS=(folds); rc=$(L); ok "folds prints 4 commands without needing a gate" '[ $rc = 0 ] && [ $(grep -c "fleet_train.py" $SB/last.out) = 4 ] && [ ! -e $SB/calls.txt ]'

echo "T2 refusals (LAUNCH=1 probe): each must exit 4 and never call the trainer"
ARGS=(probe)
reset; echo python > $SB/gpu; rc=$(L LAUNCH=1); ok "GPU python -> refused" '[ $rc = 4 ] && [ ! -e $SB/calls.txt ] && grep -q "GPU busy" $SB/last.out'
reset; echo fail > $SB/gpu; rc=$(L LAUNCH=1); ok "nvidia-smi error -> refused" '[ $rc = 4 ] && [ ! -e $SB/calls.txt ] && grep -q "99" $SB/last.out'
reset; touch $R/PAUSE_CHAIN; rc=$(L LAUNCH=1); ok "PAUSE_CHAIN -> refused" '[ $rc = 4 ] && [ ! -e $SB/calls.txt ]'
reset; printf '#!/bin/bash\nsleep 30\n' > $SB/proc/auto_after_ttrSBX.sh; setsid bash $SB/proc/auto_after_ttrSBX.sh < /dev/null > /dev/null 2>&1 & CP=$!
sleep 1; rc=$(L LAUNCH=1); kill $CP 2>/dev/null; pkill -f "bash $SB/proc/auto_after_ttrSBX.sh" 2>/dev/null
ok "TTR288 chain bash alive -> refused" '[ $rc = 4 ] && [ ! -e $SB/calls.txt ] && grep -q "chain" $SB/last.out'
reset; ( exec -a "python fleet_trainSBX.py --fold 9" sleep 30 ) & TP=$!; sleep 1; rc=$(L LAUNCH=1); kill $TP 2>/dev/null
ok "a fleet_train trainer without CUDA yet -> refused (NEW guard)" '[ $rc = 4 ] && [ ! -e $SB/calls.txt ] && grep -q "fleet_train.py process" $SB/last.out'
reset; printf '#!/bin/bash\nsleep 30\n' > $SB/proc/queue_runnerSBX.sh; setsid bash $SB/proc/queue_runnerSBX.sh < /dev/null > /dev/null 2>&1 &
sleep 1; rc=$(L LAUNCH=1); pkill -f "bash $SB/proc/queue_runnerSBX.sh" 2>/dev/null
ok "queue runner alive -> refused (NEW guard)" '[ $rc = 4 ] && [ ! -e $SB/calls.txt ] && grep -q "queue_runner" $SB/last.out'
reset; sed 's/2026-10-20T12:00:00+05:30/2026-10-01T00:00:00+05:30/' $SB/launch.sh > $SB/launch_frozen.sh
rc=$(env -u CUDA_VISIBLE_DEVICES HOME=$SB/home PATH=$SB/bin:$PATH LAUNCH=1 bash $SB/launch_frozen.sh probe > $SB/last.out 2>&1; echo $?)
ok "past the freeze -> refused (NEW guard)" '[ $rc = 4 ] && [ ! -e $SB/calls.txt ] && grep -q "freeze" $SB/last.out'
reset; ( exec -a "tail -n 0 -F /x/fleet_trainSBX.py" sleep 30 ) & XP=$!; sleep 1; rc=$(L LAUNCH=1); kill $XP 2>/dev/null
ok "a tail naming fleet_train.py is NOT a trainer (exact match): launch proceeds" '[ $rc = 0 ] && [ $(wc -l < $SB/calls.txt) = 1 ]'

echo "T3 launch (all free): argv, log, skip rules"
reset; rc=$(L LAUNCH=1)
( HOME=$SB/home; source $SB/home/novel/frac75/argv_frac75.sh; frac75_argv 0; echo "fleet_train.py ${FRAC75_ARGV[*]}" ) > $SB/want_call0.txt
ok "probe launched once with exactly frac75_argv 0, rc 0" '[ $rc = 0 ] && [ $(wc -l < $SB/calls.txt) = 1 ] && diff -q $SB/calls.txt $SB/want_call0.txt > /dev/null'
ok "probe log written at OUT.log with the closing line" 'grep -q "^{\"raw\"" $R/f0_s1043_ttr_frac75.log'
rm -f $SB/calls.txt; mkdir -p $R/f0_s1043_ttr_frac75; rc=$(L LAUNCH=1)
ok "re-run with OUT existing -> skipped, trainer not called" '[ $rc = 0 ] && [ ! -e $SB/calls.txt ] && grep -q "skipping" $SB/last.out'
rmdir $R/f0_s1043_ttr_frac75; echo "another launcher's log" > $R/f0_s1043_ttr_frac75.log; rc=$(L LAUNCH=1)
ok "only OUT.log existing -> skipped, log NOT truncated (NEW)" '[ ! -e $SB/calls.txt ] && grep -q "another launcher" $R/f0_s1043_ttr_frac75.log'

echo "T4 folds gate"
ARGS=(folds); G=$SB/home/novel/frac75/out/gate_f0_s1043_ttr_frac75.json; RUNP=$SB/home/fleet_runs/f0_s1043_ttr_frac75
reset; rc=$(L LAUNCH=1); ok "no gate json -> refused" '[ $rc = 4 ] && [ ! -e $SB/calls.txt ]'
reset; printf '{\n "run": "%s",\n "dry": true,\n "pass_": true\n}\n' $RUNP > $G; rc=$(L LAUNCH=1); ok "dry json -> refused" '[ $rc = 4 ] && [ ! -e $SB/calls.txt ]'
reset; printf '{\n "run": "%s",\n "dry": false,\n "pass_": false\n}\n' $RUNP > $G; rc=$(L LAUNCH=1); ok "FAIL json -> refused" '[ $rc = 4 ] && [ ! -e $SB/calls.txt ]'
reset; printf '{\n "run": "%s",\n "dry": false,\n "pass_": true\n}\n' $SB/home/novel/frac75/dry/f0_s1043_ttr_frac75 > $G; rc=$(L LAUNCH=1)
ok "PASS json for another run dir -> refused (NEW)" '[ $rc = 4 ] && [ ! -e $SB/calls.txt ]'
reset; printf '{\n "run": "%s",\n "dry": false,\n "pass_": true\n}\n' $RUNP > $G; rc=$(L LAUNCH=1)
ok "real PASS json -> folds 1-4 launched in order, one call each" '[ $rc = 0 ] && [ "$(grep -o -- "--fold [0-9]" $SB/calls.txt | tr "\n" " ")" = "--fold 1 --fold 2 --fold 3 --fold 4 " ]'
ok "folds use 21 ep, gc0, seed 1043, save 6-21" '[ $(grep -c -- "--epochs 21 --seed 1043 --save_epochs 6-21" $SB/calls.txt) = 4 ] && [ $(grep -c -- "--grad_ckpt 0$" $SB/calls.txt) = 4 ]'
reset; printf '{\n "run": "%s",\n "dry": false,\n "pass_": true\n}\n' $RUNP > $G; echo 1 > $SB/rc_gc0; rc=$(L LAUNCH=1)
ok "early gc0 failure -> moved aside + ONE gc1 rerun per fold (8 calls), no loop" '[ $(wc -l < $SB/calls.txt) = 8 ] && [ $(grep -c -- "--grad_ckpt 0 --grad_ckpt 1$" $SB/calls.txt) = 4 ] && [ $(ls -d $R/*_fastfail_*.log | wc -l) = 4 ]'
reset; printf '{\n "run": "%s",\n "dry": false,\n "pass_": true\n}\n' $RUNP > $G; echo 1 > $SB/rc; rc=$(L LAUNCH=1)
ok "probe-style failure on gc1 rerun too -> still one rerun only" '[ $(wc -l < $SB/calls.txt) = 8 ]'
ARGS=(probe); reset; echo 1 > $SB/rc; rc=$(L LAUNCH=1)
ok "probe (f0) failure -> no fallback, out kept for the operator" '[ $(wc -l < $SB/calls.txt) = 1 ] && [ -e $R/f0_s1043_ttr_frac75.log ]'
echo "RESULT: $NP pass, $NF fail"
