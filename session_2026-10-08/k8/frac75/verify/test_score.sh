#!/bin/bash
# test_score.sh (frac75 verify pass, 2026-10-07): CPU-only adversarial tests of ~/novel/frac75/score_frac75.py.
# Fake probe dirs live in ~/novel/frac75_verify/fake (J1's best raw ckpt re-saved with edited args; J1's OOF + log).
# Every scorer call passes --tag verify_* so the canonical gate_f0_s1043_ttr_frac75.json is never written, and
# --gold_preds (J1's) so gold_infer5 never runs and ~/novel/frac75/gold74_preds is never written.
set -u
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=~/miniconda3/envs/rsna/bin/python; S=~/novel/frac75/score_frac75.py; V=~/novel/frac75_verify; FK=$V/fake
J1=~/fleet_runs/f0_s1043_ttr_mix50; J1G=~/jump2/gold74_preds/f0_s1043_ttr_mix50__raw.csv
rm -rf $FK; mkdir -p $FK
NP=0; NF=0
ok(){ if eval "$2"; then NP=$((NP+1)); echo "  PASS $1"; else NF=$((NF+1)); echo "  FAIL $1"; fi; }
mk(){   # name, python expr editing dict a (args) -> fake run dir $FK/$1/f0_s1043_ttr_frac75 (+ .log)
  local d=$FK/$1/f0_s1043_ttr_frac75; mkdir -p $d
  cp $J1/oof_f0_s1043_raw.csv $d/; cp $J1.log $d.log
  nice -n 19 $PY - "$J1" "$d" "$2" <<'EOF'
import sys, glob, os, torch
j1, d, edit = sys.argv[1:4]
p = glob.glob(f'{j1}/f0_s1043_raw_*_best.pt'); assert len(p) == 1
c = torch.load(p[0], map_location='cpu', weights_only=False); a = c['args']
exec(edit)
torch.save(c, f'{d}/{os.path.basename(p[0])}')
EOF
}
sc(){ nice -n 19 $PY $S --run "$1" --tag "$2" --gold_preds $J1G --boot 200 "${@:3}" > $V/$2.out 2>&1; echo $?; }

echo "S1 J1 itself as a null probe (labels mix50): FAIL on G2, exit 1"
rc=$(sc $J1 verify_J1_null --expect_labels labels_mix50_logit.csv)
ok "exit 1, FAIL (G2), G1 .8803 G2 .8936 G3 .9092" '[ $rc = 1 ] && grep -q "FRAC75 GATE: FAIL (G2)" $V/verify_J1_null.out && grep -q "G1 no harm   : OOF 11 non-Fracture labels vs v4 0.8803" $V/verify_J1_null.out && grep -q "gold74 Fracture (masked)        0.8936" $V/verify_J1_null.out && grep -q "gold74 macro (masked)           0.9092" $V/verify_J1_null.out'
ok "json pass_ false, dry false, tag verify_J1_null (canonical json untouched)" 'grep -q "\"pass_\": false" ~/novel/frac75/out/gate_verify_J1_null.json && [ ! -e ~/novel/frac75/out/gate_f0_s1043_ttr_frac75.json ]'

echo "S2 J1 weights re-tagged as the frac75 probe (only train_labels differs): integrity OK, FAIL (G2), exit 1"
mk paired "a['train_labels'] = 'labels_frac75_logit.csv'"
rc=$(sc $FK/paired/f0_s1043_ttr_frac75 verify_paired)
ok "exit 1, pairing line says only train_labels differs, label md5 checked" '[ $rc = 1 ] && grep -q "differ only in train_labels (labels_mix50_logit.csv -> labels_frac75_logit.csv)" $V/verify_paired.out && grep -q "== pre-registered" $V/verify_paired.out'

echo "S3 integrity refusals (exit 2)"
mk seed "a['train_labels'] = 'labels_frac75_logit.csv'; a['seed'] = 1044"
rc=$(sc $FK/seed/f0_s1043_ttr_frac75 verify_seed); ok "seed differs -> exit 2" '[ $rc = 2 ] && grep -q "INTEGRITY ERROR" $V/verify_seed.out'
mk amp "a['train_labels'] = 'labels_frac75_logit.csv'; a['amp'] = 'fp32'"
rc=$(sc $FK/amp/f0_s1043_ttr_frac75 verify_amp); ok "amp differs (a CPU run) -> exit 2" '[ $rc = 2 ]'
mk lab "pass"
rc=$(sc $FK/lab/f0_s1043_ttr_frac75 verify_lab); ok "train_labels still mix50 (wrong label file) -> exit 2" '[ $rc = 2 ] && grep -q "train_labels labels_mix50_logit.csv != labels_frac75_logit.csv" $V/verify_lab.out'
mk sha "a['train_labels'] = 'labels_frac75_logit.csv'; a['model_sha'] = 'deadbeefdeadbeef'"
rc=$(sc $FK/sha/f0_s1043_ttr_frac75 verify_sha); ok "model_sha differs -> exit 2" '[ $rc = 2 ]'
mk ttr "a['train_labels'] = 'labels_frac75_logit.csv'; a['ttr_size'] = 288"
rc=$(sc $FK/ttr/f0_s1043_ttr_frac75 verify_ttr); ok "ttr_size 288 -> exit 2" '[ $rc = 2 ]'
mk oof "a['train_labels'] = 'labels_frac75_logit.csv'"
$PY -c "import pandas as pd,sys; p=sys.argv[1]; o=pd.read_csv(p); o.iloc[:,1]=o.iloc[:,1][::-1].values; o.to_csv(p,index=False)" $FK/oof/f0_s1043_ttr_frac75/oof_f0_s1043_raw.csv
rc=$(sc $FK/oof/f0_s1043_ttr_frac75 verify_oof); ok "OOF file not the selected epoch (macro != ckpt cv) -> exit 2" '[ $rc = 2 ] && grep -q "not the selected epoch" $V/verify_oof.out'
mk cv "a['train_labels'] = 'labels_frac75_logit.csv'; c['cv'] = 0.9"
rc=$(sc $FK/cv/f0_s1043_ttr_frac75 verify_cv); ok "ckpt cv != closing raw -> exit 2" '[ $rc = 2 ]'

echo "S4 not scorable (exit 3)"
mk nolog "a['train_labels'] = 'labels_frac75_logit.csv'"; rm $FK/nolog/f0_s1043_ttr_frac75.log
rc=$(sc $FK/nolog/f0_s1043_ttr_frac75 verify_nolog); ok "no log -> exit 3" '[ $rc = 3 ]'
mk crash "a['train_labels'] = 'labels_frac75_logit.csv'"; L=$FK/crash/f0_s1043_ttr_frac75.log
grep -v "^{" $L | awk '!/^ep 2[0-3] /' > $L.t && mv $L.t $L
rc=$(sc $FK/crash/f0_s1043_ttr_frac75 verify_crash); ok "crashed at ep 20 (no closing line) -> exit 3" '[ $rc = 3 ]'
mk early "a['train_labels'] = 'labels_frac75_logit.csv'"; L=$FK/early/f0_s1043_ttr_frac75.log
awk '!/^ep 2[0-3] /' $L > $L.t && mv $L.t $L
rc=$(sc $FK/early/f0_s1043_ttr_frac75 verify_early); ok "closing line but only 20 of 24 epochs evaluated -> exit 3" '[ $rc = 3 ] && grep -q "raw evals for 20 epochs" $V/verify_early.out'
mk two "a['train_labels'] = 'labels_frac75_logit.csv'"; cp $FK/two/f0_s1043_ttr_frac75/*_best.pt $FK/two/f0_s1043_ttr_frac75/f0_s1043_raw_ttr224-320x24_cv1_best.pt
rc=$(sc $FK/two/f0_s1043_ttr_frac75 verify_two); ok "two raw best ckpts -> exit 3" '[ $rc = 3 ]'

echo "S5 synthetic gold preds outside dry mode: the trainer-log parity check refuses (exit 2)"
$PY - $J1G $V/fakepass.csv <<'EOF'
import pandas as pd, numpy as np, sys
g = pd.read_csv('/home/training/rsna_train/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
p = pd.read_csv(sys.argv[1], dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
y = g.reindex(p.index)['Fracture'].fillna(0.5).values
p['Fracture'] = 0.5 * p['Fracture'] + 0.5 * y          # pull Fracture toward the experts
for c in p.columns: p[c] = 0.3 * p[c] + 0.7 * g.reindex(p.index)[c].fillna(p[c]).values   # and every label
p.to_csv(sys.argv[2])
EOF
rc=$(nice -n 19 $PY $S --run $FK/paired/f0_s1043_ttr_frac75 --tag verify_fakepass --gold_preds $V/fakepass.csv --boot 100 > $V/verify_fakepass.out 2>&1; echo $?)
ok "fake PASS preds -> INTEGRITY ERROR (parity), exit 2, no PASS json" '[ $rc = 2 ] && grep -q "differs from the trainer log" $V/verify_fakepass.out && [ ! -e ~/novel/frac75/out/gate_verify_fakepass.json ]'
rc=$(nice -n 19 $PY $S --run $FK/paired/f0_s1043_ttr_frac75 --tag verify_fakepass_dry --gold_preds $V/fakepass.csv --boot 100 --dry > $V/verify_fakepass_dry.out 2>&1; echo $?)
ok "same in --dry -> PASS printed but json pass_ false, tag DRY_" '[ $rc = 0 ] && grep -q "FRAC75 GATE: PASS" $V/verify_fakepass_dry.out && grep -q "\"pass_\": false" ~/novel/frac75/out/gate_DRY_verify_fakepass_dry.json'
rm -rf $FK $V/fakepass.csv
echo "RESULT: $NP pass, $NF fail"
