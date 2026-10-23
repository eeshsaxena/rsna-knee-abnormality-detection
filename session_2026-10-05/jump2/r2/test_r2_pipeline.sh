#!/bin/bash
# test_r2_pipeline.sh (jump2 r2): end-to-end CPU test of auto_round2.sh in R2_TEST mode with fake fold runs and a fake
# trainer. Touches only ~/jump2/r2/test/. No GPU (CUDA_VISIBLE_DEVICES=''), nice 19.
set -eu
PY=$HOME/miniconda3/envs/rsna/bin/python
TD=$HOME/jump2/r2/test; rm -rf "$TD"; mkdir -p "$TD/runs" "$TD/r2" "$TD/code"
cp $HOME/jump2/r2/build_r2_targets.py $HOME/jump2/r2/r2_capped.py $HOME/jump2/r2/r2_gate.py "$TD/r2/"
export CUDA_VISIBLE_DEVICES=''
nice -n 19 $PY - <<'EOF'
import os, numpy as np, pandas as pd, shutil
H = os.path.expanduser; TD = H('~/jump2/r2/test')
T = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
rd = lambda p: pd.read_csv(H(p), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
v4 = rd('~/rsna_train/llm_labels_v4_blend.csv')[T]; folds = rd('~/rsna_train/folds5.csv')['fold']
rng = np.random.default_rng(0)
for F in range(5):
    d = f'{TD}/runs/f{F}_s1043_ttr_mix50'; os.makedirs(d)
    if F == 0:
        shutil.copy(H('~/fleet_runs/f0_s1043_ttr_mix50/oof_f0_s1043_raw.csv'), f'{d}/oof_f0_s1043_raw.csv')
    else:
        ids = folds.index[folds == F]; p = np.clip(v4.loc[ids].values, .01, .99)
        z = np.log(p / (1 - p)) * 0.5 + rng.normal(0, 1, p.shape) + 0.3 * F      # fold-specific calibration shift
        pd.DataFrame(1 / (1 + np.exp(-z)), index=ids, columns=T).reset_index().to_csv(f'{d}/oof_f{F}_s1043_raw.csv', index=False)
    with open(d + '.log', 'w') as f:
        f.write('labels 4407 | fake\nep 0 raw | x\n{"raw": 0.88%d, "ema": 0.87}\n' % F)
EOF
cat > "$TD/code/fleet_train.py" <<'EOF'
# FAKE trainer for test_r2_pipeline.sh: checks the label csv, exercises a forked DataLoader with a __main__ class,
# writes a fake OOF and closing JSON. Never touches the GPU.
import argparse, os, sys, shutil, json, numpy as np, pandas as pd, torch
ap = argparse.ArgumentParser(); ap.add_argument('--labels'); ap.add_argument('--out'); ap.add_argument('--fold', type=int)
ap.add_argument('--seed', type=int); ap.add_argument('--grad_ckpt', type=int, default=1); ap.add_argument('--model', default='f1')
A, rest = ap.parse_known_args()
print('FAKE argv', ' '.join(sys.argv[1:]), flush=True)
if A.model == 'ttr':   # same guard as the real fleet_train.py BJ-5 (verify 10-05: seed 2043 crashed the real trainer)
    assert 1000 <= A.seed <= 1999, f'--model ttr runs use seeds 1000-1999: got --seed {A.seed}'
lab = pd.read_csv(A.labels, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
assert len(lab) == 4407 and not lab.isna().any().any() and lab.values.min() >= 0 and lab.values.max() <= 1
class DS(torch.utils.data.Dataset):
    def __len__(self): return 8
    def __getitem__(self, i): return torch.tensor([i], dtype=torch.float32)
s = sum(float(b.sum()) for b in torch.utils.data.DataLoader(DS(), batch_size=2, num_workers=2))
assert s == 28.0, s
print(f'labels {len(lab)} | store has 4407 | held-out (fold<0 or --exclude) 74 | train 3466 | val 867 | fake loader ok', flush=True)
os.makedirs(A.out)
shutil.copy(os.path.expanduser('~/fleet_runs/f0_s1042_ttr/oof_f0_s1042_ema.csv'), f'{A.out}/oof_f0_s{A.seed}_ema.csv')
print('ep 0 raw | gold74 0.9000 (12 labels) | gold58 0.9050 (12 labels) | val AUC vs llm_labels_v4_blend.csv 0.8750 (12 labels) | x')
print('ep 0 ema | gold74 0.9010 (12 labels) | gold58 0.9100 (12 labels) | val AUC vs llm_labels_v4_blend.csv 0.8810 (12 labels) | x')
print(json.dumps({'raw': 0.875, 'ema': 0.881}), flush=True)
EOF
cd $HOME/jump2/r2
# negative cases (verify 10-05): each must exit 2 at startup, before any wait
set +e
for BAD in "R2_SEED=2043" "R2_SEED=1043" "R2_SETS=$TD/runs/f?_s1043_ttr_mix50*"; do
  env R2_TEST=1 R2_RUNS=$TD/runs R2_DIR=$TD/r2 R2_CODE_DIR=$TD/code R2_LOCK=$TD/test.lock R2_SETS="$TD/runs/f?_s1043_ttr_mix50" \
    "$BAD" bash $HOME/auto_round2.sh --check; rc=$?
  [ $rc -eq 2 ] && echo "NEG OK ($BAD -> rc 2)" || { echo "NEG FAIL ($BAD -> rc $rc)"; exit 1; }
done
set -e
R2_TEST=1 R2_RUNS=$TD/runs R2_DIR=$TD/r2 R2_CODE_DIR=$TD/code R2_LOCK=$TD/test.lock R2_SETS="$TD/runs/f?_s1043_ttr_mix50" \
  bash $HOME/auto_round2.sh --check
R2_TEST=1 R2_RUNS=$TD/runs R2_DIR=$TD/r2 R2_CODE_DIR=$TD/code R2_LOCK=$TD/test.lock R2_SETS="$TD/runs/f?_s1043_ttr_mix50" \
  nice -n 19 bash $HOME/auto_round2.sh || echo "auto_round2 rc=$?"
echo "===== test log"; cat "$TD/runs/auto_round2.log"
echo "===== trainer log"; cat "$TD/runs/f0_s1543_ttr_r2a50.log"
ls -la "$TD/r2"
