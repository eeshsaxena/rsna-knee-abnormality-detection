#!/bin/bash
# VERIFY (10-05, adversarial review): 2-member A0 where member 2 (re-keyed fold 1) has NO DICOM val study on the box.
# CPU only, nice 19, writes only ~/k4_verify. Read-only on ~/fleet_runs and ~/k4_test.
set -euo pipefail
PYK=~/miniconda3/envs/kmimic/bin/python
V=~/k4_verify; S=$V/sim; J1=~/fleet_runs/f0_s1043_ttr_mix50; TV=~/fleet_runs/f0_s1042_ttr
rm -rf $S; mkdir -p $S/ds $S/kernel $S/runs
cd ~/k4_test/ds/rsna-knee-cnxt-fleet && cp -p fleet_*.py dataset-metadata.json parity_*.npz $S/ds/
N0=f0_s1043_raw_ttr224-320x24_cv8835_best.pt; N1=f1_s1043_raw_ttr224-320x24_sim1_best.pt
cp -p $J1/$N0 $S/ds/$N0; cp -p $(ls $TV/f0_s1042_raw_*_best.pt) $S/ds/$N1
CUDA_VISIBLE_DEVICES= $PYK - $S $N0 $N1 <<'EOF'
import hashlib, json, sys, os, torch
S, n0, n1 = sys.argv[1:4]; H = os.path.expanduser('~')
src = {}
for n in (n0, n1):
    a = torch.load(f'{S}/ds/{n}', map_location='cpu', weights_only=False)['args']
    src[n] = dict(original=n, run_dir='members', sha256=hashlib.sha256(open(f'{S}/ds/{n}', 'rb').read()).hexdigest()[:16],
                  model_mod=a.get('model_mod'), model_sha=a.get('model_sha'))
json.dump({'checkpoints': [n0, n1], 'sources': src}, open(f'{S}/ds/fleet_manifest.json', 'w'), indent=1)
m = json.load(open(f'{H}/k4_verify/members_sim2.json'))
m['members'][1]['file'] = n1
json.dump(m, open(f'{H}/k4_verify/members_sim2_e2e.json', 'w'), indent=1)
nb = json.load(open(f'{H}/k4_test/kernel/rsna_sub_k4_fleet.ipynb'))
s31 = ''.join(nb['cells'][31]['source']); hdr, body = s31.split('\n', 1)
cfg = eval(hdr.split('=', 1)[1].split('   #')[0]); cfg['MEMBERS'] = [n0, n1]
nb['cells'][31]['source'] = (f'K2_CFG = {cfg!r}   # VERIFY sim2\n' + body).splitlines(keepends=True)
json.dump(nb, open(f'{S}/kernel/sim2_main.ipynb', 'w'), indent=1)
print('manifest', [n0, n1], 'cfg', cfg)
EOF
# the FIXED launch snippet, paths redirected to the sim
sed -n "/^OOFARGS=\$(\$PYK - \"\$IDS\" <<'EOF'/,/^EOF/p" ~/k4_verify/k4_box_launch.sh | sed '1d;$d' \
  | sed "s#{H}/k4_test/ds/rsna-knee-cnxt-fleet#{H}/k4_verify/sim/ds#; s#{H}/k4_test/members.json#{H}/k4_verify/members_sim2_e2e.json#" > $S/oofargs.py
OOFARGS=$($PYK $S/oofargs.py $V/a0_sim2.csv)
echo "OOFARGS: $OOFARGS"
cd $V
CUDA_VISIBLE_DEVICES= nohup setsid nice -n 19 $PYK -W ignore ~/k4_verify/k2_box_stage.py --cpu --nb $S/kernel/sim2_main.ipynb \
  --ids $V/a0_sim2.csv --fleet_ds $S/ds --age_h 7.9 --threads 4 $OOFARGS --out $S/runs/SIM2_main > $S/runs/SIM2_main.log 2>&1 < /dev/null &
echo "started SIM2 pid $!"
