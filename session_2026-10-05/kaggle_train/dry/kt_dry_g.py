# kt_dry_g.py: phase G of the dry test, run after kt_dry.py (uses its A and B outputs). RUNBOOK sec 6 copy logic into
# empty placeholders of a FAKE ~/fleet_runs (~/kt_test/dry/fake_fleet_runs), verify_outputs.py on it, and the K4
# closing-line rule. Never touches the real ~/fleet_runs.
import os, sys, json, shutil, subprocess

H = os.path.expanduser('~')
KT = f'{H}/kt_test'
DRY = f'{KT}/dry'
PY = f'{H}/miniconda3/envs/rsna/bin/python'
VO = f'{KT}/dry_src/verify_outputs.py'
RES = {}


def check(name, cond, info=''):
    RES[name] = bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{info}]' if info else ''), flush=True)


FR = f'{DRY}/fake_fleet_runs'
WA = f'{DRY}/A/kaggle/working'
WB = f'{DRY}/B/kaggle/working'
shutil.rmtree(FR, ignore_errors=True)
for F in (3, 4):
    os.makedirs(f'{FR}/f{F}_s1043_ttr_mix50')                 # the empty placeholders of RUNBOOK sec 3
script = r'''
FR="$1"; OUT="$2"
for F in 3 4; do R=f${F}_s1043_ttr_mix50
  test -d "$FR/$R" && [ -z "$(ls -A "$FR/$R")" ] && [ ! -e "$FR/$R.log" ] || { echo "STOP: $R is not an empty placeholder"; exit 9; }
  cp -r "$OUT/$R/"* "$FR/$R/"
  cp "$OUT/$R.log" "$FR/$R.log"
done
cp "$OUT/run_receipt.json" "$FR/kaggle_f34_run_receipt.json"
'''
cp = subprocess.run(['bash', '-c', script, 'x', FR, WA], capture_output=True, text=True)
check('G copy into the empty placeholders (RUNBOOK sec 6 logic)', cp.returncode == 0, cp.stdout + cp.stderr)
cp2 = subprocess.run(['bash', '-c', script, 'x', FR, WA], capture_output=True, text=True)
check('G second copy refused (placeholder no longer empty)', cp2.returncode == 9 and 'STOP' in cp2.stdout, cp2.stdout)
v = subprocess.run([PY, VO, FR, '--receipt', f'{FR}/kaggle_f34_run_receipt.json'], capture_output=True, text=True)
print(v.stdout[-2500:])
check('G verify_outputs on the box layout: exit 0, 0 mismatch, 0 absent', v.returncode == 0 and '0 mismatch, 0 absent' in v.stdout,
      v.returncode)
v = subprocess.run([PY, VO, WB], capture_output=True, text=True)
check('G verify_outputs on the watchdog run: exit 2 (PARTIAL)', v.returncode == 2 and 'PARTIAL' in v.stdout, v.returncode)
os.remove(f'{FR}/f3_s1043_ttr_mix50/oof_f3_s1043_raw.csv')
p = [f for f in os.listdir(f'{FR}/f4_s1043_ttr_mix50') if f.endswith('.pt')][0]
with open(f'{FR}/f4_s1043_ttr_mix50/{p}', 'r+b') as f:
    f.seek(1000)
    b = f.read(1)
    f.seek(1000)
    f.write(bytes([b[0] ^ 1]))
v = subprocess.run([PY, VO, FR, '--receipt', f'{FR}/kaggle_f34_run_receipt.json'], capture_output=True, text=True)
check('G verify_outputs flags an absent file and a changed ckpt (exit 1)',
      v.returncode == 1 and 'ABSENT   f3_s1043_ttr_mix50/oof_f3_s1043_raw.csv' in v.stdout and 'MISMATCH' in v.stdout,
      v.returncode)
k4 = [json.loads(open(f'{WA}/f{F}_s1043_ttr_mix50.log').read().splitlines()[-1]) for F in (3, 4)]
kb = [open(f'{WB}/f{F}_s1043_ttr_mix50.log').read().splitlines()[-1] for F in (3, 4)]
check('G K4 rule: a complete run ends with {"raw", "ema"}, a watchdog-cut run does not',
      all(set(j) == {'raw', 'ema'} for j in k4) and not any(l.lstrip().startswith('{') for l in kb), kb)
print(f'\nPHASE G: {sum(RES.values())}/{len(RES)} PASS')
sys.exit(0 if all(RES.values()) else 1)
