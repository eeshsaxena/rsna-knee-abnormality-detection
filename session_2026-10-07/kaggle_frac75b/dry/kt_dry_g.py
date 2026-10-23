# kt_dry_g.py A|B (kaggle_frac75b): phase G of the dry test, run after kt_dry.py A|B (uses its A and B phase outputs).
# RUNBOOK copy logic into empty placeholders of a FAKE ~/fleet_runs (~/kt_frac75b/dry_<KEY>/fake_fleet_runs), verify_outputs.py
# on it, the fetch --file-pattern, and the closing-line rule. Never touches the real ~/fleet_runs.
import os, re, sys, json, shutil, subprocess

H = os.path.expanduser('~')
KT = f'{H}/kt_frac75b'
KEY = sys.argv[1]
NBK = {'A': ('rsna-train-ttr-frac75-f23', 'f23', [(2, 1043), (3, 1043)]),
       'B': ('rsna-train-ttr-frac75-f4x', 'f4x', [(4, 1043), (0, 1044)])}[KEY]
RUNS = [f'f{F}_s{s}_ttr_frac75' for F, s in NBK[2]]
RCPT = f'kaggle_frac75_{NBK[1]}_run_receipt.json'
DRY = f'{KT}/dry_{KEY}'
PY = f'{H}/miniconda3/envs/rsna/bin/python'
VO = f'{KT}/verify_outputs.py'
RES = {}


def check(name, cond, info=''):
    RES[name] = bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{info}]' if info else ''), flush=True)


FR = f'{DRY}/fake_fleet_runs'
WA = f'{DRY}/A/kaggle/working'
WB = f'{DRY}/B/kaggle/working'
shutil.rmtree(FR, ignore_errors=True)
for r in RUNS:
    os.makedirs(f'{FR}/{r}')                                   # the empty placeholders of the RUNBOOK
# the RUNBOOK's fetch pattern, applied the way the kaggle CLI does (re.search on the output-relative path)
PAT = r'^(?!_[a-z0-9]+_env/)(?!.*epochs/.*\.pt$)'
rel = sorted(os.path.relpath(os.path.join(r, f), WA) for r, _, fs in os.walk(WA) for f in fs)
keep = [p for p in rel if re.search(PAT, p)]
check(f'G fetch pattern: drops epochs/*.pt and _xxx_env/, keeps 4 best ckpts, 12 epoch OOFs, both logs ({RUNS}), receipt',
      not any(p.endswith('.pt') and '/epochs/' in p for p in keep) and
      sum(p.endswith('_best.pt') and '/epochs/' not in p for p in keep) == 4 and
      sum('/epochs/oof_' in p for p in keep) == 12 and 'run_receipt.json' in keep and
      all(f'{r}.log' in keep for r in RUNS) and all(any(p.startswith(f'{r}/') for p in keep) for r in RUNS) and
      re.search(PAT, '_abc123_env/lib/x.py') is None and re.search(PAT, f'{RUNS[0]}/epochs/oof_x_ep24.csv'),
      f'{len(keep)} of {len(rel)} files kept')
DL = f'{DRY}/fetched'                                         # what the laptop holds after the filtered fetch
shutil.rmtree(DL, ignore_errors=True)
for p in keep:
    os.makedirs(os.path.dirname(f'{DL}/{p}'), exist_ok=True)
    shutil.copy2(f'{WA}/{p}', f'{DL}/{p}')
v = subprocess.run([PY, VO, DL], capture_output=True, text=True)
check('G verify_outputs on the filtered fetch: exit 0, 0 mismatch, 12 absent (= the epochs/*.pt left on Kaggle)',
      v.returncode == 0 and '0 mismatch, 12 absent' in v.stdout, v.stdout[-200:])
script = r'''
FR="$1"; OUT="$2"; RC="$3"; shift 3
for R in "$@"; do
  test -d "$FR/$R" && [ -z "$(ls -A "$FR/$R")" ] && [ ! -e "$FR/$R.log" ] || { echo "STOP: $R is not an empty placeholder"; exit 9; }
  cp -r "$OUT/$R/"* "$FR/$R/"
  cp "$OUT/$R.log" "$FR/$R.log"
done
cp "$OUT/run_receipt.json" "$FR/$RC"
'''
cp = subprocess.run(['bash', '-c', script, 'x', FR, DL, RCPT] + RUNS, capture_output=True, text=True)
check('G copy into the empty placeholders (RUNBOOK logic)', cp.returncode == 0, cp.stdout + cp.stderr)
cp2 = subprocess.run(['bash', '-c', script, 'x', FR, DL, RCPT] + RUNS, capture_output=True, text=True)
check('G second copy refused (placeholder no longer empty)', cp2.returncode == 9 and 'STOP' in cp2.stdout, cp2.stdout)
v = subprocess.run([PY, VO, FR, '--receipt', f'{FR}/{RCPT}'], capture_output=True, text=True)
print(v.stdout[-2500:])
check(f'G verify_outputs on the box layout ({RCPT}): exit 0, 0 mismatch', v.returncode == 0 and '0 mismatch,' in v.stdout,
      v.returncode)
v = subprocess.run([PY, VO, WB], capture_output=True, text=True)
check('G verify_outputs on the watchdog run: exit 2 (PARTIAL)', v.returncode == 2 and 'PARTIAL' in v.stdout, v.returncode)
r0, r1 = RUNS
s0 = r0.split('_')[1]
os.remove(f'{FR}/{r0}/oof_{r0.split("_")[0]}_{s0}_raw.csv')
p = [f for f in os.listdir(f'{FR}/{r1}') if f.endswith('.pt')][0]
with open(f'{FR}/{r1}/{p}', 'r+b') as f:
    f.seek(1000)
    b = f.read(1)
    f.seek(1000)
    f.write(bytes([b[0] ^ 1]))
v = subprocess.run([PY, VO, FR, '--receipt', f'{FR}/{RCPT}'], capture_output=True, text=True)
check('G verify_outputs flags an absent file and a changed ckpt (exit 1)',
      v.returncode == 1 and f'ABSENT   {r0}/oof_{r0.split("_")[0]}_{s0}_raw.csv' in v.stdout and 'MISMATCH' in v.stdout,
      v.returncode)
k4 = [json.loads(open(f'{WA}/{r}.log').read().splitlines()[-1]) for r in RUNS]
kb = [open(f'{WB}/{r}.log').read().splitlines()[-1] for r in RUNS]
check('G closing-line rule: a complete run ends with {"raw", "ema"}, a watchdog-cut run does not',
      all(set(j) == {'raw', 'ema'} for j in k4) and not any(l.lstrip().startswith('{') for l in kb), kb)
print(f'\nPHASE G {KEY}: {sum(RES.values())}/{len(RES)} PASS')
sys.exit(0 if all(RES.values()) else 1)
