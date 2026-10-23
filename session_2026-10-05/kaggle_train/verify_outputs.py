# verify_outputs.py <dir> [--receipt <run_receipt.json>]
# Checks the rsna-train-ttr-f34 outputs in <dir> against the run receipt the kernel wrote: every file listed for
# folds 3/4 present with the receipt's sha256 (files skipped on purpose, e.g. epochs/*.pt, are reported as ABSENT),
# then prints the facts the operator decides on (status, E, s/it, warnings, best val per fold, closing line).
# Works on the laptop download dir AND on the box (~/fleet_runs: same relative layout). Exit 0 = status DONE and
# no sha mismatch; 2 = PARTIAL or ABORTED (read the warnings); 1 = a present file differs from the receipt, or a ckpt
# the kernel itself could not load / had other keys (UNUSABLE, e.g. a kill inside torch.save).
import os, sys, json, hashlib

d = sys.argv[1]
rp = sys.argv[sys.argv.index('--receipt') + 1] if '--receipt' in sys.argv else os.path.join(d, 'run_receipt.json')
rec = json.load(open(rp))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


bad = absent = 0
print(f'receipt {rp}\nstatus  {rec["status"]}')
pl = rec.get('plan') or {}
pr = rec.get('probe') or {}
print(f'plan    E={pl.get("E")} save_epochs={pl.get("save_epochs")} s/it used={pr.get("s_it_used")} per fold '
      f'{pr.get("s_per_it")} epoch est {pl.get("epoch_min_est")} min | launch +{pl.get("elapsed_at_launch_h")} h | est '
      f'finish {pl.get("est_finish_ist")} | wall {pl.get("wall_ist")}')
print(f'probe   workers {pr.get("workers")} | CPU busy {pr.get("cpu_busy_frac")} | GPU util {pr.get("gpu_util_mean_pct")} '
      f'| {pr.get("verdict")}')
for w in rec.get('warnings', []):
    print('WARNING', w)
for F, x in sorted((rec.get('train') or {}).get('folds', {}).items()):
    print(f'\nfold {F}: complete={x["complete"]} ended={x["ended"]} rc={x["rc"]} closing={x["closing_line"]} '
          f'epochs done {x["progress"]["epochs_done"]} best {x["progress"]["best"]}')
    for e in x['files']:
        p = os.path.join(d, e['path'])
        if not os.path.exists(p):
            absent += 1
            print(f'  ABSENT   {e["path"]}')
            continue
        ok = os.path.getsize(p) == e['bytes'] and sha(p) == e['sha256']
        unusable = e.get('load_ok') is False or e.get('keys_match') is False    # kernel could not load it / wrong keys
        bad += (not ok) or unusable
        extra = '' if 'load_ok' not in e else f'  (load_ok {e["load_ok"]}, keys_match {e.get("keys_match")}, cv {e.get("cv")}, epoch {e.get("epoch")})'
        print(f'  {"MISMATCH" if not ok else "UNUSABLE" if unusable else "OK      "} {e["path"]}{extra}')
    lp = os.path.join(d, os.path.basename(x['log']))
    print(f'  log {"present" if os.path.exists(lp) else "ABSENT"}: {lp}')
print(f'\n{bad} mismatch, {absent} absent')
sys.exit(1 if bad else (0 if rec['status'] == 'DONE' else 2))
