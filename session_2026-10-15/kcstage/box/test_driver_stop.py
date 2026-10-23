# test_driver_stop.py (10-15, Claude subagent, kcstage): the KC driver's own stop rule on Linux (box CPU): --stop-at = start + 25 s, so the
# side pass completes and the two serve processes (CPU, stand-in members, still loading / decoding) are SIGKILLed with their groups at the
# stop time. Expect: driver exit 0, receipt ok, serve_timed_out True, 2 worker groups killed, kc_probs.csv written with every study unscored
# (coverage 0.0 -> the BLEND cell would fire the KC canary), and no process naming the test root left.
#   python test_driver_stop.py --driver kc_driver.py --serve kc_serve.py --code CODE --root FAKEROOT --members m1 m2
import os, sys, json, time, shutil, argparse, subprocess, tempfile

ap = argparse.ArgumentParser()
ap.add_argument('--driver', required=True); ap.add_argument('--serve', required=True); ap.add_argument('--code', required=True)
ap.add_argument('--root', required=True); ap.add_argument('--members', nargs='+', required=True)
A = ap.parse_args()
T = tempfile.mkdtemp(prefix='kcstop_') + '/kc'
shutil.copytree(A.code, T + '/code')
shutil.copyfile(A.driver, T + '/kc_driver.py'); shutil.copyfile(A.serve, T + '/kc_serve.py')
os.makedirs(T + '/work'); out = T + '/out'
env = dict(os.environ, KC_TMP_ROOT=T, KC_OUT_DIR=out, KC_TEST_ROOT=A.root, KC_DEVICE='cpu', KC_WORKERS='2', KC_WORKER_THREADS='2',
           PYTHONPATH=T + '/code', CUDA_VISIBLE_DEVICES='')
t0 = time.time()
p = subprocess.run([sys.executable, '-u', T + '/kc_driver.py', '--stop-at', '%.3f' % (t0 + 25), '--nb-t0', '%.3f' % (t0 - 3600),
                    '--members', *A.members], env=env, cwd=T + '/work', capture_output=True, text=True, timeout=600)
time.sleep(2)
rec = json.load(open(out + '/kc_driver_receipt.json'))
left = []
for d in os.listdir('/proc'):
    if d.isdigit():
        try:
            if (T + '/').encode() in open('/proc/%s/cmdline' % d, 'rb').read().replace(b'\0', b' '):
                left.append(int(d))
        except Exception:
            pass
import pandas as pd
kp = pd.read_csv(out + '/kc_probs.csv')
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


print(p.stdout[-1500:])
chk('driver exit 0, receipt ok', p.returncode == 0 and rec['ok'] is True, (p.returncode, rec.get('error')))
chk('serve_timed_out True, 2 worker groups killed at the stop time', rec.get('serve_timed_out') is True and len(rec.get('killed_worker_groups') or []) == 2,
    rec.get('killed_worker_groups'))
chk('kc_probs.csv: every test study present, none scored (coverage 0.0)', rec['B']['coverage'] == 0.0 and len(kp) == rec['B']['n'] and
    kp['kc_err'].notna().all(), rec['B'].get('errors'))
chk('no process naming the test root left', not left, left)
print('KC_DRIVER_STOP %s %d/%d (%.0f s)' % ('PASS' if all(R) else 'FAIL', sum(R), len(R), time.time() - t0), flush=True)
sys.exit(0 if all(R) else 1)
