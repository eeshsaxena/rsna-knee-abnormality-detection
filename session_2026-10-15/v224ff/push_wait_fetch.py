# push_wait_fetch.py KERNEL_DIR SLUG OUTDIR [MAX_WAIT_MIN]: push a private kernel (commit run) as soon as Kaggle accepts it (the
# API refuses while 2 GPU sessions of this account run: "Maximum batch GPU session count of 2 reached"), then poll its status every
# 60 s until COMPLETE / ERROR and download its output into OUTDIR. Never submits anything.
import sys, time, subprocess
from kaggle.api.kaggle_api_extended import KaggleApi

kdir, slug, out = sys.argv[1], sys.argv[2], sys.argv[3]
mx = float(sys.argv[4]) if len(sys.argv) > 4 else 240
t0 = time.time()
api = KaggleApi(); api.authenticate()
while True:
    r = subprocess.run(['kaggle', 'kernels', 'push', '-p', kdir], capture_output=True, text=True)
    msg = (r.stdout + r.stderr).strip()
    print(time.strftime('%H:%M:%S'), 'PUSH', msg[-300:], flush=True)
    if 'successfully pushed' in msg:
        break
    if 'GPU session count' not in msg:
        print('PUSH_FAILED', flush=True); sys.exit(3)
    if time.time() - t0 > mx * 60:
        print('PUSH_TIMEOUT', flush=True); sys.exit(2)
    time.sleep(60)
time.sleep(30)
while True:
    try:
        st = str(api.kernels_status(slug).status)
    except Exception as e:
        st = 'ERR %r' % e
    print(time.strftime('%H:%M:%S'), slug, st, flush=True)
    if 'COMPLETE' in st or 'ERROR' in st or 'CANCEL' in st:
        break
    if time.time() - t0 > mx * 60:
        print('WAIT_TIMEOUT', flush=True); sys.exit(2)
    time.sleep(60)
r = subprocess.run(['kaggle', 'kernels', 'output', slug, '-p', out], capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-2000:], flush=True)
print('FETCHED', slug, out, st, flush=True)
