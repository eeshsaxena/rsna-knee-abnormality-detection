# wait_fetch.py KERNEL OUTDIR [MAX_MIN]: poll a kernel's status every 60 s until COMPLETE / ERROR (read-only), then download
# its output (kaggle kernels output) into OUTDIR. Prints one status line per poll.
import sys, time, subprocess
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
k, out = sys.argv[1], sys.argv[2]
mx = float(sys.argv[3]) if len(sys.argv) > 3 else 40
t0 = time.time()
while True:
    try:
        st = str(api.kernels_status(k).status)
    except Exception as e:
        st = 'ERR %r' % e
    print(time.strftime('%H:%M:%S'), k, st, flush=True)
    if 'COMPLETE' in st or 'ERROR' in st or 'CANCEL' in st:
        break
    if time.time() - t0 > mx * 60:
        print('WAIT_TIMEOUT', flush=True); sys.exit(2)
    time.sleep(60)
r = subprocess.run(['kaggle', 'kernels', 'output', k, '-p', out], capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-2000:], flush=True)
print('FETCHED', k, out, st, flush=True)
