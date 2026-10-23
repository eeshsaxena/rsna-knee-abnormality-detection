# Stage 2b [s5]: nartaa's 224-pixel, 0.8-crop CoAtNet (public, 0.945 alone; same dataset, second checkpoint).
# Runs only if the 9 h clock allows; any failure or timeout leaves the blend exactly as the 0.954 S3 recipe.
import os, sys, time, subprocess
R224_OK = False
_left = 9 * 3600 - (time.time() - NB_T0) - 40 * 60          # keep 40 min for the blend and the margin
if _left < 30 * 60:
    print(f"R224_SKIPPED only {_left / 60:.0f} min left", flush=True)
else:
    _t = time.time()
    try:
        _rc = subprocess.run([sys.executable, '-u', '/tmp/raptor224_run.py'], timeout=min(_left, 2.5 * 3600)).returncode
        R224_OK = _rc == 0 and os.path.isfile('/kaggle/working/raptor224_sub.csv')
        print(f"R224_STAGE rc={_rc} ok={R224_OK} seconds {time.time() - _t:.0f}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"R224_TIMEOUT after {time.time() - _t:.0f}s - blending without it", flush=True)