# Stage 1: public Raptor model (checkpoint cell + scored source above, run as its own process so its GPU memory is
# released before stage 2).  A failed Raptor stage stops the notebook: no silent single-model submission.
import os, sys, time, subprocess
NB_T0 = time.time()
rc = subprocess.run([sys.executable, '-u', '/tmp/raptor_run.py'], timeout=2 * 3600).returncode
if rc != 0:
    raise RuntimeError('Raptor stage failed with exit code %d' % rc)
for f in ('raptor_sub.csv', 'raptor_raw.csv'):
    assert os.path.isfile('/kaggle/working/' + f), f + ' missing'
RAPTOR_SECONDS = time.time() - NB_T0
print('RAPTOR_STAGE_DONE seconds %.0f' % RAPTOR_SECONDS, flush=True)
subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader'], check=False)
