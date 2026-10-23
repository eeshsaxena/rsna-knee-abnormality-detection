# run_trainer.py <code_dir> [fleet_train.py args...] -- run <code_dir>/fleet_train.py in THIS process with the round3
# dry-run harness patches only: cv2 -> PIL shim (laptop has no OpenCV), timm pretrained -> False (no HF download),
# 4 CPU threads (== fleet2/tests/run_trainer.py).
import os, sys, runpy
from _paths import SHIM
code = sys.argv[1]
sys.path.insert(0, SHIM)
import torch, timm
torch.set_num_threads(int(os.environ.get('RT_THREADS', '4')))
_cm = timm.create_model


def _create_model(*a, **k):
    k['pretrained'] = False
    return _cm(*a, **k)


timm.create_model = _create_model
sys.argv = [f'{code}/fleet_train.py'] + sys.argv[2:]
runpy.run_path(f'{code}/fleet_train.py', run_name='__main__')
