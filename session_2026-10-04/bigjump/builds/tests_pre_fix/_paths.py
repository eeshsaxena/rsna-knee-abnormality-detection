# Shared paths for the bigjump build tests (harness only; nothing here ships). Read-only inputs elsewhere in SP.
import os, sys, shutil
T = os.path.dirname(os.path.abspath(__file__)).replace(chr(92), '/')
B = os.path.dirname(T)                                # bigjump/builds
SP = os.path.dirname(os.path.dirname(B))              # scratchpad
CODE = f'{B}/code'                                    # the bigjump code under test
ORIG = f'{SP}/round3/fleet'                           # == box ~/fleet (family 1, running; md5-identical)
F2 = f'{SP}/fleet2/code'                              # == box ~/fleet2_code (md5-identical)
K2C = f'{SP}/k2/code'                                 # the K2 dataset code (K2 worker + pack tool)
K2 = f'{SP}/k2'
DRY = f'{SP}/round3/fleet_dry'                        # round3 synthetic assets (read only): 64 px store, fake comp
SHIM = f'{DRY}/shim'                                  # cv2 stand-in (laptop has no OpenCV)
OUT = f'{T}/out'
PY = sys.executable
os.makedirs(OUT, exist_ok=True)
ALL = ['fleet_common.py', 'fleet_model.py', 'fleet_model_f2.py', 'fleet_model_ttr.py', 'fleet_aug.py', 'fleet_train.py',
       'fleet_worker.py', 'pack_fleet_ds.py', 'tta_bench.py', 'gt2_read.py']


def stage64(src, dst, files):
    """Copy `files` from src to dst with ONLY fleet_common's `STORE = 320` -> `STORE = 64` (round3 dry-run rule)."""
    shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
    for f in files:
        s = open(f'{src}/{f}', encoding='utf-8', newline='').read()
        if f == 'fleet_common.py':
            assert s.count('STORE = 320 ') == 1
            s = s.replace('STORE = 320 ', 'STORE = 64 ')
        open(f'{dst}/{f}', 'w', encoding='utf-8', newline='').write(s)
    return dst


def relabel_fp16(src, dst):
    """Dry ckpts are trained on CPU (amp fp32); the worker admits fp16-trained members only. Harness-only relabel."""
    import torch
    ck = torch.load(src, weights_only=False)
    assert ck['args']['amp'] == 'fp32'
    ck['args']['amp'] = 'fp16'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save(ck, dst)
    return dst
