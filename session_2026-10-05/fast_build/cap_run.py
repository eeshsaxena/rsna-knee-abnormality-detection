# cap_run.py -- run an UNEDITED trainer script (runpy, as __main__) under the box GPU rules: free-memory gate, then a
# per-process CUDA memory cap; optional deterministic mode (--det: cudnn deterministic, cudnn.benchmark pinned to False
# even though the trainer sets it True, torch deterministic algorithms warn-only; the runner exports
# CUBLAS_WORKSPACE_CONFIG). Prints the process's peak CUDA memory and the nvidia-smi view at exit.
# Usage: cap_run.py --cap_gib 9.8 --need_mib 11000 [--det] -- /path/fleet_train.py <trainer args>
import os, sys, json, atexit, runpy
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import fbcommon as fb

argv = sys.argv[1:]
k = argv.index('--')
opts, rest = argv[:k], argv[k + 1:]
cap_gib = float(opts[opts.index('--cap_gib') + 1])
need = int(opts[opts.index('--need_mib') + 1])
det = '--det' in opts
fb.gpu_gate(need)
import torch
fb.cap(cap_gib)
if det:
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG'), 'det mode: export CUBLAS_WORKSPACE_CONFIG=:4096:8'
    from torch.backends import ContextProp
    _set = torch._C._set_cudnn_benchmark
    type(sys.modules['torch.backends.cudnn']).benchmark = ContextProp(torch._C._get_cudnn_benchmark, lambda v: _set(False))
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    print('[cap_run] DET mode: cudnn.deterministic True, cudnn.benchmark pinned False', flush=True)
smp = fb.SmiSampler(1.0); smp.start()


@atexit.register
def _peak():
    d = dict(det=det, cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
             peak_alloc_MiB=round(torch.cuda.max_memory_allocated() / 2 ** 20),
             peak_reserved_MiB=round(torch.cuda.max_memory_reserved() / 2 ** 20), **smp.snap())
    print('CAPRUN ' + json.dumps(d), flush=True)


sys.argv = rest
sys.path.insert(0, os.path.dirname(os.path.abspath(rest[0])))
runpy.run_path(rest[0], run_name='__main__')
