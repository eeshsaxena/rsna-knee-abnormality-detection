# kc_test_oom_atexit.py -- test of the KC-5 atexit verdict path: runs a KC trainer with DataLoader construction forced to
# raise an error of the chosen KIND. Expect (trainer >= f1d266cf, KC-R4 = queue_runner.sh is_oom's regex):
#   --grad_ckpt 0 + an OOM kind (torch | cudnn | cublas) -> NO verdict written (the queue runner re-runs with grad_ckpt 1)
#   --grad_ckpt 1 (any kind), or kind 'other' (any gc)    -> 'SMOKE_TEST_ONLY FAIL kc_f0 crashed ...'
# The pre-KC-R4 trainer (5a22d1eb) wrote a FAIL for gc0 + cudnn / cublas, while the runner retried those (KC-R4 bug).
#   python kc_test_oom_atexit.py <gc 0|1> <out> [kind: torch|cudnn|cublas|other] [trainer path]
# Run with PYTHONPATH=~/kc/code:~/fast_build/code2 (kc_common first), cores 28-39, GPU hidden.
import sys, os, runpy, torch
gc = sys.argv[1]; out = sys.argv[2]
kind = sys.argv[3] if len(sys.argv) > 3 else 'torch'
H = os.path.expanduser('~')
trainer = sys.argv[4] if len(sys.argv) > 4 else os.path.join(H, 'kc', 'code', 'fleet_train_kc.py')
ERR = {'torch': lambda: torch.cuda.OutOfMemoryError('CUDA out of memory. Tried to allocate 2.00 GiB (TEST)'),
       'cudnn': lambda: RuntimeError('cuDNN error: CUDNN_STATUS_NOT_ENOUGH_MEMORY (TEST)'),
       'cublas': lambda: RuntimeError('CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)` (TEST)'),
       'other': lambda: RuntimeError('CUDA error: an illegal memory access was encountered (TEST)')}[kind]


def boom(*a, **k):
    raise ERR()


torch.utils.data.DataLoader.__init__ = boom
sys.argv = ['fleet_train_kc.py', '--device', 'cpu', '--max_steps', '1', '--threads', '4', '--workers', '0', '--model', 'ttr', '--ttr_size', '256',
            '--backbone', 'convnext_small.fb_in22k_ft_in1k', '--mirror', '0', '--seed', '2000', '--fold', '0', '--min_cover', '0',
            '--store', f'{H}/kc_store/test458', '--labels', f'{H}/r7t_labels/real/labels_r7t_b.csv', '--folds', f'{H}/kc/smoke/folds_smoke.csv',
            '--exclude', f'{H}/rsna_train/gold74_eval_set.csv', '--gate_out', f'{out}.verdict', '--out', out, '--grad_ckpt', gc]
runpy.run_path(trainer, run_name='__main__')
