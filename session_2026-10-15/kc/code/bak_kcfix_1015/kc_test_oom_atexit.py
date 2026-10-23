# kc_test_oom_atexit.py -- test of the KC-5 atexit verdict path: runs fleet_train_kc.py with DataLoader construction forced to
# raise a CUDA-OOM-like error. Expect: --grad_ckpt 0 -> NO verdict written; --grad_ckpt 1 -> 'SMOKE_TEST_ONLY FAIL ... OutOfMemoryError'.
import sys, os, runpy, torch
gc = sys.argv[1]; out = sys.argv[2]
def boom(*a, **k):
    raise torch.cuda.OutOfMemoryError('CUDA out of memory. Tried to allocate 2.00 GiB (TEST)')
torch.utils.data.DataLoader.__init__ = boom
H = os.path.expanduser('~')
sys.argv = ['fleet_train_kc.py', '--device', 'cpu', '--max_steps', '1', '--threads', '4', '--workers', '0', '--model', 'ttr', '--ttr_size', '256',
            '--backbone', 'convnext_small.fb_in22k_ft_in1k', '--mirror', '0', '--seed', '2000', '--fold', '0', '--min_cover', '0',
            '--store', f'{H}/kc_store/test458', '--labels', f'{H}/r7t_labels/real/labels_r7t_b.csv', '--folds', f'{H}/kc/smoke/folds_smoke.csv',
            '--exclude', f'{H}/rsna_train/gold74_eval_set.csv', '--gate_out', f'{out}.verdict', '--out', out, '--grad_ckpt', gc]
runpy.run_path(os.path.join(H, 'kc', 'code', 'fleet_train_kc.py'), run_name='__main__')
