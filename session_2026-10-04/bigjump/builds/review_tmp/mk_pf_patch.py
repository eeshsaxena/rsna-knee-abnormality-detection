s = open('code/ttr_preflight.py', encoding='utf-8', newline='').read()
crlf = '\r\n' in s; s = s.replace('\r\n', '\n')
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
rep("ap.add_argument('--cases', default='f1,ttr'); ap.add_argument('--ttr_size', type=int, default=224)", "ap.add_argument('--cases', default='ttr'); ap.add_argument('--ttr_size', type=int, default=224)   # R-6: f1 case (~7 GB) only on a free GPU")
rep("ap.add_argument('--device', default='cuda'); ap.add_argument('--no_pretrained', action='store_true')",
    "ap.add_argument('--device', default='cuda'); ap.add_argument('--no_pretrained', action='store_true')\n"
    "ap.add_argument('--max_gpu_gb', type=float, default=5.0)   # R-6 hard cap: an over-budget case OOMs HERE, not in a trainer\n"
    "ap.add_argument('--min_free_gb', type=float, default=6.5)  # R-6 refuse to start below this (cap + context + margin)")
rep("""    print(f'GPU {torch.cuda.get_device_name()} | free {fr / 2**30:.2f} / {tot / 2**30:.2f} GB before start', flush=True)""",
"""    print(f'GPU {torch.cuda.get_device_name()} | free {fr / 2**30:.2f} / {tot / 2**30:.2f} GB before start', flush=True)
    need = A.min_free_gb + (4.0 if 'f1' in A.cases.split(',') else 0.0)                  # R-6
    assert fr / 2**30 >= need, f'GPU free {fr / 2**30:.1f} GB < {need} GB: refusing (a running trainer could hit CUDA OOM)'
    torch.cuda.set_per_process_memory_fraction(min(1.0, (A.max_gpu_gb + (4.0 if 'f1' in A.cases.split(',') else 0.0)) * 2**30 / tot))""")
open('review_tmp/ttr_preflight_new_lf.py', 'w', encoding='utf-8', newline='').write(s)
open('review_tmp/ttr_preflight_orig_lf.py', 'w', encoding='utf-8', newline='').write(open('code/ttr_preflight.py', encoding='utf-8', newline='').read().replace('\r\n', '\n'))
print('ok crlf', crlf)
