"""compact_train.py -> compact_train_cpu.py for a CPU-only pipeline test (device and autocast device only; maths unchanged)."""
import sys
src = open(sys.argv[1], encoding="utf-8").read()
reps = [('torch.device("cpu") if a.check_targets else torch.device("cuda")', 'torch.device("cpu")'),
        ('torch.autocast("cuda", dtype=torch.bfloat16)', 'torch.autocast("cpu", dtype=torch.bfloat16)'),
        ('torch.cuda.get_rng_state_all()', '[]')]
for a_, b_ in reps:
    n = src.count(a_)
    assert n >= 1, a_
    src = src.replace(a_, b_)
    print("replaced %d x %s" % (n, a_))
open(sys.argv[2], "w", encoding="utf-8").write(src)
