#!/usr/bin/env python
# r2_capped.py (jump2 r2, 2026-10-05): run a training script unchanged, but with a CUDA memory cap first
# (torch.cuda.set_per_process_memory_fraction, env R2_MEMFRAC, default 0.75 = 12 GiB of the 16 GiB V100), so a
# co-resident 3 GB trainer can never be pushed out of memory by this process.
# usage: python r2_capped.py <script.py> <script args...>   (run from the script's own directory)
import os, sys, runpy
frac = float(os.environ.get('R2_MEMFRAC', '0.75'))
script = os.path.abspath(sys.argv[1])
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(script))
import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    print(f'[r2_capped] cuda memory fraction {frac} = {frac * tot:.1f} of {tot:.1f} GiB | script {script}', flush=True)
else:
    print('[r2_capped] no CUDA visible: cap not applied', flush=True)
runpy.run_path(script, run_name='__main__')
