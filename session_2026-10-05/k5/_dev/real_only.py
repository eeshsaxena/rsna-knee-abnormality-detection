# dev helper: re-run only the smoke's D/D5 real-build block (build_k2.py on the real M1 + M2 from k3/ckpt, + --tta)
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests'))
import smoke_k2 as s
t0 = time.time()
ok = s.real_build()
print('REAL_ONLY', 'PASS' if ok else 'FAIL', f'{time.time() - t0:.0f} s')
open(f'{s.T}/smoke_rerun_D.txt', 'w', encoding='utf-8').write('\n'.join(s.LOG) + '\n')
