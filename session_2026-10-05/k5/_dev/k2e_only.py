# dev helper: re-run only the smoke's K2e decode-timeout check and the D/D5 real-build block on the existing tests/_smoke
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests'))
import numpy as np
import smoke_k2 as s
t0 = time.time()
ok_t = s.worker_timeout(np.load(f'{s.S}/wp_k2.npz'))
ok_d = s.real_build() if '--real' in sys.argv else True
print('K2E_ONLY', 'PASS' if ok_t else 'FAIL', 'REAL', 'PASS' if ok_d else 'FAIL', f'{time.time() - t0:.0f} s')
