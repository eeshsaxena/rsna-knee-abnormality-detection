# ops_selftest_ck.py (checker, 10-21): runs final/ops_tick_a23.py's offline selftest() WITHOUT touching final/: the module is loaded
# from its real path (so R, GO, PREREG, KDIR are the live ones), no bytecode is written, and only its log path F is redirected here.
# Also re-checks the GO-parse gotchas from mech.md. No Kaggle import (selftest never imports kaggle).
import sys, os, importlib.util
sys.dont_write_bytecode = True
P = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/final/ops_tick_a23.py'
spec = importlib.util.spec_from_file_location('ops_ck', P); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.F = os.path.dirname(os.path.abspath(__file__))
m.selftest()
for g in ['GO c1 c2 MAX=3 Eesh 10-21 23:00: "ok"', 'GO c1 c1 c2', 'GO c1 c2 MAX=0', 'GO c1 c2 ok', 'GO c1 c2 2300', 'GO c1 c2 by eesh',
          'GO c1 c2 MAX=x', 'GO c1 C2 Eesh', 'GO c1 c2 c3 Eesh 10-21 23:50', 'GO c1 c2, Eesh']:
    print('PARSE %-40s -> %s' % (g, m.parse_go(g)))
print('GO path', m.GO, 'exists', os.path.exists(m.GO), '| prereg_ok', m.prereg_ok(), '| static', m.static_ok())
