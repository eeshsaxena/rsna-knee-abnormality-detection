# review-only: patch COPIES of the smoke harness under k2/_review (never the shipped k2 files)
import os
R = os.path.dirname(os.path.abspath(__file__))
p = f'{R}/tests/run_scenario.py'; s = open(p, encoding='utf-8').read()
s = s.replace("SP = os.path.dirname(K2)", "SP = os.path.dirname(os.path.dirname(K2))")
a = "    'comb_normal':      dict(nb='combined', crop=True, expect_crop='ok'),"
assert a in s
s = s.replace(a, "    'fleet_blank':      dict(fds='stub', stub='blank', expect_fleet='fail'),\n"
                 "    'fleet_views2':     dict(fds='stub', stub='views2', expect_fleet='fail'),\n" + a)
b = "('fleet_nods', 'not attached'), ('fleet_writefail', 'No space left')):"
assert b in s
s = s.replace(b, "('fleet_nods', 'not attached'), ('fleet_writefail', 'No space left'),\n"
                 "               ('fleet_blank', 'failed studies'), ('fleet_views2', 'views/study')):")
open(p, 'w', encoding='utf-8').write(s)
p = f'{R}/tests/_smoke/ds/fleet_stub/fleet_worker.py'; s = open(p, encoding='utf-8').read()
c = "P[keep:] = np.nan\n"
assert c in s and "NV=np.full(len(ids), 4)," in s
s = s.replace(c, c + "NV = np.full(len(ids), 4)\nif mode == 'blank' and len(ids):\n    P[:, 0] = np.nan; NV[0] = 0\n"
                     "if mode == 'views2':\n    NV[:] = 2\n")
s = s.replace("NV=np.full(len(ids), 4),", "NV=NV,")
open(p, 'w', encoding='utf-8').write(s)
print('patched copies under', R)
