import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_lib import patch
K2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
patch(f'{K2}/cells/k2_fleet_cell.py', [
("""#   P3 one wall: kernel start = min(/proc start of this process, parent T0); no worker starts a new chunk after
#      kernel start + WALL_H (8.6 h); workers are killed GRACE_S after it; the stage starts only if >= MIN_START_MIN
#      minutes remain (round3: 0.35 h margin + 30 min, which threw away a feasible fleet at P 8.05-8.15 h).
""",
"""#   P3 one wall: kernel start = min(/proc start of this process, parent T0); no worker starts a new chunk after
#      kernel start + WALL_H (8.6 h); workers are killed GRACE_S after it; the stage starts only if >= MIN_START_MIN
#      minutes remain (round3: 0.35 h margin + 30 min, which threw away a feasible fleet at P 8.05-8.15 h). The wall
#      is the only time guard (P6: never cut a fleet that could still finish before it). Only if NO kernel clock is
#      readable (no /proc start AND no parent T0; not expected, the pinned parent sets T0 in cell 5) the stage gets
#      NOCLOCK_MIN minutes from its own start: a parent at its 8.0 h TIME_BUDGET + 40 min + GRACE_S stays < 9 h.
"""),
("""K2 = {'W_MAIN': 0.30, 'W_VARIANTS': [0.20, 0.40], 'WALL_H': 8.6, 'MIN_START_MIN': 20.0, 'BUDGET_MIN': 150.0,
""",
"""K2 = {'W_MAIN': 0.30, 'W_VARIANTS': [0.20, 0.40], 'WALL_H': 8.6, 'MIN_START_MIN': 20.0, 'NOCLOCK_MIN': 40.0,
"""),
("""    deadline = t_cell + float(K2['BUDGET_MIN']) * 60                   # backstop only (no clock information at all)
    if k0 is not None:
        deadline = min(deadline, k0 + float(K2['WALL_H']) * 3600)
    left = deadline - now
""",
"""    if k0 is not None:
        deadline = k0 + float(K2['WALL_H']) * 3600                      # the wall is the guard (P6)
    else:
        deadline = t_cell + float(K2['NOCLOCK_MIN']) * 60               # no clock at all: short backstop (P3)
    left = deadline - now
"""),
])
