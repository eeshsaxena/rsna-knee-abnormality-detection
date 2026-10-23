import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_lib import patch
K2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
patch(f'{K2}/box/d1_pick.py', [
("""    o = pd.read_csv(f'{d}/oof_f0_s42_{tag}.csv', index_col='StudyInstanceUID')
""",
"""    try:
        o = pd.read_csv(f'{d}/oof_f0_s42_{tag}.csv', index_col='StudyInstanceUID')
    except Exception as e:
        if done:
            raise
        raise Provisional(f'{name}: oof_f0_s42_{tag}.csv not readable while training/saving ({type(e).__name__})')
"""),
("""    pick = R[win]
    if not pick['g1']:
        other = R['v4' if win == 'mix50' else 'mix50']
        if other['g1']:
            why += f' | {win} FAILS G1 -> ship {other["name"]}'; pick = other
        else:
            print(f'D1 {win} ({why}) | BOTH runs fail G1: NO K2 (ENDGAME G1: EV2S f0 is the family switch)')
            return 2
    not_final = [k for k, r in R.items() if not r['final']]
""",
"""    pick = R[win]
    not_final = [k for k, r in R.items() if not r['final']]
    if not pick['g1']:
        other = R['v4' if win == 'mix50' else 'mix50']
        if other['g1']:
            why += f' | {win} FAILS G1 -> ship {other["name"]}'; pick = other
        else:
            print(f'D1 {win} ({why}) | BOTH runs fail G1: NO K2 (ENDGAME G1: EV2S f0 is the family switch)'
                  + (f' | PROVISIONAL (exit 3): {not_final} not final' if not_final else ''))
            return 3 if not_final else 2
"""),
])
