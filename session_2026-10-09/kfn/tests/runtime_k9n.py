#!/usr/bin/env python3
"""runtime_k9n.py (laptop; MODEL on MEASURED + MODEL inputs; K9N 2026-10-08, Claude): does K9N (K9 = 5 x TTR288 +
5 x TTR224 A+B, then the N stage) fit the hidden rerun (2 x T4, 8.6 h wall, 9 h hard limit)?

Fleet stage: SP/k8/tests/runtime_k8.py, unchanged (MEASURED T4 microbench from the K7i + K5 commit receipts; K9 has the
  K8 composition). Parent ~6.0 h (K5/K7 runbooks), hidden ~1,322 studies.
N stage (MODEL, PUBLIC949_ASSESSMENT sec 2.2): ~25 min central (20-29), 40 min slow (x1.6; the assessment's 35-40):
  upper bound from the author's own 26.1 min submit-to-score turnaround (incl. queue) of the same notebook on the same
  2 x T4 hidden rerun; decode <= ~7 min (the 7.2 min efficiency run uses the same reader). + ~1 min child start / model
  load / weights sha256, blend seconds.
Guard (cell 32): the N child is killed at kernel start + 8.6 h - 2 min (RESERVE_MIN); the stage is skipped when < 20 min
  remain (MIN_START_MIN). Kill or skip = strict: no submission.csv (the slot errors, it never scores P+F or P).
  python tests/runtime_k9n.py
"""
import importlib.util, io, os, contextlib

T_ = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
SP = os.path.dirname(os.path.dirname(T_))
spec = importlib.util.spec_from_file_location('rt8', f'{SP}/k8/tests/runtime_k8.py')
rt8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt8)
WALL, HARD, RESERVE_H, MIN_START_H = 8.6, 9.0, 2 / 60, 20 / 60
N_C, N_S, N_FIX = 25.0, 40.0, 1.0


def main():
    import glob, statistics as stt
    k7 = sorted(glob.glob(f'{SP}/k7/build_i10/out_*_v1/fleet_receipt.json'))
    k5 = sorted(glob.glob(f'{SP}/k5/build/out_*_v1/fleet_receipt.json'))
    b224, b288, dec = rt8.bench(k7 + k5)
    rows = []
    for case, c224, c288, dmin, nmin in (('central', stt.median(b224), stt.median(b288), 15.0, N_C),
                                         ('slow', max(b224), max(b288) * 1.10, rt8.N_PER_T4 * max(dec) / 60.0, N_S)):
        ms = 5 * c224 + 5 * c288
        fleet = rt8.N_PER_T4 * ms / 60000.0 + dmin + 1.0
        nst = nmin + N_FIX
        k9_end = rt8.PARENT_H + fleet / 60
        end = k9_end + nst / 60
        rows.append((case, fleet, nst, k9_end, end, WALL - end, HARD - end, WALL - RESERVE_H - (fleet + nst) / 60))
    print(f'MEASURED T4 microbench (A+B ms/member/study): TTR224 median {stt.median(b224):.0f} max {max(b224):.0f}; TTR288 '
          f'median {stt.median(b288):.0f} max {max(b288):.0f}; decode s/study/thread median {stt.median(dec):.2f} max {max(dec):.2f}')
    print(f'{"case":8s} {"fleet min":>9s} {"N min":>6s} {"K9 end h":>8s} {"K9N end h":>9s} {"margin 8.6":>10s} '
          f'{"margin 9.0":>10s} {"max parent h":>12s}')
    for r in rows:
        print(f'{r[0]:8s} {r[1]:9.1f} {r[2]:6.1f} {r[3]:8.2f} {r[4]:9.2f} {r[5]:10.2f} {r[6]:10.2f} {r[7]:12.2f}')
    c, s = rows
    print(f'RUNTIME_K9N: K9N ends ~{c[4]:.2f} h central / ~{s[4]:.2f} h slow at a {rt8.PARENT_H} h parent; margin to the '
          f'{WALL} h wall {c[5]:.2f} h / {s[5]:.2f} h (K9 alone {WALL - c[3]:.2f} / {WALL - s[3]:.2f}); to the 9 h limit '
          f'{c[6]:.2f} / {s[6]:.2f} h. The N stage is killed (strict: no submission) only if the parent runs past ~{s[7]:.2f} h '
          f'in a slow session (~{c[7]:.2f} h central); it is skipped outright when < 20 min remain after the fleet.')


if __name__ == '__main__':
    main()
