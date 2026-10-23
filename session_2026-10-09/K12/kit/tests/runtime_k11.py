#!/usr/bin/env python3
"""runtime_k11.py (laptop; MODEL on MEASURED inputs; 2026-10-09, Claude; = k10 tests/runtime_k10.py + the K11 kits).
Hidden rerun of the parent-free FN kernels (2 x T4, ~1,322 studies, 8.6 h wall in the cells, 9 h hard limit).
Inputs exactly as runtime_k10.py: fleet stage = SP/k8/tests/runtime_k8.py bench (MEASURED T4 microbench of the K7i + K5
commit receipts: A+B ms per member per study per GPU; decode per study); N stage 25 / 40 min + 1 min child start; boot +
FN-IDS 3 min (GUESS). The per-member cost is additive (FN10-3 = the first 15-member T4 fleet; same assumption).
PREREG_FN11 sec 5 rule: slow-case end <= 0.65 x 9 h = 5.85 h (>= 35 % of the 9 h limit unused).
Cross-checks (MEASURED): the scored FN (K9 fleet 5x224 + 5x288 + N, swapiy v1, 10-09) went submit 11:22 -> COMPLETE by
12:34 IST (<= 1.2 h incl. queue + scoring) vs MODEL 1.45 h central; its 3-study commit MICROBENCH (kfn/build/
out_fn_swapiy_v1) TTR224 283-289, TTR288 474-488 ms/member/study per GPU.
  python tests/runtime_k11.py
"""
import glob, importlib.util, os, statistics as stt

T_ = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
SP = os.path.dirname(os.path.dirname(T_))
spec = importlib.util.spec_from_file_location('rt8', f'{SP}/k8/tests/runtime_k8.py')
rt8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt8)
WALL, HARD, RULE = 8.6, 9.0, 0.65 * 9.0
N_C, N_S, N_FIX, BOOT = 25.0, 40.0, 1.0, 3.0


def main():
    k7 = sorted(glob.glob(f'{SP}/k7/build_i10/out_*_v1/fleet_receipt.json'))
    k5 = sorted(glob.glob(f'{SP}/k5/build/out_*_v1/fleet_receipt.json'))
    b224, b288, dec = rt8.bench(k7 + k5)
    kits = {'FN (K9: 5x224 + 5x288 + N) [scored]': (5, 5), 'FN10-3 (5x224 + 10x288 + N)': (5, 10),
            'FN11, E2 4/5 (5x224 + 14x288 + N)': (5, 14), 'FN11 (5x224 + 15x288 + N)': (5, 15),
            'FN11-D2 (10x224 + 15x288 + N)': (10, 15)}
    print(f'MEASURED T4 microbench (A+B ms/member/study): TTR224 median {stt.median(b224):.0f} max {max(b224):.0f}; TTR288 '
          f'median {stt.median(b288):.0f} max {max(b288):.0f}; decode s/study/thread median {stt.median(dec):.2f} max {max(dec):.2f}; '
          f'studies per T4 {rt8.N_PER_T4}')
    print(f'{"kernel":38s} {"case":8s} {"fleet min":>9s} {"N min":>6s} {"end h":>6s} {"unused of 9 h":>13s} {"rule <= 5.85 h":>14s}')
    out = {}
    for name, (n224, n288) in kits.items():
        for case, c224, c288, dmin, nmin in (('central', stt.median(b224), stt.median(b288), 15.0, N_C),
                                             ('slow', max(b224), max(b288) * 1.10, rt8.N_PER_T4 * max(dec) / 60.0, N_S)):
            fleet = rt8.N_PER_T4 * (n224 * c224 + n288 * c288) / 60000.0 + dmin + 1.0
            end = (BOOT + fleet + nmin + N_FIX) / 60
            out[(name, case)] = end
            print(f'{name:38s} {case:8s} {fleet:9.1f} {nmin + N_FIX:6.1f} {end:6.2f} {100 * (1 - end / HARD):12.0f}% '
                  f'{"OK" if end <= RULE else "OVER":>14s}')
    s11 = out[('FN11 (5x224 + 15x288 + N)', 'slow')]
    c11 = out[('FN11 (5x224 + 15x288 + N)', 'central')]
    sd2 = out[('FN11-D2 (10x224 + 15x288 + N)', 'slow')]
    print(f'RUNTIME_K11: FN11 ~{c11:.2f} h central / ~{s11:.2f} h slow ({100 * (1 - s11 / HARD):.0f} % of the 9 h limit unused; '
          f'rule <= {RULE:.2f} h: {"PASS" if s11 <= RULE else "FAIL"}); FN11-D2 slow {sd2:.2f} h '
          f'({"PASS" if sd2 <= RULE else "FAIL"}); 5 more TTR288 members cost '
          f'+{(s11 - out[("FN10-3 (5x224 + 10x288 + N)", "slow")]) * 60:.0f} min slow.')


if __name__ == '__main__':
    main()
