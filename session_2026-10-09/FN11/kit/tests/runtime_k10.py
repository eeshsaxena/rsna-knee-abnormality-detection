#!/usr/bin/env python3
"""runtime_k10.py (laptop; MODEL on MEASURED + MODEL inputs; 2026-10-09, Claude; from kfn tests/runtime_kfn.py). Hidden
rerun (2 x T4, ~1,322 studies, 8.6 h wall in the cells, 9 h hard limit) of every K10 kernel.

Inputs (unchanged from the kfn / k9n models): fleet stage = SP/k8/tests/runtime_k8.py (MEASURED T4 microbench, K7i + K5
commit receipts: A+B ms per member per study, TTR224 / TTR288; decode per study); N stage 25 min central / 40 min slow
+ 1 min child start; boot + FN-IDS 3 min (GUESS); parent 6.0 h (runbook, MEASURED on the scored K-kits; PREREG R4 logs
the real value on 10-10).
Composition (asserted from the builds): K10 / K10-solo / K10N-a / FN10 = 5 x TTR224 (set D) + 5 x TTR288 (set E) == the
K9 composition (set C is TTR288 too), so their runtime == K9 / K9N-a / FN exactly. FN10-3 = 5 x TTR224 + 10 x TTR288.
  python tests/runtime_k10.py
"""
import glob, importlib.util, json, os, statistics as stt

T_ = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
SP = os.path.dirname(os.path.dirname(T_))
spec = importlib.util.spec_from_file_location('rt8', f'{SP}/k8/tests/runtime_k8.py')
rt8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt8)
WALL, HARD = 8.6, 9.0
N_C, N_S, N_FIX, BOOT, PARENT = 25.0, 40.0, 1.0, 3.0, 6.0 * 60


def main():
    k7 = sorted(glob.glob(f'{SP}/k7/build_i10/out_*_v1/fleet_receipt.json'))
    k5 = sorted(glob.glob(f'{SP}/k5/build/out_*_v1/fleet_receipt.json'))
    b224, b288, dec = rt8.bench(k7 + k5)
    kits = {'K10 main / solo (P + 5x224 + 5x288)': (5, 5, True, False), 'K10N-a (P + 5x224 + 5x288 + N)': (5, 5, True, True),
            'FN10 (5x224 + 5x288 + N)': (5, 5, False, True), 'FN10-3 (5x224 + 10x288 + N)': (5, 10, False, True)}
    print(f'MEASURED T4 microbench (A+B ms/member/study): TTR224 median {stt.median(b224):.0f} max {max(b224):.0f}; TTR288 '
          f'median {stt.median(b288):.0f} max {max(b288):.0f}; decode s/study/thread median {stt.median(dec):.2f} max {max(dec):.2f}')
    print(f'{"kernel":38s} {"case":8s} {"fleet min":>9s} {"N min":>6s} {"end h":>6s} {"margin 8.6":>10s} {"margin 9":>8s}')
    out = {}
    for name, (n224, n288, parent, n) in kits.items():
        for case, c224, c288, dmin, nmin in (('central', stt.median(b224), stt.median(b288), 15.0, N_C),
                                             ('slow', max(b224), max(b288) * 1.10, rt8.N_PER_T4 * max(dec) / 60.0, N_S)):
            fleet = rt8.N_PER_T4 * (n224 * c224 + n288 * c288) / 60000.0 + dmin + 1.0
            nst = (nmin + N_FIX) if n else 0.0
            end = ((PARENT if parent else BOOT) + fleet + nst) / 60
            out[(name, case)] = end
            print(f'{name:38s} {case:8s} {fleet:9.1f} {nst:6.1f} {end:6.2f} {WALL - end:10.2f} {HARD - end:8.2f}')
    c3, s3 = out[('FN10-3 (5x224 + 10x288 + N)', 'central')], out[('FN10-3 (5x224 + 10x288 + N)', 'slow')]
    cn, sn = out[('K10N-a (P + 5x224 + 5x288 + N)', 'central')], out[('K10N-a (P + 5x224 + 5x288 + N)', 'slow')]
    print(f'RUNTIME_K10: K10N-a {cn:.2f} / {sn:.2f} h (== K9N-a: E replaces C, same 288 px); FN10 == FN; FN10-3 ends ~{c3:.2f} h '
          f'central / ~{s3:.2f} h slow, margin {WALL - s3:.2f} h to the {WALL} h wall in the slow case: the parent-free kit fits '
          f'three sets (a parent + three sets would end ~{(PARENT + (s3 * 60 - BOOT)) / 60:.2f} h slow: refused by the cap).')


if __name__ == '__main__':
    main()
