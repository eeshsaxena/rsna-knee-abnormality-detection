#!/usr/bin/env python3
"""runtime_kfn.py (laptop; MODEL on MEASURED + MODEL inputs; kfn 2026-10-09, Claude): FN (FN-IDS + K9 fleet + N) and
N-only (FN-IDS + N) on the hidden rerun (2 x T4, ~1,322 studies, 8.6 h wall in the cells, 9 h hard limit), and their
3-study commits.

Inputs (same as tests/runtime_k9n.py, which is unchanged in this kit):
  fleet stage: SP/k8/tests/runtime_k8.py (MEASURED T4 microbench from the K7i + K5 commit receipts; K9 = the K8
    composition 5 x TTR224 + 5 x TTR288, A+B): central 58.2 min / slow 76.1 min.
  N stage: 25 min central (20-29) / 40 min slow (x1.6) + 1 min child start (MODEL, PUBLIC949_ASSESSMENT sec 2.2; upper
    bound = the author's 26.1 min submit-to-score turnaround of the same notebook on the same hidden rerun).
  FN-IDS + kernel boot before cell 1: 3 min (GUESS; the cell itself reads two csv files, seconds).
  No parent: K9N's 6.0 h parent term is 0.
Commit (3 public studies): MEASURED fleet stage on the K7i 10-member commits 1.21 / 1.67 min (receipts); N on 3 studies
  ~2 min (model load + cudnn warm-up + 3 studies; the K9N commit has not run yet, GUESS); + boot.
  python tests/runtime_kfn.py
"""
import glob, importlib.util, json, os, statistics as stt

T_ = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
SP = os.path.dirname(os.path.dirname(T_))
spec = importlib.util.spec_from_file_location('rt8', f'{SP}/k8/tests/runtime_k8.py')
rt8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt8)
WALL, HARD = 8.6, 9.0
N_C, N_S, N_FIX, BOOT = 25.0, 40.0, 1.0, 3.0


def main():
    k7 = sorted(glob.glob(f'{SP}/k7/build_i10/out_*_v1/fleet_receipt.json'))
    k5 = sorted(glob.glob(f'{SP}/k5/build/out_*_v1/fleet_receipt.json'))
    b224, b288, dec = rt8.bench(k7 + k5)
    rows = []
    for case, c224, c288, dmin, nmin in (('central', stt.median(b224), stt.median(b288), 15.0, N_C),
                                         ('slow', max(b224), max(b288) * 1.10, rt8.N_PER_T4 * max(dec) / 60.0, N_S)):
        fleet = rt8.N_PER_T4 * (5 * c224 + 5 * c288) / 60000.0 + dmin + 1.0
        nst = nmin + N_FIX
        fn_end = (BOOT + fleet + nst) / 60
        no_end = (BOOT + nst) / 60
        rows.append((case, fleet, nst, fn_end, WALL - fn_end, HARD - fn_end, no_end, WALL - no_end))
    print(f'MEASURED T4 microbench (A+B ms/member/study): TTR224 median {stt.median(b224):.0f} max {max(b224):.0f}; TTR288 '
          f'median {stt.median(b288):.0f} max {max(b288):.0f}; decode s/study/thread median {stt.median(dec):.2f} max {max(dec):.2f}')
    print(f'{"case":8s} {"fleet min":>9s} {"N min":>6s} {"FN end h":>8s} {"FN margin 8.6":>13s} {"FN margin 9":>11s} '
          f'{"N-only end h":>12s} {"N-only margin 8.6":>17s}')
    for r in rows:
        print(f'{r[0]:8s} {r[1]:9.1f} {r[2]:6.1f} {r[3]:8.2f} {r[4]:13.2f} {r[5]:11.2f} {r[6]:12.2f} {r[7]:17.2f}')
    cm = [json.load(open(p)).get('stage_min') for p in k7]
    c, s = rows
    print(f'commit (3 studies): fleet stage MEASURED {min(cm):.2f}-{max(cm):.2f} min (K7i 10-member commits); FN commit '
          f'~{BOOT + max(cm) + 2.0:.0f} min kernel time, N-only ~{BOOT + 2.0:.0f} min (GUESS for N on 3 studies)')
    print(f'RUNTIME_KFN: FN ends ~{c[3]:.2f} h central / ~{s[3]:.2f} h slow (K9N-a: 7.40 / 7.95 h at a 6.0 h parent); margin to '
          f'the {WALL} h wall {c[4]:.2f} / {s[4]:.2f} h. N-only ends ~{c[6]:.2f} / ~{s[6]:.2f} h. Neither kernel can reach '
          f'the N-stage skip (< 20 min left) or kill (8.6 h - 2 min) guard unless a session runs > {WALL / s[3]:.1f}x slower '
          f'than the slow case.')


if __name__ == '__main__':
    main()
