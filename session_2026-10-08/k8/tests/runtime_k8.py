#!/usr/bin/env python3
"""runtime_k8.py (laptop; MODEL on MEASURED inputs; 2026-10-07, Claude): does K8 (5 x TTR288 + 5 x TTR224, all A+B) fit the
hidden rerun (2 x T4, the cell's 8.6 h wall)?

MEASURED inputs (Kaggle T4 commit microbench, ms/member/study over BOTH served passes, 5 rounds, read from the receipts):
  K7i main + solo commits (SP/k7/build_i10/out_{main,solo}_v1/fleet_receipt.json): 9 TTR224 + 1 TTR288 member, 2 GPUs each
  K5 main / solo / w50 commits (SP/k5/build/out_*_v1): 5 TTR224 members
  decode s/study/thread from the same receipts. Hidden set: ~1322 studies -> ~661 per T4 (K7_RUNBOOK sec 4).
  Parent ~6.0 h (K5/K7 runbooks); the stage starts only with >= 20 min to the wall (MIN_START_MIN), strict/solo never
  write a silent parent score (a stage that hits the wall deletes submission.csv: the slot errors).
MODEL: per T4, GPU min = studies x sum(ms per member) / 60000; stage = GPU + decode (studies x decode s / 60, central:
  overlapped -> 15 min floor, slow: serial 1.0x) + 1 min pass-B build; kernel end = parent + stage.
  python tests/runtime_k8.py
"""
import glob, json, os, statistics as stt

T_ = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
SP = os.path.dirname(os.path.dirname(T_))
N_PER_T4 = 661
PARENT_H = 6.0
WALL_H = 8.6


def bench(paths):
    b224, b288, dec = [], [], []
    for p in paths:
        r = json.load(open(p))
        for g in r.get('bench') or []:
            for k, v in g.items():
                (b288 if '_ttr288-' in k else b224).append(float(v))
        if r.get('decode_s_per_study_per_thread'):
            dec.append(float(r['decode_s_per_study_per_thread']))
    return b224, b288, dec


def main():
    k7 = sorted(glob.glob(f'{SP}/k7/build_i10/out_*_v1/fleet_receipt.json'))
    k5 = sorted(glob.glob(f'{SP}/k5/build/out_*_v1/fleet_receipt.json'))
    b224, b288, dec = bench(k7 + k5)
    print(f'MEASURED T4 microbench (A+B, ms/member/study): TTR224 n {len(b224)} median {stt.median(b224):.0f} '
          f'max {max(b224):.0f} | TTR288 n {len(b288)} median {stt.median(b288):.0f} max {max(b288):.0f} '
          f'(ratio 288/224 median {stt.median(b288) / stt.median(b224):.2f}) | decode s/study/thread median '
          f'{stt.median(dec):.2f} max {max(dec):.2f}  [{len(k7)} K7i + {len(k5)} K5 commit receipts]')
    rows = []
    for name, n224, n288 in (('K5 (5 x 224, scored .946)', 5, 0), ('K7i (9 x 224 + 1 x 288, scored)', 9, 1),
                             ('K8 (5 x 224 + 5 x 288)', 5, 5), ('5 x 288 + 5 x 288 (later kit)', 0, 10),
                             ('15 members (10 x 224 + 5 x 288)', 10, 5)):
        for case, c224, c288, dmin in (('central', stt.median(b224), stt.median(b288), 15.0),
                                       ('slow', max(b224), max(b288) * 1.10, N_PER_T4 * max(dec) / 60.0)):
            ms = n224 * c224 + n288 * c288
            gpu = N_PER_T4 * ms / 60000.0
            stage = gpu + dmin + 1.0
            end = PARENT_H + stage / 60.0
            rows.append((name, case, ms / 1000.0, gpu, stage, end, WALL_H - end, WALL_H - stage / 60.0))
    print(f'{"kit":36s} {"case":8s} {"s/study/T4":>10s} {"GPU min":>8s} {"stage min":>9s} {"end h":>6s} '
          f'{"margin h":>8s} {"max parent h":>12s}')
    for r in rows:
        print(f'{r[0]:36s} {r[1]:8s} {r[2]:10.2f} {r[3]:8.1f} {r[4]:9.1f} {r[5]:6.2f} {r[6]:8.2f} {r[7]:12.2f}')
    k8s = [r for r in rows if r[0].startswith('K8') and r[1] == 'slow'][0]
    k8c = [r for r in rows if r[0].startswith('K8') and r[1] == 'central'][0]
    ok = k8s[6] >= 0.75
    print(f'RUNTIME_K8 {"FITS" if ok else "TIGHT"}: K8 stage ~{k8c[4]:.0f} min central / ~{k8s[4]:.0f} min slow; kernel end '
          f'~{k8c[5]:.2f} h / ~{k8s[5]:.2f} h at a {PARENT_H} h parent; margin to the {WALL_H} h wall >= {k8s[6]:.2f} h; the '
          f'parent may run to ~{k8s[7]:.2f} h before the slow-case stage would hit the wall')
    return 0


if __name__ == '__main__':
    main()
