#!/usr/bin/env python3
"""k5_ids.py, K8 COPY (= the K7 copy with ~/k8_test default paths; K7 text: K7: studies are picked PER FOLD, not per member: every member of a fold (K7: A, B and C
all have a fold-0 member) gets the SAME A0 studies, chosen from the studies present in ALL of that fold's OOFs, so each
fold-0 member gets its own per-member OOF parity on the same 2 studies and the CPU A0 stays 2 studies long).
(K6 text) k5_ids.py (box, CPU, read-only; == k4_ids.py with ~/k7_test paths): the A0 study list for the K5 box CPU test = PER_FOLD val studies of EVERY member's
own fold (from its OOF csv, so each member gets a train/serve parity read on studies it never trained on), only
studies whose DICOMs exist in the box mirror. Studies listed in --prefer (default ~/k3_test/two.csv, the K3 A0 pair)
come first, so a J1-only run uses exactly the K3 A0 studies.

  ~/miniconda3/envs/kmimic/bin/python ~/k7_test/k5/box/k5_ids.py [--members ~/k7_test/members.json] [--per_fold 2]
        [--out ~/k7_test/ids/a0.csv]
"""
import argparse, json, os
import pandas as pd

H = os.path.expanduser('~')
COMP = f'{H}/kmimic/kaggle/input/rsna-knee-abnormality-detection'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', default=f'{H}/k8_test/members.json')
    ap.add_argument('--per_fold', type=int, default=2)
    ap.add_argument('--prefer', default=f'{H}/k3_test/two.csv')
    ap.add_argument('--out', default=f'{H}/k8_test/ids/a0.csv')
    a = ap.parse_args()
    m = json.load(open(a.members))
    trs = set(pd.read_csv(f'{COMP}/train_series.csv', dtype=str)['StudyInstanceUID'])
    pref = (pd.read_csv(a.prefer, dtype=str)['StudyInstanceUID'].tolist() if os.path.isfile(a.prefer) else [])
    ids = []
    byfold = {}
    for r in m['members']:
        byfold.setdefault(r['fold'], []).append(r)
    for F in sorted(byfold):                                            # K7: one pick per fold, shared by its members
        r = byfold[F][0]
        oofs = [set(pd.read_csv(x['oof'], dtype={'StudyInstanceUID': str})['StudyInstanceUID']) for x in byfold[F]]
        oof = pd.read_csv(r['oof'], dtype={'StudyInstanceUID': str})['StudyInstanceUID'].tolist()
        ok = [s for s in oof if all(s in o for o in oofs) and s in trs and os.path.isdir(f'{COMP}/train_series/{s}')]
        order = [s for s in pref if s in ok] + sorted(s for s in ok if s not in pref)
        pick = [s for s in order if s not in ids][:a.per_fold]
        # VERIFY fix 10-05: the box DICOM mirror holds fold 0 + gold58 only (458 studies; folds 1-4: 0), so a fold 1-4
        # member has no val study here. Its per-member OOF parity is skipped by k4_box_launch.sh (it still runs in the
        # A0 and is covered by the npz mean identity, decode probe and n_fail/n_blank); it used to assert here.
        if len(pick) < a.per_fold:
            print(f'NOTE fold {r["fold"]}: only {len(pick)} val studies with DICOMs on the box (mirror = fold 0 + gold): '
                  f'no per-member OOF parity for this member on the box')
        ids += pick
        print(f'fold {r["fold"]} ({len(byfold[F])} member(s)): {len(ok)} val studies with DICOMs in every member OOF, '
              f'picked {[s[-12:] for s in pick]}')
    assert len(ids) >= a.per_fold, f'only {len(ids)} A0 studies with DICOMs on the box (need >= {a.per_fold})'
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame({'StudyInstanceUID': ids}).to_csv(a.out, index=False)
    print(f'wrote {a.out}: {len(ids)} studies')


if __name__ == '__main__':
    main()
