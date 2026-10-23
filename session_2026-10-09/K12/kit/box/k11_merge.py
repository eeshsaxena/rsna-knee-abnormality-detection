#!/usr/bin/env python3
"""k11_merge.py (box, CPU, read-only except DEST/members.json; 2026-10-09, Claude; from k10_merge.py): ONE members.json
for every K11 (FN11) build = the FINAL K10 members C + D + E (~/k10_test/members.json, read only, every pick asserted ==
the shipped k10w bytes) + set E2 (the K11 picker output DEST/members_E2.json: f{F}_s1046_ttr288frac75, Modal) [+ D2].

  python k11_merge.py --dest ~/k11_test [--base ~/k10_test/members.json] [--allow_interim] [--e2_ship 4of5] [--with_d2]

PRE-REGISTERED (PREREG_FN11.md sec 2 / 4, written 10-09 ~16:50 IST before any E2 fold was final):
  E2 fold F ACCEPTED iff every K10 set-E picker check + K11-MREC (Modal receipt) passed (picker) AND
     cv >= E_F - .005 (E_F = set E's picked raw cv of the same fold, from the base members.json): no override.
  A fold under its bar or with a failed check is EXCLUDED; E2 is gated only with >= 4 accepted folds.
  E2 5/5 -> FINAL. E2 = folds 0-3 only -> INTERIM (do_not_submit; TEST builds) unless --e2_ship 4of5 (the operator's call
     after the 10-10 12:00 IST cutoff of sec 4, or after f4 failed): then FINAL with e2_ship '4of5'.
  D2 (--with_d2, optional, sec 6): bar D2_F >= D_F - .005, 5/5 only, Kaggle f0 / Vast f1-f4 receipts (picker records).
Exit 0 = written (FINAL, or INTERIM with --allow_interim); 1 = check failed; 3 = not complete (no --allow_interim).
"""
import argparse, json, os, sys, time

H = os.path.expanduser('~')
E_SHA16 = {0: '35c626874e1aeb70', 1: '3af5d74bcf929393', 2: '2575ba5833d5cbfe', 3: 'bc92742b54270fd4',
           4: '8f1bb372aa20cf07'}                   # the shipped K10 / k10w set E picks (k10/.k10_want.txt, 10-09 07:05)
K9_SHA = {('C', 0): 'aa394cbf83b0084e', ('C', 1): '2aa098e78541e33c', ('C', 2): '47d6f35bf0f1a4ff',
          ('C', 3): 'a62a6be3750bc390', ('C', 4): '33078bd461d504d7', ('D', 0): 'c478a39d75b9657f',
          ('D', 1): 'fea35749550a275c', ('D', 2): 'ded00943d1b22a1a', ('D', 3): '71815926cc6f286b',
          ('D', 4): '79e301ed41d5cb12'}             # == k10_merge.py K9_SHA
PREREG_E2_BAR = {0: .884325, 1: .887201, 2: .886670, 3: .873837, 4: .886362}   # PREREG_FN11 sec 2 (rounded, 6 dp)
NEW = {'E2': dict(seed=1046, ttr=288, labels='labels_frac75_logit.csv', suffix='ttr288frac75', key='t288f75', ref='E',
                  receipt='modal_receipt'),
       'D2': dict(seed=1044, ttr=224, labels='labels_frac75_logit.csv', suffix='ttr_frac75', key='f75', ref='D',
                  receipt=None)}
MARGIN = 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default=f'{H}/k11_test')
    ap.add_argument('--base', default=f'{H}/k10_test/members.json')
    ap.add_argument('--allow_interim', action='store_true')
    ap.add_argument('--e2_ship', default='', choices=('', '4of5'))
    ap.add_argument('--with_d2', action='store_true')
    a = ap.parse_args()
    assert os.path.realpath(a.dest).startswith(os.path.realpath(f'{H}/k11_test')), 'k11_merge writes only under ~/k11_test'
    base = json.load(open(a.base))
    assert base.get('k10') is True and not base.get('interim') and not base.get('do_not_submit') and not base.get('failed')
    bm = base['members']
    cnt = {s: sorted(r['fold'] for r in bm if r['set'] == s) for s in ('C', 'D', 'E')}
    assert cnt == {s: [0, 1, 2, 3, 4] for s in ('C', 'D', 'E')} and len(bm) == 15, cnt
    for r in bm:
        want = E_SHA16[r['fold']] if r['set'] == 'E' else K9_SHA[(r['set'], r['fold'])]
        if r['sha256'][:16] != want or r.get('tag') != 'raw':
            print(f'K11MERGE FAIL: base {r["set"]} f{r["fold"]} {r["file"]} sha256 {r["sha256"][:16]} != shipped {want}')
            return 1
    ref = {s: {r['fold']: r['cv'] for r in bm if r['set'] == s} for s in ('D', 'E')}
    for F, b in PREREG_E2_BAR.items():
        assert abs(ref['E'][F] - MARGIN - b) < 6e-7, (F, ref['E'][F] - MARGIN, b)
    members = [dict(r) for r in bm]
    info, skipped, final, counts = {}, [], True, {}
    for s in (['E2'] + (['D2'] if a.with_d2 else [])):
        cfg = NEW[s]
        p = f'{a.dest}/members_{s}.json'
        X = json.load(open(p)) if os.path.isfile(p) else dict(members=[], skipped=[dict(fold=F, reason='no picker output')
                                                                                for F in range(5)], failed=[])
        bars = {F: ref[cfg['ref']][F] - MARGIN for F in range(5)}
        st = dict(margin=MARGIN, ref=f'set {cfg["ref"]} picked raw cv', bars=bars)
        bad = [(r['fold'], r.get('tag'), r.get('seed'), r.get('ttr_size'), r.get('train_labels'), r.get('suffix'))
               for r in X['members'] if r.get('tag') != 'raw' or int(r.get('seed', -1)) != cfg['seed'] or
               int(r.get('ttr_size', -1)) != cfg['ttr'] or r.get('train_labels') != cfg['labels'] or r.get('suffix') != cfg['suffix']]
        if bad:
            print(f'K11MERGE FAIL: set {s} picks not (raw, seed {cfg["seed"]}, ttr {cfg["ttr"]}, {cfg["labels"]}): {bad}')
            return 1
        if cfg['receipt']:
            nor = [r['fold'] for r in X['members'] if not r.get(cfg['receipt']) or r.get('source') != 'modal']
            if nor:
                print(f'K11MERGE FAIL: set {s} folds {nor} carry no Modal receipt (picker --modal_receipt_folds)'); return 1
        acc, excl = [], []
        for F in range(5):
            r = {x['fold']: x for x in X['members']}.get(F)
            fl = {x['fold']: x for x in X.get('failed', [])}.get(F)
            if r is None:
                why = (fl or {}).get('reason') or ({x['fold']: x for x in X.get('skipped', [])}.get(F) or {}).get('reason', 'not picked')
                stt = 'FAILED A CHECK (excluded)' if fl else 'NOT FINISHED'
                (excl if fl else skipped).append(dict(fold=F, set=s, seed=cfg['seed'], suffix=cfg['suffix'], reason=why))
                print(f'  set {s} fold {F}: {stt}: {str(why)[:150]}')
                continue
            ok = r['cv'] >= bars[F]
            print(f'  set {s} fold {F}: cv {r["cv"]:.6f} ({r["cv"] - ref[cfg["ref"]][F]:+.4f} vs {cfg["ref"]}) E {r.get("epochs")} '
                  f'source {r.get("source")} bar {bars[F]:.6f} [{cfg["ref"]} f{F} cv {ref[cfg["ref"]][F]:.6f} - {MARGIN}] -> '
                  f'{"ACCEPTED" if ok else "BELOW BAR (excluded, no override)"}')
            if ok:
                acc.append(dict(r, set=s))
            else:
                excl.append(dict(fold=F, set=s, seed=cfg['seed'], suffix=cfg['suffix'], reason=f'cv {r["cv"]:.6f} < bar {bars[F]:.6f}'))
        folds = [r['fold'] for r in acc]
        st.update(cv={r['fold']: r['cv'] for r in acc}, accepted=folds, excluded=excl)
        counts[s] = folds
        if folds == [0, 1, 2, 3, 4]:
            st['status'] = 'FINAL: 5/5 accepted'
        elif s == 'E2' and folds == [0, 1, 2, 3] and a.e2_ship == '4of5':
            st['status'] = 'FINAL 4/5 (e2_ship 4of5, PREREG_FN11 sec 4)'
        elif len(folds) >= 4 or (s == 'E2' and len(folds) >= 1):
            final = False
            st['status'] = f'INTERIM: {len(folds)}/5 accepted {folds}'
        else:
            final = False
            st['status'] = f'INTERIM: {len(folds)}/5 accepted {folds}'
        if s == 'D2' and folds != [0, 1, 2, 3, 4]:
            final = False
        if excl and s == 'E2' and len(folds) + len([x for x in skipped if x['set'] == 'E2']) < 4:
            print(f'K11MERGE: set E2 has {len(folds)} accepted fold(s) and cannot reach 4: E2 NOT GATED (PREREG sec 2)')
        members += acc
        info[s] = st
    if a.e2_ship == '4of5':
        assert counts['E2'] == [0, 1, 2, 3], f'--e2_ship 4of5 needs E2 folds exactly 0-3 accepted, got {counts["E2"]}'
    if not final and not a.allow_interim:
        for sk in skipped:
            print(f'  NOT FINISHED set {sk["set"]}: f{sk["fold"]}_s{sk["seed"]}_{sk["suffix"]}: {str(sk["reason"])[:110]}')
        print(f'K11MERGE NOT FINAL: ' + ', '.join(f'{s} {len(v)}/5' for s, v in counts.items()) +
              '. TEST builds only with --allow_interim (members.json do_not_submit)')
        return 3
    ids = [(r['seed'], r['fold'], int(r['ttr_size']), r['train_labels']) for r in members]
    files = [r['file'] for r in members]
    if len(set(ids)) != len(ids) or len(set(files)) != len(files):
        print(f'K11MERGE FAIL: duplicate members {sorted(ids)}'); return 1
    out = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), k10=True, k11=True,
               set='K11 = C + D + E (the FINAL K10 members, == k10w) + E2 f{0..4}_s1046_ttr288frac75 (TTR288 frac75, Modal)'
                   + (' + D2 f{0..4}_s1044_ttr_frac75' if a.with_d2 else ''),
               base=a.base, members=members, skipped=skipped, failed=[], copied=True,
               set_c=base['set_c'], set_d=base['set_d'], set_e=base['set_e'], set_e2=info.get('E2'), set_d2=info.get('D2'),
               e2_ship=(a.e2_ship or None), interim=not final, do_not_submit=not final)
    tmp = f'{a.dest}/members.json.tmp'
    json.dump(out, open(tmp, 'w'), indent=1); os.replace(tmp, f'{a.dest}/members.json')
    for r in members:
        rc = os.path.basename(r.get('modal_receipt') or r.get('kaggle_receipt') or r.get('vast_receipt') or '-')
        print(f'  {r["set"]} s{r["seed"]} f{r["fold"]} ttr{r["ttr_size"]} {r["train_labels"]} {r["tag"]} E {r.get("epochs")} '
              f'cv {r["cv"]:.6f} {r["file"]} source {r["source"]} receipt {rc}')
    print(f'  set E2: {info["E2"]["status"]}' + (f' | set D2: {info["D2"]["status"]}' if 'D2' in info else ''))
    print(f'K11MERGE OK {len(members)} member(s): C 5/5, D 5/5, E 5/5, ' + ', '.join(f'{s} {len(v)}/5' for s, v in counts.items())
          + (' (FINAL' + (', e2_ship 4of5)' if a.e2_ship else ')') if final else ' (INTERIM: do_not_submit, NEVER SUBMIT)'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
