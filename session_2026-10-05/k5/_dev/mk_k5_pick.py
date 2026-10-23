# dev helper (not shipped): k5/box/k5_pick.py = k4_pick.py + the K5 changes below
import os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = open(f'{HERE}/box/k4_pick.py', encoding='utf-8', newline='').read()
assert '\r\n' not in s


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:70], s.count(old))
    s = s.replace(old, new)


rep('''"""k4_pick.py (box, CPU, read-only on ~/fleet_runs and ~/fleet_store; writes only under --dest, default ~/k4_test):
the K4 member set = ONE checkpoint per FINISHED run of the 5-fold TTR mix50 set f{0..4}_s1043_ttr_mix50.
Generalises k3/box/k3_pick_ttr.py (same rules) from one run to all folds.

  CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/kmimic/bin/python -W ignore ~/k4_test/k4/box/k4_pick.py [--copy]
''', '''"""k5_pick.py (box, CPU, read-only on ~/fleet_runs and ~/fleet_store; writes only under --dest, default ~/k5_test):
the K5 member set = ONE checkpoint per FINISHED run of the 5-fold TTR mix50 set f{0..4}_s1043_ttr_mix50.
= k4_pick.py (same rules) + runs whose out dir was filled from the Kaggle training notebook rsna-train-ttr-f34
(kaggle_train/RUNBOOK.md step 6: same file layout + the trainer's own log with its closing line):
  K5-E   the run's epoch count E comes from its log header ('BJ opt-in: ... | epochs E |') and must equal the
         checkpoint args 'epochs' (box folds: 24; Kaggle folds: E = min(24, what fits in 10.3 h)); the last-epoch
         lines checked are 'ep E-1 raw/ema'. E < --min_epochs (default 16, the Kaggle RUNBOOK warning line) FAILS
         (the operator's call: re-run with --min_epochs N, or K5_MIN_EPOCHS=N for k5_collect.sh).
  K5-OOF content proof for EVERY run: the picked OOF csv, scored exactly as the trainer scores (macro AUC vs the
         selection labels, sklearn roc_auc_score, NaN masked), must reproduce the checkpoint's stored cv to 1e-12
         (MEASURED 10-05 on J1 raw / J1 ema / M1: diff 0.0). A stale or other-epoch OOF cannot pass.
  K5-KAG a Kaggle-filled run (its fold is listed in the Kaggle run receipt: --kaggle_receipt, default
         <root>/kaggle_f34_run_receipt.json, or --kaggle_folds) skips the OOF-vs-ckpt mtime rule (scp / download
         set new mtimes) and instead needs the picked checkpoint AND OOF sha256 == the receipt's entries.
  K5-ROOT --extra_root DIR (repeatable): a fold whose run dir or log is missing under --root (e.g. the empty Kaggle
         placeholders) is also looked up there, same layout (DIR/f{F}_s1043_ttr_mix50/ + DIR/f{F}_s1043_ttr_mix50.log).

  CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/kmimic/bin/python -W ignore ~/k5_test/k5/box/k5_pick.py [--copy]
''')
rep('''import argparse, glob, hashlib, json, os, re, shutil, sys, time
''', '''import argparse, glob, hashlib, json, os, re, shutil, sys, time
import numpy as np, pandas as pd
''')
rep('''TRAIN_LABELS, SEL_LABELS = 'labels_mix50_logit.csv', 'llm_labels_v4_blend.csv'
''', '''TRAIN_LABELS, SEL_LABELS = 'labels_mix50_logit.csv', 'llm_labels_v4_blend.csv'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
           'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def log_epochs(log):
    """K5-E: E from the trainer's header line 'BJ opt-in: ... | epochs E | ...' (None if absent)."""
    for l in open(log, encoding='utf-8', errors='replace'):
        if l.startswith('BJ opt-in:'):
            m = re.search(r'[|] epochs (\\d+) [|]', l)
            return int(m.group(1)) if m else None
    return None


def oof_cv(oof_csv, sel_csv):
    """K5-OOF: the trainer's macro_auc (fleet_train.py) of the OOF csv vs the selection labels."""
    from sklearn.metrics import roc_auc_score
    o = pd.read_csv(oof_csv, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    sel = pd.read_csv(sel_csv, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    P = o[TARGETS].to_numpy(np.float64); Y = sel.loc[list(o.index), TARGETS].to_numpy(np.float64); a = []
    for j in range(len(TARGETS)):
        ok = np.isfinite(Y[:, j]) & np.isfinite(P[:, j]); yb = Y[ok, j] >= 0.5
        if ok.sum() and 0 < yb.mean() < 1:
            a.append(roc_auc_score(yb, P[ok, j]))
    return (float(np.mean(a)) if a else float('nan')), len(o)


def kaggle_entries(a, root, F):
    """K5-KAG: ({relative path: sha256} of fold F's files, receipt path) from the Kaggle run receipt; (None, None) when
    the run is not Kaggle-filled; ({}, None) for --kaggle_folds without a receipt."""
    cands = ([a.kaggle_receipt] if a.kaggle_receipt else []) + [f'{root}/kaggle_f34_run_receipt.json']
    for rp in cands:
        if rp and os.path.isfile(rp):
            rec = json.load(open(rp))
            folds = (rec.get('train') or {}).get('folds') or {}
            x = folds.get(str(F), folds.get(F))
            if x:
                return {e['path']: e['sha256'] for e in x.get('files', [])}, rp
    if F in a.kaggle_folds:
        return {}, None
    return None, None
''')
rep('''def pick_fold(F, a, torch):
    """-> ('member', dict) | ('skip', reason) | ('fail', reason)"""
    run = f'{a.root}/f{F}_s{a.seed}_{a.suffix}'
    log = run + '.log'
    prefix = f'f{F}_s{a.seed}'
    if not os.path.isdir(run) or not os.path.isfile(log):
        return 'skip', f'not started ({run} or its log missing)'
    ok, best, lastline, have = final_line(log, a.epochs)
    if not ok:
        return 'skip', f'not finished: last log line {lastline!r}; ep {a.epochs - 1} raw/ema logged {have}'
''', '''def pick_fold(F, a, torch):
    """-> ('member', dict) | ('skip', reason) | ('fail', reason)"""
    prefix = f'f{F}_s{a.seed}'
    tried = []
    for root in [a.root] + a.extra_root:                                # K5-ROOT
        run = f'{root}/f{F}_s{a.seed}_{a.suffix}'
        log = run + '.log'
        if os.path.isdir(run) and os.path.isfile(log):
            break
        tried.append(run)
    else:
        return 'skip', f'not started ({" / ".join(tried)}: run dir or its .log missing)'
    E = log_epochs(log) or a.epochs                                     # K5-E
    ok, best, lastline, have = final_line(log, E)
    if not ok:
        return 'skip', f'not finished: last log line {lastline!r}; ep {E - 1} raw/ema logged {have} (E {E} from the log)'
    kag, kag_rp = kaggle_entries(a, root, F)                            # K5-KAG
''')
rep('''                         bf16=bool(ar.get('bf16', False)), probe=sorted(ar.get('decode_probe', {})))''',
    '''                         bf16=bool(ar.get('bf16', False)), probe=sorted(ar.get('decode_probe', {})),
                         epochs=int(ar.get('epochs', -1)))''')
rep('''    if c['amp'] != 'fp16' or c['bf16']: errs.append(f'amp {c["amp"]} bf16 {c["bf16"]}')
    # VERIFY fix 10-05: save() writes the ckpt then the OOF ~13 ms later (MEASURED on 5 runs), so whole-second equality
    # failed ~1.3% of picks when a second boundary fell in between. Same save() call = OOF 0..5 s after the ckpt
    # (two save() calls of one tag are >= 1 epoch, ~10 min, apart).
    if not (0.0 <= fmt_o - fmt_c <= 5.0):
        errs.append(f'oof mtime - ckpt mtime = {fmt_o - fmt_c:.3f} s, not in [0, 5] (not the same save() call)')
''', '''    if c['amp'] != 'fp16' or c['bf16']: errs.append(f'amp {c["amp"]} bf16 {c["bf16"]}')
    if c['epochs'] != E: errs.append(f'ckpt args epochs {c["epochs"]} != log header epochs {E}')          # K5-E
    if E < a.min_epochs: errs.append(f'E {E} < --min_epochs {a.min_epochs} (short cosine: the operator decides)')
    cv_o, n_o = oof_cv(oof, a.sel_csv)                                                                       # K5-OOF
    print(f'  fold {F} {tag}: OOF {os.path.basename(oof)} ({n_o} studies) rescored cv {cv_o!r} vs ckpt {c["cv"]!r} '
          f'(diff {cv_o - c["cv"]:.2e})', flush=True)
    if not abs(cv_o - c['cv']) <= 1e-12:
        errs.append(f'OOF rescored cv {cv_o} != ckpt cv {c["cv"]} (not this checkpoint OOF)')
    if kag is None:
        # VERIFY fix 10-05: save() writes the ckpt then the OOF ~13 ms later (MEASURED on 5 runs), so whole-second
        # equality failed ~1.3% of picks when a second boundary fell in between. Same save() call = OOF 0..5 s after
        # the ckpt (two save() calls of one tag are >= 1 epoch, ~10 min, apart).
        if not (0.0 <= fmt_o - fmt_c <= 5.0):
            errs.append(f'oof mtime - ckpt mtime = {fmt_o - fmt_c:.3f} s, not in [0, 5] (not the same save() call; '
                        f'a run filled from Kaggle needs its run receipt: --kaggle_receipt / --kaggle_folds)')
    else:                                                                                                    # K5-KAG
        rel_c = f'{os.path.basename(run)}/{os.path.basename(c["path"])}'
        rel_o = f'{os.path.basename(run)}/{os.path.basename(oof)}'
        if kag_rp:
            for rel, pth in ((rel_c, c['path']), (rel_o, oof)):
                if kag.get(rel) != sha256(pth):
                    errs.append(f'{rel} sha256 != the Kaggle run receipt {kag_rp} ({kag.get(rel)})')
        print(f'  fold {F}: Kaggle-filled run ({kag_rp or "--kaggle_folds"}): mtime rule skipped, '
              f'{"ckpt + OOF sha256 == receipt" if kag_rp else "NO receipt sha check"}', flush=True)
''')
rep('''               sel_labels=c['sel_labels'], decode_probe=c['probe'])''',
    '''               sel_labels=c['sel_labels'], decode_probe=c['probe'], epochs=E, oof_rescored_cv=cv_o,
               source='kaggle' if kag is not None else 'box', kaggle_receipt=kag_rp)''')
rep('''    ap.add_argument('--epochs', type=int, default=24)
    ap.add_argument('--copy', action='store_true')
    ap.add_argument('--dest', default=os.path.expanduser('~/k4_test'))''',
    '''    ap.add_argument('--epochs', type=int, default=24, help='fallback when the log header has no epochs field')
    ap.add_argument('--min_epochs', type=int, default=int(os.environ.get('K5_MIN_EPOCHS', 16)))
    ap.add_argument('--extra_root', action='append', default=[])
    ap.add_argument('--kaggle_receipt', default=None)
    ap.add_argument('--kaggle_folds', default='', help='comma list: treat these folds as Kaggle-filled without a receipt')
    ap.add_argument('--sel_csv', default=os.path.expanduser('~/rsna_train/llm_labels_v4_blend.csv'))
    ap.add_argument('--copy', action='store_true')
    ap.add_argument('--dest', default=os.path.expanduser('~/k5_test'))''')
rep('''    a = ap.parse_args()
    import torch''', '''    a = ap.parse_args()
    a.extra_root = [os.path.expanduser(x) for x in a.extra_root]
    a.kaggle_folds = [int(x) for x in a.kaggle_folds.split(',') if x.strip()]
    import torch''')
rep('''    print(f'K4PICK members''', '''    print(f'K5PICK members''')
import ast
ast.parse(s)
open(f'{HERE}/box/k5_pick.py', 'w', encoding='utf-8', newline='').write(s)
print('ok')
