#!/usr/bin/env python3
"""k5_pick.py, K8 COPY (2026-10-07, Claude) of the K7 copy. K8 addition (default keeps the K5/K6/K7 behaviour):
  K8-RAW  --tag raw|ema|auto (default auto = the old rule: the tag with the higher val cv, tie -> ema). K8 ships RAW
          best checkpoints for every member (NOVEL_PLAN 10-07 sec 2 item 2; "raw best-val checkpoint" settled sec 6), so
          k8_collect.sh passes --tag raw for set A and set C. Every check below runs unchanged on the forced tag (its cv
          == closing line, its OOF rescored == its cv, ...); the record carries tag_rule and other_tag_cv.
(K7 copy text) Additions over the K6 copy (every default keeps the K5/K6 behaviour):
  K7-TTR  --ttr_size N (default 224): the expected ttr_size buffer == args ttr_size of this set (set C: TTR288 = 288).
  K7-REC  --kaggle_receipt is repeatable (one receipt per Kaggle notebook, e.g. the teammate's s1044 f01 / f23 / f4 runs);
          a receipt is used for fold F only when its cfg SEED (if recorded) == --seed AND its runs[F] (if recorded) == this
          run's folder name, so the seed-1043 f34 receipt can never vouch for a seed-1044 fold, and a Kaggle s1044 fold-0
          receipt never vouches for the box TTR288 fold-0 run. --require_receipt: a fold without a receipt entry FAILS
          (set B is Kaggle-trained only, so a missing receipt means a wrong copy).
  K7-KEY  two sets may now share fold+seed (set B f0_s1044_ttr_mix50 and set C f0_s1044_ttr288mix50, both fold 0 seed
          1044). Their files differ in the recipe field (field 4 of f{F}_s{S}_{tag}_{recipe}_cv{N}_best.pt: ttr224-... vs
          ttr288-...). --copy now removes a stale DEST/ckpt file of this fold+seed ONLY when it has the same recipe field
          (it used to remove every f{F}_s{S}_*_best.pt, which would delete the other set's pick), and the OOF copy of a
          non-224 set is named oof_f{F}_s{S}t{N}_<tag>_cv<NNNN>.csv (224 sets: unchanged name). Each member record
          carries 'suffix' (the run suffix) so the merged members.json says which run every pick came from.
(K6 header: + --train_labels so the TTR v4 set f{F}_s1042_ttr can be picked with the same checks; default unchanged) (box, CPU, read-only on ~/fleet_runs and ~/fleet_store; writes only under --dest, default ~/k7_test):
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

  CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/kmimic/bin/python -W ignore ~/k7_test/k5/box/k5_pick.py [--copy]

Per fold F (run dir ~/fleet_runs/f{F}_s1043_ttr_mix50, log <run>.log):
  FINAL only when the trainer printed its closing {"raw": cv, "ema": cv} line (fleet_train.py prints it after the LAST
  save(); save() deletes the old best before writing the new one, so files alone prove nothing) AND the log holds the
  last epoch's raw and ema lines. Otherwise the fold is SKIPPED (listed, nothing copied).
  Tag rule (K2 RUNBOOK 3.1, one tag per run): the tag whose best checkpoint has the higher val cv (read exactly from the
  checkpoint, must equal the closing line); a tie goes to EMA.
  Checks: exactly one best file per tag, stored cv == closing line, OOF csv mtime 0..5 s after the checkpoint mtime
  (same save() call), model_mod fleet_model_ttr, model_sha f141aeed136dfc01, ttr_size buffer == args == 224,
  args fold == F and seed == 1043, train labels labels_mix50_logit.csv, selection labels llm_labels_v4_blend.csv,
  amp fp16 (no bf16), every decode_probe study present in ~/fleet_store.
--copy: cp -p the picked checkpoint to DEST/ckpt/ and its OOF to DEST/oof/oof_f{F}_s1043_<tag>_cv<NNNN>.csv, re-check
  sha256 of the copy and that the source did not change; stale DEST/ckpt/f{F}_s1043_*_best.pt files of a picked fold
  are removed (DEST is ours). Writes DEST/members.json (members + skipped + failed).
Exit 0 = >= 1 member picked and no check failed; 1 = a check failed on a FINISHED run (do not ship); 3 = no run final.
"""
import argparse, glob, hashlib, json, os, re, shutil, sys, time
import numpy as np, pandas as pd

TTR_SHA = 'f141aeed136dfc01'
TRAIN_LABELS, SEL_LABELS = 'labels_mix50_logit.csv', 'llm_labels_v4_blend.csv'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
           'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def log_epochs(log):
    """K5-E: E from the trainer's header line 'BJ opt-in: ... | epochs E | ...' (None if absent)."""
    for l in open(log, encoding='utf-8', errors='replace'):
        if l.startswith('BJ opt-in:'):
            m = re.search(r'[|] epochs (\d+) [|]', l)
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
    if a.no_kaggle_receipt:                                              # K6: the TTR v4 set is all box-trained
        return None, None
    cands = list(a.kaggle_receipt) + [f'{root}/kaggle_f34_run_receipt.json']       # K7-REC: repeatable
    run_name = f'f{F}_s{a.seed}_{a.suffix}'
    for rp in cands:
        if rp and os.path.isfile(rp):
            rec = json.load(open(rp))
            seed = (rec.get('cfg') or {}).get('SEED')
            if seed is not None and int(seed) != a.seed:                 # K7-REC: another seed's receipt
                continue
            runs = rec.get('runs') or {}
            rn = runs.get(str(F), runs.get(F))
            if rn is not None and rn != run_name:                         # K7-REC: another run of this fold+seed
                continue
            folds = (rec.get('train') or {}).get('folds') or {}
            x = folds.get(str(F), folds.get(F))
            if x:
                return {e['path']: e['sha256'] for e in x.get('files', [])}, rp
    if F in a.kaggle_folds:
        return {}, None
    return None, None


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def final_line(log, epochs):
    lines = [l.strip() for l in open(log, encoding='utf-8', errors='replace') if l.strip()]
    last = epochs - 1
    have = {t: any(l.startswith(f'ep {last} {t} ') for l in lines) for t in ('raw', 'ema')}
    best = None
    if lines and lines[-1].startswith('{'):
        try:
            best = json.loads(lines[-1])
        except Exception:
            best = None
    ok = isinstance(best, dict) and set(best) >= {'raw', 'ema'} and all(have.values())
    return ok, best, (lines[-1][:120] if lines else None), have


def pick_fold(F, a, torch):
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
    cand = {}
    for tag in ('raw', 'ema'):
        fs = sorted(glob.glob(f'{run}/{prefix}_{tag}_*_best.pt'))
        if len(fs) != 1:
            return 'fail', f'expected exactly one {tag} best file in {run}, found {fs}'
        ck = torch.load(fs[0], map_location='cpu', weights_only=False)
        ar = ck['args']
        cand[tag] = dict(path=fs[0], cv=float(ck['cv']), epoch1=int(ck['epoch']) + 1, mod=ar.get('model_mod'),
                         msha=ar.get('model_sha'), tsz=int(ck['model'].get('ttr_size_t', -1)),
                         ttr=int(ar.get('ttr_size') or -1), fold=int(ar.get('fold', -1)), seed=int(ar.get('seed', -1)),
                         train_labels=ar.get('train_labels'), sel_labels=ar.get('sel_labels'), amp=ar.get('amp'),
                         bf16=bool(ar.get('bf16', False)), probe=sorted(ar.get('decode_probe', {})),
                         epochs=int(ar.get('epochs', -1)))
        del ck
        c = cand[tag]
        print(f'  fold {F} {tag}: {os.path.basename(c["path"])} cv {c["cv"]:.6f} (closing line {best[tag]:.6f}) '
              f'epoch {c["epoch1"]} {c["mod"]} sha {c["msha"]} ttr_size_t {c["tsz"]}', flush=True)
        if abs(c['cv'] - float(best[tag])) > 1e-12:
            return 'fail', f'{tag} checkpoint cv {c["cv"]} != closing line {best[tag]}'
    tag = 'ema' if cand['ema']['cv'] >= cand['raw']['cv'] else 'raw'
    if a.tag != 'auto':                                                  # K8-RAW: forced tag (both tags checked above)
        tag = a.tag
    c = cand[tag]
    oof = f'{run}/oof_{prefix}_{tag}.csv'
    if not os.path.isfile(oof):
        return 'fail', f'missing OOF {oof}'
    fmt_c, fmt_o = os.path.getmtime(c['path']), os.path.getmtime(oof)
    mt_c, mt_o = int(fmt_c), int(fmt_o)
    errs = []
    if c['mod'] != 'fleet_model_ttr': errs.append(f'model_mod {c["mod"]}')
    if c['msha'] != TTR_SHA: errs.append(f'model_sha {c["msha"]} != {TTR_SHA}')
    if not (c['tsz'] == c['ttr'] == a.ttr_size): errs.append(f'ttr_size_t {c["tsz"]} args {c["ttr"]} (want --ttr_size {a.ttr_size})')   # K7-TTR
    if c['fold'] != F or c['seed'] != a.seed: errs.append(f'args fold/seed {c["fold"]}/{c["seed"]}')
    if c['train_labels'] != a.train_labels: errs.append(f'train_labels {c["train_labels"]} != --train_labels {a.train_labels}')   # K6: per set
    if c['sel_labels'] != SEL_LABELS: errs.append(f'sel_labels {c["sel_labels"]}')
    if c['amp'] != 'fp16' or c['bf16']: errs.append(f'amp {c["amp"]} bf16 {c["bf16"]}')
    if c['epochs'] != E: errs.append(f'ckpt args epochs {c["epochs"]} != log header epochs {E}')          # K5-E
    if E < a.min_epochs: errs.append(f'E {E} < --min_epochs {a.min_epochs} (short cosine: the operator decides)')
    cv_o, n_o = oof_cv(oof, a.sel_csv)                                                                       # K5-OOF
    print(f'  fold {F} {tag}: OOF {os.path.basename(oof)} ({n_o} studies) rescored cv {cv_o!r} vs ckpt {c["cv"]!r} '
          f'(diff {cv_o - c["cv"]:.2e})', flush=True)
    if not abs(cv_o - c['cv']) <= 1e-12:
        errs.append(f'OOF rescored cv {cv_o} != ckpt cv {c["cv"]} (not this checkpoint OOF)')
    if kag is None and a.require_receipt:                                                                   # K7-REC
        errs.append(f'no Kaggle run receipt vouches for {os.path.basename(run)} (--require_receipt; receipts tried: '
                    f'{list(a.kaggle_receipt)} + {root}/kaggle_f34_run_receipt.json)')
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
    miss = [s for s in c['probe'] if not os.path.isfile(f'{a.store}/{s}.npz')]
    if miss: errs.append(f'decode_probe npz missing in {a.store}: {miss}')
    if not c['probe']: errs.append('empty decode_probe')
    if errs:
        return 'fail', '; '.join(errs)
    base = os.path.basename(c['path'])
    m = re.search(r'_cv(\d+)_best\.pt$', base)
    src_sha = sha256(c['path'])
    rec = dict(fold=F, seed=a.seed, tag=tag, epoch=c['epoch1'], cv=c['cv'], file=base, sha256=src_sha,
               other_tag_cv=cand['raw' if tag == 'ema' else 'ema']['cv'], closing_line=best, run=run,
               src=c['path'], oof_src=oof, mtime=mt_c, mb=round(os.path.getsize(c['path']) / 2**20, 1),
               model_mod=c['mod'], model_sha=c['msha'], ttr_size=c['ttr'], train_labels=c['train_labels'],
               sel_labels=c['sel_labels'], decode_probe=c['probe'], epochs=E, oof_rescored_cv=cv_o,
               source='kaggle' if kag is not None else 'box', kaggle_receipt=kag_rp, suffix=a.suffix,          # K7-KEY
               tag_rule=a.tag)                                                                          # K8-RAW
    if a.copy:
        os.makedirs(f'{a.dest}/ckpt', exist_ok=True); os.makedirs(f'{a.dest}/oof', exist_ok=True)
        recipe = base.split('_')[3]                                      # K7-KEY: ttr224-320x24 / ttr288-320x24
        for q in glob.glob(f'{a.dest}/ckpt/{prefix}_*_best.pt'):          # stale pick of this fold (our dir only)
            qp = os.path.basename(q).split('_')
            if os.path.basename(q) != base and len(qp) > 3 and qp[3] == recipe:     # K7-KEY: never the other set's pick
                os.remove(q); print(f'  removed stale {q}')
        dst = f'{a.dest}/ckpt/{base}'
        tkey = '' if a.ttr_size == 224 else f't{a.ttr_size}'             # K7-KEY: 224 sets keep the old name
        doof = f'{a.dest}/oof/oof_{prefix}{tkey}_{tag}_cv{m.group(1)}.csv'
        if not (os.path.isfile(dst) and sha256(dst) == src_sha):
            shutil.copy2(c['path'], dst)
        shutil.copy2(oof, doof)
        s2 = sha256(dst)
        if s2 != src_sha or int(os.path.getmtime(c['path'])) != mt_c or sha256(c['path']) != src_sha:
            return 'fail', 'the source changed during the copy (a save() raced it?); re-run'
        rec.update(copy=dst, oof=doof)
        print(f'  copied {dst} (sha256 {s2[:16]}) and {doof}')
    return 'member', rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--seed', type=int, default=1043)
    ap.add_argument('--suffix', default='ttr_mix50')
    ap.add_argument('--root', default=os.path.expanduser('~/fleet_runs'))
    ap.add_argument('--store', default=os.path.expanduser('~/fleet_store'))
    ap.add_argument('--epochs', type=int, default=24, help='fallback when the log header has no epochs field')
    ap.add_argument('--min_epochs', type=int, default=int(os.environ.get('K5_MIN_EPOCHS', 16)))
    ap.add_argument('--extra_root', action='append', default=[])
    ap.add_argument('--kaggle_receipt', action='append', default=[], help='K7-REC: repeatable')
    ap.add_argument('--require_receipt', action='store_true', help='K7-REC: a fold no receipt vouches for FAILS')
    ap.add_argument('--ttr_size', type=int, default=224, help='K7-TTR: expected ttr_size of this set (TTR288: 288)')
    ap.add_argument('--kaggle_folds', default='', help='comma list: treat these folds as Kaggle-filled without a receipt')
    ap.add_argument('--sel_csv', default=os.path.expanduser('~/rsna_train/llm_labels_v4_blend.csv'))
    ap.add_argument('--train_labels', default=TRAIN_LABELS, help='K6: expected ckpt args train_labels of this set (mix50 default; TTR v4 set: llm_labels_v4_blend.csv)')
    ap.add_argument('--no_kaggle_receipt', action='store_true', help='K6: never treat a fold as Kaggle-filled (the v4 set; the f34 receipt belongs to the mix50 set)')
    ap.add_argument('--tag', choices=('auto', 'raw', 'ema'), default='auto',
                    help='K8-RAW: force this tag for every fold (default auto: higher val cv, tie -> ema)')
    ap.add_argument('--copy', action='store_true')
    ap.add_argument('--dest', default=os.path.expanduser('~/k8_test'))
    a = ap.parse_args()
    a.extra_root = [os.path.expanduser(x) for x in a.extra_root]
    a.kaggle_receipt = [os.path.expanduser(x) for x in a.kaggle_receipt]
    a.kaggle_folds = [int(x) for x in a.kaggle_folds.split(',') if x.strip()]
    import torch
    members, skipped, failed = [], [], []
    for F in [int(x) for x in a.folds.split(',') if x.strip()]:
        kind, x = pick_fold(F, a, torch)
        if kind == 'member':
            members.append(x)
            print(f'PICK fold {F}: {x["file"]} tag {x["tag"]} epoch {x["epoch"]} cv {x["cv"]:.6f} '
                  f'(other tag {x["other_tag_cv"]:.6f}) sha256 {x["sha256"][:16]}', flush=True)
        elif kind == 'skip':
            skipped.append(dict(fold=F, reason=x)); print(f'SKIP fold {F}: {x}', flush=True)
        else:
            failed.append(dict(fold=F, reason=x)); print(f'FAIL fold {F}: {x}', flush=True)
    out = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S %Z'), host=os.uname().nodename, seed=a.seed,
               set=f'f{{0..4}}_s{a.seed}_{a.suffix}', members=members, skipped=skipped, failed=failed, copied=bool(a.copy))
    if a.copy:
        os.makedirs(a.dest, exist_ok=True)
        tmp = f'{a.dest}/members.json.tmp'
        json.dump(out, open(tmp, 'w'), indent=1); os.replace(tmp, f'{a.dest}/members.json')
        print(f'wrote {a.dest}/members.json')
    print(f'K5PICK members {[m["fold"] for m in members]} skipped {[s["fold"] for s in skipped]} '
          f'failed {[f["fold"] for f in failed]}', flush=True)
    if failed:
        return 1
    return 0 if members else 3


if __name__ == '__main__':
    sys.exit(main())
