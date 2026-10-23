#!/usr/bin/env python3
"""k4_pick.py (box, CPU, read-only on ~/fleet_runs and ~/fleet_store; writes only under --dest, default ~/k4_test):
the K4 member set = ONE checkpoint per FINISHED run of the 5-fold TTR mix50 set f{0..4}_s1043_ttr_mix50.
Generalises k3/box/k3_pick_ttr.py (same rules) from one run to all folds.

  CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/kmimic/bin/python -W ignore ~/k4_test/k4/box/k4_pick.py [--copy]

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

TTR_SHA = 'f141aeed136dfc01'
TRAIN_LABELS, SEL_LABELS = 'labels_mix50_logit.csv', 'llm_labels_v4_blend.csv'


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
    run = f'{a.root}/f{F}_s{a.seed}_{a.suffix}'
    log = run + '.log'
    prefix = f'f{F}_s{a.seed}'
    if not os.path.isdir(run) or not os.path.isfile(log):
        return 'skip', f'not started ({run} or its log missing)'
    ok, best, lastline, have = final_line(log, a.epochs)
    if not ok:
        return 'skip', f'not finished: last log line {lastline!r}; ep {a.epochs - 1} raw/ema logged {have}'
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
                         bf16=bool(ar.get('bf16', False)), probe=sorted(ar.get('decode_probe', {})))
        del ck
        c = cand[tag]
        print(f'  fold {F} {tag}: {os.path.basename(c["path"])} cv {c["cv"]:.6f} (closing line {best[tag]:.6f}) '
              f'epoch {c["epoch1"]} {c["mod"]} sha {c["msha"]} ttr_size_t {c["tsz"]}', flush=True)
        if abs(c['cv'] - float(best[tag])) > 1e-12:
            return 'fail', f'{tag} checkpoint cv {c["cv"]} != closing line {best[tag]}'
    tag = 'ema' if cand['ema']['cv'] >= cand['raw']['cv'] else 'raw'
    c = cand[tag]
    oof = f'{run}/oof_{prefix}_{tag}.csv'
    if not os.path.isfile(oof):
        return 'fail', f'missing OOF {oof}'
    fmt_c, fmt_o = os.path.getmtime(c['path']), os.path.getmtime(oof)
    mt_c, mt_o = int(fmt_c), int(fmt_o)
    errs = []
    if c['mod'] != 'fleet_model_ttr': errs.append(f'model_mod {c["mod"]}')
    if c['msha'] != TTR_SHA: errs.append(f'model_sha {c["msha"]} != {TTR_SHA}')
    if not (c['tsz'] == c['ttr'] == 224): errs.append(f'ttr_size_t {c["tsz"]} args {c["ttr"]} (want 224)')
    if c['fold'] != F or c['seed'] != a.seed: errs.append(f'args fold/seed {c["fold"]}/{c["seed"]}')
    if c['train_labels'] != TRAIN_LABELS: errs.append(f'train_labels {c["train_labels"]}')
    if c['sel_labels'] != SEL_LABELS: errs.append(f'sel_labels {c["sel_labels"]}')
    if c['amp'] != 'fp16' or c['bf16']: errs.append(f'amp {c["amp"]} bf16 {c["bf16"]}')
    # VERIFY fix 10-05: save() writes the ckpt then the OOF ~13 ms later (MEASURED on 5 runs), so whole-second equality
    # failed ~1.3% of picks when a second boundary fell in between. Same save() call = OOF 0..5 s after the ckpt
    # (two save() calls of one tag are >= 1 epoch, ~10 min, apart).
    if not (0.0 <= fmt_o - fmt_c <= 5.0):
        errs.append(f'oof mtime - ckpt mtime = {fmt_o - fmt_c:.3f} s, not in [0, 5] (not the same save() call)')
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
               sel_labels=c['sel_labels'], decode_probe=c['probe'])
    if a.copy:
        os.makedirs(f'{a.dest}/ckpt', exist_ok=True); os.makedirs(f'{a.dest}/oof', exist_ok=True)
        for q in glob.glob(f'{a.dest}/ckpt/{prefix}_*_best.pt'):          # stale pick of this fold (our dir only)
            if os.path.basename(q) != base:
                os.remove(q); print(f'  removed stale {q}')
        dst = f'{a.dest}/ckpt/{base}'
        doof = f'{a.dest}/oof/oof_{prefix}_{tag}_cv{m.group(1)}.csv'
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
    ap.add_argument('--epochs', type=int, default=24)
    ap.add_argument('--copy', action='store_true')
    ap.add_argument('--dest', default=os.path.expanduser('~/k4_test'))
    a = ap.parse_args()
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
    print(f'K4PICK members {[m["fold"] for m in members]} skipped {[s["fold"] for s in skipped]} '
          f'failed {[f["fold"] for f in failed]}', flush=True)
    if failed:
        return 1
    return 0 if members else 3


if __name__ == '__main__':
    sys.exit(main())
