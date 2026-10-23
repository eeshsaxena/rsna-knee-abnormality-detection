#!/usr/bin/env python3
"""k3_pick_ttr.py (box, CPU, read-only on ~/fleet_runs; writes only ~/k3_test/ckpt and ~/k3_test/oof): the final TTR
member for K3 = M2.

  CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/kmimic/bin/python -W ignore ~/k3_test/k3/box/k3_pick_ttr.py [--run DIR] [--copy]

FINAL only when the trainer printed its closing {"raw": cv, "ema": cv} line (fleet_train.py prints it after the LAST
save(); save() deletes the old best file before writing the new one, so files alone are never proof) AND the log holds
the last epoch's raw and ema lines. Until then: exit 3 PROVISIONAL, nothing copied.
Tag rule (K2 RUNBOOK sec 3.1, one tag per run): the tag whose best checkpoint has the higher val cv (read exactly from
the checkpoint, must equal the closing line); a tie goes to EMA.
Checks: the chosen best file is the only <tag> best file, its stored cv == the closing line, its OOF csv mtime == the
checkpoint mtime to the second (same save() call), model_mod fleet_model_ttr, model_sha f141aeed136dfc01, ttr_size buffer.
--copy: cp -p the checkpoint to ~/k3_test/ckpt/ and the OOF to ~/k3_test/oof/oof_f0_s1042_<tag>_cv<NNNN>.csv, re-check
sha256 of the copy. Prints SHIP <file> and the laptop scp line. Exit 0 = final (+ copied with --copy), 3 = not final,
1 = a check failed (do not ship).
"""
import argparse, glob, hashlib, json, os, re, shutil, sys

TTR_SHA = 'f141aeed136dfc01'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=os.path.expanduser('~/fleet_runs/f0_s1042_ttr'))
    ap.add_argument('--log', default=None)
    ap.add_argument('--epochs', type=int, default=24)
    ap.add_argument('--copy', action='store_true')
    ap.add_argument('--prefix', default='f0_s1042')
    ap.add_argument('--dest', default=os.path.expanduser('~/k3_test'))
    a = ap.parse_args()
    log = a.log or a.run.rstrip('/') + '.log'
    lines = [l.strip() for l in open(log, encoding='utf-8', errors='replace') if l.strip()]
    last = a.epochs - 1
    have = {t: any(l.startswith(f'ep {last} {t} ') for l in lines) for t in ('raw', 'ema')}
    best = None
    if lines and lines[-1].startswith('{'):
        try:
            best = json.loads(lines[-1])
        except Exception:
            best = None
    print(f'log {log}: last line {lines[-1][:120] if lines else None!r}; ep {last} raw/ema logged {have}')
    if not (isinstance(best, dict) and set(best) >= {'raw', 'ema'} and all(have.values())):
        print('PROVISIONAL: no closing {"raw": .., "ema": ..} line after the last epoch yet; re-run later, nothing copied')
        return 3
    import torch
    cand = {}
    for tag in ('raw', 'ema'):
        fs = sorted(glob.glob(f'{a.run}/{a.prefix}_{tag}_*_best.pt'))
        if len(fs) != 1:
            print(f'FAIL: expected exactly one {tag} best file, found {fs}'); return 1
        ck = torch.load(fs[0], map_location='cpu', weights_only=False)
        ar = ck['args']
        cand[tag] = dict(path=fs[0], cv=float(ck['cv']), epoch1=int(ck['epoch']) + 1, mod=ar.get('model_mod'),
                         msha=ar.get('model_sha'), tsz=int(ck['model'].get('ttr_size_t', -1)), ttr=int(ar.get('ttr_size') or -1))
        del ck
        c = cand[tag]
        print(f'{tag}: {os.path.basename(c["path"])} cv {c["cv"]:.6f} (closing line {best[tag]:.6f}) epoch {c["epoch1"]} '
              f'{c["mod"]} sha {c["msha"]} ttr_size_t {c["tsz"]}')
        if abs(c['cv'] - float(best[tag])) > 1e-12:
            print(f'FAIL: {tag} checkpoint cv != closing line'); return 1
    tag = 'ema' if cand['ema']['cv'] >= cand['raw']['cv'] else 'raw'
    c = cand[tag]
    oof = f'{a.run}/oof_{a.prefix}_{tag}.csv'
    mt_c, mt_o = int(os.path.getmtime(c['path'])), int(os.path.getmtime(oof))
    ok = (c['mod'] == 'fleet_model_ttr' and c['msha'] == TTR_SHA and c['tsz'] == c['ttr'] == 224 and mt_c == mt_o)
    if not ok:
        print(f'  model_mod {c["mod"]} (want fleet_model_ttr), model_sha {c["msha"]} (want {TTR_SHA}), ttr_size_t {c["tsz"]} '
              f'args {c["ttr"]} (want 224), mtimes equal {mt_c == mt_o}')
    print(f'pick {tag} (raw {cand["raw"]["cv"]:.6f} vs ema {cand["ema"]["cv"]:.6f}; tie -> ema) | ckpt mtime {mt_c} oof mtime {mt_o} '
          f'| checks {"OK" if ok else "FAILED"}')
    if not ok:
        return 1
    base = os.path.basename(c['path'])
    m = re.search(r'_cv(\d+)_best\.pt$', base)
    if a.copy:
        os.makedirs(f'{a.dest}/ckpt', exist_ok=True); os.makedirs(f'{a.dest}/oof', exist_ok=True)
        dst = f'{a.dest}/ckpt/{base}'
        doof = f'{a.dest}/oof/oof_{a.prefix}_{tag}_cv{m.group(1)}.csv'
        shutil.copy2(c['path'], dst); shutil.copy2(oof, doof)
        s1, s2 = sha256(c['path']), sha256(dst)
        if s1 != s2 or int(os.path.getmtime(c['path'])) != mt_c:
            print('FAIL: the source changed during the copy (a save() raced it); re-run'); return 1
        print(f'copied {dst} (sha256 {s2[:16]}) and {doof}')
    print(f'SHIP {base} epoch {c["epoch1"]} cv {c["cv"]:.6f} tag {tag}')
    print(f'laptop: scp -o BatchMode=yes training@192.168.10.30:k3_test/ckpt/{base} "$SP/k3/ckpt/"')
    return 0


if __name__ == '__main__':
    sys.exit(main())
