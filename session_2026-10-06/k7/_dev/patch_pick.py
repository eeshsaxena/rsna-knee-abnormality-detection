"""dev helper (not shipped): patch k7/box/k5_pick.py from the K6 copy (K7-TTR, K7-REC, K7-KEY)."""
import sys

p = sys.argv[1]
s = open(p, encoding='utf-8').read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:80], s.count(old))
    s = s.replace(old, new)


rep('"""k5_pick.py, K6 COPY (2026-10-06, Claude: + --train_labels so the TTR v4 set f{F}_s1042_ttr can be picked with the same checks; default unchanged) (box, CPU, read-only on ~/fleet_runs and ~/fleet_store; writes only under --dest, default ~/k6_test):',
    '''"""k5_pick.py, K7 COPY (2026-10-06 evening, Claude). Additions over the K6 copy (every default keeps the K5/K6 behaviour):
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
(K6 header: + --train_labels so the TTR v4 set f{F}_s1042_ttr can be picked with the same checks; default unchanged) (box, CPU, read-only on ~/fleet_runs and ~/fleet_store; writes only under --dest, default ~/k7_test):''')
rep('''    if a.no_kaggle_receipt:                                              # K6: the TTR v4 set is all box-trained
        return None, None
    cands = ([a.kaggle_receipt] if a.kaggle_receipt else []) + [f'{root}/kaggle_f34_run_receipt.json']
    for rp in cands:
        if rp and os.path.isfile(rp):
            rec = json.load(open(rp))
            folds = (rec.get('train') or {}).get('folds') or {}''',
    '''    if a.no_kaggle_receipt:                                              # K6: the TTR v4 set is all box-trained
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
            folds = (rec.get('train') or {}).get('folds') or {}''')
rep('''    if not (c['tsz'] == c['ttr'] == 224): errs.append(f'ttr_size_t {c["tsz"]} args {c["ttr"]} (want 224)')''',
    '''    if not (c['tsz'] == c['ttr'] == a.ttr_size): errs.append(f'ttr_size_t {c["tsz"]} args {c["ttr"]} (want --ttr_size {a.ttr_size})')   # K7-TTR''')
rep('''    if kag is None:
        # VERIFY fix''', '''    if kag is None and a.require_receipt:                                                                   # K7-REC
        errs.append(f'no Kaggle run receipt vouches for {os.path.basename(run)} (--require_receipt; receipts tried: '
                    f'{list(a.kaggle_receipt)} + {root}/kaggle_f34_run_receipt.json)')
    if kag is None:
        # VERIFY fix''')
rep('''               source='kaggle' if kag is not None else 'box', kaggle_receipt=kag_rp)''',
    '''               source='kaggle' if kag is not None else 'box', kaggle_receipt=kag_rp, suffix=a.suffix)          # K7-KEY''')
rep('''        for q in glob.glob(f'{a.dest}/ckpt/{prefix}_*_best.pt'):          # stale pick of this fold (our dir only)
            if os.path.basename(q) != base:''',
    '''        recipe = base.split('_')[3]                                      # K7-KEY: ttr224-320x24 / ttr288-320x24
        for q in glob.glob(f'{a.dest}/ckpt/{prefix}_*_best.pt'):          # stale pick of this fold (our dir only)
            qp = os.path.basename(q).split('_')
            if os.path.basename(q) != base and len(qp) > 3 and qp[3] == recipe:     # K7-KEY: never the other set's pick''')
rep('''        doof = f'{a.dest}/oof/oof_{prefix}_{tag}_cv{m.group(1)}.csv\'''',
    '''        tkey = '' if a.ttr_size == 224 else f't{a.ttr_size}'             # K7-KEY: 224 sets keep the old name
        doof = f'{a.dest}/oof/oof_{prefix}{tkey}_{tag}_cv{m.group(1)}.csv\'''')
rep('''    ap.add_argument('--kaggle_receipt', default=None)''',
    '''    ap.add_argument('--kaggle_receipt', action='append', default=[], help='K7-REC: repeatable')
    ap.add_argument('--require_receipt', action='store_true', help='K7-REC: a fold no receipt vouches for FAILS')
    ap.add_argument('--ttr_size', type=int, default=224, help='K7-TTR: expected ttr_size of this set (TTR288: 288)')''')
rep('''    ap.add_argument('--dest', default=os.path.expanduser('~/k6_test'))''',
    '''    ap.add_argument('--dest', default=os.path.expanduser('~/k7_test'))''')
rep('''~/k6_test/k5/box/k5_pick.py [--copy]''', '''~/k7_test/k5/box/k5_pick.py [--copy]''')
rep('''    a.extra_root = [os.path.expanduser(x) for x in a.extra_root]''',
    '''    a.extra_root = [os.path.expanduser(x) for x in a.extra_root]
    a.kaggle_receipt = [os.path.expanduser(x) for x in a.kaggle_receipt]''')
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched', p)
