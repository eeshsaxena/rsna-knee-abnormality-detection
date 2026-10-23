# pack_check.py (rental_kc, BOX, CPU, read only): the checks of pack.sh, fail loud. No em dashes.
#   oai BUNDLE                    OAI assertion over the staged bundle: no oai / union in any name, path, csv cell or code
#                                 text; symlinks only inside hf/ and pointing inside the bundle; the store = the 4,407 RSNA
#                                 studies (RSNA UID pattern) == folds5 + gold74 ids; every csv id is an RSNA UID inside
#                                 folds5 + gold74; folds5 / gold74 md5 == PREREG
#   tables TB DREAD --ready LABELS_READY --dread_receipt R
#                                 T_b = the 10-14 verified table (md5 d206bbcf, its sha256 listed in final/LABELS_READY; its
#                                 provenance v4 + OAI-free fleet OOF + R7 OOF was asserted by the rental_student packer);
#                                 dread = make_dread_labels.py output of the RSNA-report soft labels (receipt src md5
#                                 96e6723f), same 12 columns as T_b, values in [0, 1] or empty, ids inside folds5 + gold74
# exit 0 = PASS (one PASS line per check), 3 = STOP (prints the reasons)
import argparse, csv, hashlib, json, os, re, sys
UID = re.compile(r'^1\.2\.826\.0\.1\.3680043\.8\.498\.\d+$')
OAI = re.compile(r'oai|union', re.I)
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
          "Baker's", 'Contusion', 'Fracture']
TB_MD5 = 'd206bbcf40694a153fd78d1a898b4db6'
DREAD_SRC_MD5 = '96e6723f7dcc997f8951060a23dd22cf'
FOLDS_MD5 = '9c5117e06e5c6ecc77aabbeeca5db44e'
GOLD_MD5 = 'a0ff70f7af1d458b9cfe9509e6122740'
ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('paths', nargs='+')
ap.add_argument('--subset', action='store_true'); ap.add_argument('--ready', default=''); ap.add_argument('--dread_receipt', default='')
ap.add_argument('--folds', default=os.path.expanduser('~/rsna_train/folds5.csv'))
ap.add_argument('--gold', default=os.path.expanduser('~/rsna_train/gold74_eval_set.csv'))
A = ap.parse_args()
bad = []
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def rows(p):
    with open(p, newline='') as fh:
        return list(csv.DictReader(fh))


if A.cmd == 'oai':
    B = os.path.realpath(A.paths[0])
    for d, ds, fs in os.walk(B):
        for f in ds + fs:
            p = os.path.join(d, f); rel = os.path.relpath(p, B)
            if OAI.search(rel):
                bad.append('name ' + rel)
            if os.path.islink(p):
                if not rel.startswith('hf/'):
                    bad.append('symlink outside hf/: ' + rel)
                elif not os.path.realpath(p).startswith(B + os.sep):
                    bad.append('hf symlink leaves the bundle: ' + rel)
            elif os.path.isfile(p) and OAI.search(os.path.realpath(p)):
                bad.append('real path mentions oai/union: ' + os.path.realpath(p))
    store = sorted(os.listdir(f'{B}/kc_store/store'))
    sids = [f[:-4] for f in store if f.endswith('.npz')]
    if len(sids) != len(store):
        bad.append('non-npz file in kc_store/store')
    bad += ['store id ' + s for s in sids if not UID.match(s)][:5]
    if len(sids) != 4407 and not A.subset:
        bad.append(f'store holds {len(sids)} studies, want the 4,407 RSNA studies')
    fo = {r['StudyInstanceUID'] for r in rows(f'{B}/rsna_train/folds5.csv')}
    go = {r['StudyInstanceUID'] for r in rows(f'{B}/rsna_train/gold74_eval_set.csv')}
    if (set(sids) != (fo | go)) if not A.subset else (set(sids) - (fo | go)):
        bad.append(f'store ids != folds5 + gold74 ids ({len(set(sids) ^ (fo | go))} differ)')
    ncsv = 0
    for d, _, fs in os.walk(B):
        for f in fs:
            if not f.endswith('.csv'):
                continue
            p = os.path.join(d, f); rel = os.path.relpath(p, B); ncsv += 1
            if OAI.search(open(p, errors='replace').read()):
                bad.append('csv content mentions oai/union: ' + rel)
            ids = [r.get('StudyInstanceUID', '') for r in rows(p)]
            bad += [f'{rel} id {s[:40]}' for s in ids if not UID.match(s)][:5]
            if set(ids) - (fo | go):
                bad.append(f'{rel}: {len(set(ids) - (fo | go))} ids outside folds5 + gold74')
    for d, _, fs in os.walk(f'{B}/fast_build'):
        for f in fs:
            if not f.endswith('.py'):
                bad.append('non-py file in code: ' + f)
            elif re.search(r'oai|union_store', open(os.path.join(d, f), errors='replace').read(), re.I):
                bad.append('code mentions oai: ' + f)
    for c, want in (('rsna_train/folds5.csv', FOLDS_MD5), ('rsna_train/gold74_eval_set.csv', GOLD_MD5)):
        if md5(f'{B}/{c}') != want:
            bad.append(f'{c} md5 {md5(f"{B}/{c}")} != PREREG {want}')
    if not bad:
        print(f'OAI_ASSERT_OK: {len(sids)} RSNA studies (UID pattern) == folds5 + gold74 ids; {ncsv} csv files: every id an RSNA UID '
              f'inside folds5 + gold74, no oai/union cell; no oai/union name, path or code reference; symlinks only inside hf/')
elif A.cmd == 'tables':
    TB, DR = A.paths[0], A.paths[1]
    fo = {r['StudyInstanceUID'] for r in rows(A.folds)}; go = {r['StudyInstanceUID'] for r in rows(A.gold)}
    if md5(TB) != TB_MD5:
        bad.append(f'T_b md5 {md5(TB)} != the 10-14 verified {TB_MD5}')
    rd = open(A.ready, errors='replace').read() if A.ready and os.path.exists(A.ready) else ''
    if sha(TB) not in rd:
        bad.append('T_b sha256 not listed in ' + (A.ready or 'LABELS_READY'))
    tb = rows(TB)
    hdr_tb = list(tb[0].keys()) if tb else []
    if [c for c in hdr_tb if c in LABELS] != LABELS:
        bad.append('T_b label columns')
    try:
        rc = json.load(open(A.dread_receipt))
    except Exception as e:
        rc = {}; bad.append(f'dread receipt unreadable: {A.dread_receipt} ({type(e).__name__})')
    if rc.get('src_md5') != DREAD_SRC_MD5:
        bad.append(f"dread receipt src_md5 {rc.get('src_md5')} != {DREAD_SRC_MD5}")
    if rc.get('out_md5') and rc.get('out_md5') != md5(DR):
        bad.append(f"dread table md5 {md5(DR)} != its receipt out_md5 {rc.get('out_md5')}")
    dr = rows(DR)
    hdr = list(dr[0].keys()) if dr else []
    if [c for c in hdr if c in LABELS] != LABELS:
        bad.append('dread label columns != the T_b columns')
    ids = [r['StudyInstanceUID'] for r in dr]
    if len(ids) != len(set(ids)):
        bad.append('dread duplicate ids')
    if set(ids) - (fo | go):
        bad.append(f'dread: {len(set(ids) - (fo | go))} ids outside folds5 + gold74')
    pool = {s for s in fo} - go
    nv = 0
    for r in dr:
        for c in LABELS:
            v = r[c].strip()
            if v in ('', 'nan', 'NaN'):
                continue
            try:
                x = float(v)
            except ValueError:
                bad.append(f'dread non-numeric cell {c}'); break
            if not 0.0 <= x <= 1.0:
                nv += 1
    if nv:
        bad.append(f'dread: {nv} cells outside [0, 1]')
    cov = len(pool & set(ids))
    if not bad:
        print(f'TABLES_OK: T_b md5 {TB_MD5[:8]} (sha256 {sha(TB)[:16]} listed in LABELS_READY) | dread md5 {md5(DR)[:8]} sha256 '
              f'{sha(DR)[:16]} rows {len(ids)}, train pool covered {cov}/{len(pool)}, src md5 {DREAD_SRC_MD5[:8]} (RSNA report soft labels)')
else:
    bad.append('unknown command ' + A.cmd)
if bad:
    print('STOP: ' + ' | '.join(bad[:12]))
    sys.exit(3)
