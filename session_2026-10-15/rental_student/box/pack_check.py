# pack_check.py (BOX, CPU, read only): the checks of pack.sh, fail loud. No em dashes.
#   oai BUNDLE [--subset]        OAI assertion over the staged bundle (names, store ids, csv cells, code text, symlinks)
#   table TABLE --mode REAL|DRY [--ready LABELS_READY] [--receipt R] [--fleet_receipt F] [--real_copy P] [--verdict V]
#                                T_b table content + provenance: built only from v4 + the OAI-FREE fleet OOF (C, D, E, E2,
#                                D2; md5 3479835c) + R7 OOF per PREREG_TEACHER; any other / OAI-trained component -> STOP
# exit 0 = PASS (prints one PASS line per check), 3 = STOP (prints the reasons)
import argparse, csv, hashlib, json, os, re, sys
UID = re.compile(r'^1\.2\.826\.0\.1\.3680043\.8\.498\.\d+$')
OAI = re.compile(r'oai|union', re.I)
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
          "Baker's", 'Contusion', 'Fracture']
FLEET_OOF_MD5 = '3479835c4056e86a74511ad68e9abb48'          # PREREG_TEACHER sec 2: OAI-free fleet mean of C, D, E, E2, D2
FLEET_GOLD_MD5 = 'ba7956c85c6ec57d08b9e3cc18a2203b'         # PREREG_TEACHER sec 2: the same 25 members' gold74 preds (T_b gold rows)
BUILDER_MD5 = 'a7f8823d7216fb0a55d5fc7821607c78'            # PREREG_TEACHER A1 code freeze (A2: unchanged)
V4_MD5 = '44874d8ded7699d356b935ca8d7e8df7'
FOLDS_MD5 = '9c5117e06e5c6ecc77aabbeeca5db44e'
GOLD_MD5 = 'a0ff70f7af1d458b9cfe9509e6122740'
FLEET_SETS = {'C': 's1044_ttr288mix50', 'D': 's1043_ttr_frac75', 'E': 's1045_ttr288frac75', 'E2': 's1046_ttr288frac75', 'D2': 's1044_ttr_frac75'}
ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('path')
ap.add_argument('--subset', action='store_true'); ap.add_argument('--mode', default='DRY')
ap.add_argument('--ready', default=''); ap.add_argument('--receipt', default=''); ap.add_argument('--fleet_receipt', default='')
ap.add_argument('--real_copy', default=''); ap.add_argument('--verdict', default=''); ap.add_argument('--table_name', default='labels_r7t_b.csv')
A = ap.parse_args()
bad = []
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def ids(p):
    with open(p, newline='') as fh:
        return [r['StudyInstanceUID'] for r in csv.DictReader(fh)]


if A.cmd == 'oai':
    B = os.path.realpath(A.path)
    for d, ds, fs in os.walk(B):
        for f in ds + fs:
            p = os.path.join(d, f)
            if OAI.search(os.path.relpath(p, B)):
                bad.append('name ' + os.path.relpath(p, B))
            if os.path.islink(p):
                t = os.path.realpath(p)
                if not os.path.relpath(p, B).startswith('hf/'):
                    bad.append('symlink outside hf/: ' + os.path.relpath(p, B))
                elif not t.startswith(B + os.sep):
                    bad.append('hf symlink leaves the bundle: ' + os.path.relpath(p, B))
            elif os.path.isfile(p):
                r = os.path.realpath(p)
                if OAI.search(r):
                    bad.append('real path mentions oai/union: ' + r)
    store = sorted(os.listdir(f'{B}/fleet_store'))
    sids = [f[:-4] for f in store if f.endswith('.npz')]
    if len(sids) != len(store):
        bad.append('non-npz file in fleet_store')
    bad += ['store id ' + s for s in sids if not UID.match(s)][:5]
    if not A.subset and len(sids) != 4407:
        bad.append(f'store holds {len(sids)} studies, want the 4,407 RSNA studies')
    for c in ('r7t_labels/real/labels_r7t_b.csv', 'rsna_train/folds5.csv', 'rsna_train/gold74_eval_set.csv'):
        t = open(f'{B}/{c}', errors='replace').read()
        if OAI.search(t):
            bad.append('content ' + c)
        bad += [f'{c} id {s}' for s in ids(f'{B}/{c}') if not UID.match(s)][:5]
    tab = ids(f'{B}/r7t_labels/real/labels_r7t_b.csv')
    fo = set(ids(f'{B}/rsna_train/folds5.csv')); go = set(ids(f'{B}/rsna_train/gold74_eval_set.csv'))
    if set(sids) - (fo | go):
        bad.append(f'{len(set(sids) - (fo | go))} store ids outside folds5 + gold74')
    if A.subset:
        if set(sids) - set(tab):
            bad.append('subset store ids missing from the table')
    elif set(tab) != set(sids):
        bad.append(f'table ids != store ids ({len(set(tab) ^ set(sids))} differ)')
    for d, _, fs in os.walk(f'{B}/fast_build'):
        for f in fs:
            if not f.endswith('.py'):
                bad.append('non-py file in code: ' + f)
            elif re.search(r'oai|union_store', open(os.path.join(d, f), errors='replace').read(), re.I):
                bad.append('code mentions oai: ' + f)
    for c, want in (('rsna_train/folds5.csv', FOLDS_MD5), ('rsna_train/gold74_eval_set.csv', GOLD_MD5)):
        if md5(f'{B}/{c}') != want:
            bad.append(f'{c} md5 {md5(f"{B}/{c}")} != PREREG {want}')
    if bad:
        print('STOP OAI ASSERTION FAILED:', bad[:10]); sys.exit(3)
    print(f'PASS OAI_ASSERT_OK: {len(sids)} RSNA studies (UID pattern, all in folds5 + gold74), table ids '
          f'{"cover the subset" if A.subset else "== store ids"} ({len(tab)}), no oai / union name, real path, csv cell or '
          f'code reference, symlinks only inside hf/; folds5 + gold74 md5 == PREREG')
    sys.exit(0)

if A.cmd == 'table':
    T = A.path; real = A.mode == 'REAL'
    with open(T, newline='') as fh:
        rd = csv.reader(fh); hdr = next(rd); rows = list(rd)
    if hdr != ['StudyInstanceUID'] + LABELS:
        bad.append(f'header {hdr[:3]}... != StudyInstanceUID + the 12 labels')
    if len(rows) != 4407:
        bad.append(f'{len(rows)} rows, want 4,407')
    if len({r[0] for r in rows}) != len(rows):
        bad.append('duplicate ids')
    bad += [f'id {r[0]}' for r in rows if not UID.match(r[0])][:5]
    nn = 0
    for r in rows:
        for v in r[1:]:
            if v in ('', 'nan', 'NaN'):
                nn += 1; continue
            x = float(v)
            if not (0.0 <= x <= 1.0):
                bad.append(f'value {x} outside [0, 1]'); break
    if OAI.search(open(T, errors='replace').read()):
        bad.append('table text mentions oai / union')
    tsha, tmd5 = sha(T), md5(T)
    # LABELS_READY: a line naming the table must carry its sha256 (64 hex)
    if A.ready:
        lines = [l for l in open(A.ready, errors='replace').read().splitlines() if os.path.basename(T) in l or A.table_name in l]
        hx = [h for l in lines for h in re.findall(r'\b[0-9a-f]{64}\b', l)]
        if not hx:
            bad.append(f'LABELS_READY has no sha256 line for {os.path.basename(T)}')
        elif tsha not in hx:
            bad.append(f'table sha256 {tsha[:16]} != LABELS_READY {hx[0][:16]}')
    elif real:
        bad.append('REAL mode needs LABELS_READY')
    # provenance from the label build receipt (PREREG_TEACHER sec 2-3)
    if A.receipt:
        r = json.load(open(A.receipt))
        tb = r.get('tables', {}).get('b', {})
        if {k: round(float(v), 6) for k, v in tb.items()} != {'v4': 0.4, 'fleet': 0.3, 'r7': 0.3}:
            bad.append(f'receipt T_b weights {tb} != v4 .4 / fleet .3 / r7 .3')
        inp = r.get('inputs', {})
        if inp.get('fleet_oof', {}).get('md5') != FLEET_OOF_MD5:
            bad.append(f'fleet OOF md5 {inp.get("fleet_oof", {}).get("md5")} != the OAI-free fleet mean {FLEET_OOF_MD5}')
        if inp.get('fleet_gold', {}).get('md5') != FLEET_GOLD_MD5:   # review 10-14: the gold rows of T_b carry the fleet gold mean
            bad.append(f'fleet gold md5 {inp.get("fleet_gold", {}).get("md5")} != the OAI-free 25-member gold mean {FLEET_GOLD_MD5}')
        if inp.get('v4', {}).get('md5') != V4_MD5:
            bad.append('v4 md5 differs from PREREG')
        extra = sorted(set(inp) - {'v4', 'frac75', 'folds', 'gold', 'train_csv', 'fleet_oof', 'fleet_gold', 'teacher'})
        if extra:
            bad.append(f'receipt lists unexpected inputs {extra}')
        tp = str(inp.get('teacher', {}).get('path', ''))
        if OAI.search(json.dumps(inp)):
            bad.append('receipt inputs mention oai / union')
        rb = r.get('table_b', {})
        if rb.get('sha256') and rb.get('sha256') != tsha and rb.get('md5') != tmd5:
            # the receipt may record the table under its build name; equal bytes are what matters
            bad.append(f'receipt table_b sha256 {str(rb.get("sha256"))[:16]} != table {tsha[:16]}')
        if real:
            if not r.get('real') or str(r.get('teacher_tag', '')).startswith('SIM'):
                bad.append('REAL mode: the receipt is a simulation')
            if 'SIM' in tp:
                bad.append('REAL mode: the teacher input is a SIM file')
            if r.get('builder_md5') != BUILDER_MD5:
                bad.append(f'REAL mode: receipt builder md5 {r.get("builder_md5")} != PREREG A1 frozen {BUILDER_MD5}')
            # the PREREG sha the build ran under == the one finalize_labels.sh published in LABELS_READY (A2)
            pw = re.findall(r'^prereg_sha256 ([0-9a-f]{64})\s*$', open(A.ready, errors='replace').read(), re.M) if A.ready else []
            if not pw or r.get('prereg_sha256') != pw[0]:
                bad.append(f'REAL mode: receipt prereg_sha256 {str(r.get("prereg_sha256"))[:12]} != LABELS_READY prereg_sha256 {(pw or ["none"])[0][:12]}')
            if r.get('label_gate', {}).get('train_table') not in (A.table_name, os.path.basename(T)):
                bad.append(f'label gate chose {r.get("label_gate", {}).get("train_table")}, not the T_b table')
        prov = f'receipt {os.path.basename(A.receipt)}: T_b = .4 v4 + .3 fleet(md5 {FLEET_OOF_MD5[:8]}) + .3 R7 ({tp}), teacher_tag {r.get("teacher_tag")}, real {r.get("real")}'
    else:
        if real:
            bad.append('REAL mode needs the label build receipt')
        prov = 'no receipt (DRY)'
    if A.fleet_receipt:
        fr = json.load(open(A.fleet_receipt)); sets = fr.get('sets', {})
        if set(sets) != set(FLEET_SETS):
            bad.append(f'fleet receipt sets {sorted(sets)} != the OAI-free C, D, E, E2, D2')
        for k, v in sets.items():
            if OAI.search(json.dumps(v)) or re.search(r'(^|_)(dt|l13)(_|$)', str(v.get('tag', '')), re.I):
                bad.append(f'fleet set {k} tag {v.get("tag")} looks OAI-trained')
            if FLEET_SETS.get(k) and v.get('tag') != FLEET_SETS[k]:
                bad.append(f'fleet set {k} tag {v.get("tag")} != {FLEET_SETS[k]}')
        fm, fg = fr.get('fleet_mean', {}), fr.get('fleet_gold_mean', {})   # review 10-14: link build inputs <-> this receipt
        if sorted(fm.get('members', [])) != sorted(FLEET_SETS):
            bad.append(f'fleet receipt mean members {fm.get("members")} != C, D, E, E2, D2')
        if A.receipt:
            ri = json.load(open(A.receipt)).get('inputs', {})
            if not fm.get('sha256') or fm.get('sha256') != ri.get('fleet_oof', {}).get('sha256'):
                bad.append('the fleet OOF the build used is not the mean of the fleet receipt (OAI-free sets)')
            if not fg.get('sha256') or fg.get('sha256') != ri.get('fleet_gold', {}).get('sha256'):
                bad.append('the fleet gold the build used is not the gold mean of the fleet receipt (OAI-free sets)')
        if not bad:
            prov += f' | fleet sets {", ".join(f"{k}={v.get(chr(116) + chr(97) + chr(103))}" for k, v in sets.items())}'
    elif real:
        bad.append('REAL mode needs the fleet OOF receipt')
    if A.real_copy and os.path.isfile(A.real_copy) and sha(A.real_copy) != tsha:
        bad.append(f'{A.real_copy} exists with different bytes than the final table: ambiguous, STOP')
    if A.verdict and os.path.isfile(A.verdict):
        v = open(A.verdict).read().strip()
        if not v.startswith('PASS'):
            bad.append(f'labels_gate verdict is not PASS: {v[:120]}')
    elif real:
        bad.append(f'REAL mode needs the labels_gate verdict file ({A.verdict or "none given"})')
    if bad:
        print('STOP T_b TABLE CHECK FAILED:', bad[:10]); sys.exit(3)
    print(f'PASS TABLE_OK ({A.mode}): {os.path.basename(T)} sha256 {tsha[:16]} md5 {tmd5[:12]}, 4,407 RSNA rows x 12 labels in '
          f'[0, 1] ({nn} empty cells), no oai text | {prov}')
    sys.exit(0)
print('usage: see header'); sys.exit(2)
