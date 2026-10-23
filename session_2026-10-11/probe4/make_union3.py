#!/usr/bin/env python
# make_union3.py (2026-10-11, Claude): make_union2.py + --pl_cols (PROBE4). The listed NEVER labels (MCL, Contusion,
# Fracture) may carry pseudo-label soft targets on OAI rows: they must then be finite in [0, 1] on EVERY OAI row; the
# other NEVER labels stay NaN. Everything else is make_union2.py unchanged.
# make_union2.py (2026-10-09, box CPU only): make_union.py (md5 f431de57, UNCHANGED; it built the running T1 probe's
# inputs) generalised to several OAI visits and label tiers, each union in its OWN dirs:
#   --union  ~/union_store_<tag>/           symlinks: every ~/fleet_store/*.npz + every ready OAI npz of the label file
#   --outdir ~/oai_build/union_<tag>/       labels_frac75oai_<TAG>.csv  = labels_frac75_logit.csv BYTES verbatim + OAI rows
#                                           folds5_oai.csv              = folds5.csv BYTES verbatim + "oai_...,9"
#                                           union_receipt.json          md5s, counts per visit (the registered record)
# OAI row = id in --oai_labels with >= 1 finite label AND a row in its visit's manifest (converted_manifest.csv for
# V00, converted_manifest_vNN.csv for vNN) whose npz sha256 still matches AND the npz passes the store contract.
# Asserts (research_1008/D sec 4.3, verify_D F5, plus the visit ids): every OAI id matches oai_<digits>_<R|L>(_vNN)?;
# every OAI id sorts after every RSNA id; a follow-up id sorts right after its own V00 id when both exist;
# sorted(ids)[:8] (decode probe + parity npz that ship to Kaggle) are RSNA; no OAI id in folds 0-4, gold74 or v4 eval
# labels; MCL/Contusion/Fracture NaN on every OAI row; RSNA rows bit-identical through the trainer's own read path;
# ~/fleet_store holds no OAI file; every OAI npz symlink points into ~/oai_store.
# --min_frac_visit F: refuse (exit 3) unless, for every visit present in the label file, ready/labelled >= F.
# Refuses to touch a union dir that a running fleet_train.py uses as --store (exact path match, so the T1 probe's
# ~/union_store does not block a different union).
import os, re, sys, json, glob, time, hashlib, argparse, subprocess
import numpy as np, pandas as pd

H = os.path.expanduser('~')
sys.path.insert(0, f'{H}/fast_build/code2')
import fleet_common as fc
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
RSNA_LAB, RSNA_MD5 = f'{H}/novel/frac75/labels_frac75_logit.csv', 'ad7582b817243f8f22e8bcfdd7e3f84a'
FOLDS, GOLD, V4 = f'{H}/rsna_train/folds5.csv', f'{H}/rsna_train/gold74_eval_set.csv', f'{H}/rsna_train/llm_labels_v4_blend.csv'
OAI_STORE = f'{H}/oai_store'
IDRX = re.compile(r'^oai_\d+_[RL](_v\d\d)?$')
ap = argparse.ArgumentParser()
ap.add_argument('--oai_labels', required=True)
ap.add_argument('--tag', required=True)                   # e.g. T12v00 -> labels_frac75oai_T12v00.csv
ap.add_argument('--union', required=True)
ap.add_argument('--outdir', required=True)
ap.add_argument('--work', default=f'{H}/oai_build')       # where the per-visit manifests live
ap.add_argument('--min_oai', type=int, default=0)
ap.add_argument('--min_frac_visit', type=float, default=0.0)
ap.add_argument('--pl_cols', default='')                # e.g. MCL,Contusion,Fracture
A = ap.parse_args()
NEV = ['MCL', 'Contusion', 'Fracture']
PLC = [c for c in A.pl_cols.split(',') if c]
assert set(PLC) <= set(NEV), PLC
NAN3 = [c for c in NEV if c not in PLC]
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
visit_of = lambda k: k[-3:] if re.search(r'_v\d\d$', k) else 'v00'
phys = lambda k: k[:-4] if re.search(r'_v\d\d$', k) else k


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


union = os.path.abspath(A.union)
ps = subprocess.run(['pgrep', '-af', 'fleet_train.py'], capture_output=True, text=True).stdout
for u in {union, os.path.realpath(union)}:
    assert not re.search(r'--store\s+' + re.escape(u) + r'/?(\s|$)', ps), \
        f'a fleet_train.py process uses {u} as --store: refusing to rebuild it'
assert os.path.realpath(union) != os.path.realpath(f'{H}/fleet_store') and os.path.realpath(union) != os.path.realpath(f'{H}/union_store'), \
    'never rebuild ~/fleet_store or the T1 probe union ~/union_store here'
assert not glob.glob(f'{H}/fleet_store/oai_*'), 'OAI file inside ~/fleet_store'
assert md5(RSNA_LAB) == RSNA_MD5, 'frac75 label file changed'
for d in (union, A.outdir):
    os.makedirs(d, exist_ok=True); os.chmod(d, 0o700)

# ---- OAI rows: labelled AND converted AND npz unchanged since conversion
ol = pd.read_csv(A.oai_labels, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
assert list(ol.columns) == T, list(ol.columns)
ol = ol[ol.notna().any(axis=1)]
assert ol.index.is_unique and all(IDRX.match(k) for k in ol.index), [k for k in ol.index if not IDRX.match(k)][:5]
mans = {}
for v in sorted({visit_of(k) for k in ol.index}):
    p = f'{A.work}/converted_manifest.csv' if v == 'v00' else f'{A.work}/converted_manifest_{v}.csv'
    mans[v] = pd.read_csv(p, dtype=str).set_index('kid') if os.path.exists(p) else pd.DataFrame(columns=['npz_sha256'])
ok, bad, per_visit = [], [], {}
for k in ol.index:
    v = visit_of(k); man = mans[v]; p = f'{OAI_STORE}/{k}.npz'
    pv = per_visit.setdefault(v, dict(labelled=0, ready=0, sha_mismatch=0))
    pv['labelled'] += 1
    if k in man.index and os.path.isfile(p):
        if sha256(p) == man.at[k, 'npz_sha256']:
            with np.load(p, allow_pickle=False) as z:
                fc.check_store_contract(z, k)
            ok.append(k); pv['ready'] += 1
        else:
            bad.append(k); pv['sha_mismatch'] += 1
ol = ol.loc[ok]
for v, pv in sorted(per_visit.items()):
    pv['frac_ready'] = round(pv['ready'] / max(pv['labelled'], 1), 4)
print(f'OAI labelled {sum(p["labelled"] for p in per_visit.values())} | ready {len(ok)} | sha mismatch {len(bad)} | '
      f'per visit {json.dumps(per_visit)}', flush=True)
low = {v: pv['frac_ready'] for v, pv in per_visit.items() if pv['frac_ready'] < A.min_frac_visit}
if len(ok) < A.min_oai or low:
    print(f'REFUSED: ready {len(ok)} < --min_oai {A.min_oai} or visits below --min_frac_visit {A.min_frac_visit}: {low}'); sys.exit(3)
assert ol.index.str.startswith('oai_').all() and ol[NAN3].isna().all().all()
if PLC:
    _v = ol[PLC].to_numpy(float); assert np.isfinite(_v).all() and (_v >= 0).all() and (_v <= 1).all(), 'PL cols not finite in [0,1]'

# ---- union store (symlinks only)
for e in os.scandir(union):
    assert e.is_symlink(), f'non-symlink entry in {union}: {e.name}'
want = {os.path.basename(p): p for p in glob.glob(f'{H}/fleet_store/*.npz')}
assert not [n for n in want if n.startswith('oai_')]
want.update({f'{k}.npz': f'{OAI_STORE}/{k}.npz' for k in ol.index})
have = {e.name: os.readlink(e.path) for e in os.scandir(union)}
for n, t in have.items():
    if want.get(n) != t:
        os.unlink(f'{union}/{n}')
for n, t in want.items():
    if have.get(n) != t:
        os.symlink(t, f'{union}/{n}')
n_rsna_store = sum(1 for n in want if not n.startswith('oai_'))
assert all(os.readlink(e.path).startswith(OAI_STORE + '/') for e in os.scandir(union) if e.name.startswith('oai_'))


# ---- labels / folds: RSNA bytes verbatim + OAI rows
def append_rows(src, dst, header_want, rows):
    b = open(src, 'rb').read()
    nl = b'\r\n' if b'\r\n' in b[:4096] else b'\n'
    head = b.split(nl, 1)[0].decode()
    assert head == header_want, (head, header_want)
    if not b.endswith(nl):
        b += nl
    tmp = dst + '.part'
    with open(tmp, 'wb') as f:
        f.write(b + nl.join(r.encode() for r in rows) + nl)
    os.replace(tmp, dst)
    return nl


fmt = lambda v: '' if not np.isfinite(v) else format(float(v), '.6g')
lab_rows = [k + ',' + ','.join(fmt(v) for v in ol.loc[k, T].to_numpy(float)) for k in ol.index]
LAB = f'{A.outdir}/labels_frac75oai_{A.tag}.csv'; FO = f'{A.outdir}/folds5_oai.csv'
nl = append_rows(RSNA_LAB, LAB, 'StudyInstanceUID,' + ','.join(T), lab_rows)
append_rows(FOLDS, FO, 'StudyInstanceUID,fold', [f'{k},9' for k in ol.index])

# ---- verification through the trainer's own read path (fleet_train.py l.129-158)
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
lab, lab0 = rd(LAB), rd(RSNA_LAB)
folds = rd(FO)['fold']; folds0 = rd(FOLDS)['fold']
assert lab.index.is_unique and folds.index.is_unique
assert list(lab.index[:len(lab0)]) == list(lab0.index) and list(folds.index[:len(folds0)]) == list(folds0.index)
a, b = lab.loc[lab0.index, T].to_numpy(np.float64), lab0[T].to_numpy(np.float64)
assert np.array_equal(a, b) and np.array_equal(a.astype(np.float32), b.astype(np.float32)), 'RSNA label rows changed'
assert (folds.loc[folds0.index] == folds0).all()
oai = [s for s in lab.index if s.startswith('oai_')]
assert len(oai) == len(ol) and set(oai) == set(folds.index[folds == 9]) and not (set(folds0.index) & set(oai))
assert np.allclose(lab.loc[oai, T].to_numpy(float), ol[T].to_numpy(float), atol=5e-6, equal_nan=True)
assert lab.loc[oai, NAN3].isna().all().all()
if PLC:
    assert np.isfinite(lab.loc[oai, PLC].to_numpy(float)).all()
assert max(lab0.index) < min(oai), 'an OAI id would sort before an RSNA id'
so = sorted(oai); pos = {k: i for i, k in enumerate(so)}
for k, i in pos.items():                         # a follow-up id sorts right after its own knee's V00 id (beside it)
    if visit_of(k) != 'v00' and phys(k) in pos:
        j = pos[phys(k)]
        assert j < i <= j + 4 and all(phys(s) == phys(k) for s in so[j:i + 1]), (k, i, j)
excl = set(rd(GOLD).index); v4 = set(rd(V4).index)
assert not (excl | v4) & set(oai)
held = set(folds.index[folds < 0]) | excl
havef = {s for s in lab.index if os.path.exists(f'{union}/{s}.npz')}
ids = [s for s in lab.index if s in folds.index and s in havef]
assert not [s for s in sorted(ids)[:8] if s.startswith('oai_')], 'decode-probe / parity ids would include OAI'
assert all(s.startswith('1.2.826.') for s in sorted(ids)[:8])
per_fold = {}
for F in range(5):
    tr = [s for s in ids if s not in held and folds[s] != F]
    va = [s for s in ids if s not in held and folds[s] == F]
    assert not [s for s in va if s.startswith('oai_')]
    n_want = sum(1 for s in lab.index if s in folds.index and s not in held)
    assert len(tr) + len(va) >= 0.99 * n_want
    per_fold[F] = dict(train=len(tr), train_oai=sum(s.startswith('oai_') for s in tr), val=len(va))
print('trainer view per fold:', per_fold, flush=True)
lab_oai = lab.loc[oai, T]
rec = dict(ts=time.strftime('%Y-%m-%d %H:%M:%S'), tag=A.tag, labels=LAB, labels_md5=md5(LAB), folds=FO, folds_md5=md5(FO),
           oai_labels=os.path.abspath(A.oai_labels), oai_labels_md5=md5(A.oai_labels), union=union,
           n_rsna_store=n_rsna_store, n_oai=len(oai), n_oai_sha_mismatch=len(bad), per_visit=per_visit,
           n_phys_knees=len({phys(k) for k in oai}), per_fold=per_fold, line_ending=repr(nl),
           oai_finite_cells_per_label={c: int(lab_oai[c].notna().sum()) for c in T},
           decode_probe_ids=sorted(ids)[:4], first8_rsna=True, script_md5=md5(os.path.abspath(__file__)), pl_cols=PLC,
           store_contract=fc.STORE_CONTRACT)
json.dump(rec, open(f'{A.outdir}/union_receipt.json', 'w'), indent=1)
print(json.dumps({k: rec[k] for k in ('tag', 'labels_md5', 'folds_md5', 'n_rsna_store', 'n_oai', 'n_phys_knees',
                                      'oai_finite_cells_per_label')}), flush=True)
print('ALL ASSERTIONS PASSED', flush=True)
