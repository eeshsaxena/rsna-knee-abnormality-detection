# kc_side_validate.py (box CPU only, read-only unless --install): parse + validate the kc_side v2 kernel output
# (eeshsaxena/rsna-kc-side -> kc_side_train.csv) before it becomes ~/kc_store/side_train.csv.
# Integrity checks (any FAIL -> exit 1, no install):
#   I1 schema: v1 columns first and unchanged, the v2 columns present, 4,407 unique studies = train_series.csv studies
#   I2 side rule re-derived from (mean_x, tag) matches side / rule on every row
#   I3 slot series re-picked with the VERBATIM d96 reader (sha 30aa72fd, from ~/raptor2/prep_kernel_s0.py) = s0..s4_uid
#   I4 d96 mask (~/raptor_store/rsna_index.csv): a KC slot block with pixels must have a kernel series with files
#   I5 458 overlap vs v1 box table (~/kcrop_probe/kc_side_train.csv): n, side, rule, tag equal; mean_x / sag_nx within 1e-6
#   I6 (with --box_csv) 458 overlap vs the SAME v2 code run on the box against ~/eda_data: every column equal
#   I7 derived columns re-computed: s{k}_ok, s{k}_flip, weak, geo_ok, serve
#   I8 gate: side U rate <= 2 %
# Reported, not failing: GEO_ASSERT (every present KC-slot series has the expected sign, sagittal normal x < 0,
# coronal / axial row x > 0; unexpected ones are served as both canons by construction), presence counts, weak zone,
# lat458 header side agreement, kernel summary cross-check, sha256 of the CSV.
import os, sys, re, json, argparse, hashlib, shutil, time
import numpy as np, pandas as pd

H = os.path.expanduser('~')
ap = argparse.ArgumentParser()
ap.add_argument('--csv', required=True, help='kc_side_train.csv fetched from the Kaggle kernel')
ap.add_argument('--summary', default='', help='kc_side_summary.json from the kernel (optional cross-check)')
ap.add_argument('--box_csv', default=f'{H}/kcrop_probe/v2/box_out/kc_side_train.csv')
ap.add_argument('--ref', default=f'{H}/kcrop_probe/kc_side_train.csv')
ap.add_argument('--lat', default=f'{H}/kcrop_probe/lat458.csv')
ap.add_argument('--series', default=f'{H}/eda_data/rsna-knee-abnormality-detection/train_series.csv')
ap.add_argument('--index', default=f'{H}/raptor_store/rsna_index.csv')
ap.add_argument('--reader', default=f'{H}/raptor2/prep_kernel_s0.py')
ap.add_argument('--out_json', default='')
ap.add_argument('--install', default='', help='copy the CSV here only on PASS (keeps a .bak of any different file)')
ap.add_argument('--self_test', action='store_true', help='box self-test only: I4 / I8 on the studies with DICOMs on the box')
a = ap.parse_args()
assert not (a.self_test and a.install), '--self_test output is never installed'

V1 = ['StudyInstanceUID', 'mean_x', 'n', 'sag_nx', 'tag', 'side', 'rule']
KC = (0, 1, 2, 4)
SLOT_NAME = {0: 'SagFS', 1: 'SagPD', 2: 'CorFS', 3: 'CorPD', 4: 'Ax'}
EXPECT = {0: -1, 1: -1, 2: 1, 4: 1}
BLOCK = {0: (0, 26), 1: (26, 48), 2: (48, 66), 3: (66, 78), 4: (78, 96)}
V2 = ['side_uid', 'tag_all', 'man', 'weak', 'geo_ok', 'serve', 'nread'] + [f's{k}_uid' for k in range(5)] + \
     [f's{k}_{c}' for k in KC for c in ('n', 'nh', 'sign', 'frac', 'mag', 'dom', 'mx', 'ok', 'flip')]
READER_SHA = '30aa72fd972e29cf42c752aa5394a6c88a175f17b98740b1f6b0e0f55553ac40'
T_MM, LAT_MM = 30.0, 5.0
res, fails = {}, []


def check(name, ok, detail):
    res[name] = dict(ok=bool(ok), detail=detail)
    print(f'{"PASS" if ok else "FAIL"} {name}: {detail}', flush=True)
    if not ok:
        fails.append(name)


def info(name, detail):
    res[name] = dict(info=detail)
    print(f'INFO {name}: {detail}', flush=True)


def rd(p):
    return pd.read_csv(p, dtype={'StudyInstanceUID': str, 'tag': str, 'tag_all': str, 'man': str, 'side_uid': str,
                                 **{f's{k}_uid': str for k in range(5)}, **{f's{k}_dom': str for k in KC}},
                       keep_default_na=False, na_values=[''])


sha = hashlib.sha256(open(a.csv, 'rb').read()).hexdigest()
df = rd(a.csv)
for c in ('tag', 'tag_all', 'man', 'side_uid') + tuple(f's{k}_uid' for k in range(5)) + tuple(f's{k}_dom' for k in KC):
    if c in df:
        df[c] = df[c].fillna('').astype(str)
print(f'CSV {a.csv} | sha256 {sha} | rows {len(df)} | cols {len(df.columns)}', flush=True)

# I1 schema
ser = pd.read_csv(a.series)
ser['StudyInstanceUID'] = ser['StudyInstanceUID'].astype(str); ser['SeriesInstanceUID'] = ser['SeriesInstanceUID'].astype(str)
SER = {k: v.to_dict('records') for k, v in ser.groupby('StudyInstanceUID')}
miss = [c for c in V2 if c not in df.columns]
check('I1_schema', list(df.columns[:7]) == V1 and not miss and df.StudyInstanceUID.is_unique
      and set(df.StudyInstanceUID) == set(SER) and len(df) == len(SER),
      f'v1 head {list(df.columns[:7]) == V1} | missing v2 cols {miss} | rows {len(df)} | unique {df.StudyInstanceUID.is_unique}'
      f' | = train_series studies ({len(SER)}) {set(df.StudyInstanceUID) == set(SER)}')
df = df.set_index('StudyInstanceUID', drop=False)


# I2 side rule
def rule_of(mx, tag):
    tags = set(tag) if isinstance(tag, str) else set()
    if np.isfinite(mx) and abs(mx) > T_MM:
        return ('R' if mx < 0 else 'L'), 'mean_x'
    if tags in ({'R'}, {'L'}):
        return next(iter(tags)), 'tag'
    if np.isfinite(mx) and abs(mx) > LAT_MM:
        return ('R' if mx < 0 else 'L'), 'mean_x_weak'
    return 'U', 'none'


rr = [rule_of(float(m), t) for m, t in zip(df.mean_x.astype(float), df.tag)]
bad = int(sum((s, r) != (a_, b_) for (s, r), a_, b_ in zip(rr, df.side, df.rule)))
check('I2_side_rule', bad == 0, f'rows where (side, rule) != rule(mean_x, tag): {bad}')

# I3 slot picks with the verbatim reader
src = open(a.reader).read()
m = re.search(r'^_READER_SRC = (.*)$', src, re.M)
rsrc = eval(m.group(1))
assert hashlib.sha256(rsrc.encode()).hexdigest() == READER_SHA, 'box reader text drift'
ns = {'np': np}
exec(compile(rsrc, 'raptor_reader', 'exec'), ns)
pick, SLOTS = ns['_pick_series_for_slot'], ns['SLOTS']
mism = {k: 0 for k in range(5)}
for sid, rows in SER.items():
    used = set()
    for k, (plane, fluid, _) in enumerate(SLOTS):
        r = pick(rows, plane, fluid, used)
        if r is not None:
            used.add(r['SeriesInstanceUID'])
        if (r['SeriesInstanceUID'] if r is not None else '') != df.at[sid, f's{k}_uid']:
            mism[k] += 1
check('I3_slot_picks', sum(mism.values()) == 0, f'mismatches per slot {mism} over {len(SER)} studies (reader sha {READER_SHA[:8]})')

# I4 presence vs the d96 mask
idx = pd.read_csv(a.index, dtype={'id': str, 'mask': str}).set_index('id')
pres = {}
both_bad = []
sub = df[df.n > 0] if a.self_test else df
if a.self_test:
    print(f'SELF-TEST: I4 / I8 restricted to the {len(sub)} studies with DICOMs on the box', flush=True)
for k in KC:
    lo, hi = BLOCK[k]
    mk = idx['mask'].reindex(sub.index).fillna('').map(lambda s: '1' in s[lo:hi])
    kn = sub[f's{k}_n'].fillna(0).astype(int) > 0
    pres[SLOT_NAME[k]] = dict(mask_present=int(mk.sum()), kernel_files=int(kn.sum()),
                              mask_without_series=int((mk & ~kn).sum()), series_without_mask=int((kn & ~mk).sum()))
    both_bad += sub.index[mk & ~kn].tolist()
check('I4_presence_vs_d96_mask', len(both_bad) == 0, f'{pres} (series_without_mask = pixel reads failed in the corpus, '
      f'allowed; mask_without_series must be 0); studies in index {int(sub.index.isin(idx.index).sum())}/{len(sub)}')

# I5 v1 box table on the 458
ref = rd(a.ref).set_index('StudyInstanceUID')
ref['tag'] = ref['tag'].fillna('').astype(str)
ov = ref.index[ref.n > 0]
d = df.loc[ov]
r5 = dict(studies=len(ov), n=int((d.n == ref.loc[ov, 'n']).sum()), side=int((d.side == ref.loc[ov, 'side']).sum()),
          rule=int((d.rule == ref.loc[ov, 'rule']).sum()), tag=int((d.tag == ref.loc[ov, 'tag']).sum()),
          mean_x=int((np.abs(d.mean_x - ref.loc[ov, 'mean_x']) <= 1e-6).sum()),
          sag_nx=int((np.abs(d.sag_nx - ref.loc[ov, 'sag_nx']) <= 1e-6).sum()))
check('I5_vs_v1_box458', len(ov) > 0 and all(v == len(ov) for v in r5.values()), f'agree {r5}')

# I6 same v2 code on the box
if a.box_csv and os.path.exists(a.box_csv) and os.path.abspath(a.box_csv) != os.path.abspath(a.csv):
    bx = rd(a.box_csv).set_index('StudyInstanceUID', drop=False)
    for c in ('tag', 'tag_all', 'man', 'side_uid') + tuple(f's{k}_uid' for k in range(5)) + tuple(f's{k}_dom' for k in KC):
        bx[c] = bx[c].fillna('').astype(str)
    ovb = bx.index[bx.n > 0]
    diff = {}
    for c in bx.columns:
        x, y = df.loc[ovb, c], bx.loc[ovb, c]
        if pd.api.types.is_numeric_dtype(y) and pd.api.types.is_numeric_dtype(x):
            same = np.isclose(x.astype(float), y.astype(float), atol=1e-6, rtol=0, equal_nan=True)
        else:
            same = (x.astype(str) == y.astype(str)).values
        if (~same).sum():
            diff[c] = int((~same).sum())
    check('I6_vs_v2_box458', len(ovb) > 0 and not diff, f'{len(ovb)} studies, columns with differences {diff}')
else:
    info('I6_vs_v2_box458', 'skipped (no separate --box_csv)')

# I7 derived columns
r7 = {}
for k in KC:
    n = df[f's{k}_n'].fillna(0).astype(int); sg = df[f's{k}_sign'].fillna(0).astype(int); fr = df[f's{k}_frac']
    ok = (n == 0) | ((sg == EXPECT[k]) & (fr == 1.0))
    flip = (df.side == 'R').astype(int) ^ (sg == -EXPECT[k]).astype(int)
    r7[f's{k}_ok'] = int((ok.astype(int) != df[f's{k}_ok']).sum())
    r7[f's{k}_flip'] = int((flip != df[f's{k}_flip']).sum())
geo = np.logical_and.reduce([df[f's{k}_ok'] == 1 for k in KC])
weak = (df.rule != 'mean_x').astype(int)
serve = np.where((weak == 0) & geo, 'single', 'both')
r7.update(weak=int((weak != df.weak).sum()), geo_ok=int((geo.astype(int) != df.geo_ok).sum()),
          serve=int((serve != df.serve).sum()))
check('I7_derived_columns', sum(r7.values()) == 0, f'mismatching rows {r7}')

# I8 gate
u = float((sub.side == 'U').mean())
check('I8_U_rate', u <= 0.02, f'U {int((sub.side == "U").sum())}/{len(sub)} = {u:.4f} (gate <= 0.02)')

# GEO_ASSERT + counts (reported)
g = {}
for k in KC:
    p = df[f's{k}_n'].fillna(0).astype(int) > 0
    sg = df[f's{k}_sign'].fillna(0).astype(int)
    g[SLOT_NAME[k]] = dict(present=int(p.sum()), expected=int((p & (sg == EXPECT[k])).sum()),
                           opposite=int((p & (sg == -EXPECT[k])).sum()), undefined=int((p & (sg == 0)).sum()),
                           mixed=int((p & (df[f's{k}_frac'] < 1)).sum()),
                           dom=dict(df.loc[p, f's{k}_dom'].value_counts()),
                           mag_min=float(df.loc[p, f's{k}_mag'].min()) if p.any() else None,
                           not_ok_sids=df.index[p & (df[f's{k}_ok'] == 0)].tolist()[:20])
geo_all = all(v['present'] == v['expected'] and v['mixed'] == 0 for v in g.values())
info('GEO_ASSERT', f'{"PASS" if geo_all else "FAIL (unexpected series are served as both canons)"} {g}')
mx = df.mean_x.abs()
info('side_counts', f'sides {dict(df.side.value_counts())} | rules {dict(df.rule.value_counts())} | '
     f'weak {int(df.weak.sum())} ({df.weak.mean():.4f}) | serve {dict(df.serve.value_counts())} | '
     f'|mean_x| q01/q02/q05/q10 {[round(float(mx.quantile(q)), 1) for q in (.01, .02, .05, .10)]}')
info('man_x_side', str(pd.crosstab(df.man.replace('', '?'), df.side).to_dict()))

# lat458 header side (independent probe_lat.py pass)
if os.path.exists(a.lat):
    lat = pd.read_csv(a.lat, dtype=str).set_index('sid')
    o = lat.index.intersection(df.index)
    hs = lat.loc[o, 'hdr']
    info('lat458', f'{len(o)} studies | kernel side == probe_lat hdr side {int((df.loc[o, "side"] == hs).sum())}'
         f' | probe_lat sag_nx all -1 {bool(lat.sag_nx.fillna("").str.split(",").map(lambda v: all(x == "-1" for x in v if x)).all())}'
         f' | cor_rx all 1 {bool(lat.cor_rx.fillna("").str.split(",").map(lambda v: all(x == "1" for x in v if x)).all())}'
         f' | ax_rx all 1 {bool(lat.ax_rx.fillna("").str.split(",").map(lambda v: all(x == "1" for x in v if x)).all())}')

# kernel summary cross-check
if a.summary and os.path.exists(a.summary):
    s = json.load(open(a.summary))['train']
    same = (s['studies'] == len(df) and s['weak'] == int(df.weak.sum()) and s['geo_ok'] == int(df.geo_ok.sum())
            and s['sides'] == {k: int(v) for k, v in df.side.value_counts().items()})
    info('kernel_summary', f'consistent {same} | kernel seconds {s["seconds"]} | files read {s["files_read"]} | '
         f'versions {s["versions"]} | KC_SIDE_GEO {s["KC_SIDE_GEO"]}')

verdict = ('PASS' if not fails else 'FAIL ' + ','.join(fails)) + (' (SELF-TEST, box subset)' if a.self_test else '')
print(f'KC_SIDE_VALIDATE {verdict} | sha256 {sha}', flush=True)
out = dict(csv=os.path.abspath(a.csv), sha256=sha, verdict=verdict, checks=res,
           time=time.strftime('%Y-%m-%d %H:%M:%S'))
if a.out_json:
    json.dump(json.loads(json.dumps(out, default=lambda o: o.item() if hasattr(o, 'item') else str(o))),
              open(a.out_json, 'w'), indent=1)
if a.install:
    if fails:
        print('NOT installed (FAIL)'); sys.exit(1)
    dst = os.path.expanduser(a.install)
    if os.path.exists(dst):
        if hashlib.sha256(open(dst, 'rb').read()).hexdigest() == sha:
            print(f'already installed: {dst}'); sys.exit(0)
        bak = f'{dst}.bak.{time.strftime("%Y%m%d_%H%M%S")}'
        shutil.copy2(dst, bak); print(f'kept previous copy: {bak}')
    shutil.copy2(a.csv, dst)
    print(f'installed {dst} sha256 {sha}')
sys.exit(1 if fails else 0)
