# G6 offline gate for G-FLAG (the series-flag canonicaliser), CPU only. The canonicaliser is the exact function a kernel cell would
# run on the hidden test_series.csv BEFORE any member reads it; the members' readers are reproduced from c4's code:
#   Raptor r384 / r224 picker = _fastread._pick_series_for_slot (decoded _FASTREAD_B64, c4 cells 4 / 11);
#   R6 / S6fall series record = native_cache.prepare l.112-117 (sha c53ce898 = c4 CODE_SHA pin): slot from Fat_Suppression via
#   common.SLOTS, meta 'fluid' from Fluid_Sensitive; NaN -> -1.
import numpy as np, pandas as pd, io
B = 'C:/Users/eeshs/Downloads/rsna-knee/'
PL = ('Sagittal', 'Coronal', 'Axial')

def gflag_canon(df):
    """Returns (canonical copy, n_rows_changed). A row is already canonical iff both flags are the same value in {0, 1} and the plane is
    one of the three exact names; then it is returned unchanged. Otherwise both flags := min(Fluid, Fat) (the train labelling rule:
    1 only for fluid-bright fat-suppressed sequences), a missing / invalid flag takes the other one, both missing -> -1 (no
    preference; R6 already maps NaN to -1); a plane that matches one of the three names after strip / case-fold is renamed."""
    out = df.copy()
    f = pd.to_numeric(df['Fluid_Sensitive'], errors='coerce'); z = pd.to_numeric(df['Fat_Suppression'], errors='coerce')
    f = f.where(f.isin([0, 1])); z = z.where(z.isin([0, 1]))
    c = np.where(f.notna() & z.notna(), np.fmin(f, z), np.where(f.notna(), f, np.where(z.notna(), z, -1))).astype(np.int64)
    p = df['Anatomical_Plane'].astype(str).str.strip().str.lower().map({k.lower(): k for k in PL})
    plane = np.where(p.notna(), p, df['Anatomical_Plane'])
    ok = f.notna() & z.notna() & (f == z) & df['Anatomical_Plane'].isin(PL)
    n = int((~ok).sum())
    if n:
        out['Fluid_Sensitive'] = c; out['Fat_Suppression'] = c; out['Anatomical_Plane'] = plane
    return out, n

SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", None, 18)]
def raptor_pick(rows):                     # verbatim logic of _fastread._pick_series_for_slot over SLOTS
    used, out = set(), []
    for plane, fluid, k in SLOTS:
        cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
        r = None
        if fluid in (0, 1):
            pref = [r_ for r_ in cands if int(r_.get('Fluid_Sensitive', 0) or 0) == fluid]
            if pref: r = pref[0]
        if r is None: r = cands[0] if cands else None
        out.append(None if r is None else r['SeriesInstanceUID'])
        if r is not None: used.add(r['SeriesInstanceUID'])
    return tuple(out)
R6SLOT = {("Sagittal", 1): 1, ("Sagittal", 0): 2, ("Coronal", 1): 3, ("Coronal", 0): 4, ("Axial", 1): 5, ("Axial", 0): 6,
          ("Sagittal", -1): 7, ("Coronal", -1): 8, ("Axial", -1): 9}
def r6_rec(rows):                          # native_cache.prepare: (series, slot, fluid meta) for cardinal planes
    out = []
    for row in rows:
        if row['Anatomical_Plane'] not in PL: continue
        fs = int(row['Fat_Suppression']) if pd.notna(row.get('Fat_Suppression')) else -1
        fs = fs if fs in (0, 1) else -1
        fl = int(row['Fluid_Sensitive']) if pd.notna(row.get('Fluid_Sensitive')) else -1
        out.append((row['SeriesInstanceUID'], R6SLOT[(row['Anatomical_Plane'], fs)], fl))
    return tuple(out)
def members(df):
    res, err = {}, 0
    for u, g in df.groupby('StudyInstanceUID', sort=False):
        rows = g.to_dict('records')
        try: rp = raptor_pick(rows)
        except Exception: rp = 'RAISES'; err += 1             # c4: _pre_one catches it -> all-zero volume, unflagged
        res[u] = (rp, r6_rec(rows))
    return res, err
def read(p): return pd.read_csv(p)                         # Raptor reads with default dtypes (cell 4 l.524)

tr = read(B + '_data/train_series.csv'); te = read(B + '_data/test_series.csv')
for nm, df in (('visible test_series.csv', te), ('train_series.csv (24,371 rows)', tr)):
    c, n = gflag_canon(df)
    print('%-32s rows changed %d | canonical frame identical to input: %s' % (nm, n, c.equals(df)))
# definition-following DEF flags on the 458 header studies (g2): fired, and the member inputs equal BASE exactly
F = pd.read_csv(B + 'session_2026-10-20/more/work/g2_def_flags.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
base = F[['StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane']].assign(Fluid_Sensitive=F.F_base, Fat_Suppression=F.F_base)
dfn = F[['StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane']].assign(Fluid_Sensitive=F.F_def, Fat_Suppression=F.Fat_def)
cb, _ = members(base); cd, _ = members(dfn); gd, n = members(gflag_canon(dfn)[0])[0], gflag_canon(dfn)[1]
print('DEF (458 studies): rows changed %d | status quo: Raptor picks differ in %d studies, R6 records differ in %d | G-FLAG: Raptor %d, R6 %d'
      % (n, sum(cb[u][0] != cd[u][0] for u in cb), sum(cb[u][1] != cd[u][1] for u in cb), sum(cb[u][0] != gd[u][0] for u in cb), sum(cb[u][1] != gd[u][1] for u in cb)))
# NaN / invalid / plane-case injections on train (stress): status quo vs G-FLAG
rng = np.random.default_rng(5)
for kind in ('Fluid NaN', 'both NaN', 'Fluid=2', 'plane lower-case'):
    x = tr.copy(); ix = rng.choice(len(x), 200, replace=False)
    if kind == 'Fluid NaN': x.loc[ix, 'Fluid_Sensitive'] = np.nan
    if kind == 'both NaN': x.loc[ix, ['Fluid_Sensitive', 'Fat_Suppression']] = np.nan
    if kind == 'Fluid=2': x.loc[ix, 'Fluid_Sensitive'] = 2
    if kind == 'plane lower-case': x.loc[ix, 'Anatomical_Plane'] = x.loc[ix, 'Anatomical_Plane'].str.lower()
    studs = set(x.StudyInstanceUID.iloc[ix]); xs = x[x.StudyInstanceUID.isin(studs)]; ts = tr[tr.StudyInstanceUID.isin(studs)]
    ref = members(ts)[0]; sq, e_sq = members(xs); gc, n = gflag_canon(xs); gm, e_g = members(gc)
    print('%-17s %d studies | status quo: Raptor raises %d (-> all-zero volume), picks differ %d, R6 recs differ %d | G-FLAG (%d rows): raises %d, picks differ %d, R6 differ %d'
          % (kind, len(studs), e_sq, sum(ref[u][0] != sq[u][0] for u in ref), sum(ref[u][1] != sq[u][1] for u in ref), n, e_g,
             sum(ref[u][0] != gm[u][0] for u in ref), sum(ref[u][1] != gm[u][1] for u in ref)))
