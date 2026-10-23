# box_rubric.py (10-20 levers / rubric lens, Claude). BOX ONLY, read-only, CPU. Prints AGGREGATES ONLY (n, prevalence, AUC,
# AUC deltas, sd, P, Spearman). Never prints an id, a row or a per-knee value; writes nothing.
# Part A: T-VOL lever on the OAI expert ruler: Effusion (MOAKS m_eff_wk 0-3; rubric = grade >= 2) and Baker's (m_popcys
#         presence; WORMS w_popcys size 0-3) with R6's own fluid volumes (fluid_stacker.csv, computed on the store rebuild).
#         out = pct(score) + b * pct(volume) (missing -> .5).
# Part B: regime-subclass headroom. For labels whose rubric-negative subclass is report-positive (meniscal intrasubstance signal,
#         effusion grade 1, ACL code 1, OA cartilage loss without full-thickness), how high do OAI-free models rank the subclass,
#         and what is the perfect-demotion bound (subclass forced below every positive)?
import os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr

H = os.path.expanduser('~')
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
       "Baker's", 'Contusion', 'Fracture']
BS = (.1, .2, .3, .5)
rng = np.random.default_rng(1020)
B = int(os.environ.get('BOOT', '1000'))


def pct(v):
    return rankdata(v) / len(v)


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def rdp(p):
    d = pd.read_csv(p, dtype={'StudyInstanceUID': str})
    return d.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')


def tvol(s, v, b):
    pv = np.full(len(v), .5); ok = np.isfinite(v); pv[ok] = pct(v[ok])
    return pct(s) + b * pv


S = pd.read_csv(H + '/oai_build/oai_label_sources.csv', dtype={'kid': str}).set_index('kid')
T = rdp(H + '/oai_build/oai_labels_T123.csv')
r6 = rdp(H + '/r5_oai/run_r6/worker0/predictions.csv')[LAB].astype(float)
fl = rdp(H + '/r5_oai/run_r6/worker0/fluid_stacker.csv')
st = rdp(H + '/r5_oai/run_r6/worker0/state_stacker.csv')
s6 = rdp(H + '/r5_oai/s6_r5/out/predictions.csv')[LAB].astype(float)
d5 = rdp(H + '/oai_build/teacher/teacher_D5_mean_v00.csv')[LAB].astype(float)
r6o = r6[r6.index.str.startswith('oai_')].dropna()
common = r6o.index.intersection(s6.dropna().index).intersection(d5.dropna().index).intersection(S.index)
print('common OAI knees (R6 & S6fall & D5 & sources):', len(common))
st_ok = fl.reindex(common)['status'].astype(str)
print('fluid status share ok: %.3f | n_sag_fluid_series dist %s' % ((st_ok == 'ok').mean(),
      fl.reindex(common)['n_sag_fluid_series'].value_counts().to_dict()))


def models(c):
    R6 = r6o.loc[common, c].values; S6 = s6.loc[common, c].values; D5 = d5.loc[common, c].values
    wr = {"Baker's": .5, 'Lateral Meniscus': .4, 'Lateral OA': .2, 'Medial OA': .5, 'PF OA': .5}.get(c, .5)
    c4d5 = .8 * pct(wr * pct(D5) + (1 - wr) * pct(R6)) + .2 * pct(S6)
    out = {'R6': R6, 'S6fall': S6, 'D5': D5, 'R6.8S6.2': .8 * pct(R6) + .2 * pct(S6), 'c4D5': c4d5}
    return out


def boot_d(s, v, y, b):
    n = len(y); d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].sum() in (0, n):
            continue
        d.append(auc(tvol(s[i], v[i], b), y[i]) - auc(s[i], y[i]))
    d = np.array(d); return d.mean(), d.std(), (d > 0).mean()

# ================= Part A: T-VOL =================
print('\n=== PART A: T-VOL, out = pct(s) + b pct(R6 volume); AUC deltas at b', BS)
for c, vcol, pm_src in (('Effusion', 'eff_sag_p99_open', 'fluid'), ("Baker's", 'bak_sag_p99_open', 'state')):
    v_all = pd.to_numeric(fl.reindex(common)[vcol], errors='coerce').values
    M = models(c)
    M['R6_pmodel'] = pd.to_numeric((fl if pm_src == 'fluid' else st).reindex(common)[c + '|p_model'], errors='coerce').values
    if c == 'Effusion':
        g = pd.to_numeric(S.loc[common, 'm_eff_wk'], errors='coerce').values
        targets = {'grade>=1 (report-like)': g >= 1, 'grade>=2 (RUBRIC)': g >= 2, 'grade>=3 (large)': g >= 3}
    else:
        g = pd.to_numeric(S.loc[common, 'm_popcys'], errors='coerce').values
        w = pd.to_numeric(S.loc[common, 'w_popcys'], errors='coerce').values
        targets = {'MOAKS popcys present': g >= 1}
        okw = np.isfinite(w)
        targets_w = {'WORMS size>=1': w >= 1, 'WORMS size>=2 (RUBRIC-like)': w >= 2}
    okg = np.isfinite(g)
    ok_v = np.isfinite(v_all) & okg
    print(f'-- {c}: n graded {okg.sum()} | volume finite {np.isfinite(v_all).mean():.3f} | Spearman(volume, grade) '
          f'{spearmanr(v_all[ok_v], g[ok_v]).correlation:.3f}')
    for tn, yy in targets.items():
        y = yy[okg].astype(int)
        print(f'   target {tn}: n {len(y)} prev {y.mean():.3f} | volume alone AUC {auc(np.nan_to_num(v_all[okg], nan=np.nanmedian(v_all)), y):.4f}')
        for m, s in M.items():
            sm = s[okg]
            if not np.isfinite(sm).all():
                continue
            base = auc(sm, y)
            ds = [auc(tvol(sm, v_all[okg], b), y) - base for b in BS]
            line = f'      {m:10s} base {base:.4f} | ' + ' '.join(f'{d:+.4f}' for d in ds)
            if m in ('c4D5', 'R6') and 'RUBRIC' in tn:
                bm, bsd, bp = boot_d(sm, v_all[okg], y, .2)
                line += f' | b.2 boot {bm:+.4f} sd {bsd:.4f} P {bp:.2f}'
            print(line)
    if c == "Baker's":
        for tn, yy in targets_w.items():
            y = yy[okw].astype(int)
            print(f'   target {tn}: n {len(y)} prev {y.mean():.3f}')
            for m, s in M.items():
                sm = s[okw]
                if not np.isfinite(sm).all():
                    continue
                base = auc(sm, y)
                ds = [auc(tvol(sm, v_all[okw], b), y) - base for b in BS]
                print(f'      {m:10s} base {base:.4f} | ' + ' '.join(f'{d:+.4f}' for d in ds))

# ================= Part B: subclass headroom =================
print('\n=== PART B: rubric-negative subclass g1 between clean negatives g0 and rubric positives g2')
print('cols: n0 n1 n2 | AUC(g2 vs g0+g1) [expert] | AUC(g2 vs g0) | AUC(g2 vs g1) | AUC(g1 vs g0) | bound (g1 forced to bottom) | headroom')


def oa_classes(lab):
    comp = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'],
            'PF OA': ['mp', 'lp', 'amf', 'alf']}[lab]
    C = S.loc[common, ['m_cm_' + r for r in comp]].astype(float).values
    area = np.floor(C + 1e-6); ft = np.round((C - area) * 10); fin = np.isfinite(C)
    anyfin = fin.any(1)
    g2 = ((ft >= 1) & fin).any(1)
    g1 = ((area >= 1) & fin).any(1) & ~g2
    cls = np.where(g2, 2, np.where(g1, 1, 0)).astype(float); cls[~anyfin] = np.nan
    return cls


def classes(lab):
    if lab in ('Medial Meniscus', 'Lateral Meniscus'):
        t = T.reindex(common)[lab].values.astype(float)
        return np.where(np.isnan(t), np.nan, np.where(t >= .5, 2, np.where(t > 0, 1, 0)).astype(float)), 'signal-only (.2)'
    if lab == 'Effusion':
        g = pd.to_numeric(S.loc[common, 'm_eff_wk'], errors='coerce').values
        return np.where(np.isnan(g), np.nan, np.where(g >= 2, 2, np.where(g >= 1, 1, 0)).astype(float)), 'grade 1 (small)'
    if lab == 'ACL':
        g = pd.to_numeric(S.loc[common, 'm_acl_tear'], errors='coerce').values
        rep = pd.to_numeric(S.loc[common, 'm_acl_repair'], errors='coerce').values
        g = np.where(rep == 1, np.nan, g)
        return np.where(np.isnan(g), np.nan, g.astype(float)), 'code 1 (partial / signal)'
    if lab in ('Medial OA', 'Lateral OA', 'PF OA'):
        return oa_classes(lab), 'cartilage loss, no full-thickness'
    if lab == "Baker's":
        w = pd.to_numeric(S.loc[common, 'w_popcys'], errors='coerce').values
        return np.where(np.isnan(w), np.nan, np.where(w >= 2, 2, np.where(w >= 1, 1, 0)).astype(float)), 'WORMS size 1'

for lab in ['Medial Meniscus', 'Lateral Meniscus', 'Effusion', 'ACL', 'Medial OA', 'Lateral OA', 'PF OA', "Baker's"]:
    cls, nm = classes(lab)
    ok = np.isfinite(cls)
    k = cls[ok].astype(int)
    n0, n1, n2 = (k == 0).sum(), (k == 1).sum(), (k == 2).sum()
    print(f'-- {lab}: g1 = {nm}')
    for m, s in models(lab).items():
        sm = s[ok]
        y = (k == 2).astype(int)
        base = auc(sm, y)
        a20 = auc(sm[k != 1], (k[k != 1] == 2).astype(int)) if n0 and n2 else np.nan
        a21 = auc(sm[k != 0], (k[k != 0] == 2).astype(int)) if n1 and n2 else np.nan
        a10 = auc(sm[k != 2], (k[k != 2] == 1).astype(int)) if n0 and n1 else np.nan
        bound = (n0 * a20 + n1 * 1.0) / (n0 + n1)
        print(f'   {m:9s} {n0:5d} {n1:5d} {n2:5d} | {base:.4f} | {a20:.4f} | {a21:.4f} | {a10:.4f} | {bound:.4f} | {bound - base:+.4f}')
print('DONE')
