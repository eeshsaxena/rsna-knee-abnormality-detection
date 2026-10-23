# G2: exposure of each c4 member if the hidden CSV followed the data-page DEFINITION (Fluid = T2/PD/STIR-type weighting,
# Fat = fat suppression), built from DICOM headers of the 458 header studies (train only). CPU, code-faithful pickers.
import pandas as pd, numpy as np
W = r'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/more/work/'
m = pd.read_csv(W + 'g1b_classified.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str, 'Fluid_Sensitive': str, 'Fat_Suppression': str})
S = pd.read_csv(r'C:/Users/eeshs/Downloads/rsna-knee/_data/train_series.csv', dtype=str)
S = S[S.StudyInstanceUID.isin(set(m.StudyInstanceUID))].copy()   # all series of the header studies, CSV order
k = m.set_index('SeriesInstanceUID')
base_f = S.Fluid_Sensitive.astype(int).values
def_f, def_fat = base_f.copy(), base_f.copy()
for i, sid in enumerate(S.SeriesInstanceUID):
    if sid not in k.index: continue
    r = k.loc[sid]
    if base_f[i] == 0 and r.w in ('PD', 'T2') and not r.fsd: def_f[i] = 1          # PD / T2 without fat sat: fluid-sensitive by the definition
    if base_f[i] == 0 and r.w == 'T1' and r.fsd: def_fat[i] = 1                      # T1 with fat sat: fat-suppressed, not fluid-sensitive
S['F_base'], S['F_def'], S['Fat_def'] = base_f, def_f, def_fat
S['F_guard'] = np.minimum(def_f, def_fat)                                            # G-FLAG: both flags := min(F, Fat)
print('series', len(S), '| (1,0) rows', int(((def_f == 1) & (def_fat == 0)).sum()), '| (0,1) rows', int(((def_f == 0) & (def_fat == 1)).sum()),
      '| guard == base on every row:', bool((S.F_guard == S.F_base).all()))
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", None, 18)]
def pick(rows, fcol):
    used, out = set(), []
    for plane, fl, kk in SLOTS:
        c = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
        p = [r for r in c if fl in (0, 1) and int(r[fcol]) == fl]
        r = p[0] if p else (c[0] if c else None)
        out.append(None if r is None else r['SeriesInstanceUID']); 
        if r is not None: used.add(r['SeriesInstanceUID'])
    return out
G = pd.read_csv(r'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58/gold74_eval_set.csv', dtype=str)
g58 = set(G[G.source == 'original_gold58'].StudyInstanceUID)
res = []
for u, g in S.groupby('StudyInstanceUID', sort=False):
    rows = g.to_dict('records')
    b, d, q = pick(rows, 'F_base'), pick(rows, 'F_def'), pick(rows, 'F_guard')
    # R6 / S6fall (native_cache): slot from Fat_Suppression, meta 'fluid' from Fluid_Sensitive; stacker FS-sag from meta fluid
    r6_meta = int((g.F_def != g.F_base).sum()); r6_slot = int((g.Fat_def != g.F_base).sum())
    fs_sag_b = g[(g.Anatomical_Plane == 'Sagittal') & (g.F_base == 1)].SeriesInstanceUID.tolist()
    fs_sag_d = g[(g.Anatomical_Plane == 'Sagittal') & (g.F_def == 1)].SeriesInstanceUID.tolist()
    res.append(dict(u=u, gold=u in g58, any_def=bool((g.F_def != g.F_base).any() or (g.Fat_def != g.F_base).any()),
                    rap_slots_changed=sum(x != y for x, y in zip(b, d)), rap_slot0=b[0] != d[0], rap_slot2=b[2] != d[2],
                    guard_eq=b == q, r6_meta=r6_meta, r6_slot=r6_slot, stk_cand_added=len(set(fs_sag_d) - set(fs_sag_b)) > 0))
R = pd.DataFrame(res)
for nm, X in (('458 header studies', R), ('gold58', R[R.gold])):
    print('\n%s: n %d' % (nm, len(X)))
    print('  studies with >= 1 definition-changed row      %.3f' % X.any_def.mean())
    print('  Raptor: >= 1 slot changed %.3f | slot0 (Sag FS) changed %.3f | slot2 (Cor FS) changed %.3f | mean slots changed %.2f'
          % ((X.rap_slots_changed > 0).mean(), X.rap_slot0.mean(), X.rap_slot2.mean(), X.rap_slots_changed.mean()))
    print('  R6/S6fall: series with fluid meta changed / study %.2f (studies >=1: %.3f) | slot changed (T1-FS) studies %.3f | fluid-stacker FS-sag candidate added %.3f'
          % (X.r6_meta.mean(), (X.r6_meta > 0).mean(), (X.r6_slot > 0).mean(), X.stk_cand_added.mean()))
    print('  G-FLAG output == base picks on every study: %s' % bool(X.guard_eq.all()))
S[['StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane', 'F_base', 'F_def', 'Fat_def', 'F_guard']].to_csv(W + 'g2_def_flags.csv', index=False)
