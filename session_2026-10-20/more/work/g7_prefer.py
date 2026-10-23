# G7: the narrower Raptor-only guard "prefer (1,1) over (1,0) within a plane" vs G-FLAG, on the DEF flags (458 header studies).
import pandas as pd, sys
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/more/work')
from g6_gate_lib import members, raptor_pick, r6_rec, gflag_canon, SLOTS
F = pd.read_csv('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/more/work/g2_def_flags.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
base = F[['StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane']].assign(Fluid_Sensitive=F.F_base, Fat_Suppression=F.F_base)
dfn = F[['StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane']].assign(Fluid_Sensitive=F.F_def, Fat_Suppression=F.Fat_def)
def pick_prefer(rows):
    used, out = set(), []
    for plane, fl, k in SLOTS:
        c = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
        r = None
        if fl in (0, 1):
            p2 = [x for x in c if int(x['Fluid_Sensitive']) == fl and int(x['Fat_Suppression']) == fl]
            p1 = [x for x in c if int(x['Fluid_Sensitive']) == fl]
            r = p2[0] if p2 else (p1[0] if p1 else None)
        if r is None: r = c[0] if c else None
        out.append(None if r is None else r['SeriesInstanceUID']); used.add(None if r is None else r['SeriesInstanceUID'])
    return tuple(out)
b = {u: raptor_pick(g.to_dict('records')) for u, g in base.groupby('StudyInstanceUID', sort=False)}
p = {u: pick_prefer(g.to_dict('records')) for u, g in dfn.groupby('StudyInstanceUID', sort=False)}
print('prefer-(1,1) Raptor picks differ from BASE in %d / %d studies (G-FLAG: 0); R6 records untouched by it (differ in 299)' % (sum(b[u] != p[u] for u in b), len(b)))
