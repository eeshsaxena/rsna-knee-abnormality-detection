# recal_routeO.py (10-20 night, Claude, TRACK 1). RESEARCH ONLY, local; OAI enters ONLY as aggregates already on the laptop
# (session_2026-10-19/ex/box_coupling.log JSON block: per-direction rho_ns / q / d of the 3 OAI-free models; coupling_rows.json tets).
# Writes recal_routeO.log.
# Route O = the estimator that reproduces the E-OA trio: c4's gap = c4's LABEL-FREE score coupling rho (gold58, normal scores) minus the
# cross-predictiveness q measured on OAI (2,399 knees) after moving the OAI label coupling to gold74's. Trio check: c4 rho .816 - D5 q at
# gold74 coupling .710 = .106 (gold58 direct .10-.117; public-implied .128 [.081, .176]).
# q at gold74 coupling is not in the pair aggregates, only raw q; it is moved by kq x (gold74 tet - OAI tet), kq in {.43 (trio-measured:
# D5 q .508 -> .710 for tet +.47), .9 (single-partner upper)}. Directions whose OAI test is invalid are flagged, not re-scored.
import json
import numpy as np, pandas as pd
OUTD = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/'
log = open(OUTD + 'recal_routeO.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()
s = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/box_coupling.log', encoding='utf-8', errors='replace').read()
box = json.loads(s.split('JSON_BEGIN')[1].split('JSON_END')[0].strip())
rows = {r['pair']: r for r in json.load(open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/coupling_rows.json'))}
cd = pd.read_csv('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push/census_dirs.csv').set_index('dir')
c4dir = pd.read_csv(OUTD + 'recal_reliab_dirs.csv').set_index('dir')
AB = {"ACL": "ACL", "MCL": "MCL", "Medial Meniscus": "MM", "Lateral Meniscus": "LM", "Medial OA": "MOA", "Lateral OA": "LOA",
      "PF OA": "PFOA", "Effusion": "Eff", "Synovitis": "Syn", "Baker's": "Bak", "Contusion": "Cont", "Fracture": "Frac"}
OM = ['D5mean_s1043', 's1044_mix50', 's1045_288']
INVALID = {'Synovitis': 'OAI Syn = Hoffa; OAI-free Syn output = Eff output (rho .99)', 'ACL': 'OAI ACL is degenerative (ACL-OA tet .43-.53 vs RSNA -.38..+.04)'}
P('direction | c4 rho_ns (gold58, eoa20 base) | proxy rho_ns raw (3) | proxy q raw (3) | OAI tet -> gold74 tet | proxy gap raw | Route O gap (kq .43 / .9) | gold58 direct gap (z) | OAI d raw / R2 a.1 (3) | note')
OUT = {}
for t, o, a in [('Lateral Meniscus', 'Medial Meniscus', .1), ('Medial Meniscus', 'Lateral Meniscus', .1),
                ('Medial OA', 'Medial Meniscus', .1), ('Lateral OA', 'Lateral Meniscus', .1), ('Medial Meniscus', 'Medial OA', .1),
                ('Lateral Meniscus', 'Lateral OA', .1), ('Lateral OA', 'Medial Meniscus', .1), ('Medial Meniscus', 'Lateral OA', .1),
                ('Lateral OA', 'Effusion', -.1), ('Effusion', "Baker's", .1), ("Baker's", 'Effusion', .1), ('Effusion', 'Synovitis', .1),
                ('Synovitis', 'Effusion', .1), ('ACL', 'Lateral Meniscus', .1), ('ACL', 'Medial Meniscus', .1), ('ACL', 'Effusion', .1),
                ('PF OA', 'Lateral Meniscus', .1), ('Medial OA', "Baker's", -.1), ('Lateral Meniscus', "Baker's", -.1)]:
    pair = None
    for p, b in box.items():
        x, y = p.split(' | ')
        if {x, y} == {t, o}: pair = p; break
    dd = box[pair]['dir'].get(t + ' <- ' + o)
    cr = None
    for p_, r_ in rows.items():
        if set(p_.split(' | ')) == {t, o}: cr = r_; break
    tet_o, tet_g = cr['oai_tet'], cr['g74_tet']
    rp = np.array([dd[k]['rho_ns'] for k in OM]); qp = np.array([dd[k]['q'] for k in OM])
    draw = np.array([dd[k]['d']['0.1'] for k in OM]); dR2 = np.array([dd[k]['rw']['R2']['d.1'] for k in OM])
    nm = '%s <- %s' % (AB[t], AB[o]); c = c4dir.loc[nm]
    rho_c4 = float(cd.loc[nm, 'rho_ns']) if nm in cd.index else np.nan
    gO = {kq: rho_c4 - (qp + kq * (tet_g - tet_o)) for kq in (.43, .9)}
    note = INVALID.get(t, INVALID.get(o, ''))
    P('%-12s a %+.1f | %.3f | %s | %s | %+.2f -> %+.2f | %+.3f | %+.3f / %+.3f (sd over proxies %.3f) | %+.3f (z %+.1f) | %s / %s | %s' % (
        nm, a, rho_c4, np.round(rp, 2), np.round(qp, 2), tet_o, tet_g, (rp - qp).mean(), gO[.43].mean(), gO[.9].mean(), gO[.43].std(),
        c.gap, (c.gap - .021) / c.se_rob, np.round(draw, 4), np.round(dR2, 4), note))
    OUT[nm] = dict(a=a, rho_c4=rho_c4, gO43=gO[.43].tolist(), gO90=gO[.9].tolist(), g58=float(c.gap), g58_se=float(c.se_rob), note=note,
                   d_raw=draw.tolist(), d_R2=dR2.tolist(), proxy_rho=rp.tolist(), proxy_q=qp.tolist(), tet=[tet_o, tet_g])
json.dump(OUT, open(OUTD + 'recal_routeO.json', 'w'), indent=1)
P('RECAL_ROUTEO DONE')
