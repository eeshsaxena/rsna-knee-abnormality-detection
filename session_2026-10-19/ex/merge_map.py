# merge_map.py (10-19 ex): merges local_coupling.json (gold74 / gold58 / R7 OOF / report v2) with the box aggregates in
# box_coupling.log (JSON block) into ranked tables (printed as markdown). Local CPU only.
import json, re, sys
import numpy as np
E = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/'
loc = json.load(open(E + 'local_coupling.json'))['pairs']
txt = open(E + (sys.argv[1] if len(sys.argv) > 1 else 'box_coupling.log'), encoding='utf-8', errors='replace').read()
box = json.loads(txt.split('JSON_BEGIN')[1].split('JSON_END')[0].strip())
AB = {"ACL": "ACL", "Medial Meniscus": "MM", "Lateral Meniscus": "LM", "Medial OA": "MOA", "Lateral OA": "LOA", "PF OA": "PFOA",
      "Effusion": "Eff", "Synovitis": "Syn", "Baker's": "Bak"}
ab = lambda p: '-'.join(AB[x] for x in p.split(' | '))
abd = lambda d: ' <- '.join(AB[x] for x in d.split(' <- '))
M = ['D5mean_s1043', 's1044_mix50', 's1045_288']
rows = []
for p, b in box.items():
    l = loc[p]
    oai_m = np.mean([b['model_rho'][k] for k in M])
    g_c4 = l['g58_model_rho']['c4']; g_4 = np.mean(list(l['g58_model_rho'].values()))
    r = dict(pair=p, oai_n=b['n'], oai_prev=b['prev'], oai_tet=b['tet'], oai_phi=b['phi'], oai_b2tet=b['B2']['tet'], oai_grho=b['graded_rho'],
             oai_m=oai_m, g74_tet=l['g74']['tet'], g74_sd=l['g74']['tet_sd'], g74_phi=l['g74']['phi'], g74_n=l['g74']['n'], g58_tet=l['g58']['tet'],
             c4=g_c4, g4=g_4, r7=l['r7_rho'], r7g=l['r7_g58_rho'], rep=l['rep_dropUNK']['tet'], rep0=l['rep_UNK0']['tet'], repphi=l['rep_dropUNK']['phi'])
    r['gap_oai'] = oai_m - b['tet']                      # OAI: OAI-free models vs OAI expert (same knees)
    r['gap_c4'] = g_c4 - l['g58']['tet']                 # gold58: c4 vs gold58 expert (same 58 studies)
    r['gap_c4_74'] = g_c4 - l['g74']['tet']              # c4 (gold58) vs the larger gold74 expert estimate
    r['gap_r7g'] = l['r7_g58_rho'] - l['g58']['tet']     # R7 OOF on gold58 vs gold58 expert (same studies)
    r['gap_rep'] = l['r7_rho'] - l['rep_dropUNK']['tet'] # report regime: R7 OOF on train vs report labels (same studies)
    gs = [r['gap_oai'], r['gap_c4'], r['gap_c4_74']]
    r['cons'] = min(gs) if all(x > 0 for x in gs) else (max(gs) if all(x < 0 for x in gs) else 0.0)
    rows.append(r)
rows.sort(key=lambda r: -r['cons'])
json.dump(rows, open(E + 'coupling_rows.json', 'w'), indent=1)
print('## Coupling table (sorted by consistent gap = min of gap OAI, gap g58, gap g74 when all > 0; max when all < 0; 0 if signs disagree)\n')
print('| pair | OAI n | OAI label tet (phi; B2 tet; graded rho) | OAI model rho (3) | gap OAI | gold58 tet | gold74 tet +- sd (phi) | c4 rho g58 (4-serve mean) | R7 rho g58 / train | gap g58 c4 | gap g74 c4 | gap g58 R7 | report tet dropUNK (UNK=0) | R7 train - report | consistent |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for r in rows:
    print('| %s | %d | %+.2f (%+.2f; %+.2f; %+.2f) | %+.2f | %+.2f | %+.2f | %+.2f +- %.2f (%+.2f) | %+.2f (%+.2f) | %+.2f / %+.2f | %+.2f | %+.2f | %+.2f | %+.2f (%+.2f) | %+.2f | %+.2f |' % (
        ab(r['pair']), r['oai_n'], r['oai_tet'], r['oai_phi'], r['oai_b2tet'], r['oai_grho'], r['oai_m'], r['gap_oai'], r['g58_tet'],
        r['g74_tet'], r['g74_sd'], r['g74_phi'], r['c4'], r['g4'], r['r7g'], r['r7'], r['gap_c4'], r['gap_c4_74'], r['gap_r7g'], r['rep'], r['rep0'],
        r['gap_rep'], r['cons']))

# directional decorrelation table
print('\n## Directional pair transform out_t = pct(f_t) - a * pct(f_o): dAUC on label t (single-label AUC units; macro = /12)\n')
print('| direction | OAI d a-.1 (3 models) | OAI d a.1 (P min) | OAI d a.2 (P min) | theory gap rho-q (3) | R2 OR-matched d.1 / d.2 (3) | R2hi d.1 | R1 cell-matched d.1 (ess) | g58 c4 d.1 / d.2 | R7 vs report d.1 / d.2 | crit1 a.1 | crit1 a.2 | crit2 a.1 | crit2 a.2 | borrow crit1 (a-.1) |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
D = []
for p, b in box.items():
    for dk, dd in b['dir'].items():
        l = loc[p]; g = l['g58_c4_dir'][dk]; rr = l['r7_rep_dir'][dk]
        d1 = [dd[k]['d']['0.1'] for k in M]; d2 = [dd[k]['d']['0.2'] for k in M]; dn = [dd[k]['d']['-0.1'] for k in M]
        P1 = [dd[k]['P']['0.1'] for k in M]; P2 = [dd[k]['P']['0.2'] for k in M]; Pn = [dd[k]['P']['-0.1'] for k in M]
        r21 = [dd[k]['rw']['R2']['d.1'] for k in M]; r22 = [dd[k]['rw']['R2']['d.2'] for k in M]
        r2h = [dd[k]['rw']['R2hi']['d.1'] for k in M]; r11 = [dd[k]['rw']['R1']['d.1'] for k in M]
        gap = [dd[k]['rho_ns'] - dd[k]['q'] for k in M]
        c1a = all(x > 0 for x in d1) and min(P1) >= .9 and all(x < 0 for x in dn)
        c1b = all(x > 0 for x in d2) and min(P2) >= .9 and all(x < 0 for x in dn)
        c2a = all(x > 0 for x in r21) and all(x > 0 for x in r11)
        c2b = all(x > 0 for x in r22)
        bor = all(x > 0 for x in dn) and min(Pn) >= .9 and all(x < 0 for x in d1)
        D.append((np.mean(d1), dk, p))
        print('| %s | %s | %s (%.2f) | %s (%.2f) | %s | %s / %s | %s | %s (%d) | %+.4f / %+.4f | %+.4f / %+.4f | %s | %s | %s | %s | %s |' % (
            abd(dk), '/'.join('%+.4f' % x for x in dn), '/'.join('%+.4f' % x for x in d1), min(P1), '/'.join('%+.4f' % x for x in d2), min(P2),
            '/'.join('%+.2f' % x for x in gap), '/'.join('%+.4f' % x for x in r21), '/'.join('%+.4f' % x for x in r22), '/'.join('%+.4f' % x for x in r2h),
            '/'.join('%+.4f' % x for x in r11), dd[M[0]]['rw']['R1']['ess'], g['0.1'], g['0.2'], rr['0.1'], rr['0.2'],
            'PASS' if c1a else '-', 'PASS' if c1b else '-', 'PASS' if c2a else '-', 'PASS' if c2b else '-', 'PASS' if bor else '-'))
