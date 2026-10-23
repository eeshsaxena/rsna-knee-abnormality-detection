# table_md.py: renders the condensed coupling table (markdown) from coupling_rows.json + local_gapboot.log. Local only.
import json, re
E = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/'
rows = json.load(open(E + 'coupling_rows.json'))
AB = {"ACL": "ACL", "Medial Meniscus": "MM", "Lateral Meniscus": "LM", "Medial OA": "MOA", "Lateral OA": "LOA", "PF OA": "PFOA",
      "Effusion": "Eff", "Synovitis": "Syn", "Baker's": "Bak"}
gb = {}
for ln in open(E + 'local_gapboot.log'):
    m = re.match(r'\| (\S+) \| ([+-][\d.]+) \| ([\d.]+) \| ([\d.]+) \|', ln)
    if m: gb[m.group(1)] = (float(m.group(3)), float(m.group(4)))
print('| # | pair | OAI expert tet (phi) | OAI models rho | gap OAI | gold58 tet | gold74 tet +- sd | c4 rho g58 | gap g58 (sd, P>0) | gap g74 | R7 rho g58 / train | report tet (UNK=0) | consistent gap |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for i, r in enumerate(rows, 1):
    p = '-'.join(AB[x] for x in r['pair'].split(' | ')); sd, P = gb[p]
    print('| %d | %s | %+.2f (%+.2f) | %+.2f | %+.2f | %+.2f | %+.2f +- %.2f | %+.2f | %+.2f (%.2f, %.2f) | %+.2f | %+.2f / %+.2f | %+.2f (%+.2f) | %s |' % (
        i, p, r['oai_tet'], r['oai_phi'], r['oai_m'], r['gap_oai'], r['g58_tet'], r['g74_tet'], r['g74_sd'], r['c4'], r['gap_c4'], sd, P,
        r['gap_c4_74'], r['r7g'], r['r7'], r['rep'], r['rep0'], ('%+.2f' % r['cons']) if r['cons'] != 0 else 'mixed'))
