# oai_claimed.py (10-21 levers, Claude). LOCAL ONLY: parses the OAI AGGREGATE JSON already on the laptop
# (session_2026-10-19/ex/box_coupling.log, written 10-19 by a read-only box script; per direction x model means / P only, no ids or rows).
# Prints, per direction: OAI raw (OAI's own expert labels, unweighted) d at a -.1 / +.1 / +.2, and the gold74-OR reweighted R2, R2lo (-1 sd),
# R2hi (+1 sd) d at a +.1, for D5 / s1044 / s1045, plus a sign summary. Writes oai_claimed.log + oai_claimed.json.
import json
AB = {"ACL": "ACL", "MCL": "MCL", "Medial Meniscus": "MM", "Lateral Meniscus": "LM", "Medial OA": "MOA", "Lateral OA": "LOA",
      "PF OA": "PF", "Effusion": "Eff", "Synovitis": "Syn", "Baker's": "Bak", "Contusion": "Cont", "Fracture": "Frac"}
oai = json.loads(open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/box_coupling.log').read().split('JSON_BEGIN')[1].strip().split('\n')[0])
out = {}; f = open('oai_claimed.log', 'w')
def P(s): print(s); f.write(s + '\n')
P('OAI CLAIMED (box_coupling.log JSON): x1e4 label AUC; models D5 / s1044 / s1045; sign: + all 3 > 0')
for pk, e in oai.items():
    for dk, dd in e['dir'].items():
        t, o = dk.split(' <- '); ms = list(dd)
        g = lambda f_: [f_(dd[m]) for m in ms]
        row = {'raw-.1': g(lambda x: x['d']['-0.1']), 'raw.1': g(lambda x: x['d']['0.1']), 'raw.2': g(lambda x: x['d']['0.2']),
               'R2.1': g(lambda x: x['rw']['R2']['d.1']), 'R2lo.1': g(lambda x: x['rw']['R2lo']['d.1']), 'R2hi.1': g(lambda x: x['rw']['R2hi']['d.1']),
               'P.1': g(lambda x: x['P']['0.1']), 'gap': g(lambda x: x['rho_ns'] - x['q'])}
        out[AB[t] + '<-' + AB[o]] = row
        s = lambda v: ('+' if all(x > 0 for x in v) else ('-' if all(x < 0 for x in v) else ' ')) + '/'.join('%+.0f' % (x * 1e4) for x in v)
        P('%-10s raw a-.1 %-16s raw a.1 %-16s R2 %-16s R2lo %-16s R2hi %-16s gap %s' % (AB[t] + '<-' + AB[o], s(row['raw-.1']), s(row['raw.1']),
          s(row['R2.1']), s(row['R2lo.1']), s(row['R2hi.1']), '/'.join('%+.2f' % x for x in row['gap'])))
json.dump(out, open('oai_claimed.json', 'w'), indent=1)
