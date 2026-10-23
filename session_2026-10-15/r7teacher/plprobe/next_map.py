#!/usr/bin/env python
# next_map.py CURRENT_BEST_MAP.json GROUP KEY=VALUE NAME [--note TEXT]: write maps/NAME.json = the current best map with ONE change
# (a group's Raptor share s or fleet share f), so every probe is "team best + one change" (the only readable design).
# GROUP: OA | TRAUMA | MEN | FLUID | ACL | a single label name. Example:
#   python next_map.py maps/oa65.json TRAUMA s=0.35 oa65-tr35
import sys, json, os, argparse
GROUPS = {'OA': ['Medial OA', 'Lateral OA', 'PF OA'], 'TRAUMA': ['Fracture', 'MCL', 'Contusion'],
          'MEN': ['Medial Meniscus', 'Lateral Meniscus'], 'FLUID': ['Effusion', 'Synovitis', "Baker's"], 'ACL': ['ACL']}
ap = argparse.ArgumentParser()
ap.add_argument('cur'); ap.add_argument('group'); ap.add_argument('kv'); ap.add_argument('name'); ap.add_argument('--note', default='')
A = ap.parse_args()
m = json.load(open(A.cur, encoding='utf-8'))
k, v = A.kv.split('='); assert k in ('s', 'f'); v = float(v)
labs = GROUPS.get(A.group, [A.group])
m = {'name': A.name, 'note': (A.note or f'probe: {os.path.basename(A.cur)} + {A.group} {k}={v}'),
     'default': m.get('default', {'s': 0.5, 'f': 0.0}), 'groups': dict(m.get('groups') or {}), 'labels': dict(m.get('labels') or {})}
for l in labs:   # a label-level entry wins over groups in the builder, so write the change at label level
    e = dict(m['labels'].get(l, {})); e[k] = v; m['labels'][l] = e
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maps', A.name + '.json')
assert not os.path.exists(out), out + ' exists'
json.dump(m, open(out, 'w', encoding='utf-8'), indent=1)
print('WROTE', out)
