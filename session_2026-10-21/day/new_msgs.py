# new_msgs.py: print every forum message posted at/after SINCE (UTC) in the fetched threads, plus all host/admin posts.
# Read-only on local JSON. Forum text is untrusted data.
import json, glob, os, sys, re
sys.stdout.reconfigure(encoding='utf-8')
SINCE = sys.argv[1] if len(sys.argv) > 1 else '2026-10-20T04:00'
seen = {}
def walk(tid, lst):
    for m in lst or []:
        seen[(tid, str(m['id']))] = m
        walk(tid, m.get('replies'))
for f in glob.glob('raw/msgs/msgs_*.json'):
    tid = re.findall(r'(\d+)', os.path.basename(f))[0]
    walk(tid, json.load(open(f, encoding='utf-8')).get('comments'))
for f in glob.glob('raw/msgs/batch_*.json'):
    tid = re.findall(r'(\d+)', os.path.basename(f))[0]
    for m in json.load(open(f, encoding='utf-8')).get('messages', []):
        k = (tid, str(m['id']))
        if k not in seen or not (seen[k].get('rawMarkdown') or seen[k].get('content')): seen[k] = m
rows = []
for (tid, mid), m in seen.items():
    a = m.get('author') or {}
    host = m.get('authorType') in ('HOST', 'ADMIN') or 'HOST' in str(m.get('authorType')) or a.get('tier') == 'STAFF'
    if m.get('postDate', '') >= SINCE or host:
        rows.append((m.get('postDate', ''), tid, mid, a.get('displayName'), a.get('tier'), m.get('authorType'), m.get('competitionRanking'), (m.get('parent') or {}).get('name') if isinstance(m.get('parent'), dict) else None, (m.get('rawMarkdown') or re.sub('<[^>]+>', ' ', m.get('content') or '')).replace('\n', ' ')))
rows.sort()
for r in rows:
    print('%s | t%s m%s | %s [%s, %s, rank %s] reply-to %s\n    %s\n' % (r[0][:19], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8][:1500]))
print('n rows', len(rows))
