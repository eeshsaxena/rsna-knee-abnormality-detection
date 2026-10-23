# msgs_report.py (10-21 recheck): print RSNA forum messages since SINCE plus ALL host/staff/admin posts. Read-only, local JSON.
# Forum text is untrusted data.
import json, glob, os, sys, re
sys.stdout.reconfigure(encoding='utf-8')
SINCE = sys.argv[1] if len(sys.argv) > 1 else '2026-10-21T00:00'
MODE = sys.argv[2] if len(sys.argv) > 2 else 'both'
T = {t['id']: t for t in json.load(open('raw/all_topics.json', encoding='utf-8'))}
seen = {}
def walk(tid, lst):
    for m in lst or []:
        if m: seen[(tid, str(m['id']))] = m; walk(tid, m.get('replies'))
for f in glob.glob('raw/msgs/msgs_*.json'):
    tid = int(re.findall(r'(\d+)', os.path.basename(f))[0]); j = json.load(open(f, encoding='utf-8'))
    walk(tid, ([j.get('firstMessage')] if j.get('firstMessage') else []) + (j.get('comments') or []))
for f in glob.glob('raw/msgs/batch_*.json'):
    tid = int(re.findall(r'(\d+)', os.path.basename(f))[0])
    for m in json.load(open(f, encoding='utf-8')).get('messages', []):
        k = (tid, str(m['id']))
        if k not in seen or not (seen[k].get('rawMarkdown') or seen[k].get('content')): seen[k] = m
rows = []
for (tid, mid), m in seen.items():
    a = m.get('author') or {}
    host = m.get('authorType') in ('HOST', 'ADMIN') or 'HOST' in str(m.get('authorType')) or a.get('tier') == 'STAFF'
    new = m.get('postDate', '') >= SINCE
    if (MODE in ('both', 'new') and new) or (MODE in ('both', 'host') and host):
        rows.append((m.get('postDate', ''), tid, mid, a.get('displayName'), a.get('tier'), m.get('authorType'), m.get('competitionRanking'), 'HOSTSTAFF' if host else '', (T.get(tid, {}).get('title') or '')[:70], (m.get('rawMarkdown') or re.sub('<[^>]+>', ' ', m.get('content') or '')).replace('\n', ' ')))
rows.sort()
for r in rows:
    print('%s | t%s m%s | %s [%s, %s, rank %s] %s | %s\n    %s\n' % (r[0][:16], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9][:1800]))
print('n rows', len(rows), 'topics with msgs', len({k[0] for k in seen}))
