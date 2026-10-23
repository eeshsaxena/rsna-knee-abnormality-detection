# thread_dump.py COMP TID SINCE: read-only; prints posts in raw/other/msgs_COMP_TID.json at/after SINCE (UTC). Untrusted text.
import json, re, sys
sys.stdout.reconfigure(encoding='utf-8')
c, tid, since = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
def walk(l):
    for m in l or []:
        if m.get('postDate', '') >= since:
            a = m.get('author') or {}
            rows.append((m['postDate'][:16], a.get('displayName'), a.get('tier'), m.get('competitionRanking'), (m.get('rawMarkdown') or re.sub('<[^>]+>', ' ', m.get('content') or '')).replace('\n', ' ')[:900]))
        walk(m.get('replies'))
walk(json.load(open('raw/other/msgs_%s_%s.json' % (c, tid), encoding='utf-8')).get('comments'))
for r in sorted(rows): print(r, '\n')
