# staff_other.py: print STAFF / host posts since 10-12 UTC in the fetched other-comp threads (raw/other). Read-only.
import json, glob, re, sys, os
sys.stdout.reconfigure(encoding='utf-8')
rows = []
for f in glob.glob('raw/other/msgs_*.json'):
    c, tid = re.findall(r'msgs_(.+)_(\d+)[.]json', os.path.basename(f))[0]
    def walk(l):
        for m in l or []:
            a = m.get('author') or {}
            if (a.get('tier') == 'STAFF' or m.get('authorType') in ('HOST', 'ADMIN')) and m.get('postDate', '') >= '2026-10-12':
                rows.append((m['postDate'][:16], c, tid, a.get('displayName'), m.get('authorType'), (m.get('rawMarkdown') or re.sub('<[^>]+>', ' ', m.get('content') or '')).replace('\n', ' ')[:1600]))
            walk(m.get('replies'))
    walk(json.load(open(f, encoding='utf-8')).get('comments'))
for r in sorted(rows): print(r, '\n')
print(len(rows))
