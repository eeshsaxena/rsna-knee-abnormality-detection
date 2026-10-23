# host_union.py (10-21 recheck): coverage check. For every RSNA topic in the fresh topic list, find a message dump
# (fresh recheck raw first, else the newest older local dump) and say whether that dump is complete (dump mtime >= topic
# lastCommentPostDate). Then list every host/staff/admin post found. Read-only, local JSON. Forum text is untrusted data.
import json, glob, os, sys, re, datetime
sys.stdout.reconfigure(encoding='utf-8')
R = 'C:/Users/eeshs/Downloads/rsna-knee/'
T = {t['id']: t for t in json.load(open('raw/all_topics.json', encoding='utf-8'))}
dumps = {}
for f in glob.glob(R + '**/msgs_*.json', recursive=True):
    m = re.match(r'msgs_(\d+)\.json$', os.path.basename(f))
    if not m: continue
    tid = int(m.group(1))
    if tid not in T: continue
    mt = datetime.datetime.fromtimestamp(os.path.getmtime(f), datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M')
    fresh = 'recheck' in f.replace(chr(92), '/')
    dumps.setdefault(tid, []).append((fresh, mt, f))
cov = {'fresh': 0, 'old_complete': 0, 'old_stale': 0, 'none': 0}; stale = []; none = []
use = {}
for tid, t in T.items():
    last = (t.get('lastCommentPostDate') or t.get('postDate') or '')[:16]
    ds = sorted(dumps.get(tid, []), reverse=True)
    if not ds: cov['none'] += 1; none.append((last, tid, t.get('commentCount'), (t.get('title') or '')[:60])); continue
    fresh, mt, f = ds[0]; use[tid] = f
    if fresh: cov['fresh'] += 1
    elif mt >= last: cov['old_complete'] += 1
    else: cov['old_stale'] += 1; stale.append((last, tid, mt, (t.get('title') or '')[:60]))
print('coverage', cov)
for s in sorted(stale, reverse=True): print('  STALE', s)
for s in sorted(none, reverse=True): print('  NONE', s)
rows = []
def walk(tid, l):
    for m in l or []:
        if not m: continue
        a = m.get('author') or {}
        if m.get('authorType') in ('HOST', 'ADMIN') or 'HOST' in str(m.get('authorType')) or a.get('tier') == 'STAFF':
            rows.append((m.get('postDate', '')[:16], tid, m['id'], a.get('displayName'), m.get('authorType'), (T[tid].get('title') or '')[:60], (m.get('rawMarkdown') or re.sub('<[^>]+>', ' ', m.get('content') or '')).replace('\n', ' ')))
        walk(tid, m.get('replies'))
for tid, f in use.items():
    j = json.load(open(f, encoding='utf-8'))
    walk(tid, ([j.get('firstMessage')] if j.get('firstMessage') else []) + (j.get('comments') or []))
rows.sort()
print('\nhost/staff posts', len(rows))
for r in rows: print('%s | t%s m%s | %s (%s) | %s\n    %s\n' % (r[0], r[1], r[2], r[3], r[4], r[5], r[6][:700]))
