# diff_topics.py: compare forum topic lists now vs 10-21 03:09 UTC read (session_2026-10-21/day/raw). Read-only, prints metadata only.
import json, glob, sys
sys.stdout.reconfigure(encoding='utf-8')
def load(pat):
    d = {}
    for f in glob.glob(pat):
        for t in json.load(open(f, encoding='utf-8')):
            d[t['id']] = t
    return d
old = load('../day/raw/l_TOPIC_LIST_SORT_BY_*.json'); new = load('raw/l_TOPIC_LIST_SORT_BY_*.json')
k = next(iter(new.values())); print('keys:', list(k.keys()))
def lc(t):
    return t.get('lastCommentPostDate') or t.get('lastCommentDate') or t.get('postDate') or ''
print('unique old', len(old), 'new', len(new))
rows = []
for i, t in new.items():
    o = old.get(i)
    tag = 'NEW' if o is None else ''
    if o is not None:
        if t.get('commentCount') != o.get('commentCount'): tag += 'posts %s->%s ' % (o.get('commentCount'), t.get('commentCount'))
        if lc(t) != lc(o): tag += 'lastc '
        if t.get('votes') != o.get('votes'): tag += 'votes %s->%s' % (o.get('votes'), t.get('votes'))
    rows.append((lc(t), i, tag, t))
rows.sort(reverse=True)
for d, i, tag, t in rows[:40]:
    print(d[:19], i, t.get('commentCount'), '|', (t.get('title') or t.get('name') or '')[:70], '|', (t.get('authorUser') or {}).get('displayName') if isinstance(t.get('authorUser'), dict) else t.get('authorUserName'), '|', t.get('lastCommenter', {}).get('displayName') if isinstance(t.get('lastCommenter'), dict) else '', '|', tag)
print('fell off:', [ (i, lc(o)[:10]) for i, o in old.items() if i not in new])
