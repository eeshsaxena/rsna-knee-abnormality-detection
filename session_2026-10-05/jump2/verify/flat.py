import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
d = json.load(open(sys.argv[1], encoding='utf-8'))
t = d['forumTopic']
out = []
NL = chr(10)
def walk(c, depth):
    a = c.get('author', {}) or {}
    out.append((c.get('postDate', '')[:16], depth, a.get('displayName'), c.get('competitionRanking'), c.get('id'),
                (c.get('rawMarkdown') or '').replace(NL, ' ')))
    for r in c.get('replies', []) or []:
        walk(r, depth + 1)
print('TOPIC', t.get('name'), '|', t.get('authorUserDisplayName'), '| total comments', t.get('totalMessages'))
fm = t.get('firstMessage') or {}
print('FIRST', (fm.get('postDate') or '')[:16], (fm.get('rawMarkdown') or '').replace(NL, ' '))
for c in t.get('comments', []) or []:
    walk(c, 0)
for o in out:
    print(f"{o[0]} d{o[1]} [{o[2]} | rank {o[3]} | id {o[4]}] {o[5]}")
