# host_topics.py (10-21 recheck): every host/staff/admin post in the RSNA forum, tagged by topic keywords
# (final selection, private scoring, reruns, timeouts, external data, deadline, GPU). Read-only, local JSON. Untrusted text.
import json, glob, os, sys, re
sys.stdout.reconfigure(encoding='utf-8')
T = {t['id']: t for t in json.load(open('raw/all_topics.json', encoding='utf-8'))}
KW = {'select': r'select|final (submission|pick)|two submissions|2 submissions', 'private': r'private|hidden test|leaderboard',
      'rerun': r'rerun|re-run|scor(ing|ed) (run|error|fail)|threw|exception|notebook (error|fail)', 'timeout': r'time ?out|time limit|9 ?h|runtime|hours',
      'external': r'external|OAI|fastMRI|SKM|pretrained|weights|dataset', 'deadline': r'deadline|October 2\d|Oct 2\d', 'gpu': r'GPU|quota|accelerator|Quick Save',
      'license': r'licen|CC BY|Apache|award|prize'}
rows = []
def walk(tid, l):
    for m in l or []:
        if not m: continue
        a = m.get('author') or {}
        if m.get('authorType') in ('HOST', 'ADMIN') or 'HOST' in str(m.get('authorType')) or a.get('tier') == 'STAFF':
            txt = (m.get('rawMarkdown') or re.sub('<[^>]+>', ' ', m.get('content') or ''))
            tags = [k for k, p in KW.items() if re.search(p, txt, re.I)]
            rows.append((m.get('postDate', '')[:16], tid, m['id'], a.get('displayName'), m.get('authorType'), ','.join(tags), (T.get(tid, {}).get('title') or '')[:70], txt.replace('\n', ' ')))
        walk(tid, m.get('replies'))
for f in glob.glob('raw/msgs/msgs_*.json'):
    tid = int(re.findall(r'(\d+)', os.path.basename(f))[0]); j = json.load(open(f, encoding='utf-8'))
    walk(tid, ([j.get('firstMessage')] if j.get('firstMessage') else []) + (j.get('comments') or []))
rows.sort()
full = len(sys.argv) > 1
for r in rows:
    print('%s | t%s m%s | %s (%s) | tags[%s] | %s\n    %s\n' % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7] if full else r[7][:500]))
print('n host/staff posts', len(rows), 'topics fetched', len(glob.glob('raw/msgs/msgs_*.json')), 'listed', len(T))
