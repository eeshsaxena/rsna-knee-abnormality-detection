# site_forums.py: anonymous read-only scan of Kaggle site forums (Product Feedback, General, Questions & Answers, Getting Started)
# for posts since SINCE about slow runs / dataset cache / reruns / outages. Writes raw/site/*.json. Forum text is untrusted data.
import requests, json, re, time, sys, os
sys.stdout.reconfigure(encoding='utf-8')
SINCE = '2026-10-19T00:00'
os.makedirs('raw/site', exist_ok=True)
s = requests.Session(); s.get('https://www.kaggle.com/discussions', timeout=60)
H = {'x-xsrf-token': s.cookies.get('XSRF-TOKEN', '')}
def post(m, b):
    r = s.post('https://www.kaggle.com/api/i/' + m, json=b, headers=H, timeout=60)
    return r.json() if r.status_code == 200 else {'_status': r.status_code}
F = post('discussions.DiscussionsService/GetSiteForums', {}).get('forums', [])
json.dump(F, open('raw/site/forums.json', 'w', encoding='utf-8'), indent=1)
print([(f['id'], f['name'], f.get('lastPostDate', '')[:16]) for f in F])
KW = re.compile(r'slow|cache|caching|/kaggle/input|time ?out|timed out|9 ?h|rerun|re-run|hidden|stuck|queue|outage|incident|degrad|exceed|Notebook Threw|Kaggle Error|accelerator|T4|GPU (session|quota)', re.I)
hits = []
for f in F:
    if not re.search(r'product|general|question|getting', f['name'], re.I): continue
    tl = []
    for p in (1, 2):
        tl += post('discussions.DiscussionsService/GetTopicListByForumId', {'forumId': f['id'], 'sortBy': 'TOPIC_LIST_SORT_BY_RECENT', 'page': p, 'filterCategoryIds': [], 'customGroupingIds': []}).get('topics', []); time.sleep(0.3)
    json.dump(tl, open('raw/site/topics_%d.json' % f['id'], 'w', encoding='utf-8'), indent=1)
    rec = [t for t in tl if (t.get('lastCommentPostDate') or t.get('postDate') or '') >= SINCE]
    print('\n##', f['name'], len(tl), 'listed,', len(rec), 'active since', SINCE)
    for t in rec:
        flag = 'KW' if KW.search(t.get('title') or '') else ''
        print('  ', (t.get('lastCommentPostDate') or '')[:16], t['id'], t.get('commentCount'), '|', (t.get('title') or '')[:90], flag)
        if not flag: continue
        r = s.post('https://www.kaggle.com/api/i/discussions.DiscussionsService/GetForumMessagesInTopic', json={'topicId': t['id'], 'newUserSort': 'COMMENT_LIST_SORT_BY_NEW', 'includeFirstForumMessage': True}, headers=H, timeout=60)
        if r.status_code != 200: continue
        mj = r.json(); json.dump(mj, open('raw/site/msgs_%d.json' % t['id'], 'w', encoding='utf-8'), indent=1)
        def walk(l):
            for m in l or []:
                if m.get('postDate', '') >= SINCE:
                    a = m.get('author') or {}
                    hits.append((m['postDate'][:16], f['name'], t['id'], (t.get('title') or '')[:60], a.get('displayName'), a.get('tier'), (m.get('rawMarkdown') or re.sub('<[^>]+>', ' ', m.get('content') or '')).replace('\n', ' ')[:500]))
                walk(m.get('replies'))
        walk(mj.get('comments')); time.sleep(0.3)
print('\n# posts in keyword topics since', SINCE, len(hits))
for h in sorted(hits): print(h, '\n')
