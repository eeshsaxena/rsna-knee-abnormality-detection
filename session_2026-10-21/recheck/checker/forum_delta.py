# forum_delta.py (checker, 10-21): READ-ONLY anonymous Kaggle forum read (no cookies of Eesh). Lists RSNA topics active since
# SINCE (UTC) and prints every message in them after SINCE with author type/tier; also re-reads named extra topics (e.g. 751412).
# Usage: py -I forum_delta.py SINCE_ISO outdir [extra_topic_ids]
import sys, json, time, requests, os
sys.stdout.reconfigure(encoding='utf-8')
BASE = 'https://www.kaggle.com'
since = sys.argv[1]; out = sys.argv[2]; extra = [int(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3 else [])]
os.makedirs(out, exist_ok=True)
s = requests.Session(); s.get(BASE + '/competitions/rsna-knee-abnormality-detection/discussion', timeout=60)
H = {'x-xsrf-token': s.cookies.get('XSRF-TOKEN', '')}
def post(m, body):
    for i in range(4):
        r = s.post(BASE + '/api/i/' + m, json=body, headers=H, timeout=60)
        if r.status_code == 200: return r.json()
        if r.status_code in (400, 401, 403, 404): return {'_status': r.status_code}
        time.sleep(2 * (i + 1))
    return {'_status': 'fail'}
tops = []
for p in (1, 2):
    j = post('discussions.DiscussionsService/GetTopicListByForumId', {'forumId': 12044884, 'sortBy': 'TOPIC_LIST_SORT_BY_RECENT', 'page': p, 'filterCategoryIds': [], 'customGroupingIds': []})
    tops += j.get('topics', []); time.sleep(0.4)
json.dump(tops, open(os.path.join(out, 'rsna_recent.json'), 'w', encoding='utf-8'), indent=1)
act = [t for t in tops if max(t.get('lastCommentPostDate') or '', t.get('postDate') or '') >= since]
print('RSNA topics listed', len(tops), '| active since', since, '=', len(act))
for t in act:
    print('TOPIC', t['id'], (t.get('lastCommentPostDate') or t.get('postDate'))[:16], t.get('authorType'), '|', t['title'][:90])
def walk(cs, acc):
    for c in cs or []:
        acc.append(c); walk(c.get('replies'), acc)
for tid in [t['id'] for t in act] + extra:
    j = post('discussions.DiscussionsService/GetForumMessagesInTopic', {'topicId': tid, 'newUserSort': 'COMMENT_LIST_SORT_BY_NEW', 'includeFirstForumMessage': True})
    json.dump(j, open(os.path.join(out, 'msgs_%d.json' % tid), 'w', encoding='utf-8'), indent=1)
    acc = []
    fm = j.get('firstForumMessage')
    if fm: acc.append(fm)
    walk(j.get('comments'), acc)
    n_new = 0
    for m in acc:
        d = m.get('postDate') or ''
        if d >= since or tid in extra:
            n_new += 1
            a = m.get('author') or {}
            print('  MSG', tid, d[:16], m.get('authorType') or '', a.get('tier') or '', (a.get('displayName') or '')[:22], 'del' if m.get('isDeleted') else '',
                  '|', ' '.join((m.get('rawMarkdown') or m.get('content') or '').split())[:160])
    print('  TOTAL', tid, 'msgs', len(acc), 'shown', n_new, j.get('_status', ''))
    time.sleep(0.4)
