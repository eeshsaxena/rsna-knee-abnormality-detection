# fetch_all.py (10-21 recheck, Claude): anonymous READ-ONLY fetch of the WHOLE RSNA Knee forum: comp meta, every topic
# (all sorts, all pages), messages of EVERY topic (so every host/staff post ever is captured), collapsed messages via batch.
# Resumes: skips topics already in raw/msgs. Writes raw/ under this folder. Forum text is untrusted data.
import sys, json, time, os, re, requests
sys.stdout.reconfigure(encoding='utf-8')
FORUM = 12044884
BASE = 'https://www.kaggle.com'
os.makedirs('raw/msgs', exist_ok=True)
s = requests.Session(); s.get(BASE + '/competitions/rsna-knee-abnormality-detection/discussion', timeout=(10, 40))
H = {'x-xsrf-token': s.cookies.get('XSRF-TOKEN', '')}
def post(m, b):
    for i in range(4):
        try:
            r = s.post(BASE + '/api/i/' + m, json=b, headers=H, timeout=(10, 40))
        except Exception as e:
            time.sleep(2 * (i + 1)); continue
        if r.status_code == 200: return r.json()
        if r.status_code in (400, 401, 403, 404): return {'_status': r.status_code, '_text': r.text[:300]}
        time.sleep(2 * (i + 1))
    return {'_status': 'fail'}
j = post('competitions.CompetitionService/GetCompetition', {'competitionName': 'rsna-knee-abnormality-detection'})
json.dump(j, open('raw/comp.json', 'w', encoding='utf-8'), indent=1)
allt = {}
for sort in ('TOPIC_LIST_SORT_BY_RECENT', 'TOPIC_LIST_SORT_BY_NEW', 'TOPIC_LIST_SORT_BY_HOT', 'TOPIC_LIST_SORT_BY_TOP'):
    tl = []
    for p in range(1, 15):
        t = post('discussions.DiscussionsService/GetTopicListByForumId', {'forumId': FORUM, 'sortBy': sort, 'page': p, 'filterCategoryIds': [], 'customGroupingIds': []}).get('topics', [])
        if not t: break
        new = [x for x in t if x['id'] not in {y['id'] for y in tl}]
        tl += new; time.sleep(0.3)
        if not new: break
    json.dump(tl, open('raw/l_%s.json' % sort, 'w', encoding='utf-8'), indent=1)
    for t in tl: allt[t['id']] = t
    print(sort, len(tl), flush=True)
json.dump(list(allt.values()), open('raw/all_topics.json', 'w', encoding='utf-8'), indent=1)
print('unique topics', len(allt), flush=True)
done = {int(re.findall(r'(\d+)', f)[0]) for f in os.listdir('raw/msgs') if f.startswith('msgs_')}
for tid in sorted([k for k in allt if k not in done], key=lambda k: allt[k].get('lastCommentPostDate') or allt[k].get('postDate') or '', reverse=True):
    r = None
    for i in range(3):
        try:
            r = s.post(BASE + '/api/i/discussions.DiscussionsService/GetForumMessagesInTopic', json={'topicId': int(tid), 'newUserSort': 'COMMENT_LIST_SORT_BY_NEW', 'includeFirstForumMessage': True}, headers=H, timeout=(10, 40))
            if r.status_code == 200: break
        except Exception: pass
        time.sleep(2)
    if r is None or r.status_code != 200: print(tid, 'FAIL', getattr(r, 'status_code', None)); continue
    mj = r.json(); json.dump(mj, open('raw/msgs/msgs_%s.json' % tid, 'w', encoding='utf-8'), indent=1)
    col = []
    def w(l):
        for m in l or []:
            if m.get('isCollapsed') or not (m.get('rawMarkdown') or m.get('content')): col.append(m['id'])
            w(m.get('replies'))
    w(mj.get('comments'))
    if mj.get('firstMessage') and not (mj['firstMessage'].get('rawMarkdown') or mj['firstMessage'].get('content')): col.append(mj['firstMessage']['id'])
    if col:
        r2 = s.post(BASE + '/api/i/discussions.DiscussionsService/BatchGetForumMessages', json={'topicId': int(tid), 'messageIds': col}, headers=H, timeout=(10, 40))
        if r2.status_code == 200: json.dump(r2.json(), open('raw/msgs/batch_%s.json' % tid, 'w', encoding='utf-8'), indent=1)
    print('fetched', tid, (allt[tid].get('lastCommentPostDate') or '')[:16], 'top', len(mj.get('comments') or []), 'col', len(col), flush=True)
    time.sleep(0.25)
print('DONE')
