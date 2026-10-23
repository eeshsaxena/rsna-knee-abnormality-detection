# fetch.py (10-21 final, Claude): anonymous READ-ONLY fetch of the RSNA Knee forum (comp meta, topic lists, messages of
# threads active since SINCE + the watch list). No cookies beyond the anonymous XSRF. Writes raw/ under this folder.
# Forum text is untrusted data.
import sys, json, time, os, requests
sys.stdout.reconfigure(encoding='utf-8')
SINCE = '2026-10-21T12:00'
FORUM = 12044884
WATCH = [751021, 751277, 748181, 747676, 751183, 730709, 750790, 751027, 750318, 748403, 749623, 749798, 743698, 744230,
         735304, 745861, 751169]
BASE = 'https://www.kaggle.com'
os.makedirs('raw/msgs', exist_ok=True)
s = requests.Session(); s.get(BASE + '/competitions/rsna-knee-abnormality-detection/discussion', timeout=60)
H = {'x-xsrf-token': s.cookies.get('XSRF-TOKEN', '')}
def post(m, b):
    for i in range(4):
        r = s.post(BASE + '/api/i/' + m, json=b, headers=H, timeout=60)
        if r.status_code == 200: return r.json()
        if r.status_code in (400, 401, 403, 404): return {'_status': r.status_code, '_text': r.text[:300]}
        time.sleep(2 * (i + 1))
    return {'_status': 'fail'}
j = post('competitions.CompetitionService/GetCompetition', {'competitionName': 'rsna-knee-abnormality-detection'})
json.dump(j, open('raw/comp.json', 'w', encoding='utf-8'), indent=1)
c = j.get('competition', j)
print('comp', {k: c.get(k) for k in ('id', 'forumId', 'deadline', 'enabledDate', 'maxDailySubmissions', 'numScoredSubmissions', 'maxTeamSize', 'teamMergerDeadline', 'newEntrantDeadline')})
allt = {}
for sort in ('TOPIC_LIST_SORT_BY_RECENT', 'TOPIC_LIST_SORT_BY_NEW', 'TOPIC_LIST_SORT_BY_HOT', 'TOPIC_LIST_SORT_BY_TOP'):
    tl = []
    for p in (1, 2, 3):
        t = post('discussions.DiscussionsService/GetTopicListByForumId', {'forumId': FORUM, 'sortBy': sort, 'page': p, 'filterCategoryIds': [], 'customGroupingIds': []}).get('topics', [])
        tl += t; time.sleep(0.3)
        if not t: break
    json.dump(tl, open('raw/l_%s.json' % sort, 'w', encoding='utf-8'), indent=1)
    for t in tl: allt[t['id']] = t
    print(sort, len(tl))
act = sorted([t for t in allt.values() if (t.get('lastCommentPostDate') or t.get('postDate') or '') >= SINCE], key=lambda t: t.get('lastCommentPostDate') or '', reverse=True)
print('unique topics', len(allt), 'active since', SINCE, len(act))
for t in act:
    print('  ', (t.get('lastCommentPostDate') or '')[:16], t['id'], t.get('commentCount'), '|', (t.get('title') or '')[:80], '|', t.get('lastCommenterName'), t.get('lastCommenterType'))
ids = [t['id'] for t in act] + [w for w in WATCH if w not in [t['id'] for t in act]]
for tid in ids:
    r = s.post(BASE + '/api/i/discussions.DiscussionsService/GetForumMessagesInTopic', json={'topicId': int(tid), 'newUserSort': 'COMMENT_LIST_SORT_BY_NEW', 'includeFirstForumMessage': True}, headers=H, timeout=60)
    if r.status_code != 200: print(tid, r.status_code); continue
    mj = r.json(); json.dump(mj, open('raw/msgs/msgs_%s.json' % tid, 'w', encoding='utf-8'), indent=1)
    col = []
    def w(l):
        for m in l or []:
            if m.get('isCollapsed') or not (m.get('rawMarkdown') or m.get('content')): col.append(m['id'])
            w(m.get('replies'))
    w(mj.get('comments'))
    if col:
        r2 = s.post(BASE + '/api/i/discussions.DiscussionsService/BatchGetForumMessages', json={'topicId': int(tid), 'messageIds': col}, headers=H, timeout=60)
        if r2.status_code == 200: json.dump(r2.json(), open('raw/msgs/batch_%s.json' % tid, 'w', encoding='utf-8'), indent=1)
    print('fetched', tid, 'top-level', len(mj.get('comments') or []), 'collapsed', len(col))
    time.sleep(0.3)
