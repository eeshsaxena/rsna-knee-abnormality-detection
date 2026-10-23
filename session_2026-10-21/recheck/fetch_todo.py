# fetch_todo.py (10-21 recheck): READ-ONLY anonymous fetch of older RSNA topics whose titles touch selection / private /
# reruns / timeouts / external data / rules and that have >= 1 reply (only replies can be host answers). Untrusted text.
import sys, json, time, requests
sys.stdout.reconfigure(encoding='utf-8')
T = {t['id']: t for t in json.load(open('raw/all_topics.json', encoding='utf-8'))}
ids = [i for i in json.load(open('todo_ids.json')) if (T[i].get('commentCount') or 0) >= 1]
BASE = 'https://www.kaggle.com'
s = requests.Session(); s.get(BASE + '/competitions/rsna-knee-abnormality-detection/discussion', timeout=(10, 40))
H = {'x-xsrf-token': s.cookies.get('XSRF-TOKEN', '')}
print('todo', len(ids), flush=True)
for tid in ids:
    r = None
    for i in range(5):
        try:
            r = s.post(BASE + '/api/i/discussions.DiscussionsService/GetForumMessagesInTopic', json={'topicId': tid, 'newUserSort': 'COMMENT_LIST_SORT_BY_NEW', 'includeFirstForumMessage': True}, headers=H, timeout=(10, 40))
            if r.status_code != 429: break
        except Exception as e:
            print(tid, 'EXC', e, flush=True)
        time.sleep(6 * (i + 1))
    if r is None: continue
    if r.status_code != 200: print(tid, 'HTTP', r.status_code, flush=True); time.sleep(2); continue
    json.dump(r.json(), open('raw/msgs/msgs_%d.json' % tid, 'w', encoding='utf-8'), indent=1)
    print('fetched', tid, flush=True); time.sleep(0.25)
print('DONE')
