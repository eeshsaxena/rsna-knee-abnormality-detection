# fetch_forum.py (10-16, Claude): anonymous READ-ONLY fetch of the competition forum topic list (Kaggle web API, no cookies
# beyond the anonymous XSRF). Writes topics_<tag>.json and prints topics with activity after --since (UTC ISO).
import sys, json, time, requests, argparse
ap = argparse.ArgumentParser(); ap.add_argument('--since', default='2026-10-15T10:25'); ap.add_argument('--tag', default='now')
A = ap.parse_args()
s = requests.Session(); s.get('https://www.kaggle.com/competitions', timeout=60)
h = {'x-xsrf-token': s.cookies.get('XSRF-TOKEN', '')}
def api(svc, m, b):
    r = s.post(f'https://www.kaggle.com/api/i/{svc}/{m}', json=b, headers=h, timeout=90)
    return r.status_code, (r.json() if r.status_code == 200 else r.text[:300])
out = {}
for sort in ('TOPIC_LIST_SORT_BY_LAST_COMMENT_DATE', 'TOPIC_LIST_SORT_BY_CREATION_DATE'):
    code, j = api('discussions.DiscussionsService', 'GetTopicListByForumId', {'forumId': 12044884, 'page': 1, 'sortBy': sort})
    print(sort, code, (list(j.keys()) if isinstance(j, dict) else j)[:10] if code == 200 else j)
    out[sort] = j
json.dump(out, open(f'topics_{A.tag}.json', 'w', encoding='utf-8'))
