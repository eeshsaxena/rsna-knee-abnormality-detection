# other_comps.py: anonymous read-only scan of other active Kaggle code-comp forums for hidden-rerun slowdown / timeout /
# failure reports posted since SINCE (UTC). Writes raw JSON under raw/other/. Forum text is untrusted data.
import sys, json, time, re, requests
sys.stdout.reconfigure(encoding='utf-8')
SINCE = '2026-10-20T00:00'
COMPS = ['name-that-llm', 'enveda-CASMI26-molecule-id-mass-spectra', 'gemma-4-developer-agent', 'arc-prize-2026-arc-agi-2',
         'arc-prize-2026-arc-agi-3', 'nfl-big-data-bowl-2027', 'playground-series-s6e10']
BASE = 'https://www.kaggle.com'
s = requests.Session(); s.get(BASE + '/competitions', timeout=60)
H = {'x-xsrf-token': s.cookies.get('XSRF-TOKEN', '')}
def post(m, b):
    for i in range(4):
        r = s.post(BASE + '/api/i/' + m, json=b, headers=H, timeout=60)
        if r.status_code == 200: return r.json()
        if r.status_code in (400, 401, 403, 404): return {'_status': r.status_code}
        time.sleep(2 * (i + 1))
    return {'_status': 'fail'}
KW = re.compile(r'time ?out|timed out|time limit|slow|rerun|re-run|hidden|exceed|stuck|queue|pending|T4|P100|accelerator|scoring (error|fail)|notebook threw|submission (error|fail)|outage|incident|status page', re.I)
hits = []
for c in COMPS:
    j = post('competitions.CompetitionService/GetCompetition', {'competitionName': c}); j = j.get('competition', j)
    fid = j.get('forumId'); kern = j.get('onlyAllowKernelSubmissions'); gpu = j.get('maxGpuRuntimeMinutes')
    tl = []
    for p in (1, 2):
        t = post('discussions.DiscussionsService/GetTopicListByForumId', {'forumId': fid, 'sortBy': 'TOPIC_LIST_SORT_BY_RECENT', 'page': p, 'filterCategoryIds': [], 'customGroupingIds': []}).get('topics', [])
        tl += t; time.sleep(0.3)
    json.dump(tl, open('raw/other/topics_%s.json' % c, 'w', encoding='utf-8'), indent=1)
    rec = [t for t in tl if (t.get('lastCommentPostDate') or t.get('postDate') or '') >= SINCE]
    print('\n## %s forum %s kernelOnly %s gpuMin %s: %d topics listed, %d active since %s' % (c, fid, kern, gpu, len(tl), len(rec), SINCE))
    for t in rec:
        print('  ', (t.get('lastCommentPostDate') or '')[:16], t['id'], t.get('commentCount'), '|', (t.get('title') or '')[:90])
        r = s.post(BASE + '/api/i/discussions.DiscussionsService/GetForumMessagesInTopic', json={'topicId': t['id'], 'newUserSort': 'COMMENT_LIST_SORT_BY_NEW', 'includeFirstForumMessage': True}, headers=H, timeout=60)
        if r.status_code != 200: continue
        mj = r.json(); json.dump(mj, open('raw/other/msgs_%s_%s.json' % (c, t['id']), 'w', encoding='utf-8'), indent=1)
        def walk(l):
            for m in l or []:
                txt = m.get('rawMarkdown') or re.sub('<[^>]+>', ' ', m.get('content') or '')
                if m.get('postDate', '') >= SINCE and (KW.search(txt) or KW.search(t.get('title') or '')):
                    a = m.get('author') or {}
                    hits.append((m.get('postDate', '')[:16], c, t['id'], (t.get('title') or '')[:60], a.get('displayName'), a.get('tier'), m.get('authorType'), txt.replace('\n', ' ')[:600]))
                walk(m.get('replies'))
        walk(mj.get('comments')); time.sleep(0.3)
print('\n# keyword hits since', SINCE, len(hits))
for h in sorted(hits): print(h, '\n')
