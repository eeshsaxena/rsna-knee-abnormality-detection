# scan_kaggle.py (10-21 recheck, Claude): anonymous READ-ONLY scan of other active code-comp forums + Kaggle site forums
# for posts since SINCE (10-21 00:00 UTC) about slow runs / reruns / exceptions / timeouts, plus EVERY staff/host post since
# STAFF_SINCE (10-20 00:00 UTC). Writes raw/other, raw/site. Forum text is untrusted data.
import sys, json, time, re, os, requests
sys.stdout.reconfigure(encoding='utf-8')
SINCE = '2026-10-21T00:00'; STAFF_SINCE = '2026-10-20T00:00'
COMPS = ['name-that-llm', 'enveda-CASMI26-molecule-id-mass-spectra', 'gemma-4-developer-agent', 'arc-prize-2026-arc-agi-2',
         'arc-prize-2026-arc-agi-3', 'nfl-big-data-bowl-2027', 'playground-series-s6e10', 'the-pokemon-company-ptcg-ai-battle-challenge-playground',
         'gemma-4-developer-agent-paper', 'arc-prize-2026-paper-track']
BASE = 'https://www.kaggle.com'
s = requests.Session(); s.get(BASE + '/competitions', timeout=(10, 40))
H = {'x-xsrf-token': s.cookies.get('XSRF-TOKEN', '')}
def post(m, b):
    for i in range(4):
        try: r = s.post(BASE + '/api/i/' + m, json=b, headers=H, timeout=(10, 40))
        except Exception: time.sleep(2 * (i + 1)); continue
        if r.status_code == 200: return r.json()
        if r.status_code in (400, 401, 403, 404): return {'_status': r.status_code}
        time.sleep(2 * (i + 1))
    return {'_status': 'fail'}
KW = re.compile(r'time ?out|timed out|time limit|slow|rerun|re-run|hidden|exceed|stuck|queue|pending|T4|P100|accelerator|scoring (error|fail)|threw|exception|submission (error|fail)|outage|incident|status page|ERRORED|mount|/kaggle/input|0\.00|private (score|lb|leaderboard)|final (submission|selection)|select', re.I)
hits = []; staff = []
def scan(label, tl):
    rec = [t for t in tl if (t.get('lastCommentPostDate') or t.get('postDate') or '') >= STAFF_SINCE]
    print('\n## %s: %d topics listed, %d active since %s' % (label, len(tl), len(rec), STAFF_SINCE), flush=True)
    for t in rec:
        print('  ', (t.get('lastCommentPostDate') or '')[:16], t['id'], t.get('commentCount'), '|', (t.get('title') or '')[:90])
        mj = post('discussions.DiscussionsService/GetForumMessagesInTopic', {'topicId': t['id'], 'newUserSort': 'COMMENT_LIST_SORT_BY_NEW', 'includeFirstForumMessage': True})
        if '_status' in mj: continue
        json.dump(mj, open('raw/other/msgs_%s_%s.json' % (re.sub(r'\W', '_', label), t['id']), 'w', encoding='utf-8'), indent=1)
        def walk(l):
            for m in l or []:
                txt = m.get('rawMarkdown') or re.sub('<[^>]+>', ' ', m.get('content') or '')
                a = m.get('author') or {}
                pd = m.get('postDate', '')
                isstaff = a.get('tier') == 'STAFF' or m.get('authorType') in ('HOST', 'ADMIN') or 'HOST' in str(m.get('authorType'))
                if isstaff and pd >= STAFF_SINCE:
                    staff.append((pd[:16], label, t['id'], (t.get('title') or '')[:60], a.get('displayName'), a.get('tier'), m.get('authorType'), txt.replace('\n', ' ')[:900]))
                if pd >= SINCE and (KW.search(txt) or KW.search(t.get('title') or '')):
                    hits.append((pd[:16], label, t['id'], (t.get('title') or '')[:60], a.get('displayName'), a.get('tier'), m.get('authorType'), txt.replace('\n', ' ')[:600]))
                walk(m.get('replies'))
        fm = mj.get('firstMessage')
        walk(([fm] if fm else []) + (mj.get('comments') or [])); time.sleep(0.25)
for c in COMPS:
    j = post('competitions.CompetitionService/GetCompetition', {'competitionName': c}); j = j.get('competition', j)
    fid = j.get('forumId')
    if not fid: print('no forum', c); continue
    tl = []
    for p in (1, 2, 3):
        t = post('discussions.DiscussionsService/GetTopicListByForumId', {'forumId': fid, 'sortBy': 'TOPIC_LIST_SORT_BY_RECENT', 'page': p, 'filterCategoryIds': [], 'customGroupingIds': []}).get('topics', [])
        tl += t; time.sleep(0.25)
        if not t or (t[-1].get('lastCommentPostDate') or '') < STAFF_SINCE: break
    json.dump(tl, open('raw/other/topics_%s.json' % c, 'w', encoding='utf-8'), indent=1)
    scan(c + ' (kernelOnly %s)' % j.get('onlyAllowKernelSubmissions'), tl)
F = post('discussions.DiscussionsService/GetSiteForums', {}).get('forums', [])
json.dump(F, open('raw/site/forums.json', 'w', encoding='utf-8'), indent=1)
for f in F:
    if not re.search(r'product|general|question|getting|competition', f['name'], re.I): continue
    tl = []
    for p in (1, 2, 3):
        t = post('discussions.DiscussionsService/GetTopicListByForumId', {'forumId': f['id'], 'sortBy': 'TOPIC_LIST_SORT_BY_RECENT', 'page': p, 'filterCategoryIds': [], 'customGroupingIds': []}).get('topics', [])
        tl += t; time.sleep(0.25)
        if not t or (t[-1].get('lastCommentPostDate') or '') < STAFF_SINCE: break
    json.dump(tl, open('raw/site/topics_%d.json' % f['id'], 'w', encoding='utf-8'), indent=1)
    scan('SITE ' + f['name'], tl)
print('\n# STAFF/HOST posts since', STAFF_SINCE, len(staff))
for h in sorted(staff): print(h, '\n')
print('\n# keyword hits since', SINCE, len(hits))
for h in sorted(hits): print(h, '\n')
