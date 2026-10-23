# Anonymous read-only Kaggle forum reader (no cookies of Eesh). Writes JSON to OUT dir.
import sys, json, time, requests
BASE='https://www.kaggle.com'
s=requests.Session(); s.get(BASE+'/competitions/rsna-knee-abnormality-detection/discussion', timeout=60)
H={'x-xsrf-token': s.cookies.get('XSRF-TOKEN','')}
def post(m, body):
    for i in range(4):
        r=s.post(BASE+'/api/i/'+m, json=body, headers=H, timeout=60)
        if r.status_code==200: return r.json()
        if r.status_code in (400,401,403,404): return {'_status': r.status_code, '_text': r.text[:500]}
        time.sleep(2*(i+1))
    return {'_status':'fail'}
sys.stdout.reconfigure(encoding='utf-8')
cmd=sys.argv[1]; out=sys.argv[2]
if cmd=='comp':
    j=post('competitions.CompetitionService/GetCompetition', {'competitionName':'rsna-knee-abnormality-detection'})
    json.dump(j, open(out,'w'), indent=1)
    c=j.get('competition',j); print({k:c.get(k) for k in ('id','forumId','title','deadline')} if isinstance(c,dict) else j)
elif cmd=='list':
    fid=int(sys.argv[3]); sort=sys.argv[4]; pages=int(sys.argv[5])
    allt=[]
    for p in range(1,pages+1):
        j=post('discussions.DiscussionsService/GetTopicListByForumId', {'forumId':fid,'sortBy':sort,'page':p,'filterCategoryIds':[],'customGroupingIds':[]})
        t=j.get('topics',[]); allt+=t
        print('page',p,len(t), j.get('_status',''), j.get('_text','')[:200])
        if not t: break
        time.sleep(0.4)
    json.dump(allt, open(out,'w'), indent=1)
elif cmd=='topic':
    ids=sys.argv[3].split(',')
    for tid in ids:
        j=post('discussions.DiscussionsService/GetForumTopicById', {'forumTopicId':int(tid),'includeComments':True})
        json.dump(j, open(out+'/topic_%s.json'%tid,'w'), indent=1)
        ft=j.get('forumTopic',{}); print(tid, ft.get('name'), j.get('_status',''))
        time.sleep(0.4)
