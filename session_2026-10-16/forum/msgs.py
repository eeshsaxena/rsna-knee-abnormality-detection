import sys, json, time, requests
sys.stdout.reconfigure(encoding='utf-8')
BASE='https://www.kaggle.com'
s=requests.Session(); s.get(BASE+'/competitions/rsna-knee-abnormality-detection/discussion', timeout=60)
H={'x-xsrf-token': s.cookies.get('XSRF-TOKEN','')}
out=sys.argv[1]
for tid in sys.argv[2].split(','):
    body={'topicId':int(tid),'newUserSort':'COMMENT_LIST_SORT_BY_NEW','includeFirstForumMessage':True}
    r=s.post(BASE+'/api/i/discussions.DiscussionsService/GetForumMessagesInTopic', json=body, headers=H, timeout=60)
    if r.status_code!=200: print(tid, r.status_code, r.text[:300]); continue
    j=r.json(); json.dump(j, open(out+'/msgs_%s.json'%tid,'w',encoding='utf-8'), indent=1)
    cm=j.get('comments') or []
    # collapsed ids -> batch get
    ids=[]
    def w(l):
        for m in l or []:
            if m.get('isCollapsed') or not (m.get('rawMarkdown') or m.get('content')): ids.append(m['id'])
            w(m.get('replies'))
    w(cm)
    if ids:
        r2=s.post(BASE+'/api/i/discussions.DiscussionsService/BatchGetForumMessages', json={'topicId':int(tid),'messageIds':ids}, headers=H, timeout=60)
        if r2.status_code==200: json.dump(r2.json(), open(out+'/batch_%s.json'%tid,'w',encoding='utf-8'), indent=1)
        print(tid, 'collapsed', len(ids), r2.status_code)
    print(tid, 'top-level', len(cm), list(j.keys()))
    time.sleep(0.4)
