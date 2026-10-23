import json,glob,os,sys
sys.stdout.reconfigure(encoding='utf-8')
src=sys.argv[1]; dst=sys.argv[2]; os.makedirs(dst,exist_ok=True)
for f in sorted(glob.glob(src+'/msgs_*.json')):
    tid=os.path.basename(f)[5:-5]
    j=json.load(open(f,encoding='utf-8'))
    extra={}
    bf=src+'/batch_%s.json'%tid
    if os.path.exists(bf):
        b=json.load(open(bf,encoding='utf-8'))
        for m in b.get('forumMessages') or b.get('messages') or []:
            extra[m['id']]=m
    rows=[]
    def w(l,d):
        for m in l or []:
            mm=extra.get(m['id'],m)
            a=mm.get('author') or m.get('author') or {}
            rows.append((mm.get('postDate') or m.get('postDate') or '', d, a.get('displayName',''), m.get('competitionRanking',''), a.get('tier',''), mm.get('rawMarkdown') or mm.get('content') or ''))
            w(m.get('replies'),d+1)
    w(j.get('comments'),1)
    with open(dst+'/flat_%s.txt'%tid,'w',encoding='utf-8') as o:
        for r in sorted(rows):
            o.write('=== %s | d%d | %s | rank %s | %s\n%s\n\n'%(r[0][:16],r[1],r[2],r[3],r[4],r[5]))
    print(tid,len(rows))
