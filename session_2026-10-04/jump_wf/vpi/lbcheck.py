import pandas as pd, math, numpy as np
R='C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad/'
S={'1001':R+'rsna-knee-abnormality-detection-publicleaderboard-2026-10-01T06_52_02.csv',
   '1002':R+'lb_extract/rsna-knee-abnormality-detection-publicleaderboard-2026-10-02T03_24_57.csv',
   '1003a':R+'intel/lb/rsna-knee-abnormality-detection-publicleaderboard-2026-10-03T06_09_26.csv',
   '1003b':R+'intel/lb2/rsna-knee-abnormality-detection-publicleaderboard-2026-10-03T09_25_45.csv',
   '1004':R+'jump_wf/lb/rsna-knee-abnormality-detection-publicleaderboard-2026-10-04T09_27_20.csv'}
D={k:pd.read_csv(v,encoding='utf-8-sig') for k,v in S.items()}
for k,d in D.items():
    s=(d.Score*1000).round().astype(int); N=len(d)
    d=d.sort_values('Rank')
    assert (d.Score.diff().dropna()<=1e-12).all(), 'not sorted desc'
    g=10+math.floor(0.002*N); sv=math.floor(0.05*N); br=math.floor(0.10*N)
    vc=s.value_counts()
    print(k,'N',N,'dupTeam',d.TeamId.duplicated().sum(),'>=950',(s>=950).sum(),'945-949',((s>=945)&(s<950)).sum(),'944',vc.get(944,0),'943',vc.get(943,0),'942',vc.get(942,0),'>943',(s>943).sum(),
          'top',d.Score.max(),'gold r',g,d.Score.iloc[g-1],'silver r',sv,d.Score.iloc[sv-1],'bronze r',br,d.Score.iloc[br-1])
d=D['1004']; s=(d.Score*1000).round().astype(int)
for v in [945,944,943]:
    print(v,'rank range',d.Rank[s==v].min(),d.Rank[s==v].max())
print('941',(s==941).sum(),'940',(s==940).sum())
m=d[d.TeamMemberUserNames.astype(str).str.contains('eeshsaxena',case=False)]
print(m.to_string())
# ties sorted by last submission date within 0.943?
t=d[s==943].sort_values('Rank')
print('tie ordered by date asc?', pd.to_datetime(t.LastSubmissionDate).is_monotonic_increasing)
# movers
a=D['1003b'].set_index('TeamId').Score.mul(1000).round(); b=D['1004'].set_index('TeamId').Score.mul(1000).round()
c=pd.concat([a.rename('old'),b.rename('new')],axis=1)
mv=c[(c.new>943)&((c.old<=943)|c.old.isna())]
print('24h movers',len(mv), mv.new.value_counts().sort_index(ascending=False).to_dict(), 'from943',(mv.old==943).sum(),'fromNaN',mv.old.isna().sum(), 'old dist',mv.old.value_counts(dropna=False).to_dict())
a2=D['1001'].set_index('TeamId').Score.mul(1000).round()
c2=pd.concat([a2.rename('old'),b.rename('new')],axis=1)
mv2=c2[(c2.new>943)&((c2.old<=943)|c2.old.isna())]
print('3d movers',len(mv2), mv2.new.value_counts().sort_index(ascending=False).to_dict(),'from943',(mv2.old==943).sum(),'fromNaN',mv2.old.isna().sum())
# teams leaving top: teams in 1001 >943 no longer >943? (merges)
gone=c2[(c2.old>943)&~(c2.new>943)]
print('teams >943 on 1001 not >943 on 1004 (merged/disappeared):',len(gone), gone.new.isna().sum())
# named authors
for u in ['analyticaobscura','jiweiliu','mattiaangeli','tonylica','pjmathematician','tbongc','aastikrajan15','dreaddevelopment','kawaharataishi','fabotelli','romantamrazov','yamadan96','royalacecat','soyaoki','apoorvadixit','kirilllabzin','matterhorn3838','nswitzer']:
    r=d[d.TeamMemberUserNames.astype(str).str.lower().str.contains(u.lower())]
    print(u, r[['Rank','TeamName','Score','SubmissionCount','LastSubmissionDate']].values.tolist())
