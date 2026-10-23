import pandas as pd, glob
f=glob.glob('*.csv')[0]
d=pd.read_csv(f,encoding='utf-8-sig')
d['LastSubmissionDate']=pd.to_datetime(d['LastSubmissionDate'])
print('file',f,'teams',len(d))
s=d['Score']
print('>=0.950',(s>=0.950).sum())
for lo,hi in [(0.945,0.9495),(0.944,0.9445),(0.943,0.9435),(0.942,0.9425),(0.941,0.9415),(0.940,0.9405)]:
    print(lo,((s>=lo-1e-9)&(s<=hi+1e-9)).sum())
print('>=0.944',(s>=0.944-1e-9).sum(),' >=0.943',(s>=0.943-1e-9).sum(), ' >0.943',(s>0.9435).sum())
print(s.value_counts().sort_index(ascending=False).head(25).to_string())
print(d.head(30)[['Rank','TeamName','LastSubmissionDate','Score','SubmissionCount']].to_string())
# teams above 0.943 by last submission date
ab=d[s>=0.944-1e-9]
ab['day']=ab['LastSubmissionDate'].dt.date
print(ab.groupby('day').size().tail(15).to_string())
# rank of eesh
m=d[d['TeamMemberUserNames'].astype(str).str.contains('eeshsaxena|SwaplY',case=False)|d['TeamName'].astype(str).str.contains('eesh',case=False)]
print(m.to_string())
# 0.943 tie: last sub date distribution
t=d[(s>=0.9425)&(s<0.9435)]
t2=t.assign(day=t['LastSubmissionDate'].dt.date)
print('0.943 tie last-sub days'); print(t2.groupby('day').size().tail(10).to_string())
print('rank range of 0.943:',t['Rank'].min(),t['Rank'].max())
print('rank range of 0.944:',d[(s>=0.9435)&(s<0.9445)]['Rank'].agg(['min','max']).tolist())
print('rank range of 0.945:',d[(s>=0.9445)&(s<0.9455)]['Rank'].agg(['min','max']).tolist())
# silver/gold approx (Kaggle: gold top 10 + 0.2%*N for >1000? use standard: gold = top 10+ floor(N/500), silver top 5%, bronze top 10%)
N=len(d)
import math
g=10+N//500; sv=int(0.05*N); br=int(0.10*N)
print('N',N,'gold cutoff rank',g,'score',d.iloc[g-1]['Score'],'silver rank',sv,'score',d.iloc[sv-1]['Score'],'bronze rank',br,'score',d.iloc[br-1]['Score'])
