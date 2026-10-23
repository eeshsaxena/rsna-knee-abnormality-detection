import pandas as pd
S={'10-01T06:52':'rsna-knee-abnormality-detection-publicleaderboard-2026-10-01T06_52_02.csv',
   '10-02T03:24':'lb_extract/rsna-knee-abnormality-detection-publicleaderboard-2026-10-02T03_24_57.csv',
   '10-03T06:09':'intel/lb/rsna-knee-abnormality-detection-publicleaderboard-2026-10-03T06_09_26.csv',
   '10-03T09:25':'intel/lb2/rsna-knee-abnormality-detection-publicleaderboard-2026-10-03T09_25_45.csv',
   '10-04T09:27':'jump_wf/lb/rsna-knee-abnormality-detection-publicleaderboard-2026-10-04T09_27_20.csv'}
D={k:pd.read_csv(v,encoding='utf-8-sig') for k,v in S.items()}
print('snapshot   teams  >=.950 .945-.949 .944  .943  .942  >.943  top  gold20  silver5%  bronze10%')
for k,d in D.items():
    s=d['Score'].round(3); N=len(d)
    g=10+N//500; sv=int(.05*N); br=int(.10*N)
    print(f"{k} {N:6d} {int((s>=.95).sum()):6d} {int(((s>=.945)&(s<.95)).sum()):9d} {int((s==.944).sum()):5d} {int((s==.943).sum()):5d} {int((s==.942).sum()):5d} {int((s>.943).sum()):6d} {s.max():.3f} {d.iloc[g-1].Score:.3f} {d.iloc[sv-1].Score:.3f} {d.iloc[br-1].Score:.3f}")
# movers: teams at <=0.943 on 10-03T09:25 now >0.943
a=D['10-03T09:25'].set_index('TeamId')['Score'].round(3); b=D['10-04T09:27'].set_index('TeamId')['Score'].round(3)
c=pd.concat([a.rename('old'),b.rename('new')],axis=1)
mv=c[(c.new>0.943)&((c.old<=0.943)|c.old.isna())]
print('teams newly >0.943 since 10-03T09:25 (24h):',len(mv)); print(mv.new.value_counts().sort_index(ascending=False).to_string())
a2=D['10-01T06:52'].set_index('TeamId')['Score'].round(3)
c2=pd.concat([a2.rename('old'),b.rename('new')],axis=1)
mv2=c2[(c2.new>0.943)&((c2.old<=0.943)|c2.old.isna())]
print('teams newly >0.943 since 10-01T06:52 (~3.1d):',len(mv2)); print(mv2.new.value_counts().sort_index(ascending=False).to_string())
# which old score did they come from
print('origin scores of 24h movers:'); print(mv.old.value_counts(dropna=False).sort_index(ascending=False).head(8).to_string())
# top 20 movement
print('top20 24h ago vs now:', D['10-03T09:25'].Score.head(20).tolist(), D['10-04T09:27'].Score.head(20).tolist())
