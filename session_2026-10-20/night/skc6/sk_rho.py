# skeptic C6' (10-20 night, Claude): is gold58's c4/R7 MM-LM co-firing representative of the hidden population? local only, no OAI.
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm, spearmanr
R='C:/Users/eeshs/Downloads/rsna-knee/'
oof=pd.read_csv(R+'session_2026-10-18/next/g58/R7OOF.csv').set_index('StudyInstanceUID')
g=pd.read_csv(R+'session_2026-10-18/next/g58/gold74_eval_set.csv').set_index('StudyInstanceUID')
rep=pd.read_csv(R+'_labels/rsna-knee-llm-labels/report_labels_v2.csv').set_index('StudyInstanceUID')
def ns(x): return norm.ppf((rankdata(x)-.5)/len(x))
def rho(ids):
    a=oof.loc[ids,'p|Medial Meniscus'].values; b=oof.loc[ids,'p|Lateral Meniscus'].values
    return np.corrcoef(ns(a),ns(b))[0,1]
g58=[i for i in g.index[g.source=='original_gold58'] if i in oof.index]
g74=[i for i in g.index if i in oof.index]
tr=[i for i in oof.index if i not in g.index]
print('n oof', len(oof), 'train non-gold', len(tr), 'g58 in oof', len(g58), 'g74 in oof', len(g74))
print('R7 rho_ns MM-LM: train %.3f | g58 %.3f | g74 %.3f' % (rho(tr), rho(g58), rho(g74)))
rng=np.random.default_rng(0)
tr=np.array(tr)
bs=[rho(rng.choice(tr,58,replace=False)) for _ in range(2000)]
print('train subsample n58: mean %.3f sd %.3f p(<=g58 value) %.3f' % (np.mean(bs), np.std(bs), np.mean(np.array(bs)<=rho(g58))))
# prevalence: report v2 on train, expert on gold
for c in ['Medial Meniscus','Lateral Meniscus']:
    rv=rep.loc[[i for i in tr if i in rep.index], c]
    print(c, 'report v2 prev train %.3f' % (rv>=.5).mean(), '| expert g58 %.3f g74 %.3f' % (g.loc[g.source=='original_gold58',c].mean(), g[c].mean()),
          '| report v2 on g58 %.3f' % (rep.loc[[i for i in g58 if i in rep.index], c]>=.5).mean())
# reweight train to gold58 report-label cell mix of (MM,LM) report labels -> does rho drop?
rt=rep.loc[[i for i in tr if i in rep.index]]; rg=rep.loc[[i for i in g58 if i in rep.index]]
key=lambda d: ((d['Medial Meniscus']>=.5).astype(int).astype(str)+(d['Lateral Meniscus']>=.5).astype(int).astype(str))
kt=key(rt); kg=key(rg)
print('report cell mix train', kt.value_counts(normalize=True).round(3).to_dict(), '| g58', kg.value_counts(normalize=True).round(3).to_dict())
w=kt.map(kg.value_counts(normalize=True)/kt.value_counts(normalize=True)).fillna(0).values
ids=rt.index.values
rs=[]
for _ in range(400):
    s=rng.choice(ids, 58, replace=False, p=w/w.sum()); rs.append(rho(s))
print('train resampled to g58 report-cell mix, n58: mean %.3f sd %.3f' % (np.mean(rs), np.std(rs)))
rs=[rho(rng.choice(ids, 1500, replace=True, p=w/w.sum())) for _ in range(200)]
print('same, large n: mean %.3f' % np.mean(rs))
# within report cells
for k in ['00','01','10','11']:
    sub=ids[kt.values==k]
    if len(sub)>30: print('within report cell',k,'n',len(sub),'R7 rho %.3f'%rho(sub))
# expert vs report agreement on gold58 for MM/LM, and report tet-like coupling (phi) train vs expert phi gold
def phi(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); return np.corrcoef(a,b)[0,1]
print('report phi MM-LM train %.3f' % phi(rt['Medial Meniscus']>=.5, rt['Lateral Meniscus']>=.5))
gg=g.loc[g.source=='original_gold58']
print('expert phi MM-LM g58 %.3f g74 %.3f' % (phi(gg['Medial Meniscus'],gg['Lateral Meniscus']), phi(g['Medial Meniscus'],g['Lateral Meniscus'])))
print('report phi on g58 %.3f' % phi(rg['Medial Meniscus']>=.5, rg['Lateral Meniscus']>=.5))
