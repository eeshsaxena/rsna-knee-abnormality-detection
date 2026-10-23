# MEASURED analysis (laptop, light CPU): how clean are "all text readers agree strongly" cells vs gold58,
# and how much of the train pool would an agreement mask (Prateek Grover / hengck23 "I don't know") remove.
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
L=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
t=pd.read_csv('_data/train.csv').set_index('StudyInstanceUID')
gold=t.dropna(subset=L)[L]
def rd(p): return pd.read_csv(p).set_index('StudyInstanceUID')
v4=rd('_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv')[L]
sf=rd('_labels/rsna-knee-llm-report-labels/llm_labels_full.csv')[L]
s2=rd('_labels/rsna-knee-llm-report-labels/llm_labels_v2.csv')[L]
pk=rd('_labels/rsna-knee-llm-labels/report_labels_v2.csv')[L]
sol=rd('_labels/rsna-knee-llm-report-labels-sol56/report_labels_gpt56sol.csv')[L]
fr=rd('session_2026-10-08/k8/frac75/labels_frac75_logit.csv')[L]
ids=v4.index.intersection(sf.index).intersection(s2.index).intersection(pk.index).intersection(sol.index).intersection(fr.index)
R={'v4':v4.loc[ids],'sf':sf.loc[ids],'s2':s2.loc[ids],'pk':pk.loc[ids],'sol':sol.loc[ids]}
pos=sum((R[k]>0.5).astype(int) for k in R); n=len(R)
strong_v4=(R['v4']>=0.9)|(R['v4']<=0.1)
conf=((pos==n)|(pos==0))&strong_v4
print('studies',len(ids),'readers',list(R))
nong=[i for i in ids if i not in gold.index]
print('\nTRAIN pool (non-gold) share of cells kept by the agreement mask, per label:')
print({l:round(float(conf.loc[nong,l].mean()),3) for l in L}, 'all', round(float(conf.loc[nong].values.mean()),3))
print('positives kept share (v4>.5) per label:', {l:round(float(conf.loc[nong,l][R['v4'].loc[nong,l]>0.5].mean()),3) for l in L})
g=gold.index.intersection(ids)
print('\nGOLD58 n',len(g))
C=conf.loc[g]; G=gold.loc[g]; V=R['v4'].loc[g]; F=fr.loc[g]
agree=((V>0.5).astype(int)==G)
print('cells kept %d / %d ; text-label precision on kept cells %.3f ; on masked cells %.3f'%(C.values.sum(),C.size,agree.values[C.values].mean(),agree.values[~C.values].mean()))
rows=[]
for l in L:
    k=C[l].values; y=G[l].values
    rows.append((l,int(k.sum()),int(y[k].sum()),int((~k).sum()),int(y[~k].sum()), round(float(agree[l].values[k].mean()),3) if k.sum() else None, round(float(agree[l].values[~k].mean()),3) if (~k).sum() else None,
                 round(roc_auc_score(y,F[l].values),4)))
print(pd.DataFrame(rows,columns=['label','kept','kept_pos','masked','masked_pos','prec_kept','prec_masked','frac75_auc_all']).to_string(index=False))
# error budget: of the frac75 table's mis-orderings on gold58, how many involve a kept (confidently-agreed) cell
tot_bad=0; bad_kept_both=0; bad_any_kept=0
for l in L:
    y=G[l].values; s=F[l].values; k=C[l].values
    P=np.where(y==1)[0]; N=np.where(y==0)[0]
    for i in P:
        for j in N:
            if s[i]<s[j] or s[i]==s[j]:
                w=1 if s[i]<s[j] else .5; tot_bad+=w
                if k[i] and k[j]: bad_kept_both+=w
                if k[i] or k[j]: bad_any_kept+=w
print('\nfrac75 gold58 mis-ordered pair mass: total %.1f ; both cells confidently agreed %.1f (%.0f%%) ; at least one agreed %.1f (%.0f%%)'%(tot_bad,bad_kept_both,100*bad_kept_both/tot_bad,bad_any_kept,100*bad_any_kept/tot_bad))
