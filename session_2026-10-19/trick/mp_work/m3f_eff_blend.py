import pandas as pd, numpy as np, json
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
B='C:/Users/eeshs/Downloads/rsna-knee/'
J=pd.read_pickle('JT.pkl')
miss=(J['fluid|status']!='ok').values; okm=~miss
z=J['fluid|Effusion|z_feature'].values; vol=J['fluid|eff_sag_p99_open'].values
lv=np.log1p(np.where(np.isnan(vol),0,vol))
e=json.load(open(B+'session_2026-10-15/r7teacher/oof/src/fluid_stacker.json'))['classes']['Effusion']
b0,bz,blv,bm=e['b0'],e['b_z'],e['b_lv'],e['b_miss']
reg=LinearRegression().fit(z[okm,None],lv[okm])
dep=b0+bz*z+blv*lv+bm*miss
fix=np.where(miss,b0+bz*z+blv*reg.predict(z[:,None]),dep)
pct=lambda x: pd.Series(x).rank(pct=True).values
rT=pct(J['t|Effusion'].values); rS=pct(z)
labs={}
v=J['Effusion__verdict'].values; labs['v2_yes_no']=np.where(v=='YES',1.,np.where(v=='NO',0.,np.nan))
for nm,f in [('v4blend>.5','_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv'),('jev_bin','session_2026-10-05/jump2/labels_in/jev_hybrid_binary.csv'),('gpt56sol>.5','_labels/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv')]:
    d=pd.read_csv(B+f,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')['Effusion'].reindex(J.index).values.astype(float); labs[nm]=(d>0.5).astype(float)
rng=np.random.default_rng(2)
print('n',len(J),'miss',miss.sum())
for ln,y in labs.items():
    k=~np.isnan(y); idx=np.where(k)[0]
    for bn,(wT,wR,wS) in {'R7 alone':(0,1,0),'TTR.5+R7.5':(.5,.5,0),'c4-shape .4T+.4R7+.2S':(.4,.4,.2)}.items():
        a=wT*rT+wR*pct(dep)+wS*rS; b=wT*rT+wR*pct(fix)+wS*rS
        d0=roc_auc_score(y[k],b[k])-roc_auc_score(y[k],a[k])
        ds=[]
        for _ in range(200):
            s=rng.choice(idx,len(idx)); ds.append(roc_auc_score(y[s],b[s])-roc_auc_score(y[s],a[s]))
        print('%-12s %-24s AUC %.4f  delta(fix) %+.5f  90%% [%+.5f,%+.5f]  macro-equiv %+.6f'%(ln,bn,roc_auc_score(y[k],a[k]),d0,*np.percentile(ds,[5,95]),d0/12))
