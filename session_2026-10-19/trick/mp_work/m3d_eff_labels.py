import pandas as pd, numpy as np, json
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
B='C:/Users/eeshs/Downloads/rsna-knee/'
J=pd.read_pickle('J.pkl')
miss=(J['fluid|status']!='ok').values; okm=~miss
z=J['fluid|Effusion|z_feature'].values; zo=J['fluid|Effusion|z_out'].values; vol=J['fluid|eff_sag_p99_open'].values
lv=np.log1p(np.where(np.isnan(vol),0,vol))
res={}
for tag,path in [('R7',B+'session_2026-10-15/r7teacher/oof/src/fluid_stacker.json'),('R6',B+'session_2026-10-15/r6v1/upload/rsna-knee-r6-fe055406-v1/fluid_stacker.json')]:
    e=json.load(open(path))['classes']['Effusion']; res[tag]=(e['b0'],e['b_z'],e['b_lv'],e['b_miss'])
labs={}
v=J['Effusion__verdict'].values
labs['v2_yes_no']=np.where(v=='YES',1.,np.where(v=='NO',0.,np.nan))
for nm,f in [('v4blend>.5','_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv'),('jev_hybrid_binary','session_2026-10-05/jump2/labels_in/jev_hybrid_binary.csv'),('steven_v2>.5','session_2026-10-05/jump2/labels_in/steven_v2.csv'),('gpt56sol>.5','_labels/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv')]:
    try:
        d=pd.read_csv(B+f,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')['Effusion'].reindex(J.index).values.astype(float)
        labs[nm]=np.where(np.isnan(d),np.nan,(d>0.5).astype(float))
    except Exception as ex: print(nm,'ERR',ex)
reg=LinearRegression().fit(z[okm,None],lv[okm])
rng=np.random.default_rng(1)
rows=[]
for tag,(b0,bz,blv,bm) in res.items():
    dep=b0+bz*z+blv*lv+bm*miss
    alt={'C_impute_lv|z':np.where(miss,b0+bz*z+blv*reg.predict(z[:,None]),dep),
         'B_impute_median':np.where(miss,b0+bz*z+blv*np.median(lv[okm]),dep)}
    for ln,y in labs.items():
        k=~np.isnan(y); idx=np.where(k)[0]
        base=roc_auc_score(y[k],dep[k])
        for an,s in alt.items():
            d=[]
            for b in range(200):
                si=rng.choice(idx,len(idx)); d.append(roc_auc_score(y[si],s[si])-roc_auc_score(y[si],dep[si]))
            d=np.array(d)
            rows.append(dict(stacker=tag,labels=ln,n=int(k.sum()),prev_miss=np.nanmean(y[miss]),prev_ok=np.nanmean(y[okm]),auc_deployed=base,alt=an,
                             delta=roc_auc_score(y[k],s[k])-base,ci5=np.percentile(d,5),ci95=np.percentile(d,95)))
T=pd.DataFrame(rows); print(T.round(4).to_string()); T.to_csv('eff_misspath_counterfactual.csv',index=False)
for tag,(b0,bz,blv,bm) in res.items():
    print(tag,'miss offset vs median-impute logit: %.3f ; vs E[lv|z]-impute mean: %.3f'%(bm-blv*np.median(lv[okm]), np.mean(bm-blv*reg.predict(z[miss,None]))))
