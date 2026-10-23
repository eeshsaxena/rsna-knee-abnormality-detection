import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
J=pd.read_pickle('J.pkl')
miss=(J['fluid|status']!='ok').values
z=J['fluid|Effusion|z_feature'].values; zo=J['fluid|Effusion|z_out'].values
pm=J['fluid|Effusion|p_model'].values; p=J['p|Effusion'].values
vol=J['fluid|eff_sag_p99_open'].values
import json,sys
e=json.load(open("C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-15/r7teacher/oof/src/fluid_stacker.json"))["classes"]["Effusion"]; b0,bz,blv,bm=e["b0"],e["b_z"],e["b_lv"],e["b_miss"]
lv=np.log1p(np.where(np.isnan(vol),0,vol))
rec=b0+bz*z+blv*lv+bm*miss
print('stacker recon max err', np.nanmax(np.abs(rec-zo)), 'p vs sigmoid(zo)', np.nanmax(np.abs(p-1/(1+np.exp(-zo)))))
v=J['Effusion__verdict'].values
y_yn=np.where(v=='YES',1.,np.where(v=='NO',0.,np.nan)); y_yr=(v=='YES').astype(float)
print('Effusion report YES rate: miss %.3f  ok %.3f | NO rate miss %.3f ok %.3f | UNK miss %.3f ok %.3f'%(
 y_yr[miss].mean(),y_yr[~miss].mean(),(v[miss]=='NO').mean(),(v[~miss]=='NO').mean(),(v[miss]=='UNK').mean(),(v[~miss]=='UNK').mean()))
print('mean pct rank of deployed p: miss %.3f ok %.3f ; of p_model: miss %.3f ok %.3f'%(
 pd.Series(p).rank(pct=True)[miss].mean(),pd.Series(p).rank(pct=True)[~miss].mean(),pd.Series(pm).rank(pct=True)[miss].mean(),pd.Series(pm).rank(pct=True)[~miss].mean()))
okm=~miss
reg=LinearRegression().fit(z[okm,None],lv[okm]); print('lv~z on ok: coef %.3f int %.3f R2 %.3f'%(reg.coef_[0],reg.intercept_,reg.score(z[okm,None],lv[okm])))
alts={'A_deployed':zo,'P_model_logit':z,
 'B_impute_median_lv':np.where(miss,b0+bz*z+blv*np.median(lv[okm]),zo),
 'C_impute_lv_given_z':np.where(miss,b0+bz*z+blv*reg.predict(z[:,None]),zo),
 'D_miss_offset_x0.5':np.where(miss,zo-0.5*(bm+b0+bz*z - (b0+bz*z+blv*np.median(lv[okm]))) ,zo)}
def auc(y,s,m=None):
    k=~np.isnan(y) if m is None else (~np.isnan(y))&m
    return roc_auc_score(y[k],s[k])
rng=np.random.default_rng(0)
print('\nEffusion AUC vs report labels (all 4406 / ok only / miss only):')
for k,s in alts.items():
    print('%-22s yn %.4f  yes_vs_rest %.4f | ok %.4f | miss %.4f'%(k,auc(y_yn,s),auc(y_yr,s),auc(y_yn,s,okm),auc(y_yn,s,miss)))
# bootstrap paired diff deployed vs B and C on yn
idx=np.where(~np.isnan(y_yn))[0]
for k in ['B_impute_median_lv','C_impute_lv_given_z','P_model_logit']:
    d=[]
    for b in range(300):
        s=rng.choice(idx,len(idx))
        d.append(roc_auc_score(y_yn[s],alts[k][s])-roc_auc_score(y_yn[s],zo[s]))
    d=np.array(d); print('%s - deployed: mean %+.4f  90%% CI [%+.4f, %+.4f]'%(k,d.mean(),*np.percentile(d,[5,95])))
# miss-group offset: average z_out difference (miss) vs median-impute
print('miss-path minus median-impute logit offset: %.3f (b_miss %.3f - b_lv*median lv %.3f)'%(bm-blv*np.median(lv[okm]),bm,blv*np.median(lv[okm])))
