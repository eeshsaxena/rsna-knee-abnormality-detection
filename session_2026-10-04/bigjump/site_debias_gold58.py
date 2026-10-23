# CPU-only. Does removing per-site (report-language) reporting offsets from a predictor raise its gold58 AUC?
# Offsets are estimated on TRAIN studies only (gold58 excluded), so gold58 stays a clean referee.
import pandas as pd, numpy as np, sys, io
from sklearn.metrics import roc_auc_score
from scipy.special import logit
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
S="C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad"
D="C:/Users/eeshs/Downloads/rsna-knee/_data"
LAB=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
lang=pd.read_csv(D+"/_reports_lang.csv",usecols=["StudyInstanceUID","lang"]).set_index("StudyInstanceUID")["lang"]
v4=pd.read_csv(S+"/kneelabels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")[LAB]
T={"v5":pd.read_csv(S+"/teacher_v2_out/teacher_v2_maxspan_v5_shard0.csv").set_index("StudyInstanceUID")[LAB],
   "v8":pd.read_csv(S+"/teacher_v2_out/teacher_v2_native384_v8_shard0.csv").set_index("StudyInstanceUID")[LAB],
   "wd":pd.read_csv(S+"/teacher_out/teacher_train_probs.csv").set_index("StudyInstanceUID")[LAB]}
g=pd.read_csv(S+"/kneelabels/jev/gold74_eval_set.csv"); g=g[g.source=="original_gold58"].set_index("StudyInstanceUID")[LAB]
gid=g.index
tr=v4.index.intersection(T["v5"].index).intersection(lang.index).difference(gid)
def lg(df): return pd.DataFrame(logit(np.clip(df.values,1e-4,1-1e-4)),index=df.index,columns=df.columns)
def macro(pred,ids=gid):
    a=[roc_auc_score(g.loc[ids,c],pred.loc[ids,c]) for c in LAB]; return np.mean(a),np.array(a)
def offsets(pred_tr,ref_tr,L):
    # per-site mean logit gap pred-ref, centred so the global mean shift is 0
    gap=(lg(pred_tr)-lg(ref_tr)).groupby(L).mean()
    w=L.value_counts(normalize=True).reindex(gap.index)
    return gap-gap.mul(w,axis=0).sum()
Ltr=lang.reindex(tr); Lg=lang.reindex(gid)
rng=np.random.default_rng(0)
def boot(a,b,n=2000):
    # paired bootstrap over gold studies of macro(b)-macro(a)
    ids=np.array(gid); out=[]
    A=a.loc[ids]; B=b.loc[ids]; G=g.loc[ids]
    for _ in range(n):
        s=ids[rng.integers(0,len(ids),len(ids))]
        try:
            da=np.mean([roc_auc_score(G.loc[s,c],B.loc[s,c])-roc_auc_score(G.loc[s,c],A.loc[s,c]) for c in LAB])
        except ValueError: continue
        out.append(da)
    out=np.array(out); return out.mean(),out.std(),(out>0).mean()
for refname in ["v5","v8","wd"]:
    ref=T[refname]
    off=offsets(v4.reindex(tr),ref.reindex(tr),Ltr)   # reporting offset of the LLM labels relative to an image reader
    adj=lg(v4.reindex(gid)) - off.reindex(Lg.values).set_index(gid).fillna(0)
    m0,a0=macro(lg(v4.reindex(gid))); m1,a1=macro(adj)
    mu,sd,p=boot(lg(v4.reindex(gid)),adj,600)
    print(f"[labels steven_v4, offsets vs teacher {refname}] gold58 macro {m0:.4f} -> {m1:.4f} (d {m1-m0:+.4f}; boot {mu:+.4f} sd {sd:.4f} P>0 {p:.2f})")
    print("   per-label d:", " ".join(f"{c[:5]}{d:+.3f}" for c,d in zip(LAB,a1-a0)))
# Control: shuffled site assignment (same offsets, random languages) -> expected ~0 or negative
off=offsets(v4.reindex(tr),T["v5"].reindex(tr),Ltr)
res=[]
for k in range(200):
    Lsh=pd.Series(rng.permutation(Lg.values),index=gid)
    adj=lg(v4.reindex(gid)) - off.reindex(Lsh.values).set_index(gid).fillna(0)
    res.append(macro(adj)[0]-macro(lg(v4.reindex(gid)))[0])
print(f"control (shuffled languages on gold58): mean d {np.mean(res):+.4f}, 95th pct {np.percentile(res,95):+.4f}")
# Teachers themselves: do in-sample image readers carry a site offset vs each other? (teacher v8 corrected toward v5)
for a,b in [("v8","v5"),("wd","v5")]:
    off=offsets(T[a].reindex(tr),T[b].reindex(tr),Ltr)
    adj=lg(T[a].reindex(gid)) - off.reindex(Lg.values).set_index(gid).fillna(0)
    print(f"[teacher {a}, offsets vs {b}] gold58 {macro(lg(T[a].reindex(gid)))[0]:.4f} -> {macro(adj)[0]:.4f}")
