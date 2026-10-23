# lab2_selfteach.py (jump2 label track): what does an OWN image teacher add to report labels, measured cleanly on gold58?
# gold74 is excluded from every fleet run, so the fold-0 checkpoints' gold predictions (~/jump2/gold74_preds, CPU fp32)
# are out-of-sample. Mixing recipe = mix50's: logit-match teacher to the text table, then (1-w)*text + w*teacher.
# Parent-likeness: mean per-label Spearman vs the parent's Raptor proxy on the 867 fold-0 val studies (OOF of same runs).
import numpy as np, pandas as pd
from scipy.special import logit, expit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
L=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
ID="StudyInstanceUID"; RT="/home/training/rsna_train"; J="/home/training/jump2"; FR="/home/training/fleet_runs"
tr=pd.read_csv(f"{RT}/train.csv"); gold=tr[tr["ACL"].notna()].set_index(ID)[L].astype(int); g=gold.index
def rd(p, idx): return pd.read_csv(p).drop_duplicates(ID).set_index(ID).reindex(idx)[L].astype(float)
v4all=pd.read_csv(f"{RT}/llm_labels_v4_blend.csv").drop_duplicates(ID).set_index(ID)[L].astype(float)
pk=pd.read_csv(f"{RT}/report_labels_v2.csv").drop_duplicates(ID).set_index(ID)
G={'v4':rd(f"{RT}/llm_labels_v4_blend.csv",g),'mix50':rd(f"{RT}/labels_mix50_logit.csv",g),'wd':rd(f"{RT}/teacher_train_probs.csv",g),
   'TTRv4':rd(f"{J}/gold74_preds/f0_s1042_ttr__ema.csv",g),'F1v4':rd(f"{J}/gold74_preds/f0_s42_v4__raw.csv",g),
   'TTRmix':rd(f"{J}/gold74_preds/f0_s1043_ttr_mix50__raw.csv",g),'F1mix':rd(f"{J}/gold74_preds/f0_s42__raw.csv",g)}
O={'TTRv4':f"{FR}/f0_s1042_ttr/oof_f0_s1042_ema.csv",'F1v4':f"{FR}/f0_s42_v4/oof_f0_s42_raw.csv",
   'TTRmix':f"{FR}/f0_s1043_ttr_mix50/oof_f0_s1043_raw.csv",'F1mix':f"{FR}/f0_s42/oof_f0_s42_raw.csv"}
oo={k:pd.read_csv(p).drop_duplicates(ID).set_index(ID) for k,p in O.items()}
vid=sorted(set.intersection(*[set(v.index) for v in oo.values()]))
rp=rd(f"{J}/labels_in/T_raptor_proxy.csv",vid)
V={'v4':v4all.reindex(vid),'mix50':rd(f"{RT}/labels_mix50_logit.csv",vid),'wd':rd(f"{RT}/teacher_train_probs.csv",vid),**{k:oo[k].reindex(vid)[L].astype(float) for k in oo}}
def lm(a, ref):
    la=logit(np.clip(a,1e-4,1-1e-4)); lr=logit(np.clip(ref,1e-4,1-1e-4)); return expit((la-la.mean())/la.std()*lr.std()+lr.mean())
def unk(idx): return pd.DataFrame({l: pk.reindex(idx)[l+"__verdict"].fillna("UNK").values=="UNK" for l in L}, index=idx)
def build(D, idx, text, teachers, w=.5, sa=False):
    out=pd.DataFrame(index=idx); U=unk(idx)
    # logit-match uses the FULL-table moments of the text labels for stability: use the text table on idx (gold58 or val)
    for l in L:
        t=np.mean([lm(D[k][l].values, D[text][l].values) for k in teachers], axis=0)
        ww=np.where(U[l].values, .75, w) if sa else w
        out[l]=(1-ww)*D[text][l].values+ww*t
    return out
def gauc(S): return np.mean([roc_auc_score(gold[l], S[l]) for l in L]), [roc_auc_score(gold[l], S[l]) for l in L]
rpr=np.column_stack([rp[l].rank().values for l in L])
def plike(S): return np.mean([spearmanr(S[l].values, rp[l].values).correlation for l in L])
VAR=[('v4 (text only)', 'v4', []), ('mix50_logit file', 'mix50', None),
     ('v4 + wd (= mix50 recipe, rebuilt)', 'v4', ['wd']),
     ('v4 + own TTRv4', 'v4', ['TTRv4']), ('v4 + own F1v4', 'v4', ['F1v4']), ('v4 + own TTRv4+F1v4', 'v4', ['TTRv4','F1v4']),
     ('v4 + own TTRmix', 'v4', ['TTRmix']), ('v4 + own F1mix', 'v4', ['F1mix']), ('v4 + own TTRmix+F1mix', 'v4', ['TTRmix','F1mix']),
     ('v4 + wd + own TTRv4', 'v4', ['wd','TTRv4']), ('v4 + wd + own TTRmix', 'v4', ['wd','TTRmix'])]
print(f"{'label set (gold58 n=58; parent-likeness on {len(vid)} fold-0 val)':58s} g58   | sp_rp | per-label")
for nm, text, te in VAR:
    for sa in ([False] if not te else [False, True]):
        if te is None: Sg=G['mix50']; Sv=V['mix50']
        elif not te: Sg=G['v4']; Sv=V['v4']
        else: Sg=build(G, g, text, te, sa=sa); Sv=build(V, vid, text, te, sa=sa)
        m,a=gauc(Sg); tag=nm+(' SA(UNK .75)' if sa else '')
        print(f"{tag:58s} {m:.4f} | {plike(Sv):.3f} | "+' '.join(f'{x:.3f}' for x in a))
