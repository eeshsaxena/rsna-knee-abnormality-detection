#!/usr/bin/env python3
"""Retrain the deployment MIL head on the box's cached DINOv2 embeddings with NEW (7B) labels.
Saves head_weights.pt = {'heads':[state_dict,...]} exactly as the Kaggle inference kernel expects.
Compares gold-58 vs the old 0.59. Run after box_extract7b.py finishes."""
import argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import os

ap=argparse.ArgumentParser()
ap.add_argument('--emb', default=os.path.expanduser('~/rsna_shards_full/emb_cache.npz'))
ap.add_argument('--labels', default=os.path.expanduser('~/rsna_train/pseudo_labels_qwen7b.csv'))
ap.add_argument('--train', default=os.path.expanduser('~/rsna_train/train.csv'))
ap.add_argument('--lang', default=os.path.expanduser('~/rsna_train/_reports_lang.csv'))
ap.add_argument('--out', default=os.path.expanduser('~/rsna_train/head_weights_7b.pt'))
a=ap.parse_args()
dev='cuda' if torch.cuda.is_available() else 'cpu'
LAB=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
     'Effusion','Synovitis',"Baker's",'Contusion','Fracture']

z=np.load(a.emb, allow_pickle=True)
emb=torch.tensor(z['emb'].astype(np.float32)).to(dev); meta=z['meta']; gids=np.array(z['study_ids'])
by={}
for r,(si,pid,fb) in enumerate(meta): by.setdefault(int(si),[]).append((r,int(pid),int(fb)))
soft=pd.read_csv(a.labels).set_index('StudyInstanceUID').reindex(gids)
tr=pd.read_csv(a.train).set_index('StudyInstanceUID').reindex(gids)
lg=pd.read_csv(a.lang).set_index('StudyInstanceUID').reindex(gids); lc=[c for c in lg.columns if 'lang' in c.lower()][0]
lang=lg[lc].fillna('unk').values
Y=soft[LAB].values.astype(np.float32); Yg=tr[LAB].values.astype(np.float32)
gold=tr['ACL'].notna().values; tri=np.where(~gold)[0]; gi=np.where(gold)[0]

class MIL(nn.Module):
    def __init__(s,di=384,d=256,n=12):
        super().__init__(); s.proj=nn.Sequential(nn.Linear(di,d),nn.LayerNorm(d),nn.GELU())
        s.plane=nn.Embedding(3,d); s.fluid=nn.Embedding(2,d)
        s.V=nn.Linear(d,128); s.U=nn.Linear(d,128); s.w=nn.Linear(128,1)
        s.head=nn.Sequential(nn.Dropout(0.3), nn.Linear(d,n))
    def forward(s,x,pid,fb):
        h=s.proj(x)+s.plane(pid)+s.fluid(fb)
        a=torch.softmax(s.w(torch.tanh(s.V(h))*torch.sigmoid(s.U(h))),0)
        return s.head((a*h).sum(0))
BAG={si:(torch.tensor([r[0] for r in rows],device=dev),
         torch.tensor([r[1] for r in rows],device=dev),
         torch.tensor([r[2] for r in rows],device=dev)) for si,rows in by.items()}
def bag(si): idx,pid,fb=BAG[si]; return emb[idx],pid,fb
def macro(yv,pv):
    a=[roc_auc_score(yv[:,j],pv[:,j]) for j in range(12) if len(set(yv[:,j]))>1]; return float(np.mean(a))

strat=(Y[tri]>0.5).sum(1).clip(0,4)
splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=0).split(tri,strat,lang[tri]))
heads=[]; scores=[]
for fold,(a_,_) in enumerate(splits):
    tix=tri[a_]; torch.manual_seed(fold); np.random.seed(fold)
    m=MIL(di=emb.shape[1]).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-2)
    best=-1; best_sd=None
    for ep in range(12):
        m.train(); np.random.shuffle(tix)
        for si in tix:
            x,pid,fb=bag(si); y=torch.tensor(Y[si],device=dev)
            opt.zero_grad(); F.binary_cross_entropy_with_logits(m(x,pid,fb),y).backward(); opt.step()
        m.eval(); pg=[]
        with torch.no_grad():
            for si in gi: x,pid,fb=bag(si); pg.append(torch.sigmoid(m(x,pid,fb)).cpu().numpy())
        au=macro(Yg[gi],np.array(pg))
        if au>best: best=au; best_sd={k:v.cpu().clone() for k,v in m.state_dict().items()}
    heads.append(best_sd); scores.append(best); print(f'fold{fold} gold-58 {best:.4f}',flush=True)
mods=[MIL(di=emb.shape[1]).to(dev).eval() for _ in heads]
for mm,sd in zip(mods,heads): mm.load_state_dict(sd)
ens=[]
with torch.no_grad():
    for si in gi: x,pid,fb=bag(si); ens.append(np.mean([torch.sigmoid(mm(x,pid,fb)).cpu().numpy() for mm in mods],0))
em=macro(Yg[gi],np.array(ens))
print(f'\n=== 7B-labels head: fold mean {np.mean(scores):.4f} | ENSEMBLE gold-58 {em:.4f}  (old 3B-fusion was 0.586) ===',flush=True)
torch.save({'heads':heads,'labels':LAB,'gold58_ensemble':em,'di':int(emb.shape[1])}, a.out)
print('WROTE', a.out, flush=True)
