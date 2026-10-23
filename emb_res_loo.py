#!/usr/bin/env python3
"""Embed a gold-58 npz (imgs at native RES) with a DINOv2 model on the RTX 4050 and run the
gold-58 leave-one-out macro-AUC. Tests whether higher resolution breaks the 0.6325 (224px)
frozen ceiling. Usage: python emb_res_loo.py <npz> <model_dir>"""
import sys, os, numpy as np, pandas as pd, torch
from transformers import AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut

D=r"C:\Users\eeshs\Downloads\rsna-knee"
LAB=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
     'Effusion','Synovitis',"Baker's",'Contusion','Fracture']
npz, mdir = sys.argv[1], sys.argv[2]
TRAIN_CSV = sys.argv[3] if len(sys.argv)>3 else D+r"\_data\train.csv"
z=np.load(npz, allow_pickle=True); img=z['img']; meta=z['meta']; sids=np.array(z['study_ids'])
RES=img.shape[-1]; print('npz', npz, 'slices', len(img), 'RES', RES, 'studies', len(sids))
dev='cuda' if torch.cuda.is_available() else 'cpu'
if dev=='cuda' and torch.cuda.get_device_capability(0)==(7,0):
    torch.backends.cudnn.enabled=False   # V100 fp16/cudnn conv workaround
if dev=='cpu': torch.set_num_threads(os.cpu_count() or 4)
print('device', dev, torch.cuda.get_device_name(0) if dev=='cuda' else '')
enc=AutoModel.from_pretrained(mdir).eval().float().to(dev)   # fp32 (V100-safe)   # fp32 on CPU
IM_MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); IM_STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
by={}
for r,(si,pid,fb) in enumerate(meta): by.setdefault(int(si),[]).append(r)
d=enc.config.hidden_size
X=np.zeros((len(sids),d),np.float32)
with torch.no_grad():
    for si,rows in by.items():
        im=img[rows]; e=[]
        for i in range(0,len(im),8):
            x=torch.from_numpy(im[i:i+8].astype(np.float32)/255.0)[:,None].repeat(1,3,1,1)
            x=((x-IM_MEAN)/IM_STD).float().to(dev)
            e.append(enc(pixel_values=x).last_hidden_state[:,0].float().cpu().numpy())
        X[si]=np.concatenate(e).mean(0)
tr=pd.read_csv(TRAIN_CSV).set_index('StudyInstanceUID').reindex(sids)
Y=tr[LAB].values.astype(float)
Xs=StandardScaler().fit_transform(X); aucs=[]
for j,l in enumerate(LAB):
    y=Y[:,j].astype(int)
    if len(set(y))<2: aucs.append(np.nan); continue
    p=np.zeros(len(y))
    for tri,tei in LeaveOneOut().split(Xs):
        if len(set(y[tri]))<2: p[tei]=y[tri].mean(); continue
        c=LogisticRegression(max_iter=500,C=0.5).fit(Xs[tri],y[tri]); p[tei]=c.predict_proba(Xs[tei])[:,1]
    aucs.append(roc_auc_score(y,p))
for l,a in zip(LAB,aucs): print(f'  {l:18s} {a:.3f}')
print(f'=== RES {RES} gold-58 LOO macro = {np.nanmean(aucs):.4f}  (224px baseline was 0.6325) ===')
