#!/usr/bin/env python3
"""Generate the CPU-capable inference notebook (device-agnostic, batched embed)."""
import json, os
C0='''# RSNA Knee - INFERENCE kernel (Kaggle, internet OFF). CPU-capable (GPU quota safe).
# Embeds TEST with frozen DINOv2-S + applies the deployment MIL head ensemble -> submission.csv.
import os, glob, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel
import pydicom
torch.set_num_threads(os.cpu_count() or 4)
K = 8; IMG = 224; EMB_BS = 32
LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
          'Effusion','Synovitis',"Baker's",'Contusion','Fracture']
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; HALF = (dev=='cuda')
print('torch', torch.__version__, '| device', dev, '| half', HALF)
base=None
for r in sorted(glob.glob('/kaggle/input/*')):
    h=glob.glob(r+'/**/test_series.csv', recursive=True)
    if h: base=os.path.dirname(h[0]); break
print('base', base)
test_series=pd.read_csv(os.path.join(base,'test_series.csv'))
sub=pd.read_csv(os.path.join(base,'sample_submission.csv'))
print('test studies', sub.shape[0])'''

C1='''# encoder + deployment head ensemble
mdir=None
for c in glob.glob('/kaggle/input/**/config.json', recursive=True):
    d=os.path.dirname(c)
    if glob.glob(d+'/pytorch_model.bin') or glob.glob(d+'/*.safetensors'): mdir=d; break
enc=AutoModel.from_pretrained(mdir).eval()
enc=enc.half().to(dev) if HALF else enc.to(dev)
print('encoder', mdir)
ckpt=torch.load(glob.glob('/kaggle/input/**/head_weights.pt', recursive=True)[0], map_location='cpu')
print('heads', len(ckpt['heads']), '| gold58', ckpt.get('gold58_ensemble'))

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
heads=[]
for sd in ckpt['heads']:
    m=MIL().to(dev).eval(); m.load_state_dict(sd); heads.append(m)
IM_MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); IM_STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
PL={'Sagittal':0,'Coronal':1,'Axial':2}'''

C2='''import cv2
def sampled(sd,k):
    fs=sorted(glob.glob(sd+'/*.dcm')); n=len(fs)
    if n==0: return []
    if n<=k: return fs
    lo,hi=int(n*.12),int(n*.88)
    if hi<=lo: lo,hi=0,n
    return [fs[int(round(x))] for x in np.linspace(lo,hi-1,k)]
def slc(p):
    try:
        d=pydicom.dcmread(p); a=d.pixel_array.astype(np.float32)
        a=a*float(getattr(d,'RescaleSlope',1) or 1)+float(getattr(d,'RescaleIntercept',0) or 0)
        lo,hi=np.percentile(a,0.5),np.percentile(a,99.5); a=np.clip((a-lo)/max(hi-lo,1e-6),0,1)
        return cv2.resize(a,(IMG,IMG),interpolation=cv2.INTER_AREA).astype(np.float32)
    except Exception: return None
@torch.no_grad()
def embed_batch(imgs):
    x=torch.from_numpy(np.stack(imgs))[:,None].repeat(1,3,1,1)
    x=((x-IM_MEAN)/IM_STD)
    x=x.half().to(dev) if HALF else x.to(dev)
    return enc(pixel_values=x).last_hidden_state[:,0].float()
@torch.no_grad()
def predict_study(study):
    imgs=[]; pids=[]; fbs=[]
    for _,sr in test_series[test_series['StudyInstanceUID']==study].iterrows():
        sd=os.path.join(base,'test_series',study,sr['SeriesInstanceUID'])
        if not os.path.isdir(sd):
            alt=glob.glob(os.path.join(base,'**',sr['SeriesInstanceUID']),recursive=True); sd=alt[0] if alt else None
        if not sd: continue
        pid=PL.get(sr['Anatomical_Plane'],0); fb=int(sr['Fluid_Sensitive'])
        for p in sampled(sd,K):
            im=slc(p)
            if im is not None: imgs.append(im); pids.append(pid); fbs.append(fb)
    if not imgs: return np.full(12,0.5)
    E=[]
    for i in range(0,len(imgs),EMB_BS): E.append(embed_batch(imgs[i:i+EMB_BS]))
    X=torch.cat(E,0); pid=torch.tensor(pids,device=dev); fb=torch.tensor(fbs,device=dev)
    return np.mean([torch.sigmoid(h(X,pid,fb)).cpu().numpy() for h in heads],0)
print('ready')'''

C3='''ids=list(sub['StudyInstanceUID']); t0=time.time(); out={}
for i,s in enumerate(ids):
    out[s]=predict_study(s)
    if i%100==0: print(f'{i}/{len(ids)} {time.time()-t0:.0f}s', flush=True)
res=pd.DataFrame({'StudyInstanceUID':ids})
for j,l in enumerate(LABELS): res[l]=[out[s][j] for s in ids]
res=res[['StudyInstanceUID']+LABELS]; res.to_csv('submission.csv',index=False)
print('WROTE submission.csv', res.shape, f'{time.time()-t0:.0f}s'); print(res.head())'''

cells=[C0,C1,C2,C3]
nb={"cells":[{"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
     "source":c} for c in cells],
    "metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},
    "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
p=r"C:\Users\eeshs\Downloads\rsna-knee\kernels\infer\infer.ipynb"
json.dump(nb, open(p,'w',encoding='utf-8'), indent=1)
print('wrote', p, len(cells),'cells')
