import json, os
cells=[]
def code(src): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src})

code("""# RSNA Knee 3-ENCODER ENSEMBLE (CPU, internet OFF): frozen DINOv2@384 + LoRA DINOv2@384 + RadImageNet R50@224.
import os, glob, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel
from torchvision.models import resnet50
import pydicom, cv2
torch.set_num_threads(os.cpu_count() or 4)
K=6; IMG=384; RAD_RES=224; EMB_BS=32
W_FROZEN, W_LORA, W_RAD = 1.0, 1.0, 1.0
LABELS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
        'Effusion','Synovitis',"Baker's",'Contusion','Fracture']
dev='cuda' if torch.cuda.is_available() else 'cpu'
print('torch',torch.__version__,'device',dev)
base=None
for r in sorted(glob.glob('/kaggle/input/*')):
    h=glob.glob(r+'/**/test_series.csv', recursive=True)
    if h: base=os.path.dirname(h[0]); break
print('base',base)
test_series=pd.read_csv(os.path.join(base,'test_series.csv'))
sub=pd.read_csv(os.path.join(base,'sample_submission.csv'))
print('test studies', sub.shape[0])""")

code("""def find(pat, must, notmust=None):
    for p in glob.glob('/kaggle/input/**/'+pat, recursive=True):
        pl=p.lower()
        if must in pl and (notmust is None or notmust not in pl): return p
    return None
lora_dir=os.path.dirname(find('config.json','lora-merged'))
frozen_dir=None
for p in glob.glob('/kaggle/input/**/config.json', recursive=True):
    pl=p.lower()
    if 'lora' in pl or 'head-weights' in pl or 'radimagenet' in pl: continue
    if 'dinov2' in pl: frozen_dir=p.rsplit('/',1)[0]; break
enc_f=AutoModel.from_pretrained(frozen_dir).eval().to(dev)
enc_l=AutoModel.from_pretrained(lora_dir).eval().to(dev)
print('frozen',frozen_dir); print('lora',lora_dir)
class RadEnc(nn.Module):
    def __init__(s): super().__init__(); s.backbone=nn.Sequential(*list(resnet50(weights=None).children())[:-2])
    def forward(s,x): return s.backbone(x).mean(dim=(2,3))
rad=RadEnc(); rad.load_state_dict(torch.load(find('ResNet50_radimagenet.pt','radimagenet'),map_location='cpu',weights_only=True),strict=True)
rad=rad.eval().to(dev); print('radimagenet loaded')
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
def mk(pth):
    ck=torch.load(pth,map_location='cpu'); di=int(ck.get('di',384)); hs=[]
    for sd in ck['heads']:
        m=MIL(di=di).to(dev).eval(); m.load_state_dict(sd); hs.append(m)
    return hs,di
heads_f,_=mk(find('head_weights.pt','head-weights'))
heads_l,_=mk(find('head_weights.pt','lora-merged'))
heads_r,dir_=mk(find('head_weights_radimagenet.pt','radimagenet'))
print('heads f/l/r', len(heads_f),len(heads_l),len(heads_r),'| rad di',dir_)
IM_MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); IM_STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
PL={'Sagittal':0,'Coronal':1,'Axial':2}""")

code("""def sampled(sd,k):
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
def emb_dino(enc,imgs):
    x=torch.from_numpy(np.stack(imgs))[:,None].repeat(1,3,1,1); x=((x-IM_MEAN)/IM_STD).to(dev)
    return enc(pixel_values=x).last_hidden_state[:,0].float()
@torch.no_grad()
def emb_rad(imgs):
    x=torch.from_numpy(np.stack(imgs))[:,None].repeat(1,3,1,1)
    x=F.interpolate(x,size=(RAD_RES,RAD_RES),mode='bilinear',align_corners=False)*2.0-1.0
    return rad(x.to(dev)).float()
@torch.no_grad()
def predict_study(study):
    imgs=[];pids=[];fbs=[]
    for _,sr in test_series[test_series['StudyInstanceUID']==study].iterrows():
        sd=os.path.join(base,'test_series',study,sr['SeriesInstanceUID'])
        if not os.path.isdir(sd):
            alt=glob.glob(os.path.join(base,'**',sr['SeriesInstanceUID']),recursive=True); sd=alt[0] if alt else None
        if not sd: continue
        pid=PL.get(sr['Anatomical_Plane'],0); fb=int(sr['Fluid_Sensitive'])
        for p in sampled(sd,K):
            im=slc(p)
            if im is not None: imgs.append(im);pids.append(pid);fbs.append(fb)
    if not imgs: return np.full(12,0.5)
    pid=torch.tensor(pids,device=dev); fb=torch.tensor(fbs,device=dev)
    Ef=[];El=[];Er=[]
    for i in range(0,len(imgs),EMB_BS):
        b=imgs[i:i+EMB_BS]; Ef.append(emb_dino(enc_f,b)); El.append(emb_dino(enc_l,b)); Er.append(emb_rad(b))
    Xf=torch.cat(Ef,0); Xl=torch.cat(El,0); Xr=torch.cat(Er,0)
    pf=np.mean([torch.sigmoid(h(Xf,pid,fb)).cpu().numpy() for h in heads_f],0)
    pl=np.mean([torch.sigmoid(h(Xl,pid,fb)).cpu().numpy() for h in heads_l],0)
    pr=np.mean([torch.sigmoid(h(Xr,pid,fb)).cpu().numpy() for h in heads_r],0)
    tot=W_FROZEN+W_LORA+W_RAD
    return (W_FROZEN*pf+W_LORA*pl+W_RAD*pr)/tot
print('ready')""")

code("""ids=list(sub['StudyInstanceUID']); t0=time.time(); out={}
for i,s in enumerate(ids):
    out[s]=predict_study(s)
    if i%100==0: print(f'{i}/{len(ids)} {time.time()-t0:.0f}s',flush=True)
res=pd.DataFrame({'StudyInstanceUID':ids})
for j,l in enumerate(LABELS): res[l]=[out[s][j] for s in ids]
res=res[['StudyInstanceUID']+LABELS]; res.to_csv('submission.csv',index=False)
print('WROTE',res.shape,f'{time.time()-t0:.0f}s'); print(res.head())""")

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
os.makedirs("kernels/infer_ens3",exist_ok=True)
json.dump(nb,open("kernels/infer_ens3/infer.ipynb","w"),indent=1)
meta={"id":"eeshsaxena/rsna-knee-infer-ens3","title":"rsna knee infer ens3","code_file":"infer.ipynb",
 "language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":False,"enable_tpu":False,
 "enable_internet":False,
 "dataset_sources":["eeshsaxena/rsna-knee-head-weights","eeshsaxena/rsna-knee-lora-merged","eeshsaxena/rsna-knee-radimagenet"],
 "kernel_sources":[],"competition_sources":["rsna-knee-abnormality-detection"],
 "model_sources":["metaresearch/dinov2/pytorch/small/1"]}
json.dump(meta,open("kernels/infer_ens3/kernel-metadata.json","w"),indent=2)
print("BUILT kernels/infer_ens3")
