#!/usr/bin/env python3
"""Generate a Kaggle CPU kernel that preprocesses the 58 GOLD studies at 518px AND 392px
(K=8 slices/series, same windowing/sampling as preprocess.ipynb) and saves them for download.
Purpose: A/B test whether native-res DINOv2 features break the 0.63 frozen ceiling, using the
same clean-labeled gold studies (embed locally on RTX 4050, gold-58 LOO). CPU = no GPU quota."""
import json
C0='''import os, glob, numpy as np, pandas as pd, pydicom, cv2
K=8; PL={'Sagittal':0,'Coronal':1,'Axial':2}
base=None
for r in sorted(glob.glob('/kaggle/input/*')):
    h=glob.glob(r+'/**/train_series.csv', recursive=True)
    if h: base=os.path.dirname(h[0]); break
print('base', base)
tr=pd.read_csv(os.path.join(base,'train.csv'))
ts=pd.read_csv(os.path.join(base,'train_series.csv'))
gold=list(tr[tr['ACL'].notna()]['StudyInstanceUID']); print('gold studies', len(gold))
def sampled(sd,k):
    fs=sorted(glob.glob(sd+'/*.dcm')); n=len(fs)
    if n==0: return []
    if n<=k: return fs
    lo,hi=int(n*.12),int(n*.88)
    if hi<=lo: lo,hi=0,n
    return [fs[int(round(x))] for x in np.linspace(lo,hi-1,k)]
def slc(p,RES):
    try:
        d=pydicom.dcmread(p); a=d.pixel_array.astype(np.float32)
        a=a*float(getattr(d,'RescaleSlope',1) or 1)+float(getattr(d,'RescaleIntercept',0) or 0)
        lo,hi=np.percentile(a,0.5),np.percentile(a,99.5); a=np.clip((a-lo)/max(hi-lo,1e-6),0,1)
        return (cv2.resize(a,(RES,RES),interpolation=cv2.INTER_AREA)*255).astype(np.uint8)
    except Exception: return None'''

C1='''def build(RES):
    imgs=[]; meta=[]; sids=[]
    for si,study in enumerate(gold):
        srs=ts[ts['StudyInstanceUID']==study]
        for _,sr in srs.iterrows():
            sd=None
            for cand in glob.glob(os.path.join(base,'**',sr['SeriesInstanceUID']),recursive=True):
                if os.path.isdir(cand): sd=cand; break
            if not sd: continue
            pid=PL.get(sr['Anatomical_Plane'],0); fb=int(sr['Fluid_Sensitive'])
            for p in sampled(sd,K):
                im=slc(p,RES)
                if im is not None: imgs.append(im); meta.append((si,pid,fb))
        sids.append(study)
        if si%15==0: print(RES,'study',si,'slices',len(imgs),flush=True)
    np.savez_compressed(f'gold58_{RES}.npz', img=np.stack(imgs),
                        meta=np.array(meta), study_ids=np.array(sids))
    print(f'WROTE gold58_{RES}.npz', len(imgs),'slices',len(sids),'studies')
for RES in [518, 392]:
    build(RES)
print('done')'''

nb={"cells":[{"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":c}
     for c in [C0,C1]],
    "metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},
    "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
import os
os.makedirs(r"C:\Users\eeshs\Downloads\rsna-knee\kernels\exp518", exist_ok=True)
p=r"C:\Users\eeshs\Downloads\rsna-knee\kernels\exp518\exp518.ipynb"
json.dump(nb, open(p,'w',encoding='utf-8'), indent=1)
meta={"id":"eeshsaxena/rsna-knee-exp518","title":"rsna knee exp518","code_file":"exp518.ipynb",
 "language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":False,
 "enable_tpu":False,"enable_internet":False,"keywords":[],
 "competition_sources":["rsna-knee-abnormality-detection"],"kernel_sources":[],
 "dataset_sources":[],"model_sources":[]}
json.dump(meta, open(r"C:\Users\eeshs\Downloads\rsna-knee\kernels\exp518\kernel-metadata.json",'w'), indent=2)
print('wrote exp518 kernel (CPU, no GPU quota)')
