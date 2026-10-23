#!/usr/bin/env python3
"""Embed shards with BiomedCLIP image encoder (medical image-text pretraining -> far better MRI
features than natural-image DINOv2). This is the backbone the 0.93 cluster uses. -> emb_cache."""
import glob, os, time, argparse, numpy as np, torch, open_clip, torchvision

ap=argparse.ArgumentParser()
ap.add_argument('--shards', default=os.path.expanduser('~/rsna_shards_full'))
ap.add_argument('--out', default=os.path.expanduser('~/rsna_train/emb_cache_biomedclip.npz'))
a=ap.parse_args()
dev='cuda'
if torch.cuda.get_device_capability(0)==(7,0): torch.backends.cudnn.enabled=False  # V100
model, preprocess = open_clip.create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
model=model.eval().float().to(dev)
RES=model.visual.image_size if isinstance(getattr(model.visual,'image_size',224),int) else 224
try: RES=preprocess.transforms[0].size if isinstance(preprocess.transforms[0].size,int) else preprocess.transforms[0].size[0]
except Exception: RES=224
norm=[t for t in preprocess.transforms if isinstance(t, torchvision.transforms.Normalize)][0]
MEAN=torch.tensor(norm.mean).view(1,3,1,1); STD=torch.tensor(norm.std).view(1,3,1,1)
print('BiomedCLIP loaded | input res', RES, '| mean', [round(x,3) for x in norm.mean], flush=True)

shards=sorted(glob.glob(os.path.join(a.shards,'shard_*.npz')))
embs=[]; metas=[]; ids=[]; t0=time.time()
with torch.no_grad():
    for si,sh in enumerate(shards):
        z=np.load(sh, allow_pickle=True); img=z['img']; metas.append(z['meta']); ids.append(z['study_ids'])
        out=[]
        for i in range(0,len(img),16):
            x=torch.from_numpy(img[i:i+16].astype(np.float32)/255.0)[:,None].repeat(1,3,1,1)
            if x.shape[-1]!=RES: x=torch.nn.functional.interpolate(x, size=(RES,RES), mode='bilinear', align_corners=False)
            x=((x-MEAN)/STD).float().to(dev)
            out.append(model.encode_image(x).float().cpu().numpy())
        embs.append(np.concatenate(out))
        if si%20==0: print(f'  shard {si}/{len(shards)} {time.time()-t0:.0f}s', flush=True)
emb=np.concatenate(embs).astype(np.float32); meta=np.concatenate(metas); gids=np.concatenate(ids)
np.savez(a.out, emb=emb, meta=meta, study_ids=gids)
print(f'WROTE {a.out} emb {emb.shape} {time.time()-t0:.0f}s', flush=True)
