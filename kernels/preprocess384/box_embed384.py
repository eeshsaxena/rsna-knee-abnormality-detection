#!/usr/bin/env python3
"""Embed the 384px shards with FROZEN DINOv2-small (the winning backbone) -> emb_cache_384.npz.
Then box_head_retrain.py trains the MIL head on these higher-res features + clean 7B labels.
This is the real ceiling-break: proven approach (0.62 @224) at finer resolution."""
import glob, os, time, argparse, numpy as np, torch
from transformers import AutoModel

ap=argparse.ArgumentParser()
ap.add_argument('--shards', default=os.path.expanduser('~/rsna_shards_384'))
ap.add_argument('--model', default=os.path.expanduser('~/dinov2_small'))
ap.add_argument('--out', default=os.path.expanduser('~/rsna_train/emb_cache_384.npz'))
a=ap.parse_args()
SHARDS=a.shards; MODEL=a.model; OUT=a.out
dev='cuda'
if torch.cuda.get_device_capability(0)==(7,0): torch.backends.cudnn.enabled=False  # V100
IM_MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); IM_STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
enc=AutoModel.from_pretrained(MODEL).eval().float().to(dev)   # fp32 (V100-safe)
print('encoder loaded', flush=True)

shards=sorted(glob.glob(os.path.join(SHARDS,'shard_*.npz')))
embs=[]; metas=[]; ids=[]; t0=time.time()
with torch.no_grad():
    for si,sh in enumerate(shards):
        z=np.load(sh, allow_pickle=True); img=z['img']; metas.append(z['meta']); ids.append(z['study_ids'])
        out=[]
        for i in range(0,len(img),16):
            x=torch.from_numpy(img[i:i+16].astype(np.float32)/255.0)[:,None].repeat(1,3,1,1)
            x=((x-IM_MEAN)/IM_STD).float().to(dev)
            out.append(enc(pixel_values=x).last_hidden_state[:,0].float().cpu().numpy())
        embs.append(np.concatenate(out))
        if si%20==0: print(f'  shard {si}/{len(shards)} {time.time()-t0:.0f}s', flush=True)
emb=np.concatenate(embs).astype(np.float32); meta=np.concatenate(metas); gids=np.concatenate(ids)
np.savez(OUT, emb=emb, meta=meta, study_ids=gids)
print(f'WROTE {OUT} emb {emb.shape} {time.time()-t0:.0f}s', flush=True)
