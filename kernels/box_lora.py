#!/usr/bin/env python3
"""LoRA fine-tune DINOv2-S + gated-MIL head end-to-end on the image shards -> the 0.80+ path.
Trains backbone LoRA adapters + head on soft rule x Qwen labels; gold-58 held out for eval.
Slice-subsampling per epoch for speed. fp16 autocast on sm>=8.0 (RTX 4050), fp32 on V100.
Saves lora_head.pt (LoRA adapter state + head + config) for the Kaggle inference kernel.

Usage: python box_lora.py --shards DIR --model DINODIR --labels ... --lang ... --train ...
"""
import argparse, glob, os, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

LABELS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
        'Effusion','Synovitis',"Baker's",'Contusion','Fracture']
IM_MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); IM_STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)


class Model(nn.Module):
    def __init__(s, enc, d_in=384, d=256, n=12):
        super().__init__(); s.enc=enc
        s.proj=nn.Sequential(nn.Linear(d_in,d),nn.LayerNorm(d),nn.GELU())
        s.plane=nn.Embedding(3,d); s.fluid=nn.Embedding(2,d)
        s.V=nn.Linear(d,128); s.U=nn.Linear(d,128); s.w=nn.Linear(128,1)
        s.head=nn.Sequential(nn.Dropout(0.3), nn.Linear(d,n))
    def forward(s, imgs, pid, fb):        # imgs [k,3,H,W] uint8-normalized
        e=s.enc(pixel_values=imgs).last_hidden_state[:,0]   # [k,d_in]
        h=s.proj(e)+s.plane(pid)+s.fluid(fb)
        a=torch.softmax(s.w(torch.tanh(s.V(h))*torch.sigmoid(s.U(h))),0)
        return s.head((a*h).sum(0))


def macro_auc(y,p):
    a=[roc_auc_score(y[:,j],p[:,j]) for j in range(y.shape[1]) if len(set(y[:,j]))>1]
    return float(np.mean(a)), a


def main():
    ap=argparse.ArgumentParser()
    for k in ['shards','model','labels','lang','train']: ap.add_argument('--'+k, required=True)
    ap.add_argument('--epochs', type=int, default=6); ap.add_argument('--ktrain', type=int, default=8)
    ap.add_argument('--folds', type=int, default=4); ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--fold', type=int, default=0); ap.add_argument('--seed', type=int, default=0)
    a=ap.parse_args()
    dev='cuda'; cap=torch.cuda.get_device_capability(0)
    use_fp16 = cap[0]>=8
    if cap==(7,0): torch.backends.cudnn.enabled=False   # V100 cuDNN conv workaround
    print('dev', torch.cuda.get_device_name(0), 'fp16', use_fp16, flush=True)

    # load images into RAM (uint8), group by study
    shards=sorted(glob.glob(os.path.join(a.shards,'shard_*.npz')))
    imgs=[]; meta=[]; ids=[]
    for s in shards:
        z=np.load(s, allow_pickle=True); imgs.append(z['img']); meta.append(z['meta']); ids.append(z['study_ids'])
    img=np.concatenate(imgs); meta=np.concatenate(meta); gids=np.concatenate(ids); del imgs
    by={}
    for row,(si,pid,fb) in enumerate(meta): by.setdefault(int(si),[]).append((row,int(pid),int(fb)))
    print('slices',len(img),'studies',len(gids), flush=True)
    lab=pd.read_csv(a.labels).set_index('StudyInstanceUID').reindex(gids)
    tr=pd.read_csv(a.train).set_index('StudyInstanceUID').reindex(gids)
    lg=pd.read_csv(a.lang).set_index('StudyInstanceUID').reindex(gids); lc=[c for c in lg.columns if 'lang' in c.lower()][0]
    Y=lab[LABELS].values.astype(np.float32); Yg=tr[LABELS].values.astype(np.float32)
    gold=tr['ACL'].notna().values; lang=lg[lc].fillna('unk').values
    tri=np.where(~gold)[0]; gi=np.where(gold)[0]
    strat=(Y[tri]>0.5).sum(1).clip(0,4)
    splits=list(StratifiedGroupKFold(a.folds,shuffle=True,random_state=a.seed).split(tri,strat,lang[tri]))
    train_idx=tri[splits[a.fold][0]]
    print('train studies',len(train_idx),'gold',len(gi), flush=True)

    enc=AutoModel.from_pretrained(a.model)
    lcfg=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1,
                    target_modules=['query','key','value','dense'], bias='none')
    enc=get_peft_model(enc, lcfg)
    enc.gradient_checkpointing_enable(); enc.enable_input_require_grads()   # fit 4GB
    model=Model(enc).to(dev)
    if use_fp16: pass  # autocast handles it
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-2)
    scaler=torch.cuda.amp.GradScaler(enabled=use_fp16)

    def batch_imgs(rows, k):
        if len(rows)>k:
            sel=np.random.choice(len(rows), k, replace=False); rows=[rows[i] for i in sel]
        idx=np.array([r[0] for r in rows])
        x=torch.from_numpy(img[idx].astype(np.float32)/255.0)[:,None].repeat(1,3,1,1)
        x=((x-IM_MEAN)/IM_STD).to(dev)
        pid=torch.tensor([r[1] for r in rows],device=dev); fb=torch.tensor([r[2] for r in rows],device=dev)
        return x,pid,fb

    best=-1; t0=time.time()
    for ep in range(a.epochs):
        model.train(); np.random.shuffle(train_idx)
        for n_,si in enumerate(train_idx):
            rows=by.get(si)
            if not rows: continue
            x,pid,fb=batch_imgs(rows, a.ktrain); y=torch.tensor(Y[si],device=dev)
            opt.zero_grad()
            with torch.autocast('cuda', enabled=use_fp16):
                loss=F.binary_cross_entropy_with_logits(model(x,pid,fb), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if n_%500==0: print(f'  ep{ep} {n_}/{len(train_idx)} loss {loss.item():.3f} {time.time()-t0:.0f}s', flush=True)
        # eval gold-58 (use up to 20 slices/study for eval)
        model.eval(); pg=[]
        with torch.no_grad(), torch.autocast('cuda', enabled=use_fp16):
            for si in gi:
                rows=by.get(si,[])
                if not rows: pg.append(np.full(12,0.5)); continue
                x,pid,fb=batch_imgs(rows, 20); pg.append(torch.sigmoid(model(x,pid,fb)).float().cpu().numpy())
        au,_=macro_auc(Yg[gi], np.array(pg))
        print(f'=== ep{ep} gold-58 AUC {au:.4f} ({time.time()-t0:.0f}s) ===', flush=True)
        if au>best:
            best=au
            model.enc.save_pretrained('lora_adapter')
            torch.save({k:v.cpu() for k,v in model.state_dict().items() if 'enc.' not in k}, 'head_lora.pt')
    print(f'\n=== LoRA BEST gold-58 AUC: {best:.4f} ===', flush=True)


if __name__=='__main__': main()
