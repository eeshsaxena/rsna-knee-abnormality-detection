#!/usr/bin/env python3
"""LoRA fine-tune DINOv2-S + gated-MIL, v2 -> fixes the v1 collapse (0.54 < frozen 0.59).
Root cause of v1: random head + high LR co-trained from step 0 wrecked the pretrained features.
v2 fixes:
  * PHASE 1 (--warm epochs): LoRA frozen, train ONLY head/proj/plane/fluid -> head reaches ~frozen.
  * PHASE 2: unfreeze LoRA at a LOW lr (param groups: head hi, LoRA lo), cosine decay + warmup.
  * ktrain configurable (12), fp16 autocast on RTX 4050, grad-checkpoint for 5GB VRAM.
  * best-on-gold checkpoint saved as head_weights.pt-compatible {'heads':[sd]} + lora_adapter/.
Usage: python box_lora_v2.py --shards DIR --model DINODIR --labels L --lang G --train T
"""
import argparse, glob, os, time, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
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
    def forward(s, imgs, pid, fb):
        e=s.enc(pixel_values=imgs).last_hidden_state[:,0]
        h=s.proj(e)+s.plane(pid)+s.fluid(fb)
        a=torch.softmax(s.w(torch.tanh(s.V(h))*torch.sigmoid(s.U(h))),0)
        return s.head((a*h).sum(0))

def macro_auc(y,p):
    a=[roc_auc_score(y[:,j],p[:,j]) for j in range(y.shape[1]) if len(set(y[:,j]))>1]
    return float(np.mean(a))

def main():
    ap=argparse.ArgumentParser()
    for k in ['shards','model','labels','lang','train']: ap.add_argument('--'+k, required=True)
    ap.add_argument('--epochs', type=int, default=18); ap.add_argument('--warm', type=int, default=3)
    ap.add_argument('--ktrain', type=int, default=12); ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--fold', type=int, default=0); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr_head', type=float, default=3e-4); ap.add_argument('--lr_lora', type=float, default=3e-5)
    ap.add_argument('--out', default='lora_v2'); a=ap.parse_args()
    dev='cuda'; cap=torch.cuda.get_device_capability(0); use_fp16=cap[0]>=8
    if cap==(7,0): torch.backends.cudnn.enabled=False
    print('dev',torch.cuda.get_device_name(0),'fp16',use_fp16,flush=True)

    shards=sorted(glob.glob(os.path.join(a.shards,'shard_*.npz')))
    imgs=[]; meta=[]; ids=[]
    for s in shards:
        z=np.load(s,allow_pickle=True); imgs.append(z['img']); meta.append(z['meta']); ids.append(z['study_ids'])
    img=np.concatenate(imgs); meta=np.concatenate(meta); gids=np.concatenate(ids); del imgs
    by={}
    for row,(si,pid,fb) in enumerate(meta): by.setdefault(int(si),[]).append((row,int(pid),int(fb)))
    print('slices',len(img),'studies',len(gids),flush=True)
    lab=pd.read_csv(a.labels).set_index('StudyInstanceUID').reindex(gids)
    tr=pd.read_csv(a.train).set_index('StudyInstanceUID').reindex(gids)
    lg=pd.read_csv(a.lang).set_index('StudyInstanceUID').reindex(gids); lc=[c for c in lg.columns if 'lang' in c.lower()][0]
    Y=lab[LABELS].values.astype(np.float32); Yg=tr[LABELS].values.astype(np.float32)
    gold=tr['ACL'].notna().values; lang=lg[lc].fillna('unk').values
    tri=np.where(~gold)[0]; gi=np.where(gold)[0]
    strat=(Y[tri]>0.5).sum(1).clip(0,4)
    splits=list(StratifiedGroupKFold(a.folds,shuffle=True,random_state=a.seed).split(tri,strat,lang[tri]))
    train_idx=tri[splits[a.fold][0]]
    print('train studies',len(train_idx),'gold',len(gi),flush=True)

    enc=AutoModel.from_pretrained(a.model)
    lcfg=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.1,target_modules=['query','key','value','dense'],bias='none')
    enc=get_peft_model(enc,lcfg); enc.gradient_checkpointing_enable(); enc.enable_input_require_grads()
    model=Model(enc).to(dev)
    lora_params=[p for n,p in model.named_parameters() if p.requires_grad and 'lora' in n.lower()]
    head_params=[p for n,p in model.named_parameters() if p.requires_grad and 'lora' not in n.lower()]
    opt=torch.optim.AdamW([{'params':head_params,'lr':a.lr_head},{'params':lora_params,'lr':a.lr_lora}], weight_decay=1e-2)
    scaler=torch.cuda.amp.GradScaler(enabled=use_fp16)
    for p in lora_params: p.requires_grad=False    # PHASE 1: freeze LoRA

    def batch(rows,k):
        if len(rows)>k: rows=[rows[i] for i in np.random.choice(len(rows),k,replace=False)]
        idx=np.array([r[0] for r in rows])
        x=torch.from_numpy(img[idx].astype(np.float32)/255.0)[:,None].repeat(1,3,1,1)
        x=((x-IM_MEAN)/IM_STD).to(dev)
        return x, torch.tensor([r[1] for r in rows],device=dev), torch.tensor([r[2] for r in rows],device=dev)

    steps_p2=max(1,(a.epochs-a.warm))*len(train_idx); gstep=0; best=-1; t0=time.time()
    for ep in range(a.epochs):
        if ep==a.warm:
            for p in lora_params: p.requires_grad=True   # PHASE 2: unfreeze LoRA
            print('== phase2: LoRA unfrozen ==',flush=True)
        model.train(); np.random.shuffle(train_idx)
        for n_,si in enumerate(train_idx):
            rows=by.get(si)
            if not rows: continue
            x,pid,fb=batch(rows,a.ktrain); y=torch.tensor(Y[si],device=dev)
            opt.zero_grad()
            with torch.autocast('cuda',enabled=use_fp16):
                loss=F.binary_cross_entropy_with_logits(model(x,pid,fb),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if ep>=a.warm:   # cosine+warmup on LoRA lr in phase 2
                gstep+=1; frac=gstep/steps_p2; wu=min(1.0,gstep/(0.05*steps_p2))
                lr=a.lr_lora*wu*0.5*(1+math.cos(math.pi*frac))
                opt.param_groups[1]['lr']=lr
            if n_%800==0: print(f'  ep{ep} {n_}/{len(train_idx)} loss {loss.item():.3f} {time.time()-t0:.0f}s',flush=True)
        model.eval(); pg=[]
        with torch.no_grad(), torch.autocast('cuda',enabled=use_fp16):
            for si in gi:
                rows=by.get(si,[])
                if not rows: pg.append(np.full(12,0.5)); continue
                x,pid,fb=batch(rows,20); pg.append(torch.sigmoid(model(x,pid,fb)).float().cpu().numpy())
        au=macro_auc(Yg[gi],np.array(pg))
        print(f'=== ep{ep} gold-58 AUC {au:.4f} best {max(best,au):.4f} ({time.time()-t0:.0f}s) ===',flush=True)
        if au>best:
            best=au; model.enc.save_pretrained(a.out+'_adapter')
            torch.save({'heads':[{k:v.cpu() for k,v in model.state_dict().items() if 'enc.' not in k}],
                        'labels':LABELS,'gold58':best}, a.out+'_head.pt')
    print(f'\n=== LoRA v2 BEST gold-58 AUC {best:.4f} (frozen was 0.59) ===',flush=True)

if __name__=='__main__': main()
