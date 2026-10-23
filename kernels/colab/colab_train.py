#!/usr/bin/env python3
"""End-to-end RSNA Knee trainer for Google Colab (free T4). timm backbone + gated
attention-MIL, trained END-TO-END (not frozen) on the preprocessed shards + soft labels.
fp16 autocast on T4; checkpoints to Google Drive every epoch and RESUMES from there, so a
Colab disconnect just picks up where it left off. gold-58 held out for eval.

Usage (from the Colab notebook):
  python colab_train.py --shards /content/data --labels pseudo_labels_v1.csv \
      --train train.csv --lang _reports_lang.csv --ckpt /content/drive/MyDrive/rsna_ckpt \
      --backbone tf_efficientnet_b3 --res 224 --K 12 --epochs 20 --fold 0
"""
import argparse, glob, os, time, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import timm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

LABELS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
        'Effusion','Synovitis',"Baker's",'Contusion','Fracture']
IM_MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
IM_STD =torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)


class MILNet(nn.Module):
    def __init__(s, backbone='tf_efficientnet_b3', d=256, n=12, pretrained=True):
        super().__init__()
        s.enc=timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=3)
        try: s.enc.set_grad_checkpointing(True)   # fits high-res (384px) backprop on 16GB
        except Exception: pass
        di=s.enc.num_features
        s.proj=nn.Sequential(nn.Linear(di,d), nn.LayerNorm(d), nn.GELU())
        s.plane=nn.Embedding(3,d); s.fluid=nn.Embedding(2,d)
        s.V=nn.Linear(d,128); s.U=nn.Linear(d,128); s.w=nn.Linear(128,1)
        s.head=nn.Sequential(nn.Dropout(0.3), nn.Linear(d,n))
    def forward(s, imgs, pid, fb):          # imgs [k,3,H,W]
        e=s.enc(imgs)                        # [k,di]
        h=s.proj(e)+s.plane(pid)+s.fluid(fb) # [k,d]
        a=torch.softmax(s.w(torch.tanh(s.V(h))*torch.sigmoid(s.U(h))),0)
        return s.head((a*h).sum(0))          # [n]


def macro_auc(y,p):
    a=[roc_auc_score(y[:,j],p[:,j]) for j in range(y.shape[1]) if len(set(y[:,j]))>1]
    return float(np.mean(a))


def main():
    ap=argparse.ArgumentParser()
    for k in ['shards','labels','train','lang','ckpt']: ap.add_argument('--'+k, required=True)
    ap.add_argument('--backbone', default='tf_efficientnet_b3')
    ap.add_argument('--res', type=int, default=224); ap.add_argument('--K', type=int, default=12)
    ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--fold', type=int, default=0); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    a=ap.parse_args()
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    use_fp16 = dev=='cuda'
    if dev=='cuda' and torch.cuda.get_device_capability(0)==(7,0):
        torch.backends.cudnn.enabled=False; use_fp16=False   # V100: fp32 + cudnn conv workaround
    print('device', torch.cuda.get_device_name(0) if dev=='cuda' else 'CPU',
          '| fp16', use_fp16, '| backbone', a.backbone, '| res', a.res, flush=True)
    os.makedirs(a.ckpt, exist_ok=True)
    ckpt_path=os.path.join(a.ckpt, f'{a.backbone}_res{a.res}_fold{a.fold}.pt')
    best_path=os.path.join(a.ckpt, f'{a.backbone}_res{a.res}_fold{a.fold}_BEST.pt')

    # ---- load preprocessed shards into RAM (uint8) ----
    shards=sorted(glob.glob(os.path.join(a.shards,'**','shard_*.npz'), recursive=True))
    assert shards, f'no shards under {a.shards}'
    imgs=[]; meta=[]; ids=[]
    for s in shards:
        z=np.load(s, allow_pickle=True); imgs.append(z['img']); meta.append(z['meta']); ids.append(z['study_ids'])
    img=np.concatenate(imgs); meta=np.concatenate(meta); gids=np.concatenate(ids); del imgs
    by={}
    for row,(si,pid,fb) in enumerate(meta): by.setdefault(int(si),[]).append((row,int(pid),int(fb)))
    print('slices', len(img), 'studies', len(gids), 'shape', img.shape[1:], flush=True)

    lab=pd.read_csv(a.labels).set_index('StudyInstanceUID').reindex(gids)
    tr =pd.read_csv(a.train ).set_index('StudyInstanceUID').reindex(gids)
    lg =pd.read_csv(a.lang  ).set_index('StudyInstanceUID').reindex(gids)
    lc=[c for c in lg.columns if 'lang' in c.lower()][0]
    Y=lab[LABELS].values.astype(np.float32); Yg=tr[LABELS].values.astype(np.float32)
    gold=tr['ACL'].notna().values; lang=lg[lc].fillna('unk').values
    tri=np.where(~gold)[0]; gi=np.where(gold)[0]
    strat=(Y[tri]>0.5).sum(1).clip(0,4)
    splits=list(StratifiedGroupKFold(a.folds, shuffle=True, random_state=a.seed).split(tri,strat,lang[tri]))
    train_idx=tri[splits[a.fold][0]]
    print('train studies', len(train_idx), 'gold', len(gi), flush=True)

    model=MILNet(a.backbone).to(dev)
    opt=torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    scaler=torch.cuda.amp.GradScaler(enabled=use_fp16)
    start_ep=0; best=-1
    if os.path.exists(ckpt_path):                       # RESUME after a Colab disconnect
        ck=torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        scaler.load_state_dict(ck['scaler']); start_ep=ck['epoch']+1; best=ck['best']
        print(f'RESUMED from epoch {start_ep} (best {best:.4f})', flush=True)

    RES=a.res
    def batch(rows, k):
        if len(rows)>k: rows=[rows[i] for i in np.random.choice(len(rows), k, replace=False)]
        idx=np.array([r[0] for r in rows])
        im=img[idx].astype(np.float32)/255.0                       # [k,H,W]
        x=torch.from_numpy(im)[:,None].repeat(1,3,1,1)             # [k,3,H,W] (2D: repeat mono)
        if x.shape[-1]!=RES: x=F.interpolate(x, size=(RES,RES), mode='bilinear', align_corners=False)
        x=((x-IM_MEAN)/IM_STD).to(dev)
        pid=torch.tensor([r[1] for r in rows], device=dev); fb=torch.tensor([r[2] for r in rows], device=dev)
        return x, pid, fb

    t0=time.time()
    for ep in range(start_ep, a.epochs):
        model.train(); np.random.shuffle(train_idx)
        # cosine LR with 5% warmup
        for n_, si in enumerate(train_idx):
            rows=by.get(si)
            if not rows: continue
            gstep=ep*len(train_idx)+n_; total=a.epochs*len(train_idx)
            lr=a.lr*min(1.0, gstep/(0.05*total))*0.5*(1+math.cos(math.pi*gstep/total))
            for g in opt.param_groups: g['lr']=lr
            x,pid,fb=batch(rows, a.K); y=torch.tensor(Y[si], device=dev)
            opt.zero_grad()
            with torch.autocast('cuda', enabled=use_fp16):
                loss=F.binary_cross_entropy_with_logits(model(x,pid,fb), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if n_%600==0: print(f'  ep{ep} {n_}/{len(train_idx)} loss {loss.item():.3f} {time.time()-t0:.0f}s', flush=True)
        # eval gold-58
        model.eval(); pg=[]
        with torch.no_grad(), torch.autocast('cuda', enabled=use_fp16):
            for si in gi:
                rows=by.get(si,[])
                if not rows: pg.append(np.full(12,0.5)); continue
                x,pid,fb=batch(rows, 20); pg.append(torch.sigmoid(model(x,pid,fb)).float().cpu().numpy())
        au=macro_auc(Yg[gi], np.array(pg))
        print(f'=== ep{ep} gold-58 AUC {au:.4f} best {max(best,au):.4f} ({time.time()-t0:.0f}s) ===', flush=True)
        torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'scaler':scaler.state_dict(),
                    'epoch':ep,'best':max(best,au)}, ckpt_path)   # resume point on Drive
        if au>best:
            best=au
            torch.save({'model':model.state_dict(),'backbone':a.backbone,'res':RES,'labels':LABELS,'gold58':best}, best_path)
    print(f'\n=== DONE best gold-58 AUC {best:.4f} -> {best_path} ===', flush=True)


if __name__=='__main__': main()
