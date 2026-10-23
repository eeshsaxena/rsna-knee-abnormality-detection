#!/usr/bin/env python3
"""BOX-side: embed image shards with frozen DINOv2-S on the V100, then train a
gated-attention-MIL head on soft rule-x-Qwen labels; report gold-58 macro-AUC.
Runs entirely on the college box (zero Kaggle GPU).

Usage: python box_train.py --shards ~/rsna_shards --model ~/dinov2_small \
    --labels ~/rsna_train/pseudo_labels_v1.csv --lang ~/rsna_train/_reports_lang.csv \
    --train ~/rsna_train/train.csv
"""
import argparse, glob, os, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

# The box's cuDNN can't find a conv engine for the V100 (cuDNN/driver mismatch);
# disable cuDNN so PyTorch uses its native CUDA conv (a bit slower, but it runs).
torch.backends.cudnn.enabled = False

LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
          'Effusion','Synovitis',"Baker's",'Contusion','Fracture']
IM_MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
IM_STD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)


def embed_shards(shard_dir, model_dir, dev, bs=16):
    cache = os.path.join(shard_dir, 'emb_cache.npz')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        embs, meta, gids = z['emb'], z['meta'], z['study_ids']
        print('LOADED embedding cache', embs.shape, '- skipping embed', flush=True)
    else:
        enc = AutoModel.from_pretrained(model_dir).eval().to(dev)   # fp32: V100 cuDNN lacks fp16 conv engine
        shards = sorted(glob.glob(os.path.join(shard_dir, 'shard_*.npz')))
        print('shards:', len(shards))
        # memory-efficient: embed ONE shard at a time (~60MB imgs in RAM), keep only embeddings
        emb_l, meta_l, ids = [], [], []
        t0 = time.time()
        with torch.no_grad():
            for si_sh, s in enumerate(shards):
                z = np.load(s, allow_pickle=True)
                img = z['img']; meta_l.append(z['meta']); ids.append(z['study_ids'])
                e = np.zeros((len(img), enc.config.hidden_size), np.float32)
                for i in range(0, len(img), bs):
                    x = torch.from_numpy(img[i:i+bs].astype(np.float32) / 255.0)[:, None].repeat(1, 3, 1, 1)
                    x = ((x - IM_MEAN) / IM_STD).to(dev)
                    e[i:i+bs] = enc(pixel_values=x).last_hidden_state[:, 0].float().cpu().numpy()
                emb_l.append(e); del img, z
                if si_sh % 10 == 0: print(f'  shard {si_sh}/{len(shards)} {time.time()-t0:.0f}s', flush=True)
        embs = np.concatenate(emb_l); meta = np.concatenate(meta_l); gids = np.concatenate(ids)
        print('total slices', embs.shape[0], 'studies', len(gids))
        np.savez_compressed(cache, emb=embs, meta=meta, study_ids=gids)  # embed checkpoint
        print('SAVED embedding cache', flush=True)
        del enc; torch.cuda.empty_cache()
    by = {}
    for row, (si, pid, fb) in enumerate(meta):
        by.setdefault(int(si), []).append((row, int(pid), int(fb)))
    return embs, by, gids


class MIL(nn.Module):
    def __init__(s, di=384, d=256, n=12):
        super().__init__(); s.proj=nn.Sequential(nn.Linear(di,d),nn.LayerNorm(d),nn.GELU())
        s.plane=nn.Embedding(3,d); s.fluid=nn.Embedding(2,d)
        s.V=nn.Linear(d,128); s.U=nn.Linear(d,128); s.w=nn.Linear(128,1)
        s.head=nn.Sequential(nn.Dropout(0.3), nn.Linear(d,n))
    def forward(s, x, pid, fb):
        h=s.proj(x)+s.plane(pid)+s.fluid(fb)
        a=torch.softmax(s.w(torch.tanh(s.V(h))*torch.sigmoid(s.U(h))),0)
        return s.head((a*h).sum(0))


def macro_auc(y, p):
    a=[roc_auc_score(y[:,j], p[:,j]) for j in range(y.shape[1]) if len(set(y[:,j]))>1]
    return float(np.mean(a)), a


def main():
    ap=argparse.ArgumentParser()
    for k in ['shards','model','labels','lang','train']: ap.add_argument('--'+k, required=True)
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seeds', type=int, default=2)
    a=ap.parse_args()
    dev='cuda' if torch.cuda.is_available() else 'cpu'; print('dev', dev)
    embs, by, gids = embed_shards(a.shards, a.model, dev)
    lab=pd.read_csv(a.labels).set_index('StudyInstanceUID').reindex(gids)
    tr=pd.read_csv(a.train).set_index('StudyInstanceUID').reindex(gids)
    lg=pd.read_csv(a.lang).set_index('StudyInstanceUID').reindex(gids)
    lcol=[c for c in lg.columns if 'lang' in c.lower()][0]
    Y=lab[LABELS].values.astype(np.float32); Yg=tr[LABELS].values.astype(np.float32)
    gold=tr['ACL'].notna().values; lang=lg[lcol].fillna('unk').values
    tri=np.where(~gold)[0]; gi=np.where(gold)[0]
    print(f'train {len(tri)} gold {len(gi)}')
    et=torch.tensor(embs, device=dev)
    def st(si):
        r=by.get(si,[(0,0,0)]); idx=torch.tensor([x[0] for x in r],device=dev)
        return et[idx], torch.tensor([x[1] for x in r],device=dev), torch.tensor([x[2] for x in r],device=dev)
    strat=(Y[tri]>0.5).sum(1).clip(0,4)
    gold_oof=np.zeros((len(gi),12))
    for seed in range(a.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        for fold,(t,_) in enumerate(StratifiedGroupKFold(a.folds,shuffle=True,random_state=seed).split(tri,strat,lang[tri])):
            m=MIL().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-2)
            ts=tri[t]; best=-1
            for ep in range(a.epochs):
                m.train(); np.random.shuffle(ts)
                for si in ts:
                    x,pid,fb=st(si); loss=F.binary_cross_entropy_with_logits(m(x,pid,fb),torch.tensor(Y[si],device=dev))
                    opt.zero_grad(); loss.backward(); opt.step()
                m.eval(); pg=np.array([torch.sigmoid(m(*st(si))).detach().cpu().numpy() for si in gi])
                au,_=macro_auc(Yg[gi],pg)
                if au>best: best,bpg=au,pg
            gold_oof+=bpg; print(f'  seed{seed} fold{fold} gold-58 AUC {best:.4f}', flush=True)
    gold_oof/=(a.seeds*a.folds)
    au,aus=macro_auc(Yg[gi],gold_oof)
    print(f'\n=== BOX FROZEN MODEL gold-58 macro-AUC: {au:.4f} ===')
    np.save('gold_oof_box.npy',gold_oof)

    # ---- train deployment heads on ALL report-only studies (for the Kaggle inference kernel) ----
    print('training deployment heads on all data...', flush=True)
    deploy=[]
    for seed in range(3):
        torch.manual_seed(200+seed); np.random.seed(200+seed); m=MIL().to(dev)
        opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-2); ids=list(tri)
        for ep in range(40):
            m.train(); np.random.shuffle(ids)
            for si in ids:
                x,pid,fb=st(si); loss=F.binary_cross_entropy_with_logits(m(x,pid,fb),torch.tensor(Y[si],device=dev))
                opt.zero_grad(); loss.backward(); opt.step()
        deploy.append({k:v.cpu() for k,v in m.state_dict().items()})
    torch.save({'heads':deploy,'labels':LABELS,'d_in':384}, 'head_weights.pt')
    print('SAVED head_weights.pt (%d heads)' % len(deploy), flush=True)


if __name__=='__main__': main()
