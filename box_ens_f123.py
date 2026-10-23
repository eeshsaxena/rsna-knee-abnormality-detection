#!/usr/bin/env python3
"""Multi-fold RadImageNet ensemble gold-58 search: frozen(5-head) + fold1 + fold2 + fold3.
Encodes each fine-tuned fold on the gold-58 studies (GPU), then searches ensemble weights.
Reports best 2/3/4-way gold-58 so we know the strongest deployable ensemble (proj LB ~= gold-58)."""
import glob, os, itertools, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet50
from sklearn.metrics import roc_auc_score
B = os.path.expanduser("~/rsna_train/"); dev = "cuda" if torch.cuda.is_available() else "cpu"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Bakers", "Contusion", "Fracture"]; LAB[9] = "Baker\x27s"
tr = pd.read_csv(B + "train.csv").set_index("StudyInstanceUID")


class Enc(nn.Module):
    def __init__(s): super().__init__(); s.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
    def forward(s, x): return s.backbone(x).mean(dim=(2, 3))


class MIL(nn.Module):
    def __init__(s, di, d=256, n=12):
        super().__init__(); s.proj = nn.Sequential(nn.Linear(di, d), nn.LayerNorm(d), nn.GELU()); s.plane = nn.Embedding(3, d); s.fluid = nn.Embedding(2, d)
        s.V = nn.Linear(d, 128); s.U = nn.Linear(d, 128); s.w = nn.Linear(128, 1); s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, n))
    def forward(s, x, pid, fb):
        h = s.proj(x) + s.plane(pid) + s.fluid(fb); a = torch.softmax(s.w(torch.tanh(s.V(h)) * torch.sigmoid(s.U(h))), 0); return s.head((a * h).sum(0))


def macro(y, p): return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(12) if len(set(y[:, j])) > 1]))


z = np.load(B + "emb_cache_radimagenet.npz", allow_pickle=True); emb = torch.tensor(z["emb"].astype(np.float32)); meta = z["meta"]; gids = np.array(z["study_ids"])
by = {}; [by.setdefault(int(si), []).append((r, int(pid), int(fb))) for r, (si, pid, fb) in enumerate(meta)]
Yg = tr.reindex(gids)[LAB].values.astype(np.float32); gold = tr.reindex(gids)["ACL"].notna().values; gi = np.where(gold)[0]; Y = Yg[gi]
ckf = torch.load(B + "head_weights_radimagenet.pt", map_location="cpu"); fh = [MIL(2048).eval() for _ in ckf["heads"]]
for m, sd in zip(fh, ckf["heads"]): m.load_state_dict(sd)
PF = []
with torch.no_grad():
    for si in gi:
        rows = by[si]; idx = torch.tensor([r[0] for r in rows]); pid = torch.tensor([r[1] for r in rows]); fb = torch.tensor([r[2] for r in rows])
        PF.append(np.mean([torch.sigmoid(m(emb[idx], pid, fb)).numpy() for m in fh], 0))
PF = np.array(PF); print("FROZEN gold-58", round(macro(Y, PF), 4), flush=True)
shards = sorted(glob.glob(B.replace("rsna_train/", "") + "rsna_shards_384/shard_*.npz")); img = []; m2 = []; id2 = []
for sh in shards: zz = np.load(sh, allow_pickle=True); img.append(zz["img"]); m2.append(zz["meta"]); id2.append(zz["study_ids"])
img = np.concatenate(img); m2 = np.concatenate(m2); id2 = np.concatenate(id2)
by2 = {}; [by2.setdefault(int(si), []).append((r, int(pid), int(fb))) for r, (si, pid, fb) in enumerate(m2)]
gold2 = tr.reindex(id2)["ACL"].notna().values; gi2 = np.where(gold2)[0]


def ftp(path):
    ck = torch.load(path, map_location="cpu"); enc = Enc(); enc.load_state_dict(ck["enc"]); enc = enc.eval().to(dev)
    ft = MIL(int(ck.get("di", 2048))).eval().to(dev); ft.load_state_dict(ck["head"]); P = []
    with torch.no_grad():
        for si in gi2:
            rows = by2[si]; idx = np.array([r[0] for r in rows]); pid = torch.tensor([r[1] for r in rows], device=dev); fb = torch.tensor([r[2] for r in rows], device=dev)
            x = torch.from_numpy(img[idx].astype(np.float32)) / 127.5 - 1.0; x = x[:, None].repeat(1, 3, 1, 1); x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).to(dev)
            P.append(torch.sigmoid(ft(enc(x), pid, fb)).cpu().numpy())
    del enc, ft; torch.cuda.empty_cache() if dev == "cuda" else None
    return np.array(P)


P1 = ftp(B + "finetune_f1_radimagenet.pt"); print("FOLD1 gold-58", round(macro(Y, P1), 4), flush=True)
P2 = ftp(B + "finetune_f2_radimagenet.pt"); print("FOLD2 gold-58", round(macro(Y, P2), 4), flush=True)
P3 = ftp(B + "finetune_f3_radimagenet.pt"); print("FOLD3 gold-58", round(macro(Y, P3), 4), flush=True)
preds = {"F": PF, "1": P1, "2": P2, "3": P3}
# best 2-way frozen+fold1 (reference)
b2 = max((macro(Y, w * PF + (1 - w) * P1), round(w, 2)) for w in np.arange(0, 1.01, 0.05))
print("BEST frozen+fold1 gold-58 %.4f at wF=%.2f" % b2, flush=True)
# full 4-way grid over {F,1,2,3} weights summing to 1 (step 0.1)
best = (-1, None)
grid = np.arange(0, 1.01, 0.1)
for wf in grid:
    for w1 in grid:
        for w2 in grid:
            w3 = 1 - wf - w1 - w2
            if w3 < -1e-9 or w3 > 1 + 1e-9: continue
            au = macro(Y, wf * PF + w1 * P1 + w2 * P2 + w3 * P3)
            if au > best[0]: best = (au, (round(wf, 1), round(w1, 1), round(w2, 1), round(w3, 1)))
print("BEST 4-way frozen+f1+f2+f3 gold-58 %.4f at (wF,w1,w2,w3)=%s" % best, flush=True)
print("(frozen 0.8718, frozen+fold1 was 0.8817)", flush=True)
