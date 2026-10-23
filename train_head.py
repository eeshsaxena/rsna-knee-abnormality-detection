#!/usr/bin/env python3
"""Gated-attention-MIL head on frozen DINOv2 embeddings (the frozen-path model).
Trains on soft rule x Qwen pseudo-labels for the 4349 report-only studies, with
language-grouped K-fold OOF; the 58 gold are a held-out eval set (early-stop + final
macro-AUC) and are NEVER trained on. Runs on CPU or a small GPU slice.

Usage: python train_head.py --emb embeddings.npz --labels _data/pseudo_labels_v1.csv \
                            --lang _data/_reports_lang.csv --train _data/train.csv
"""
import argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
          'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def load(args):
    z = np.load(args.emb, allow_pickle=True)
    emb, meta, sids = z['emb'].astype(np.float32), z['meta'], np.array(z['study_ids'])
    # group slice rows by study index
    by = {}
    for row, (si, pid, fb) in enumerate(meta):
        by.setdefault(int(si), []).append((row, int(pid), int(fb)))
    studies = [sids[i] for i in range(len(sids))]
    lab = pd.read_csv(args.labels).set_index('StudyInstanceUID').reindex(studies)
    tr = pd.read_csv(args.train).set_index('StudyInstanceUID').reindex(studies)
    lang = pd.read_csv(args.lang).set_index('StudyInstanceUID').reindex(studies)
    langcol = [c for c in lang.columns if 'lang' in c.lower()][0]
    gold = tr['ACL'].notna().values
    Y = lab[LABELS].values.astype(np.float32)
    Ygold = tr[LABELS].values.astype(np.float32)
    return emb, by, studies, Y, Ygold, gold, lang[langcol].fillna('unk').values


class MIL(nn.Module):
    def __init__(self, d_in=384, d=256, n_lab=12):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d), nn.GELU())
        self.plane = nn.Embedding(3, d); self.fluid = nn.Embedding(2, d)
        self.att_V = nn.Linear(d, 128); self.att_U = nn.Linear(d, 128)
        self.att_w = nn.Linear(128, 1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, n_lab))

    def forward(self, x, pid, fb):            # x:[n,d_in]
        h = self.proj(x) + self.plane(pid) + self.fluid(fb)
        a = self.att_w(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h)))  # gated attn
        a = torch.softmax(a, 0)
        z = (a * h).sum(0)                    # [d]
        return self.head(z)                   # [n_lab] logits


def macro_auc(y, p):
    aucs = []
    for j in range(y.shape[1]):
        yj = y[:, j]
        if len(set(yj)) < 2:
            continue
        aucs.append(roc_auc_score(yj, p[:, j]))
    return float(np.mean(aucs)), aucs


def study_tensor(emb, by, si, dev):
    rows = by.get(si, [(0, 0, 0)])
    idx = torch.tensor([r for r, _, _ in rows])
    x = torch.from_numpy(emb[idx.numpy()]).to(dev)
    pid = torch.tensor([p for _, p, _ in rows]).to(dev)
    fb = torch.tensor([f for _, _, f in rows]).to(dev)
    return x, pid, fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb', required=True); ap.add_argument('--labels', required=True)
    ap.add_argument('--lang', required=True); ap.add_argument('--train', required=True)
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seeds', type=int, default=2)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    emb, by, studies, Y, Ygold, gold, lang = load(args)
    n = len(studies)
    tr_idx = np.where(~gold)[0]; gold_idx = np.where(gold)[0]
    print(f'studies {n} | train(report-only) {len(tr_idx)} | gold {len(gold_idx)} | dev {dev}')
    # stratify by a coarse positivity bucket for the grouped fold
    strat = (Y[tr_idx] > 0.5).sum(1).clip(0, 4)
    groups = lang[tr_idx]

    gold_oof = np.zeros((len(gold_idx), 12))
    for seed in range(args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (tri, _) in enumerate(sgkf.split(tr_idx, strat, groups)):
            model = MIL().to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            train_studies = tr_idx[tri]
            best, best_state = -1, None
            for ep in range(args.epochs):
                model.train(); np.random.shuffle(train_studies)
                for si in train_studies:
                    x, pid, fb = study_tensor(emb, by, si, dev)
                    logit = model(x, pid, fb)
                    y = torch.from_numpy(Y[si]).to(dev)
                    loss = nn.functional.binary_cross_entropy_with_logits(logit, y)
                    opt.zero_grad(); loss.backward(); opt.step()
                # eval on gold
                model.eval(); pg = []
                with torch.no_grad():
                    for si in gold_idx:
                        x, pid, fb = study_tensor(emb, by, si, dev)
                        pg.append(torch.sigmoid(model(x, pid, fb)).cpu().numpy())
                pg = np.array(pg)
                auc, _ = macro_auc(Ygold[gold_idx], pg)
                if auc > best:
                    best, best_state, best_pg = auc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, pg
            gold_oof += best_pg
            print(f'  seed{seed} fold{fold}: best gold-58 macroAUC {best:.4f}')
    gold_oof /= (args.seeds * args.folds)
    auc, aucs = macro_auc(Ygold[gold_idx], gold_oof)
    print(f'\n=== ENSEMBLE gold-58 macro-AUC: {auc:.4f} ===')
    for lab, a in zip([l for l in LABELS if len(set(Ygold[gold_idx][:, LABELS.index(l)])) > 1], aucs):
        print(f'  {lab:18s} {a:.3f}')
    np.save('gold_oof.npy', gold_oof)
    print('saved gold_oof.npy')


if __name__ == '__main__':
    main()
