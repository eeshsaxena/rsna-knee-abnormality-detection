# fleet_model.py -- re-implementation of aastik's KneeNet contract (his model.py is private).
# Inferred from intel/aastik_ours_script.py: ConvNeXt-tiny per 2.5D slice image, backbone run
# in chunks of 32, per-target slice attention (MultiAttnPool, temperature exp(logt)), per-target
# view attention with presence mask + anatomical view prior (QueryViewPool, vbias), optional
# contrast embedding (ctr_emb), per-target heads ("pt" in the ckpt name = per_target, a guess).
# round3/fleet: M1 forward(pos=...) so the playbook's G0 budget variant (train on K < n_slices
# triplets, infer on all n_slices) runs instead of crashing on the slice_pos broadcast. pos=None is
# bit-identical to the draft (inference and eval always use pos=None).
import math
import torch
import torch.nn as nn

IMNET_M = (0.485, 0.456, 0.406)
IMNET_S = (0.229, 0.224, 0.225)


class KneeFleetNet(nn.Module):
    def __init__(self, backbone='convnext_tiny.fb_in22k_ft_in1k', n_targets=12, n_views=4,
                 n_slices=24, dim=256, n_contrast=4, contrast_emb=True, pretrained=True,
                 drop_path=0.1, dropout=0.2, chunk=32, grad_ckpt=False):
        super().__init__()
        import timm
        self.cnn = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                     drop_path_rate=drop_path)
        if grad_ckpt:
            self.cnn.set_grad_checkpointing(True)
        D = self.cnn.num_features
        self.chunk, self.dim = chunk, dim
        self.proj = nn.Sequential(nn.Linear(D, dim), nn.LayerNorm(dim), nn.GELU())
        self.slice_pos = nn.Parameter(torch.zeros(n_views, n_slices, dim))
        self.view_emb = nn.Parameter(torch.zeros(n_views, dim))
        self.ctr_emb = nn.Embedding(n_contrast, dim) if contrast_emb else None
        self.slice_q = nn.Parameter(torch.randn(n_targets, dim) * 0.02)
        self.slice_k = nn.Linear(dim, dim)
        self.logt_s = nn.Parameter(torch.zeros(1))
        self.view_q = nn.Parameter(torch.randn(n_targets, dim) * 0.02)
        self.view_k = nn.Linear(dim, dim)
        self.vbias = nn.Parameter(torch.zeros(n_targets, n_views))
        self.logt_v = nn.Parameter(torch.zeros(1))
        self.drop = nn.Dropout(dropout)
        self.head_w = nn.Parameter(torch.randn(n_targets, dim) * 0.02)
        self.head_b = nn.Parameter(torch.zeros(n_targets))
        self.register_buffer('m', torch.tensor(IMNET_M).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('s', torch.tensor(IMNET_S).view(1, 3, 1, 1), persistent=False)

    def encode(self, x_u8):
        # x_u8: (N, 3, H, W) uint8 -> (N, D). Normalisation INSIDE the model, so mirror /
        # flip TTA can act on raw uint8 (avoids aastik's S19 channel-constant bug).
        outs = []
        for i in range(0, x_u8.shape[0], self.chunk):
            x = x_u8[i:i + self.chunk].float().div_(255.0)
            x = ((x - self.m) / self.s).contiguous(memory_format=torch.channels_last)
            outs.append(self.cnn(x))
        return torch.cat(outs)

    def forward(self, views_u8, present, ctr=None, pos=None):
        # views_u8: (B, V, S, 3, H, W) uint8; present: (B, V) bool; ctr: (B, V) long
        # pos: optional (B, V, S) long = the slice_pos row of each input triplet (M1, training only)
        B, V, S = views_u8.shape[:3]
        f = self.encode(views_u8.flatten(0, 2)).float()
        if pos is None:
            assert S == self.slice_pos.shape[1], f'{S} slices in, model built for {self.slice_pos.shape[1]}'
            sp = self.slice_pos[None]
        else:
            sp = self.slice_pos[torch.arange(V, device=pos.device)[None, :, None], pos]   # (B, V, S, d)
        h = self.proj(f).view(B, V, S, self.dim) + sp + self.view_emb[None, :, None]
        if self.ctr_emb is not None and ctr is not None:
            h = h + self.ctr_emb(ctr)[:, :, None]
        sc = 1.0 / math.sqrt(self.dim)
        a = torch.einsum('td,bvsd->bvts', self.slice_q, self.slice_k(h)) * sc * self.logt_s.exp()
        pooled = torch.einsum('bvts,bvsd->bvtd', a.softmax(-1), h)            # (B,V,T,d)
        va = torch.einsum('td,bvtd->btv', self.view_q, self.view_k(pooled)) * sc * self.logt_v.exp()
        va = va + self.vbias[None]
        pres = present.clone()
        pres[pres.sum(1) == 0] = True                                          # never all-masked
        va = va.masked_fill(~pres[:, None, :], -1e4).softmax(-1)
        z = torch.einsum('btv,bvtd->btd', va, pooled)                          # (B,T,d)
        return (self.drop(z) * self.head_w[None]).sum(-1) + self.head_b         # (B,12) logits


def mirror_views(views_u8, view_planes):
    """Train-time mirror aug / optional TTA on RAW uint8 (B,V,S,3,H,W): sagittal = reverse the
    slice axis AND the triplet channels (laterality lives on the slice axis); axial/coronal = flip W.
    Labels unchanged (a mirrored right knee is a left knee)."""
    out = views_u8.clone()
    for j, p in enumerate(view_planes):
        if p == 'Sagittal':
            out[:, j] = views_u8[:, j].flip(dims=[1, 2])
        else:
            out[:, j] = views_u8[:, j].flip(dims=[-1])
    return out
