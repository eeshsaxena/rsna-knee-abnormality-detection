"""Surrogate fold model for PX_PARITY when nartaa's E_prior_s42_fall.pt is not on the box: compact_train 14767c7a Net with the S6fall
head (--spatial-attn, in_chans 3, crop 288), seeded; every head parameter that reads meta or position (slot, view, geom, sp_q, sp_pos,
q_win, cls_w, bias) is drawn non-zero, so the logits move with every pixel, every meta column and the window set. Weights are a test
fixture only (never trained, never shipped). File name matches compact_infer's glob <arm>_s*_f*.pt."""
import argparse, hashlib, os, sys
import torch
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--backbone", required=True)
ap.add_argument("--init", default=None, help="safetensors for the trunk (convnext_small: the 2457488b init); none = timm random init")
ap.add_argument("--seed", type=int, default=4343)
ap.add_argument("--out", required=True, help="output .pt (state dict)")
a = ap.parse_args()
sys.path.insert(0, a.src)
import compact_train as CT
torch.manual_seed(a.seed)
net = CT.Net(a.backbone, a.init, freeze_blocks=16, crop=288, state_spec=None, in_chans=3, spatial=True)
g = torch.Generator().manual_seed(a.seed + 1)
with torch.no_grad():
    for name, std in (("slot.weight", .5), ("view.weight", .5), ("geom.weight", .5), ("sp_q", 1.0), ("sp_pos.weight", .5),
                      ("q_win", 1.0), ("cls_w", .15), ("bias", .5)):
        p = dict(net.named_parameters())[name]
        p.copy_(torch.randn(p.shape, generator=g) * std)
sd = net.state_dict()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
torch.save(sd, a.out)
print("SURROGATE", a.out, a.backbone, "params %d" % sum(v.numel() for v in sd.values()),
      "sha256", hashlib.sha256(open(a.out, "rb").read()).hexdigest())
