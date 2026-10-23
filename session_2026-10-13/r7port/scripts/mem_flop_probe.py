"""CPU-only probe: activation memory per window and FLOPs per window of the R7 Net (convnext_small, crop 352, 3 slices,
spatial attn, extent, syn-decomp, state head).  No GPU is touched (CUDA_VISIBLE_DEVICES='' is set by the caller)."""
import os, sys, json, time, argparse
import torch
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--init", required=True)
ap.add_argument("--spec", required=True)
ap.add_argument("--crop", type=int, default=352)
ap.add_argument("--wins", type=int, default=4)
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
torch.set_num_threads(a.threads)
sys.path.insert(0, a.src)
import compact_train as CT
spec = json.load(open(a.spec))
out = {}
def build(gc):
    m = CT.Net("convnext_small", a.init, crop=a.crop, state_spec=spec, in_chans=3, spatial=True, extent=True, syn_decomp=True,
               extent_classes=None).to(memory_format=torch.channels_last)
    if gc:
        m.enc.set_grad_checkpointing(True)
    m.train()
    return m
def batch(B, W):
    img = torch.randint(0, 255, (B, W, 3, a.crop, a.crop), dtype=torch.uint8)
    meta = torch.zeros((B, W, 9)); meta[..., 0] = 1; meta[..., 1] = 1; meta[..., 3] = 1.2; meta[..., 7] = 0; meta[..., 8] = 30
    mask = torch.ones((B, W), dtype=torch.bool)
    return img, meta, mask
for gc in (False, True):
    m = build(gc)
    img, meta, mask = batch(1, a.wins)
    seen = {}
    def pack(t):
        try:
            key = t.untyped_storage().data_ptr()
            seen[key] = max(seen.get(key, 0), t.untyped_storage().nbytes())
        except Exception:
            pass
        return t
    t0 = time.time()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            z, sl, bl = m(img, meta, mask, return_aux=True)
            loss = z.float().pow(2).mean() + sl.float().masked_fill(torch.isinf(sl), 0).pow(2).mean()
    t1 = time.time()
    loss.backward()
    t2 = time.time()
    # with gradient checkpointing the stage internals are recomputed in backward: the hooks above saw only what the forward saved
    act = sum(seen.values())
    out["gc" if gc else "nogc"] = {"saved_bytes_per_window_bf16": act / a.wins, "fwd_s_per_window_cpu": (t1 - t0) / a.wins,
                                   "bwd_s_per_window_cpu": (t2 - t1) / a.wins}
    print(("gc" if gc else "nogc"), json.dumps(out["gc" if gc else "nogc"]), flush=True)
# FLOPs of the encoder for one window at the crop
from torch.utils.flop_counter import FlopCounterMode
m = build(False).eval()
x = torch.randn(1, 3, a.crop, a.crop).contiguous(memory_format=torch.channels_last)
with FlopCounterMode(display=False) as fc:
    with torch.no_grad():
        m.enc(x)
out["enc_fwd_gflops_per_window"] = fc.get_total_flops() / 1e9
n_par = sum(p.numel() for p in m.parameters())
out["params_M"] = n_par / 1e6
out["enc_params_M"] = sum(p.numel() for p in m.enc.parameters()) / 1e6
print(json.dumps(out, indent=1))
