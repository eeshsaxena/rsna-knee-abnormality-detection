"""Per-step cost proxies for the S6 member on the V100, measured on the box CPU only (the GPU is never touched).

  F  matmul/conv FLOPs of one training step (torch.utils.flop_counter), S6 recipe vs the T_a reference whose V100 rate is MEASURED
     (T_a: ConvNeXt-T 224, bs 2 x 4 views x 16 slices = 128 images / step, no grad ckpt, 0.30 s/it on the V100):
       S6 = compact_train Net(convnext_small, spatial attn, crop 288, 3 ch) + grad ckpt, batch 2 x 64 windows, fwd + bwd (+ recompute)
       Ta = timm convnext_tiny 224, 128 images, fwd + bwd
     both counted at small window counts and extrapolated linearly (FLOPs are linear in the window count).
  W  CPU wall seconds of the same two steps (fwd + bwd + AdamW, fp32), interleaved repeats, per image -> a second ratio.
  IO cold-cache read time per study of the px cache: the original memmap reader vs the S6V100 contiguous reader (S6_PX_GAP 16 / -1 /
     0), page cache dropped for the file before each variant (fsync + POSIX_FADV_DONTNEED, no root needed).
Prints a JSON line COST_PROXY {...}."""
import argparse, importlib.util, json, os, sys, time
import numpy as np

assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""
os.environ.update({"S6_DEVICE": "cpu", "S6_AMP": "fp32"})
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def s6_step_fn(CT, W, crop=288, ckpt=True):
    torch.manual_seed(0)
    net = CT.Net("convnext_small", None, crop=crop, in_chans=3, spatial=True).to(memory_format=torch.channels_last)
    if ckpt:
        net.enc.set_grad_checkpointing(True)
    net.train()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.05)
    img = torch.randint(0, 255, (2, W, 3, crop, crop), dtype=torch.uint8)
    meta = torch.zeros((2, W, 6))
    meta[..., 1] = 0
    mask = torch.ones((2, W), dtype=torch.bool)
    tgt = torch.rand(2, 12)

    def step():
        opt.zero_grad(set_to_none=True)
        z = net(img, meta, mask)
        loss = F.binary_cross_entropy_with_logits(z, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
        opt.step()
    return step


def ta_step_fn(n_img, size=224):
    import timm
    torch.manual_seed(0)
    net = timm.create_model("convnext_tiny", pretrained=False, num_classes=12, drop_path_rate=0.2).to(memory_format=torch.channels_last)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.05)
    x = torch.randn(n_img, 3, size, size).contiguous(memory_format=torch.channels_last)
    tgt = torch.rand(n_img, 12)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(net(x), tgt)
        loss.backward()
        opt.step()
    return step


def flops(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()


def timeit(fn, reps):
    fn()                                            # warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), ts


def drop_cache(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def io_bench(CO, CN, px, uids, reps_epochs=1):
    out = {}
    path = os.path.join(px, "px.u8.npy")
    for tag in ("orig", "gap16", "gap-1", "gap0"):
        drop_cache(path)
        time.sleep(1.0)
        if tag == "orig":
            ds = CO.Studies(px, uids, 64, "train", 9731, eval_stride=3, train_sampling="uniform", slices=3, phys_mm=3.5)
        else:
            CN.S6_PX_GAP = int(tag[3:])
            ds = CN.Studies(px, uids, 64, "train", 9731, eval_stride=3, train_sampling="uniform", slices=3, phys_mm=3.5)
        ts = []
        for i in range(len(uids)):
            t = time.perf_counter()
            ds[i]
            ts.append(time.perf_counter() - t)
        r = {"s_per_study_mean": round(float(np.mean(ts)), 4), "s_per_study_median": round(float(np.median(ts)), 4),
             "s_per_study_p90": round(float(np.percentile(ts, 90)), 4)}
        if tag != "orig":
            r["preads_per_study"] = round(ds.read_stats[0] / len(uids), 1)
            r["MB_read_per_study"] = round(ds.read_stats[1] / 1e6 / len(uids), 1)
        out[tag] = r
    CN.S6_PX_GAP = 16
    idx = json.load(open(os.path.join(px, "index.json")))
    rows = [idx["done"][u][1] for u in uids]
    S = np.load(path, mmap_mode="r").shape[-1]
    out["study_MB_mean"] = round(float(np.mean(rows)) * S * S / 1e6, 1)
    out["rows_per_study_mean"] = round(float(np.mean(rows)), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--px", default=None)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-cpu", action="store_true", help="skip the CPU wall timing (IO only)")
    ap.add_argument("--ta-s-per-it", type=float, default=0.30, help="MEASURED T_a V100 s/it (critique 10-15: 1,733 it/epoch at 0.30 s/it)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    CO = load("ct_orig_cp", a.orig)
    CN = load("ct_new_cp", a.new)
    res = {"threads": a.threads, "torch": torch.__version__}
    # F (MODEL): analytic MACs, timm's published counts at 224 (convnext_tiny 4.47 G, convnext_small 8.70 G), x (crop/224)^2 for a
    # convnet; fwd + bwd = 3 x fwd, + 1 x fwd recompute with grad ckpt.  The spatial-attention head is < 0.1 % (per window: 81
    # positions x 768 x 128 key MACs + 12 x 768 x 384 proj MACs).  Net.forward pads the encoder batch to a multiple of 32
    # windows (no BatchNorm), so 2 x 64 = 128 windows run unpadded.
    g_s6 = 8.70e9 * (288 / 224) ** 2
    g_ta = 4.47e9
    f_s6 = 128 * g_s6 * 2 * 4
    f_s6_nockpt = 128 * g_s6 * 2 * 3
    f_ta = 128 * g_ta * 2 * 3
    res.update({"F_s6_step_TFLOP": round(f_s6 / 1e12, 3), "F_s6_step_nockpt_TFLOP": round(f_s6_nockpt / 1e12, 3),
                "F_ta_step_TFLOP": round(f_ta / 1e12, 3), "F_ratio": round(f_s6 / f_ta, 3), "F_ratio_nockpt": round(f_s6_nockpt / f_ta, 3)})
    # diagnostic: what torch's FlopCounterMode reports for one fwd of convnext_tiny 224 (analytic 8.94 GFLOP)
    import timm
    m = timm.create_model("convnext_tiny", pretrained=False, num_classes=12).eval()
    with torch.no_grad():
        res["flopcounter_convnext_tiny_fwd_GFLOP"] = round(flops(lambda: m(torch.randn(1, 3, 224, 224))) / 1e9, 2)
    del m
    # W (MEASURED, box CPU): fwd + bwd + AdamW wall per image, 32 images each (a multiple of 32: no padding), interleaved repeats
    if a.no_cpu:
        a.reps = 0
    s6 = s6_step_fn(CO, 16) if a.reps else None                      # 2 studies x 16 windows = 32 windows, grad ckpt (the recipe)
    ta = ta_step_fn(32) if a.reps else None      # 32 images at 224, no ckpt (T_a)
    w_s6, w_ta = [], []
    for _ in range(a.reps):
        w_s6.append(timeit(s6, 1)[0] / 32.0)
        w_ta.append(timeit(ta, 1)[0] / 32.0)
    res["V100_s_per_step_from_F"] = round(a.ta_s_per_it * res["F_ratio"], 3)
    if a.reps:
        ws, wt = float(np.median(w_s6)), float(np.median(w_ta))
        res.update({"cpu_s_per_img_s6": round(ws, 4), "cpu_s_per_img_ta": round(wt, 4), "W_ratio_step": round(ws / wt, 3),
                    "cpu_reps_s6": [round(x, 4) for x in w_s6], "cpu_reps_ta": [round(x, 4) for x in w_ta]})
        res["V100_s_per_step_from_W"] = round(a.ta_s_per_it * res["W_ratio_step"], 3)
    if a.px:
        idx = json.load(open(os.path.join(a.px, "index.json")))
        uids = [u for u in idx["order"] if idx["done"][u][1] > 0]
        res["io"] = io_bench(CO, CN, a.px, uids)
    print("COST_PROXY " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
