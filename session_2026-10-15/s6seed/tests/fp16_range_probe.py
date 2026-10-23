"""CPU probe of the fp16 headroom of the S6 model at its INIT (timm convnext_small in22k-ft-in1k + the compact_train spatial head):
fp32 forward of real fixture windows (train-mode sampling + centre crop 288), max |activation| of every Linear / Conv2d / LayerNorm
output and of every ConvNeXt stage output, compared with the fp16 max 65504.  A trained model can drift; this only shows the
starting headroom (the 300-step GPU smoke and the per-epoch skip guard cover the rest).  CPU only."""
import argparse, importlib.util, json, os, sys
import numpy as np

assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--px", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--studies", type=int, default=4)
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    spec = importlib.util.spec_from_file_location("ct_probe", a.orig)
    CT = importlib.util.module_from_spec(spec)
    sys.modules["ct_probe"] = CT
    spec.loader.exec_module(CT)
    net = CT.Net("convnext_small", a.init, crop=288, in_chans=3, spatial=True).eval()
    worst = {}

    def hook(name):
        def f(m, i, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            if torch.is_tensor(t) and t.is_floating_point():
                worst[name] = max(worst.get(name, 0.0), float(t.detach().abs().max()))
        return f
    for n, m in net.named_modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm)) or n.count(".") == 2 and n.startswith("enc.stages."):
            m.register_forward_hook(hook(n))
    idx = json.load(open(os.path.join(a.px, "index.json")))
    uids = [u for u in idx["order"] if idx["done"][u][1] > 0][: a.studies]
    ds = CT.Studies(a.px, uids, a.windows, "train", 777, eval_stride=3, train_sampling="uniform", slices=3, phys_mm=3.5)
    with torch.no_grad():
        for i in range(len(ds)):
            img, meta, _ = ds[i]
            o = (img.shape[-1] - 288) // 2
            mask = torch.ones((1, img.shape[0]), dtype=torch.bool)
            net(img[None, :, :, o:o + 288, o:o + 288], meta[None], mask)
    top = sorted(worst.items(), key=lambda kv: -kv[1])[:8]
    stages = {k: round(v, 2) for k, v in worst.items() if k.count(".") == 2 and k.startswith("enc.stages.")}
    m = max(worst.values())
    print("FP16_RANGE " + json.dumps({"max_abs_activation": round(m, 2), "headroom_x": round(65504.0 / m, 1), "top8": [[k, round(v, 2)] for k, v in top],
                                      "stage_outputs": stages, "studies": len(uids), "windows_per_study": a.windows}))


if __name__ == "__main__":
    main()
