# CPU: forward FLOPs of the Raptor backbone (one 3-slice window) at 384 and 320, and of ConvNeXt-T at 224 (our TTR224 step).
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch, timm
from torch.utils.flop_counter import FlopCounterMode
torch.set_num_threads(4)
for arch, res in (("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", 384), ("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", 320),
                  ("convnext_tiny.fb_in22k_ft_in1k", 224)):
    kw = dict(pretrained=False, num_classes=0, in_chans=3, global_pool="avg")
    if "coatnet" in arch: kw["img_size"] = res
    m = timm.create_model(arch, **kw).eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(torch.zeros(1, 3, res, res))
    print(f"{arch} @{res}: {fc.get_total_flops() / 2e9:.2f} GMACs fwd, params {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M", flush=True)
