import torch, json
ck = torch.load("/home/training/kfn70_test/wts/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", map_location="cpu", weights_only=False)
print(type(ck), list(ck.keys()) if isinstance(ck, dict) else "")
for k, v in ck.items():
    if k == "model":
        sd = v; print("model tensors", len(sd), "params", sum(t.numel() for t in sd.values() if t.dtype.is_floating_point))
        ks = list(sd.keys()); print("first keys", ks[:6]); print("last keys", ks[-8:])
        print("dtypes", sorted({str(t.dtype) for t in sd.values()}))
        print("buffers-like", [k for k in ks if "running" in k or "num_batches" in k][:6], sum(("running" in k) for k in ks))
    else:
        s = repr(v); print("KEY", k, ":", s[:3000])
