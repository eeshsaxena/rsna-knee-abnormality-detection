"""s6_head_init_check.py (10-15 review S6R-1 soup option, CPU, about a minute): does a trained S6fall checkpoint still carry the
head-init stream of a given seed?  The soup option (S6_HEAD_SEED=42 for s43) only makes sense if nartaa's trainer ae18d2e1 drew
its head init exactly like compact_train 14767c7a (torch.manual_seed(1000 + 17 * seed) right before Net()).  ae18d2e1 is not
available, but AdamW keeps part of the random init in the directions the gradient barely touches (decoupled decay alone leaves
about exp(-wd x sum lr) ~ .4 of it after 16 epochs, MODEL), so the cosine between the trained head tensors and each candidate
seed's init is a fingerprint.
  For each candidate seed: rebuild the Net exactly as the trainer does (same args, same RNG call order), take every head tensor
  (not 'enc.', not 'bias', >= 64 elements), cosine(trained, init).  MATCH = the claimed seed's median cosine >= .1 and >= 5x the
  largest median cosine of the other seeds.  Writes one verdict line (and --out file).
Usage: python s6_head_init_check.py --ckpt E_prior_s42_fall.pt --seed 42 [--others 43,0,1,7] --ct src/compact_train.py --init <timm safetensors>"""
import argparse, importlib.util, json, os, sys
import numpy as np
import torch


def build(CT, seed, init):
    torch.manual_seed(1000 + 17 * seed)                      # = compact_train 14767c7a, the line before Net()
    return CT.Net("convnext_small", init, freeze_blocks=16, crop=288, state_spec=None, in_chans=3, spatial=True, extent=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--others", default="43,0,1,7")
    ap.add_argument("--ct", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    spec = importlib.util.spec_from_file_location("ct_orig", a.ct)
    CT = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(CT)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    keys = [k for k, v in sd.items() if not k.startswith("enc.") and v.is_floating_point() and v.numel() >= 64 and k != "bias"]
    seeds = [a.seed] + [int(s) for s in a.others.split(",") if s.strip() and int(s) != a.seed]
    med = {}
    for s in seeds:
        net = build(CT, s, a.init)
        ini = net.state_dict()
        cs = []
        for k in keys:
            x, y = sd[k].double().flatten(), ini[k].double().flatten()
            cs.append(float(torch.dot(x, y) / (x.norm() * y.norm() + 1e-30)))
        med[s] = float(np.median(cs)) if cs else 0.0
        print("seed %d: head tensors %d, median cosine(trained, init) %.4f, min %.4f, max %.4f" % (
            s, len(cs), med[s], min(cs) if cs else 0, max(cs) if cs else 0), flush=True)
    other = max(abs(med[s]) for s in seeds[1:]) if len(seeds) > 1 else 0.0
    ok = med[a.seed] >= 0.1 and med[a.seed] >= 5 * other
    line = "%s HEAD_INIT seed %d median cosine %.4f vs best other %.4f (%s)" % ("MATCH" if ok else "NO_MATCH", a.seed, med[a.seed], other,
                                                                           json.dumps({str(k): round(v, 4) for k, v in med.items()}))
    print(line)
    if a.out:
        open(a.out, "w").write(line + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
