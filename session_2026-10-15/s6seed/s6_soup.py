"""s6_soup.py (10-15 review S6R-1 option, CPU, seconds): uniform weight soup of S6fall-family checkpoints (compact_train Net
deploy state dicts, e.g. nartaa's E_prior_s42_fall.pt + our E_prior_s43_fall.pt) into ONE checkpoint, so the S6F stage keeps the
1-model GPU cost (compact_infer 34f433fe runs every E_prior_s*_f*.pt as a full forward, one after the other).
  - every input must have the same keys, shapes and dtypes; floating tensors are averaged in float64 and cast back; any
    non-floating tensor must be identical in all inputs (else FAIL)
  - output name must match compact_infer's glob E_prior_s*_f*.pt (default E_prior_s4243_fsoup.pt) and the file must sit ALONE in
    the --models folder of the serve (otherwise compact_infer averages it with whatever else is there)
  - writes <out>.json: input shas, output sha, max |a - b| per input pair (a size of how far apart the members are)
Only meaningful if both members started from the SAME head init (S6_HEAD_SEED=42 for s43; nartaa's ae18d2e1 init order is not
verifiable, GUESS) and the soup passes its own gold58 bar (A7). Usage: python s6_soup.py A.pt B.pt [...] --out DIR/E_prior_s4243_fsoup.pt"""
import argparse, fnmatch, hashlib, json, os, sys
import torch


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def soup(sds):
    k0 = list(sds[0].keys())
    for i, sd in enumerate(sds[1:], 1):
        if list(sd.keys()) != k0:
            raise SystemExit("input %d: key set / order differs" % i)
    out, dist = {}, {}
    for k in k0:
        ts = [sd[k] for sd in sds]
        if any(t.shape != ts[0].shape or t.dtype != ts[0].dtype for t in ts):
            raise SystemExit("tensor %s: shape / dtype differs" % k)
        if ts[0].is_floating_point():
            out[k] = (sum(t.double() for t in ts) / len(ts)).to(ts[0].dtype)
            dist[k] = float((ts[0].double() - ts[-1].double()).abs().max()) if ts[0].numel() else 0.0
        else:
            if not all(torch.equal(t, ts[0]) for t in ts):
                raise SystemExit("non-floating tensor %s differs between inputs (cannot average)" % k)
            out[k] = ts[0].clone()
    return out, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if len(a.inputs) < 2:
        raise SystemExit("need >= 2 checkpoints")
    if not fnmatch.fnmatch(os.path.basename(a.out), "E_prior_s*_f*.pt"):
        raise SystemExit("output name must match compact_infer's E_prior_s*_f*.pt glob")
    sds = [torch.load(p, map_location="cpu", weights_only=True) for p in a.inputs]
    out, dist = soup(sds)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(out, a.out)
    rec = {"inputs": [{"path": p, "sha256": sha(p)} for p in a.inputs], "out": a.out, "out_sha256": sha(a.out), "tensors": len(out),
           "max_abs_diff_first_last": max(dist.values()) if dist else 0.0,
           "top5_diff": sorted(dist.items(), key=lambda kv: -kv[1])[:5]}
    json.dump(rec, open(a.out + ".json", "w"), indent=1)
    print("SOUP " + json.dumps({k: rec[k] for k in ("out", "out_sha256", "tensors", "max_abs_diff_first_last")}))


if __name__ == "__main__":
    sys.exit(main())
