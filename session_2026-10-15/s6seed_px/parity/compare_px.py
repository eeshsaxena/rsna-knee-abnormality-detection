"""Byte comparison of two px_build_s6 caches on their common studies: px rows, meta rows and blocks (row offsets made study-local).
Used to prove that a builder change (v2 parent guard) or a thread-count change leaves every byte of the cache unchanged."""
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
a = ap.parse_args()
ia, ib = (json.load(open(os.path.join(d, "index.json"))) for d in (a.a, a.b))
pa, pb = (np.load(os.path.join(d, "px.u8.npy"), mmap_mode="r") for d in (a.a, a.b))
ma, mb = (np.load(os.path.join(d, "meta.f32.npy"), mmap_mode="r") for d in (a.a, a.b))
common = sorted(set(ia["done"]) & set(ib["done"]))
bad = []
for u in common:
    (sa, na, _), (sb, nb, _) = ia["done"][u], ib["done"][u]
    loc = lambda bl, s: [dict(b, views=[[v, r - s] for v, r in b["views"]]) for b in bl]
    ok = (na == nb and np.array_equal(pa[sa:sa + na], pb[sb:sb + nb]) and np.array_equal(ma[sa:sa + na].view(np.uint32), mb[sb:sb + nb].view(np.uint32))
          and loc(ia["blocks"][u], sa) == loc(ib["blocks"][u], sb))
    if not ok:
        bad.append(u)
print("COMPARE_PX %s: %d common studies, %d differ %s" % ("SAME" if common and not bad else "DIFF", len(common), len(bad), bad[:5]))
