"""CPU test of s6_soup.py: soup(A, A) == A bit for bit; soup(A, A + d) == A + d/2; an int-buffer mismatch FAILs; the soup loads
STRICT into the ORIGINAL compact_train 14767c7a Net; the output name matches compact_infer's glob. -> SOUP_TESTS PASS n/n"""
import importlib.util, os, subprocess, sys, tempfile
import torch
KIT = os.path.expanduser("~/s6seed")
A = os.path.join(KIT, "tests/out_s6v100/t4_full/models/E_prior_s43_fall.pt")
R = []
def check(n, ok, d=""):
    R.append(bool(ok)); print("%s %s %s" % ("PASS" if ok else "FAIL", n, d), flush=True)
SOUP = os.path.join(KIT, sys.argv[1] if len(sys.argv) > 1 else "s6_soup.py")
spec = importlib.util.spec_from_file_location("s6_soup", SOUP)
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
sd = torch.load(A, map_location="cpu", weights_only=True)
o, _ = S.soup([sd, sd])
check("S1_SOUP_OF_IDENTICAL_IS_IDENTITY", all(torch.equal(o[k], sd[k]) for k in sd), "%d tensors" % len(sd))
g = torch.Generator().manual_seed(0)
sd2 = {k: (v + torch.randn(v.shape, generator=g, dtype=torch.float64).to(v.dtype) * 1e-3) if v.is_floating_point() else v.clone() for k, v in sd.items()}
o2, dist = S.soup([sd, sd2])
nbad = sum(int(not torch.equal(o2[k], ((sd[k].double() + sd2[k].double()) / 2).to(sd[k].dtype))) for k in sd if sd[k].is_floating_point())
check("S2_SOUP_IS_THE_MEAN", nbad == 0, "%d tensors differ from the float64 mean cast back" % nbad)
ints = [k for k, v in sd.items() if not v.is_floating_point()]
if ints:
    sd3 = dict(sd); sd3[ints[0]] = sd[ints[0]] + 1
    try:
        S.soup([sd, sd3]); bad = False
    except SystemExit:
        bad = True
    check("S3_INT_MISMATCH_FAILS", bad, ints[0])
else:
    check("S3_NO_INT_TENSORS", True, "the deploy state dict has no integer tensors")
d = tempfile.mkdtemp()
out = os.path.join(d, "E_prior_s4243_fsoup.pt")
p = subprocess.run([sys.executable, SOUP, A, A, "--out", out], capture_output=True, text=True)
check("S4_CLI_WRITES_GLOB_NAME", p.returncode == 0 and os.path.exists(out) and os.path.exists(out + ".json"), p.stdout.strip()[:160])
p2 = subprocess.run([sys.executable, SOUP, A, A, "--out", os.path.join(d, "soup.pt")], capture_output=True, text=True)
check("S5_CLI_REFUSES_NON_GLOB_NAME", p2.returncode != 0, "")
spec = importlib.util.spec_from_file_location("ct_orig", os.path.join(KIT, "src/compact_train.py"))
CO = importlib.util.module_from_spec(spec); spec.loader.exec_module(CO)
net = CO.Net("convnext_small", None, crop=288, in_chans=3, spatial=True)
try:
    net.load_state_dict(torch.load(out, map_location="cpu", weights_only=True), strict=True); ok = True
except Exception as e:
    ok = False; print(e)
check("S6_STRICT_LOAD_ORIGINAL_NET", ok, "")
print("SOUP_TESTS %s %d/%d" % ("PASS" if all(R) else "FAIL", sum(R), len(R)))
