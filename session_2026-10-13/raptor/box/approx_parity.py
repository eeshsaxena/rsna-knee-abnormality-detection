# CPU test (no GPU, RSNA gold58 only): how much does N lose when its d96/384 input is APPROXIMATED from our 320 px JPEG
# fleet_store (the only form in which OAI images still exist on the box), compared with the exact DICOM build?
#   exactB  : exact Raptor build, slot 1 (Sagittal non-FS) zeroed            (= what an exact OAI re-download gives)
#   approxB : store-derived approximation, slots 1 and 3 (Coronal non-FS) zero (= what OAI from ~/oai_store gives)
# Both scored with 24 evenly spaced windows at full field 384 (the training-time gold geometry).
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd, torch, timm, cv2
from sklearn.metrics import roc_auc_score
cv2.setNumThreads(1); torch.set_num_threads(8)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raptor_ref as R
ROOT = "/home/training/kmimic/kaggle/input/rsna-knee-abnormality-detection"
STORE = "/home/training/fleet_store"
ser = pd.read_csv(ROOT + "/train_series.csv", dtype=str); ser["Fluid_Sensitive"] = ser["Fluid_Sensitive"].astype(int)
SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
g = pd.read_csv("/home/training/rsna_train/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ck = torch.load("/home/training/kfn70_test/wts/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", map_location="cpu", weights_only=False)
bb = timm.create_model(ck["arch"], pretrained=False, num_classes=0, in_chans=3, img_size=384, global_pool="avg")
M = R.RaptorClassifier(bb, F_dim=bb.num_features); M.load_state_dict(ck["model"], strict=True); M.eval()
SIZES = [26, 22, 18, 12, 18]; B = np.cumsum([0] + SIZES)
# Raptor slot -> fleet_store view index (VIEWS = Axial FS, Sagittal nonFS, Coronal FS, Sagittal FS); None = no stored view
SLOT2VIEW = {0: 3, 1: 1, 2: 2, 3: None, 4: 0}


def decode(z, j):
    if f"v{j}_blob" not in z.files: return None
    blob, offs = z[f"v{j}_blob"], z[f"v{j}_offs"]
    return np.stack([cv2.imdecode(blob[offs[i]:offs[i + 1]], cv2.IMREAD_GRAYSCALE) for i in range(len(offs) - 1)])


def approx_vol(sid, zero_slots):
    z = np.load(f"{STORE}/{sid}.npz")
    vol = np.zeros((96, 384, 384), np.uint8)
    for s, k in enumerate(SIZES):
        j = SLOT2VIEW[s]
        if s in zero_slots or j is None: continue
        st = decode(z, j)
        if st is None: continue
        n = len(st); lo, hi = int(n * 0.02), int(n * 0.98) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        sel = st[np.minimum(picks, n - 1)].astype(np.float32)
        q2, q98 = np.percentile(sel, [2.0, 98.0])
        for t, a in enumerate(sel):
            aw = np.clip((a - q2) / (q98 - q2 + 1e-6), 0, 1)
            aw = cv2.resize(aw, (384, 384), interpolation=cv2.INTER_LINEAR)
            vol[B[s] + t] = (aw * 255).astype(np.uint8)
    mask = (vol.reshape(96, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


def wins(vol, mask, k=24):
    D = vol.shape[0]; cs = R._eval_centers(mask, D, k); out = []
    for c in cs:
        c = max(1, min(c, D - 2)); out.append(np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0)
    x = torch.from_numpy(np.ascontiguousarray(np.stack(out))); return (x - R._MEAN) / R._STD


rd = R._make_reader(); P = {"exactB": [], "approxB": []}; t0 = time.time(); diffs = []
with torch.no_grad():
    for i, sid in enumerate(g.StudyInstanceUID):
        vol, mask = R.build_study(sid, SER, ROOT + "/train_series", rd)
        vb = vol.copy(); vb[B[1]:B[2]] = 0; mb = (vb.reshape(96, -1).sum(1) > 0).astype(np.uint8)
        va, ma = approx_vol(sid, zero_slots={1, 3})
        both = (mb > 0) & (ma > 0)
        if both.any(): diffs.append(np.abs(vb[both].astype(np.int16) - va[both].astype(np.int16)).mean())
        P["exactB"].append(torch.sigmoid(M(wins(vb, mb)[None]))[0].numpy())
        P["approxB"].append(torch.sigmoid(M(wins(va, ma)[None]))[0].numpy())
        if i % 10 == 0: print(i, "%.0fs" % (time.time() - t0), flush=True)
Y = g[R.LAB].values.astype(float)
for k, p in P.items():
    p = np.array(p); a = [roc_auc_score(Y[:, j], p[:, j]) for j in range(12)]
    print(f"{k}: macro {np.mean(a):.6f} | per-label " + " ".join(f"{x:.3f}" for x in a), flush=True)
    np.save(f"gold58_N_{k}.npy", p)
print("mean |exact - approx| on shared slices (uint8): %.2f (study median %.2f)" % (np.mean(diffs), np.median(diffs)))
print("DONE %.0fs" % (time.time() - t0))
