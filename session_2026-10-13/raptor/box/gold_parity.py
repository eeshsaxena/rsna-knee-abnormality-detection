# CPU parity check (no GPU, RSNA gold58 only): rebuild the d96/384 corpus for the gold58 studies with the VERBATIM Raptor
# reference reader, load nartaa r384 SWA (N), score 24 evenly spaced windows (a) full field 384 (training geometry) and
# (b) centre 320 crop (serving geometry), compare macro AUC with the checkpoint's stored gold_auc .918952.
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd, torch, timm, cv2
from sklearn.metrics import roc_auc_score
cv2.setNumThreads(1); torch.set_num_threads(8)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raptor_ref as R
ROOT = "/home/training/kmimic/kaggle/input/rsna-knee-abnormality-detection"
ser = pd.read_csv(ROOT + "/train_series.csv", dtype=str); ser["Fluid_Sensitive"] = ser["Fluid_Sensitive"].astype(int)
SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
g = pd.read_csv("/home/training/rsna_train/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ck = torch.load("/home/training/kfn70_test/wts/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", map_location="cpu", weights_only=False)


def make(res):
    bb = timm.create_model(ck["arch"], pretrained=False, num_classes=0, in_chans=3, img_size=res, global_pool="avg")
    m = R.RaptorClassifier(bb, F_dim=bb.num_features); m.load_state_dict(ck["model"], strict=True); return m.eval()


M384, M320 = make(384), make(320)


def wins(vol, mask, k, crop):
    D = vol.shape[0]; cs = R._eval_centers(mask, D, k); out = []
    for c in cs:
        c = max(1, min(c, D - 2)); t = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        if crop: t = t[:, 32:352, 32:352]
        out.append(t)
    x = torch.from_numpy(np.ascontiguousarray(np.stack(out))); return (x - R._MEAN) / R._STD


rd = R._make_reader(); P = {"full384_k24": [], "crop320_k24": []}; t0 = time.time()
with torch.no_grad():
    for i, sid in enumerate(g.StudyInstanceUID):
        vol, mask = R.build_study(sid, SER, ROOT + "/train_series", rd)
        P["full384_k24"].append(torch.sigmoid(M384(wins(vol, mask, 24, False)[None]))[0].numpy())
        P["crop320_k24"].append(torch.sigmoid(M320(wins(vol, mask, 24, True)[None]))[0].numpy())
        if i % 10 == 0: print(i, "%.0fs" % (time.time() - t0), flush=True)
Y = g[R.LAB].values.astype(float)
for k, p in P.items():
    p = np.array(p); a = [roc_auc_score(Y[:, j], p[:, j]) for j in range(12)]
    print(f"{k}: macro {np.mean(a):.6f} | stored gold_auc {ck['gold_auc']:.6f} | per-label " + " ".join(f"{x:.3f}" for x in a), flush=True)
    np.save(f"gold58_N_{k}.npy", p)
print("stored per-label", " ".join(f"{ck['aucs'][l]:.3f}" for l in R.LAB))
print("DONE %.0fs" % (time.time() - t0))
