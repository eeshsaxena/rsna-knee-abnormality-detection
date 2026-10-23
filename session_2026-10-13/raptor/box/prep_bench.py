# CPU bench: reference Raptor d96 build_study on a few RSNA studies of the box mirror; timing + storage size options.
import os, sys, time, io, zlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd, cv2
cv2.setNumThreads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raptor_ref as R
ROOT = "/home/training/kmimic/kaggle/input/rsna-knee-abnormality-detection"
ser = pd.read_csv(ROOT + "/train_series.csv", dtype=str)
ser["Fluid_Sensitive"] = ser["Fluid_Sensitive"].astype(int)
SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
have = sorted(os.listdir(ROOT + "/train_series"))[:8]
rd = R._make_reader()
tt, raw, zc, png = [], 0, 0, 0
for sid in have:
    t = time.time(); vol, mask = R.build_study(sid, SER, ROOT + "/train_series", rd); tt.append(time.time() - t)
    raw += vol.nbytes
    b = io.BytesIO(); np.savez_compressed(b, vol=vol, mask=mask); zc += len(b.getvalue())
    png += sum(len(cv2.imencode(".png", s, [cv2.IMWRITE_PNG_COMPRESSION, 3])[1]) for s in vol)
    print(sid[-12:], vol.shape, int(mask.sum()), "%.1fs" % tt[-1], flush=True)
n = len(have)
print("per study: build %.1fs (median %.1f) | raw %.1f MB | savez_compressed %.1f MB | png3 %.1f MB" % (np.mean(tt), np.median(tt), raw/n/1e6, zc/n/1e6, png/n/1e6))
