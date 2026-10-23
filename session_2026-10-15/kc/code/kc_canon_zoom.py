# kc_canon_zoom.py -- C2b helper (RSNA only): native-resolution strip of canonical CorFS slots 8-17 for given studies,
# to check that the fibular head / LCL sit image-RIGHT (lateral) and the MCL image-LEFT (medial) after kc_common.canon.
import os, sys, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kc_common as kc
store, src, out = sys.argv[1:4]; sids = sys.argv[4:]
rows = []
for sid in sids:
    with np.load(f'{store}/{sid}.npz') as z:
        flips = json.loads(str(z['flips'])); side = str(z['side'])
    cv = kc.canon(np.load(f'{src}/{sid}.npy'), {b: bool(flips.get(b, False)) for b in kc.D96_B})
    a, _ = kc.D96_B['corfs']
    t = [cv2.cvtColor(cv[a + k][64:384, 32:352], cv2.COLOR_GRAY2BGR) for k in (9, 11, 13, 15)]
    for k, im in zip((9, 11, 13, 15), t):
        cv2.putText(im, f'{sid[-6:]} {side} slot{k}  M<- ->L', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    rows.append(np.concatenate(t, 1))
cv2.imwrite(out, np.concatenate(rows, 0)); print('wrote', out)
