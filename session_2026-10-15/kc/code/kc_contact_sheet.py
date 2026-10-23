# kc_contact_sheet.py -- BOX_PLAN_CRITIQUE_1015 C2b: visual canon check of the KC store BEFORE any GPU time (RSNA only).
# 24 test458 studies (12 R, 12 L, round-robin over manufacturers, >= 3 from the weak side zone), 2 PNG sheets of 12 rows.
# Each row (canonical LEFT-knee frame, i.e. what the model sees):
#   [0] / [1] full CorFS mid (slot 9) / posterior (slot 13) slice of the canonical d96 vol, the cor_med (cyan, image-left)
#       and cor_lat (yellow, image-right) 90 mm boxes drawn, 'M' / 'L' marks (fibular head = image-right, posterior)
#   [2] cor_med s6   [3] cor_lat s6   [4] sag_lat s3   [5] sag_lat s6   [6] sag_med s6   [7] sag_med s9   [8] sag_pd s6
#   [9] ax s6. (sag_lat s0-2 / sag_med s9-11 often lie outside the bone: the d96 sagittal span is 2-98 % of the series.)
# Tiles [1]-[8] are decoded from the store npz (the exact model input bytes); tile [0] re-cuts from the d96 npy with the
# same kc_common.canon + flips. Header per row: short sid | side / rule | |mean_x| | manufacturer | flips.
import os, sys, json, argparse
import numpy as np, pandas as pd, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kc_common as kc

ap = argparse.ArgumentParser()
ap.add_argument('--store', required=True); ap.add_argument('--src', required=True)
ap.add_argument('--side', required=True); ap.add_argument('--lat', required=True)   # lat458.csv (manufacturer)
ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=24); ap.add_argument('--tile', type=int, default=200)
A = ap.parse_args()
T = A.tile
side = pd.read_csv(A.side, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
lat = pd.read_csv(A.lat, dtype={'sid': str}).set_index('sid')
have = sorted(f[:-4] for f in os.listdir(A.store) if f.endswith('.npz'))
df = side.loc[[s for s in have if s in side.index]].copy()
df['man'] = [str(lat.man.get(s, '?')).split()[0] for s in df.index]
df['weak'] = (df.rule != 'mean_x').astype(int)
rng = np.random.default_rng(1015)
pick = []
weak = df[(df.weak == 1) & df.side.isin(['R', 'L'])]
for s in rng.permutation(weak.index)[:3]:
    pick.append(s)
need = {'R': 12 - sum(df.side[p] == 'R' for p in pick), 'L': 12 - sum(df.side[p] == 'L' for p in pick)}
for sd in ('R', 'L'):
    pool = df[(df.side == sd) & (df.weak == 0) & ~df.index.isin(pick)]
    mans = sorted(pool.man.unique(), key=lambda m: -len(pool[pool.man == m]))
    bym = {m: list(rng.permutation(pool[pool.man == m].index)) for m in mans}
    k = 0
    while need[sd] > 0 and any(bym.values()):
        m = mans[k % len(mans)]; k += 1
        if bym[m]:
            pick.append(bym[m].pop()); need[sd] -= 1
pick = sorted(pick, key=lambda s: (df.side[s], df.man[s]))


def tile(img, label):
    t = cv2.resize(img, (T, T), interpolation=cv2.INTER_AREA)
    t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
    cv2.putText(t, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)
    return t


def row(sid):
    r = side.loc[sid]
    with np.load(f'{A.store}/{sid}.npz') as z:
        flips = json.loads(str(z['flips']))
        V = {n: (kc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs']) if f'v{j}_blob' in z.files else None)
             for j, n in enumerate(kc.VIEW_NAMES)}
    vol = np.load(f'{A.src}/{sid}.npy')
    cv = kc.canon(vol, {b: bool(flips.get(b, False)) for b in kc.D96_B})
    a, e = kc.D96_B['corfs']
    fulls = []
    for k, nm in ((9, 'canon CorFS mid'), (13, 'canon CorFS post')):
        full = cv2.cvtColor(cv2.resize(cv[a + k], (T, T), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        sc = T / kc.D96_IMG
        for cx, col in ((-22, (255, 255, 0)), (22, (0, 255, 255))):
            h = 90 * kc.PX_MM; x0 = (kc.D96_IMG / 2 + cx * kc.PX_MM - h / 2) * sc; y0 = (kc.D96_IMG / 2 - h / 2) * sc
            cv2.rectangle(full, (int(x0), int(y0)), (int(x0 + h * sc), int(y0 + h * sc)), col, 1)
        cv2.putText(full, 'M', (3, T - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(full, 'L', (T - 14, T - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(full, nm, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)
        fulls.append(full)
    blank = np.zeros((kc.STORE, kc.STORE), np.uint8)
    g = lambda n, i: V[n][i] if V[n] is not None else blank
    tiles = fulls + [tile(g('cor_med', 6), 'cor_med (img-L=med)'), tile(g('cor_lat', 6), 'cor_lat'),
             tile(g('sag_lat', 3), 'sag_lat s3'), tile(g('sag_lat', 6), 'sag_lat s6'),
             tile(g('sag_med', 6), 'sag_med s6'), tile(g('sag_med', 9), 'sag_med s9'),
             tile(g('sag_pd', 6), 'sag_pd s6' + ('' if V['sag_pd'] is not None else ' ABSENT')), tile(g('ax', 6), 'ax s6')]
    body = np.concatenate(tiles, 1)
    head = np.zeros((18, body.shape[1], 3), np.uint8)
    fl = ''.join(k[:3] + ('+' if v else '-') + ' ' for k, v in flips.items() if k in ('sagfs', 'corfs', 'ax'))
    txt = (f'{sid[-12:]} | side {r.side} rule {r.rule} | |mean_x| {abs(r.mean_x):.1f} mm | {df.man[sid]} | flips {fl}'
           + (' | WEAK ZONE' if r.rule != 'mean_x' else ''))
    cv2.putText(head, txt, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([head, body], 0)


os.makedirs(A.out, exist_ok=True)
man = []
for k in range(0, len(pick), 12):
    rows = [row(s) for s in pick[k:k + 12]]
    sheet = np.concatenate(rows, 0)
    fn = f'{A.out}/kc_contact_{k // 12 + 1}.png'
    cv2.imwrite(fn, sheet)
    man.append(fn)
pd.DataFrame([dict(StudyInstanceUID=s, side=df.side[s], rule=df.rule[s], mean_x=df.mean_x[s], man=df.man[s],
                   sheet=1 + i // 12, row=1 + i % 12) for i, s in enumerate(pick)]).to_csv(f'{A.out}/kc_contact_index.csv', index=False)
print('wrote', man, '| studies', len(pick), '| R', int((df.side[pick] == 'R').sum()), 'L', int((df.side[pick] == 'L').sum()),
      '| weak', int(df.weak[pick].sum()), '| manufacturers', df.man[pick].value_counts().to_dict())
