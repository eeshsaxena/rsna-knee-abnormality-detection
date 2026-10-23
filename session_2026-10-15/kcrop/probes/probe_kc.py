# KC probe C+D (box CPU only): (C) cost of cutting knee-centred compartment views from the d96 corpus npy,
# (D) cost of the verbatim Raptor d96 reader on raw DICOM + byte parity of its vol with the corpus npy.
import os, re, sys, time, json, hashlib, glob
import numpy as np, pandas as pd, cv2

cv2.setNumThreads(1)
HOME = os.path.expanduser('~')
CORPUS = f'{HOME}/raptor_store/rsna'
ROOT = f'{HOME}/eda_data/rsna-knee-abnormality-detection'
PX_MM = 384 / 140.0
# d96 slot bounds: SagFS 26, SagPD 22, CorFS 18, CorPD 12, Ax 18
B = dict(sagfs=(0, 26), sagpd=(26, 48), corfs=(48, 66), corpd=(66, 78), ax=(78, 96))
STORE = 320


def canon(vol, side):
    """Right knee onto the left-knee frame (bliverrigpdoge canon_right on the d96 bounds): sagittal slot order
    reversed, coronal / axial flipped left-right. MEASURED on 458 box studies: sagittal normal x < 0 and
    coronal / axial row x > 0 in every series, so one rule fits all vendors."""
    if side != 'R':
        return vol
    v = vol.copy()
    for k in ('sagfs', 'sagpd'):
        a, b = B[k]
        v[a:b] = v[a:b][::-1]
    for k in ('corfs', 'corpd', 'ax'):
        a, b = B[k]
        v[a:b] = v[a:b][:, :, ::-1]
    return v


def box(img, cy_mm, cx_mm, h_mm, w_mm):
    H, W = img.shape
    cy, cx = H / 2 + cy_mm * PX_MM, W / 2 + cx_mm * PX_MM
    h, w = int(round(h_mm * PX_MM)), int(round(w_mm * PX_MM))
    y0, x0 = int(round(cy - h / 2)), int(round(cx - w / 2))
    out = np.zeros((h, w), np.uint8)
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(H, y0 + h), min(W, x0 + w)
    out[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
    return cv2.resize(out, (STORE, STORE), interpolation=cv2.INTER_AREA if h > STORE else cv2.INTER_LINEAR)


def sl(a, b, k):
    return np.linspace(a, b - 1, k).round().astype(int)


def kc_views(vol, side):
    """6 views x 12 slices (stored as single slices; triplets made at load) at 320 px.
    Canonical left knee: sagittal slot index 0 = lateral end, coronal / axial image-left = medial."""
    v = canon(vol, side)
    a, b = B['sagfs']; n = b - a
    out = {}
    out['sag_lat'] = np.stack([box(v[a + i], 0, 0, 110, 110) for i in sl(0, int(n * .55), 12)])
    out['sag_med'] = np.stack([box(v[a + i], 0, 0, 110, 110) for i in sl(int(n * .45), n, 12)])
    a, b = B['sagpd']; n = b - a
    out['sag_pd'] = np.stack([box(v[a + i], 0, 0, 120, 120) for i in sl(int(n * .15), int(n * .85), 12)])
    a, b = B['corfs']; n = b - a
    cs = sl(int(n * .1), int(n * .9), 12)
    out['cor_med'] = np.stack([box(v[a + i], 0, -22, 90, 90) for i in cs])
    out['cor_lat'] = np.stack([box(v[a + i], 0, 22, 90, 90) for i in cs])
    a, b = B['ax']; n = b - a
    out['ax'] = np.stack([box(v[a + i], 0, 0, 140, 140) for i in sl(int(n * .05), int(n * .95), 12)])
    return out


def main():
    idx = pd.read_csv(f'{HOME}/raptor_store/rsna_index.csv')
    rng = np.random.default_rng(0)
    pick = idx.sample(40, random_state=1)
    t_load, t_kc, nbytes = [], [], []
    for p in pick.path:
        t0 = time.perf_counter(); vol = np.load(p); t1 = time.perf_counter()
        assert vol.shape == (96, 384, 384) and vol.dtype == np.uint8, vol.shape
        views = kc_views(vol, 'R' if rng.random() < .5 else 'L'); t2 = time.perf_counter()
        t_load.append(t1 - t0); t_kc.append(t2 - t1); nbytes.append(sum(x.nbytes for x in views.values()))
    print('C load_npy s/study median %.3f p90 %.3f (cold-ish, ionice idle) | kc_cut s/study median %.4f p90 %.4f | kc raw bytes/study %.1f MB'
          % (np.median(t_load), np.percentile(t_load, 90), np.median(t_kc), np.percentile(t_kc, 90), np.mean(nbytes) / 1e6), flush=True)
    # warm repeat on the same 40
    t_w = []
    for p in pick.path:
        t0 = time.perf_counter(); vol = np.load(p); kc_views(vol, 'L'); t_w.append(time.perf_counter() - t0)
    print('C warm load+cut s/study median %.3f' % np.median(t_w), flush=True)
    # png-compressed size of the kc views (cache option)
    sz = []
    for p in pick.path[:10]:
        views = kc_views(np.load(p), 'L')
        sz.append(sum(len(cv2.imencode('.png', x.reshape(-1, STORE), [cv2.IMWRITE_PNG_COMPRESSION, 1])[1]) for x in views.values()))
    print('C kc png bytes/study %.2f MB (x4407 = %.1f GB)' % (np.mean(sz) / 1e6, np.mean(sz) * 4407 / 1e9), flush=True)

    # D: verbatim reader on raw DICOM (box copy of the scored Raptor r384 reader) + parity vs corpus
    src = open(f'{HOME}/raptor2/prep_kernel_s0.py').read()
    m = re.search(r"_READER_SRC = ('.*?')\n", src, re.S)
    reader_src = eval(m.group(1))
    assert hashlib.sha256(reader_src.encode()).hexdigest().startswith('30aa72fd'), 'reader sha'
    g = {'np': np, 'glob': glob, 'ThreadPoolExecutor': __import__('concurrent.futures', fromlist=['ThreadPoolExecutor']).ThreadPoolExecutor}
    exec(compile(reader_src, 'raptor_reader', 'exec'), g)
    reader = g['_make_reader']()
    ser = pd.read_csv(f'{ROOT}/train_series.csv')
    ser['StudyInstanceUID'] = ser['StudyInstanceUID'].astype(str); ser['SeriesInstanceUID'] = ser['SeriesInstanceUID'].astype(str)
    SER = {k: v.to_dict('records') for k, v in ser.groupby('StudyInstanceUID')}
    have = sorted(os.listdir(f'{ROOT}/train_series'))[:10]
    td, same = [], 0
    for sid in have:
        t0 = time.perf_counter()
        vol, mask = g['build_study'](sid, SER, f'{ROOT}/train_series', reader)
        td.append(time.perf_counter() - t0)
        ref = np.load(f'{CORPUS}/{sid}.npy')
        same += int(np.array_equal(ref, vol))
    print('D d96 reader s/study median %.2f max %.2f (box, 12 cores, its own thread pools) | vol == corpus npy %d/%d'
          % (np.median(td), np.max(td), same, len(have)), flush=True)


if __name__ == '__main__':
    main()
