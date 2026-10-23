# Fine Spacing pack builder (directive 25 item 5, ordered by directive 26 order 3). CPU only.
# One npz per study, the same layout as pack5: pack (80, 336, 336) uint8 and mask (80,) uint8.
# Geometry (dreaddevelopment, discussion 737696 and the dataset card of raptor-knee-finespacing):
#   five fixed slots in this order, 22 sagittal fluid, 18 sagittal, 15 coronal fluid, 10 coronal, 15 axial,
#   slices picked by np.linspace over int(0.02 n) .. int(0.98 n) - 1, percentile 2/98 per slot,
#   140 mm centre crop from PixelSpacing, 336 px (INTER_AREA). A slot with no series stays zeros.
# The arithmetic follows the author's reference inference line for line, so the uint8 volume is the same.
import glob
import io
import os
import time
import zipfile

import numpy as np
import pydicom
import cv2

try:
    from pydicom.pixels import apply_modality_lut
except ImportError:  # pydicom < 3
    from pydicom.pixel_data_handlers.util import apply_modality_lut

SLOTS = (("Sagittal", 1, 22), ("Sagittal", 0, 18), ("Coronal", 1, 15), ("Coronal", 0, 10), ("Axial", -1, 15))
DEPTH = sum(k for _, _, k in SLOTS)          # 80
SPAN = (0.02, 0.98)
CROP_MM = 140.0
SIZE = 336


def flag(value):
    """Fluid_Sensitive as 0 or 1, None when unknown."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value).strip().lower()
        return 1 if text in ('true', 'yes', 'y', 't') else 0 if text in ('false', 'no', 'n', 'f') else None
    if not np.isfinite(number) or number not in (0.0, 1.0):
        return None
    return int(number)


def series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if not cands:
        return None
    if fluid in (0, 1):
        preferred = [r for r in cands if flag(r.get('Fluid_Sensitive')) == fluid]
        if preferred:
            return preferred[0]
    return cands[0]


def order_and_meta(series_dir):
    """Files sorted along the slice normal. Returns ([(path, pixel spacing)], median pixel spacing)."""
    records, spacings = [], []
    for path in glob.glob(series_dir + '/*.dcm'):
        try:
            h = pydicom.dcmread(path, stop_before_pixels=True)
            iop = getattr(h, 'ImageOrientationPatient', None)
            ipp = getattr(h, 'ImagePositionPatient', None)
            if iop is not None and ipp is not None and len(iop) == 6:
                r = np.array(iop[:3], float)
                c = np.array(iop[3:], float)
                pos = float(np.dot(np.array(ipp, float), np.cross(r, c)))
            else:
                pos = float(getattr(h, 'InstanceNumber', 0) or 0)
            ps = getattr(h, 'PixelSpacing', None)
            ps = float(ps[0]) if ps is not None else 0.5
            spacings.append(ps)
            records.append((pos, path, ps))
        except Exception:
            records.append((0.0, path, 0.5))
    records.sort(key=lambda t: (t[0], t[1]))
    median = float(np.median(spacings)) if spacings else 0.5
    return [(path, ps) for _, path, ps in records], median


def read_pixels(path):
    d = pydicom.dcmread(path)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
        a = a.max() - a
    return a


def crop_resize(a, ps):
    h, w = a.shape
    cpx = min(int(round(CROP_MM / max(ps, 1e-3))), min(h, w))
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    return cv2.resize(a[y0:y0 + cpx, x0:x0 + cpx], (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def picks_for(n, count):
    lo = int(n * SPAN[0])
    hi = max(int(n * SPAN[1]) - 1, lo)
    return np.linspace(lo, hi, count).round().astype(int) if n > 1 else np.zeros(count, dtype=int)


def build_pack(uid, rows, img_root):
    """-> (pack uint8 (80,336,336), mask uint8 (80,), sidecar dict). Slots that cannot be filled stay zeros."""
    pack = np.zeros((DEPTH, SIZE, SIZE), np.uint8)
    info = {'StudyInstanceUID': uid}
    used, position = set(), 0
    for si, (plane, fluid, count) in enumerate(SLOTS):
        tag = f"slot{si}_{plane}_{'fs' if fluid == 1 else 'nfs' if fluid == 0 else 'any'}"
        record = series_for_slot(rows, plane, fluid, used)
        if record is None:
            info[tag] = 'ABSENT'
            position += count
            continue
        used.add(record['SeriesInstanceUID'])
        files, median_ps = order_and_meta(f"{img_root}/{uid}/{record['SeriesInstanceUID']}")
        if not files:
            info[tag] = 'EMPTY'
            position += count
            continue
        n = len(files)
        picks = np.minimum(picks_for(n, count), n - 1)
        arrays, spacing = [], []
        for p in picks:
            path, ps = files[int(p)]
            try:
                arrays.append(read_pixels(path))
                spacing.append(ps)
            except Exception:
                arrays.append(None)
                spacing.append(median_ps)
        valid = [a for a in arrays if a is not None]
        if valid:
            low, high = np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
        else:
            low, high = 0.0, 1.0
        filled = 0
        for a, ps in zip(arrays, spacing):
            if a is not None:
                scaled = np.clip((a - low) / (high - low + 1e-6), 0, 1)
                pack[position] = (crop_resize(scaled, ps if ps > 0 else median_ps) * 255).astype(np.uint8)
                filled += 1
            position += 1
        info[tag] = str(record['SeriesInstanceUID'])
        info[tag + '_files'] = n
        info[tag + '_filled'] = filled
        info[tag + '_px_mm'] = float(spacing[0]) if spacing else median_ps
        if valid:
            info[tag + '_rows_cols'] = f'{valid[0].shape[0]}x{valid[0].shape[1]}'
    mask = (pack.reshape(DEPTH, -1).sum(1) > 0).astype(np.uint8)
    info['filled_slices'] = int(mask.sum())
    return pack, mask, info


def npz_bytes(pack, mask):
    buf = io.BytesIO()
    np.savez(buf, pack=pack, mask=mask)       # stored, not deflated: the trainer maps pack.npy by offset
    return buf.getvalue()
