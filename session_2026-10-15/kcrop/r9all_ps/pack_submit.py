"""DICOM headers to a pack5 volume to the L2 model to submission.csv.

Series choice and slice picks match wp3/export_pack5.ipynb: CSV order,
Fluid_Sensitive, used-set, span 0.06 to 0.94, 140 mm crop, cv2 INTER_AREA
to 336. Eval windows match train_pack PackWindows with K_EVAL 24.
L2 recorded pack288 false, so the 336 windows go through gpu_prep, which
resizes to 288 in float. The checkpoint holds the encoder, so the
backbone is built with pretrained False. Gold directories are refused.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

FINDINGS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
SLOTS = (
    ("Sagittal", 1, 18),
    ("Sagittal", 0, 14),
    ("Coronal", 1, 12),
    ("Coronal", 0, 8),
    ("Axial", -1, 12),
)
CROP_MM = 140.0
SIZE = 336
IMG = 288
K_EVAL = 24
IM_MEAN = (0.485, 0.456, 0.406)
IM_STD = (0.229, 0.224, 0.225)
# Hard prevalence on fused_silver, parse_ok == 1. Not 0.5.
FALLBACK = (
    0.169234, 0.116808, 0.477581, 0.153828, 0.372959, 0.229018,
    0.458956, 0.553461, 0.138883, 0.246723, 0.275926, 0.058864,
)
SLOT_BOUNDS = [(0, 18), (18, 32), (32, 44), (44, 52), (52, 64)]
# Fine Spacing (directive 25 item 5): 80 slices, 22/18/15/10/15 per slot, every window scored (70 per complete study).
FS_SLOT_BOUNDS = [(0, 22), (22, 40), (40, 55), (55, 65), (65, 80)]
GEOMETRIES = {
    "pack5": {"depth": 64, "bounds": SLOT_BOUNDS, "k_eval": K_EVAL},
    "fs": {"depth": 80, "bounds": FS_SLOT_BOUNDS, "k_eval": 70},
}
# Item 7: per-finding pooling over slice-offset passes. rsna/eval_row.py is the tested copy of this rule.
TTA_POOL = {"Lateral Meniscus": "max", "Fracture": "max", "ACL": "top2", "MCL": "top2"}


def pool_passes(passes) -> np.ndarray:
    """[passes, studies, 12] probabilities to [studies, 12]: max for Lateral Meniscus and Fracture, the two highest averaged for ACL
    and MCL, the mean for the rest."""
    a = np.asarray(passes, dtype=np.float64)
    out = np.empty(a.shape[1:], dtype=np.float64)
    for j, name in enumerate(FINDINGS):
        col = a[:, :, j]
        rule = TTA_POOL.get(name, "mean")
        if rule == "max":
            out[:, j] = col.max(axis=0)
        elif rule == "top2":
            out[:, j] = np.sort(col, axis=0)[-min(2, col.shape[0]):].mean(axis=0)
        else:
            out[:, j] = col.mean(axis=0)
    return out


def offset_picks(n_centers: int, k: int, offset: float) -> np.ndarray:
    """The picks of one pass: linspace(0, n-1, k) shifted by `offset` spacings between neighbouring picks, rounded, clipped."""
    base = np.linspace(0, n_centers - 1, k)
    if not offset:
        return base.round().astype(np.int64)
    step = (n_centers - 1) / max(k - 1, 1)
    return np.clip(np.round(base + offset * step), 0, n_centers - 1).astype(np.int64)
BANNED = {"gold58", "gold_npy", "gold_npy_v2"}
SKIP_WALK = ("competition", "competitions", "rsna-knee-abnormality-detection")
SYNTAX: list[str] = []


def refuse(path: str) -> None:
    parts = str(path).replace("\\", "/").lower().split("/")
    if any(p in BANNED or "gold74" in p for p in parts):
        raise SystemExit(f"pack_submit refuses {path}")


def header_of(sample_csv: str | None) -> list[str]:
    header = ["StudyInstanceUID"] + FINDINGS
    if sample_csv and os.path.isfile(sample_csv):
        header = list(pd.read_csv(sample_csv, nrows=0).columns)
    return header


def _fluid(row) -> int:
    if "Fluid_Sensitive" in getattr(row, "index", ()):
        v = row["Fluid_Sensitive"]
    else:
        v = row["Fat_Suppression"] if "Fat_Suppression" in getattr(row, "index", ()) else 0
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 1 if str(v) in ("1", "1.0", "True") else 0


def series_for_slot(group, plane: str, fs: int, used: set):
    """First unused CSV row of this plane. Prefer Fluid_Sensitive when asked."""
    cands = group[(group["Anatomical_Plane"].astype(str) == plane)
                  & (~group["SeriesInstanceUID"].astype(str).isin(used))]
    if not len(cands):
        return None
    if fs in (0, 1):
        pref = [row for _, row in cands.iterrows() if _fluid(row) == fs]
        if pref:
            return pref[0]
    return cands.iloc[0]


def _picks(n: int, k: int) -> np.ndarray:
    lo, hi = int(n * 0.06), int(n * 0.94) - 1
    hi = max(hi, lo)
    if n <= 1:
        return np.zeros(k, dtype=int)
    return np.linspace(lo, hi, k).round().astype(int)


def ordered_files(directory: str):
    import pydicom
    t0 = time.perf_counter()
    recs = []
    if not os.path.isdir(directory):
        return [], 0.5, time.perf_counter() - t0
    refuse(directory)
    paths = [os.path.join(directory, fn) for fn in os.listdir(directory)
             if fn.lower().endswith(".dcm")]

    def _one(path):
        try:
            header = pydicom.dcmread(
                path, stop_before_pixels=True,
                specific_tags=["ImageOrientationPatient", "ImagePositionPatient",
                               "InstanceNumber", "PixelSpacing"])
            iop = getattr(header, "ImageOrientationPatient", None)
            ipp = getattr(header, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                row = np.array(iop[:3], float)
                col = np.array(iop[3:], float)
                normal = np.cross(row, col)
                pos = float(np.dot(np.array(ipp, float), normal))
            else:
                pos = float(getattr(header, "InstanceNumber", 0) or 0)
            spacing = getattr(header, "PixelSpacing", None)
            spacing = float(spacing[0]) if spacing is not None else 0.5
            meta = getattr(header, "file_meta", None)
            syntax = getattr(meta, "TransferSyntaxUID", None) if meta is not None else None
            return pos, path, spacing, str(syntax) if syntax is not None else "missing"
        except Exception:
            return 0.0, path, 0.5, None

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(_one, paths))
    for pos, path, spacing, syntax in got:
        if syntax is not None:
            SYNTAX.append(syntax)
        recs.append((pos, path, spacing))
    recs.sort(key=lambda item: item[0])
    med = float(np.median([spacing for _, _, spacing in recs])) if recs else 0.5
    return [(path, spacing) for _, path, spacing in recs], med, time.perf_counter() - t0


def _pixels(path: str):
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut
    t0 = time.perf_counter()
    ds = pydicom.dcmread(path)
    arr = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    return arr, time.perf_counter() - t0


def _crop(image: np.ndarray, spacing: float):
    import cv2
    t0 = time.perf_counter()
    h, w = image.shape
    side = int(round(CROP_MM / max(spacing, 1e-3)))
    side = min(side, min(h, w))
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = image[y0:y0 + side, x0:x0 + side]
    out = cv2.resize(cropped, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return out.astype(np.float32), time.perf_counter() - t0


def slot_stack(directory: str, k: int):
    files, med, header_s = ordered_files(directory)
    decode_s = header_s
    pack_s = 0.0
    if not files:
        return None, decode_s, pack_s
    n = len(files)
    idx = np.clip(_picks(n, k), 0, n - 1)
    chosen = []
    for i in idx:
        path, spacing = files[int(i)]
        chosen.append((path, spacing if spacing > 0 else med))

    def _read(item):
        path, spacing = item
        try:
            arr, dt = _pixels(path)
            return arr, spacing, dt
        except Exception:
            return None, spacing, 0.0

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(_read, chosen))
    arrs = []
    spacings = []
    for arr, spacing, dt in got:
        decode_s += dt
        arrs.append(arr)
        spacings.append(spacing)
    valid = [a for a in arrs if a is not None]
    if not valid:
        return None, decode_s, pack_s
    t0 = time.perf_counter()
    flat = np.concatenate([a.ravel() for a in valid])
    lo, hi = np.percentile(flat, [2.0, 98.0])
    pack_s += time.perf_counter() - t0
    out = np.zeros((k, SIZE, SIZE), np.float32)
    n_ok = 0
    for j, (image, spacing) in enumerate(zip(arrs, spacings)):
        if image is None:
            continue
        t1 = time.perf_counter()
        scaled = np.clip((image - lo) / (hi - lo + 1e-6), 0, 1)
        pack_s += time.perf_counter() - t1
        cropped, ct = _crop(scaled, spacing)
        pack_s += ct
        out[j] = cropped
        n_ok += 1
    if n_ok == 0:
        return None, decode_s, pack_s
    return (out * 255).astype(np.uint8), decode_s, pack_s


def build_study(uid: str, group, img_root: str):
    """pack uint8 (64, 336, 336), mask (64,), decode seconds, pack seconds."""
    refuse(img_root)
    pack = np.zeros((64, SIZE, SIZE), np.uint8)
    decode_s = 0.0
    pack_s = 0.0
    pos = 0
    used: set[str] = set()
    for plane, fs, k in SLOTS:
        row = series_for_slot(group, plane, fs, used)
        if row is None:
            pos += k
            continue
        series = str(row["SeriesInstanceUID"])
        used.add(series)
        directory = os.path.join(img_root, uid, series)
        try:
            arr, ds, ps = slot_stack(directory, k)
        except Exception:
            arr, ds, ps = None, 0.0, 0.0
        decode_s += ds
        pack_s += ps
        if arr is not None:
            pack[pos:pos + k] = arr
        pos += k
    mask = (pack.reshape(64, -1).sum(1) > 0).astype(np.uint8)
    return pack, mask, decode_s, pack_s


def build_pack_for(geometry: str, uid: str, group, img_root: str):
    """pack, mask, decode seconds, pack seconds for the checkpoint's geometry. fs uses packfs_core, the builder that made pack_fs."""
    if geometry == "pack5":
        return build_study(uid, group, img_root)
    refuse(img_root)
    import packfs_core
    t0 = time.perf_counter()
    pack, mask, _info = packfs_core.build_pack(uid, group.to_dict("records"), img_root)
    return pack, mask, 0.0, time.perf_counter() - t0


META_DEFAULT = {"arch": "convnext_nano", "img": IMG, "laterality": False, "lat_mm": None, "geometry": "pack5", "tta_offsets": None,
                "score_studies": None, "norm_mean": None, "norm_std": None}


def ckpt_meta(ckpt: str) -> dict:
    """What the checkpoint needs at serve time. <ckpt>.meta.json next to the file; none means the original recipe."""
    import json
    meta = dict(META_DEFAULT)
    path = ckpt + ".meta.json"
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            meta.update(json.load(f))
    if meta["laterality"] and meta.get("lat_mm") is None:
        raise ValueError("a laterality checkpoint needs lat_mm in its meta file")
    if meta["geometry"] not in GEOMETRIES:
        raise ValueError(f"unknown geometry {meta['geometry']!r}")
    return meta


def norm_of(meta: dict):
    """(mean, std) the checkpoint was trained with when its meta file carries them (item 10: 0.5/0.5 for tf_efficientnetv2_s), else None for
    ImageNet."""
    if meta.get("norm_mean") and meta.get("norm_std"):
        return tuple(float(x) for x in meta["norm_mean"]), tuple(float(x) for x in meta["norm_std"])
    return None


def _n_dcm(img_root: str, uid: str, series: str) -> int:
    try:
        return sum(fn.lower().endswith(".dcm") for fn in os.listdir(os.path.join(img_root, uid, series)))
    except FileNotFoundError:
        return 0


def side_series(group, img_root: str, uid: str):
    """The series wp1/export_cache took the side from: sagittal fat-suppressed with the most slices, else any sagittal, else any."""
    g = group.copy()
    g["_n"] = [_n_dcm(img_root, uid, str(s)) for s in g["SeriesInstanceUID"]]
    plane = g["Anatomical_Plane"].astype(str).str.lower()
    if "Fat_Suppression" in g.columns:
        fs = g["Fat_Suppression"].astype(str).isin(["1", "1.0", "True"])
    else:
        fs = pd.Series(False, index=g.index)
    sag = g[plane.str.startswith("sag")]
    sag_fs = sag[fs.reindex(sag.index, fill_value=False)]
    for cand in (sag_fs, sag, g):
        if len(cand):
            return cand.sort_values("_n", ascending=False).iloc[0]
    return None


def study_mean_x(uid: str, group, img_root: str):
    """Mean ImagePositionPatient x over that series, as the training sidecar's mean_x. A slice without the tag counts as 0."""
    import pydicom
    row = side_series(group, img_root, uid)
    if row is None:
        return None
    directory = os.path.join(img_root, uid, str(row["SeriesInstanceUID"]))
    refuse(directory)
    xs = []
    for fn in os.listdir(directory):
        if not fn.lower().endswith(".dcm"):
            continue
        header = pydicom.dcmread(os.path.join(directory, fn), stop_before_pixels=True,
                                 specific_tags=["ImagePositionPatient"])
        ipp = getattr(header, "ImagePositionPatient", None)
        xs.append(float(ipp[0]) if ipp is not None and len(ipp) else 0.0)
    return float(np.mean(xs)) if xs else None


def side_of(mean_x, lat_mm: float) -> str:
    """train_pack.laterality_coverage: negative is right, within lat_mm of the midline or missing is unresolved."""
    if mean_x is None or not np.isfinite(mean_x) or abs(float(mean_x)) <= float(lat_mm):
        return "U"
    return "R" if float(mean_x) < 0 else "L"


def canon_right(pack: np.ndarray, bounds=SLOT_BOUNDS) -> np.ndarray:
    """Right knee onto the left-knee frame, as train_pack PackWindows._canon_right."""
    out = np.array(pack, copy=True)
    for lo, hi in bounds[:2]:
        out[lo:hi] = out[lo:hi][::-1]
    for lo, hi in bounds[2:]:
        out[lo:hi] = np.flip(out[lo:hi], axis=-1)
    return out


def apply_laterality(uid: str, group, img_root: str, pack: np.ndarray, mask: np.ndarray, lat_mm: float, bounds=SLOT_BOUNDS):
    """Flip a right knee and rebuild the mask from the flipped pack. Left and unresolved studies pass through."""
    side = side_of(study_mean_x(uid, group, img_root), lat_mm)
    if side == "R":
        pack = canon_right(pack, bounds)
        mask = (pack.reshape(pack.shape[0], -1).sum(1) > 0).astype(np.uint8)
    return pack, mask, side


def valid_centers(mask: np.ndarray, bounds=SLOT_BOUNDS):
    out = []
    for sid, (lo, hi) in enumerate(bounds):
        n = hi - lo
        for i in range(lo + 1, hi - 1):
            if mask[i - 1] and mask[i] and mask[i + 1]:
                rel = (i - lo) / max(n - 1, 1)
                out.append((i, sid, rel))
    return out


def eval_windows(pack: np.ndarray, mask: np.ndarray, k: int = K_EVAL, bounds=SLOT_BOUNDS, offset: float = 0.0):
    centers = valid_centers(mask, bounds)
    n = len(centers)
    if not centers:
        pick = np.full(k, -1, np.int64)
    elif n >= k:
        pick = offset_picks(n, k, offset)
    else:
        pick = np.arange(k)
        pick = np.where(pick < n, pick, -1).astype(np.int64)
    hw = int(pack.shape[-1])
    x = np.zeros((k, 3, hw, hw), np.uint8)
    slot = np.zeros(k, np.int64)
    rel = np.zeros(k, np.float32)
    wmask = np.zeros(k, np.float32)
    for j, ci in enumerate(pick):
        if ci < 0 or ci >= n:
            continue
        sl, sid, r = centers[int(ci)]
        x[j] = np.asarray(pack[sl - 1:sl + 2])
        slot[j] = sid
        rel[j] = r
        wmask[j] = 1.0
    if not wmask.any():
        wmask[0] = 1.0
    return x, slot, rel, wmask


def _create_encoder(timm, arch: str):
    """timm.create_model without weights. An architecture name with a pretrained tag that this timm does not list falls back to the
    bare architecture; the strict state-dict load that follows is the check."""
    try:
        return timm.create_model(arch, pretrained=False, num_classes=0, in_chans=3)
    except Exception:
        base = arch.split(".")[0]
        if base == arch:
            raise
        return timm.create_model(base, pretrained=False, num_classes=0, in_chans=3)


def _load_model(ckpt: str, device: str, arch: str = "convnext_nano"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import timm
    import torch
    import torch.nn as nn

    class PackInfer(nn.Module):
        """train_pack.PackModel, pretrained False. Same parameter names."""

        def __init__(self):
            super().__init__()
            self.enc = _create_encoder(timm, arch)
            d = self.enc.num_features
            self.slot_emb = nn.Embedding(5, d)
            self.pos = nn.Linear(1, d)
            self.norm = nn.LayerNorm(d)
            self.att = nn.Sequential(
                nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.1), nn.Linear(256, 12))
            self.clsW = nn.Parameter(torch.zeros(12, d))
            self.clsb = nn.Parameter(torch.zeros(12))
            nn.init.trunc_normal_(self.clsW, std=0.02)

        def forward(self, x, slot, rel, wmask):
            b, k = x.shape[:2]
            x4 = x.reshape(b * k, *x.shape[2:])
            if x4.device.type == "cuda":
                x4 = x4.contiguous(memory_format=torch.channels_last)
            f = self.enc(x4).reshape(b, k, -1)
            f = f + self.slot_emb(slot) + self.pos(rel.unsqueeze(-1))
            h = self.norm(f)
            a = self.att(h)
            a = a.masked_fill(wmask.unsqueeze(-1) == 0, -1e4)
            a = torch.softmax(a, dim=1)
            pooled = torch.einsum("bkn,bkd->bnd", a, h)
            return (pooled * self.clsW).sum(-1) + self.clsb

    model = PackInfer()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    if device == "cuda":
        model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
    return model


def _score(model, pack, mask, device: str, img: int = IMG, k_eval: int = K_EVAL, bounds=SLOT_BOUNDS, norm=None):
    import torch
    import torch.nn.functional as Fn
    x, slot, rel, wmask = eval_windows(pack, mask, k_eval, bounds)
    xt = torch.from_numpy(np.ascontiguousarray(x))[None]
    st = torch.from_numpy(slot)[None]
    rt = torch.from_numpy(rel)[None]
    wt = torch.from_numpy(wmask)[None]
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        xt = xt.to(device, non_blocking=True)
        b, k, c, h, w = xt.shape
        xt = xt.reshape(b * k, c, h, w).to(dtype=torch.float32).mul_(1.0 / 255.0)
        if h != img or w != img:
            xt = Fn.interpolate(xt, size=(img, img), mode="bilinear",
                                align_corners=False)
        mean = xt.new_tensor(norm[0] if norm else IM_MEAN).view(1, 3, 1, 1)
        std = xt.new_tensor(norm[1] if norm else IM_STD).view(1, 3, 1, 1)
        xt = (xt - mean) / std
        xt = xt.reshape(b, k, c, img, img)
        logits = model(xt, st.to(device), rt.to(device), wt.to(device))
        prob = torch.sigmoid(logits.float())[0].detach().cpu().numpy()
    if device == "cuda":
        torch.cuda.synchronize()
    return prob, time.perf_counter() - t0


def _skip_dir(name: str) -> bool:
    low = name.lower()
    return any(s in low for s in SKIP_WALK)


def _walk_file(name: str):
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return None
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if not _skip_dir(d)]
        if _skip_dir(dirpath):
            continue
        if name in filenames:
            return os.path.join(dirpath, name)
    return None


def find_comp(comp: str | None) -> str | None:
    if comp and os.path.isfile(os.path.join(comp, "test.csv")):
        return comp
    cands = [
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
        "/kaggle/input/rsna-knee-abnormality-detection",
    ]
    for cand in cands:
        if os.path.isfile(os.path.join(cand, "test.csv")):
            return cand
    base = "/kaggle/input"
    if os.path.isdir(base):
        for name in os.listdir(base):
            cand = os.path.join(base, name)
            if os.path.isfile(os.path.join(cand, "test.csv")):
                return cand
    return None


def ckpt_name() -> str:
    """One parameter: a ckpt_name.txt file in the private dataset."""
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return "loop_L2.pt"
    found: list[str] = []
    preferred = None
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if not _skip_dir(d)]
        if _skip_dir(dirpath):
            continue
        if "ckpt_name.txt" in filenames:
            text = open(os.path.join(dirpath, "ckpt_name.txt"), encoding="utf-8").read().strip()
            if text:
                name = text.split()[0]
                folded = dirpath.replace("\\", "/")
                if "rsna-knee-l5mean" in folded:
                    return name
                if preferred is None and "rsna-knee-l10" in folded:
                    preferred = name
                found.append(name)
    if preferred:
        return preferred
    return found[0] if found else "loop_L2.pt"


def find_ckpt(ckpt: str | None) -> str | None:
    if ckpt and os.path.isfile(ckpt):
        return ckpt
    name = ckpt_name()
    owners = ("rsna-knee-l5mean", "rsna-knee-l8", "rsna-knee-l10",
              "rsna-knee-l3", "rsna-knee-v2pin", "rsna-knee-l2")
    direct = []
    for slug in owners:
        direct.append(f"/kaggle/input/{slug}/{name}")
        direct.append(f"/kaggle/input/datasets/bliverrigpdoge/{slug}/{name}")
    for cand in direct:
        if os.path.isfile(cand):
            return cand
    return _walk_file(name)


def find_image_root(comp: str, uids: list[str], img_root: str | None) -> str | None:
    if img_root and os.path.isdir(img_root):
        refuse(img_root)
        return img_root
    names = ("test_series", "test_images", "test", "series", "train_series")
    first = uids[0] if uids else ""
    for name in names:
        cand = os.path.join(comp, name)
        if first and os.path.isdir(os.path.join(cand, first)):
            return cand
        if os.path.isdir(cand):
            return cand
    if first and os.path.isdir(os.path.join(comp, first)):
        return comp
    return None


def _mean(xs: list[float]):
    return round(float(np.mean(xs)), 4) if xs else None


def main(comp: str | None = None, ckpt: str | None = None,
         out_csv: str | None = None, img_root: str | None = None) -> dict:
    t_all = time.perf_counter()
    comp = find_comp(comp)
    ckpt = find_ckpt(ckpt)
    if comp is None:
        return {"ok": False, "error": "test.csv not found"}
    refuse(comp)
    if ckpt is None:
        return {"ok": False, "error": "loop_L2.pt not found"}
    test = pd.read_csv(os.path.join(comp, "test.csv"), dtype={"StudyInstanceUID": str})
    series_path = os.path.join(comp, "test_series.csv")
    if not os.path.isfile(series_path):
        series_path = os.path.join(comp, "train_series.csv")
    if not os.path.isfile(series_path):
        return {"ok": False, "error": "test_series.csv not found", "comp": comp}
    series = pd.read_csv(series_path, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    grouped = {k: v for k, v in series.groupby("StudyInstanceUID", sort=False)}
    uids = test["StudyInstanceUID"].astype(str).tolist()
    img_root = find_image_root(comp, uids, img_root)
    header = header_of(os.path.join(comp, "sample_submission.csv"))
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_load = time.perf_counter()
    model = _load_model(ckpt, device)
    load_s = time.perf_counter() - t_load
    rows = []
    decode_s, pack_s, model_s = [], [], []
    n_model = 0
    errors = []
    for uid in uids:
        group = grouped.get(uid)
        if img_root is None or group is None:
            errors.append("no_series")
            decode_s.append(0.0)
            pack_s.append(0.0)
            model_s.append(0.0)
            rows.append([uid] + [0.5] * len(FINDINGS))
            continue
        try:
            pack, mask, ds, ps = build_study(uid, group, img_root)
        except Exception as exc:
            errors.append(type(exc).__name__)
            decode_s.append(0.0)
            pack_s.append(0.0)
            model_s.append(0.0)
            rows.append([uid] + [0.5] * len(FINDINGS))
            continue
        decode_s.append(ds)
        pack_s.append(ps)
        if int(mask.sum()) == 0:
            errors.append("empty_pack")
            model_s.append(0.0)
            rows.append([uid] + [0.5] * len(FINDINGS))
            continue
        prob, ms = _score(model, pack, mask, device)
        model_s.append(ms)
        n_model += 1
        rows.append([uid] + [float(v) for v in prob])
    sub = pd.DataFrame(rows, columns=["StudyInstanceUID"] + FINDINGS)
    sub = sub.reindex(columns=list(header))
    if out_csv is None:
        out_csv = "/kaggle/working/submission.csv" if os.path.isdir("/kaggle/working") else "submission.csv"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    sub.to_csv(out_csv, index=False)
    runtime_s = round(time.perf_counter() - t_all, 3)
    report = {
        "ok": n_model == len(uids) and list(sub.columns) == list(header),
        "n": len(uids),
        "n_model": n_model,
        "n_columns": int(sub.shape[1]),
        "columns": list(sub.columns),
        "header_match": list(sub.columns) == list(header),
        "sample_header": list(header),
        "submission": out_csv,
        "runtime_s": runtime_s,
        "load_s": round(load_s, 3),
        "decode_s_per_study": _mean(decode_s),
        "pack_s_per_study": _mean(pack_s),
        "model_s_per_study": _mean(model_s),
        "device": device,
        "ckpt": ckpt,
        "comp": comp,
        "img_root": img_root,
        "resize": "cv2.INTER_AREA",
        "k_eval": K_EVAL,
        "img": IMG,
        "errors": errors[:8],
        "gold": "not read",
    }
    report_path = os.path.join(os.path.dirname(out_csv) or ".", "submit_report.json")
    import json
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    report["report"] = report_path
    return report


def _median(xs: list[float]):
    if not xs:
        return None
    s = sorted(float(v) for v in xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return round(s[mid], 4)
    return round(0.5 * (s[mid - 1] + s[mid]), 4)


def find_id_file(path: str | None) -> str | None:
    if path and os.path.isfile(path):
        return path
    direct = [
        "/kaggle/input/rsna-knee-fold0-ids/parity_ids.txt",
        "/kaggle/input/datasets/bliverrigpdoge/rsna-knee-fold0-ids/parity_ids.txt",
    ]
    for cand in direct:
        if os.path.isfile(cand):
            return cand
    return _walk_file("parity_ids.txt")


def _read_ids(path: str) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            uid = line.strip()
            if not uid or uid == "StudyInstanceUID":
                continue
            out.append(uid)
    return out


def find_train_root(comp: str, uids: list[str], series) -> str | None:
    """Directory that contains uid/series for the training images."""
    grouped = {k: v for k, v in series.groupby("StudyInstanceUID", sort=False)}
    probe = None
    series_uid = None
    for uid in uids:
        group = grouped.get(uid)
        if group is not None and len(group):
            probe = uid
            series_uid = str(group.iloc[0]["SeriesInstanceUID"])
            break
    if probe is None:
        return None
    names = ("train_series", "train_images", "train")
    for name in names:
        if os.path.isdir(os.path.join(comp, name, probe, series_uid)):
            return os.path.join(comp, name)
    if os.path.isdir(os.path.join(comp, probe, series_uid)):
        return comp
    return None


def pack_study_job(payload: tuple):
    """One study: headers, then only the chosen slices. Used by spawn workers."""
    from collections import Counter
    uid, records, img_root = payload[:3]
    lat_mm = payload[3] if len(payload) > 3 else None
    geometry = payload[4] if len(payload) > 4 else "pack5"
    SYNTAX.clear()
    group = pd.DataFrame(records)
    side = None
    try:
        pack, mask, ds, ps = build_pack_for(geometry, uid, group, img_root)
        err = None if int(mask.sum()) else "empty_pack"
        if lat_mm is not None and not err:
            try:
                pack, mask, side = apply_laterality(uid, group, img_root, pack, mask, float(lat_mm), GEOMETRIES[geometry]["bounds"])
            except Exception:
                side = "U"                      # an unreadable header leaves the study unflipped, as an unresolved side does
    except Exception as exc:
        return {"uid": uid, "pack": None, "mask": None, "decode_s": 0.0,
                "pack_s": 0.0, "error": type(exc).__name__,
                "syntax": {}, "n_headers": 0}
    syntax = dict(Counter(SYNTAX))
    n_headers = len(SYNTAX)
    if err:
        return {"uid": uid, "pack": None, "mask": None, "decode_s": ds,
                "pack_s": ps, "error": err, "syntax": syntax, "n_headers": n_headers}
    return {"uid": uid, "pack": pack, "mask": mask, "decode_s": ds,
            "pack_s": ps, "error": None, "syntax": syntax, "n_headers": n_headers,
            "side": side}


def iter_job_batches(pool, jobs, fn=pack_study_job, size: int = 4):
    """Yield worker results `size` studies at a time, with bounded prefetch.

    At most 3 * size studies are held at once, however many jobs there are,
    so the hidden test (about 1300 studies) needs the same memory as three.
    A job that raises in the worker becomes an error row for that study only.
    Callers must drop r["pack"] and r["mask"] after scoring each batch.
    """
    pending = []
    job_iter = iter(jobs)

    def fill(n: int) -> None:
        for _ in range(n):
            job = next(job_iter, None)
            if job is None:
                return
            pending.append((job[0], pool.submit(fn, job)))

    fill(2 * size)
    while pending:
        head = pending[:size]
        del pending[:size]
        out = []
        for uid, fut in head:
            try:
                out.append(fut.result())
            except Exception as exc:
                out.append({"uid": uid, "pack": None, "mask": None, "decode_s": 0.0,
                            "pack_s": 0.0, "error": f"worker_{type(exc).__name__}",
                            "syntax": {}, "n_headers": 0})
        fill(size)
        yield out


def _spawn_probe(n: int) -> int:
    return n + 1


def _tree_rss_gb() -> float | None:
    """RSS of this process plus its children, in GB. None if psutil is absent."""
    try:
        import psutil
    except ImportError:
        return None
    proc = psutil.Process()
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            continue
    return total / (1024 ** 3)


def score_chunks(model, packs, masks, device: str, chunk=None, **kw):
    """score_batch over chunks of `chunk` studies (all at once when chunk is None). A large model on a small card sets score_studies in its
    meta file (item 9: 70 windows of 384 px per study). Returns (probabilities in study order, seconds)."""
    n = len(packs)
    step = n if not chunk else max(1, int(chunk))
    outs, secs = [], 0.0
    for i in range(0, n, step):
        prob, sec = score_batch(model, packs[i:i + step], masks[i:i + step], device, **kw)
        outs.append(np.asarray(prob))
        secs += sec
    return (outs[0] if len(outs) == 1 else np.concatenate(outs)), secs


def score_batch(model, packs, masks, device: str, fp16: bool, img: int = IMG, k_eval: int = K_EVAL, bounds=SLOT_BOUNDS,
                offsets=(0.0,), norm=None):
    """Several studies, one forward per window offset, pooled per finding when there is more than one offset."""
    offsets = tuple(offsets) if offsets else (0.0,)
    runs = [_score_once(model, packs, masks, device, fp16, img, k_eval, bounds, off, norm) for off in offsets]
    if len(runs) == 1:
        return runs[0]
    return pool_passes(np.stack([prob for prob, _ in runs])), sum(sec for _, sec in runs)


def _score_once(model, packs, masks, device: str, fp16: bool, img: int = IMG, k_eval: int = K_EVAL, bounds=SLOT_BOUNDS,
                offset: float = 0.0, norm=None):
    """Several studies, one forward. Resize stays on the GPU."""
    import torch
    import torch.nn.functional as Fn
    xs, slots, rels, masks_w = [], [], [], []
    for pack, mask in zip(packs, masks):
        x, slot, rel, wmask = eval_windows(pack, mask, k_eval, bounds, offset)
        xs.append(x)
        slots.append(slot)
        rels.append(rel)
        masks_w.append(wmask)
    xt = torch.from_numpy(np.ascontiguousarray(np.stack(xs)))
    st = torch.from_numpy(np.stack(slots))
    rt = torch.from_numpy(np.stack(rels))
    wt = torch.from_numpy(np.stack(masks_w))
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        xt = xt.to(device, non_blocking=True)
        b, k, c, h, w = xt.shape
        xt = xt.reshape(b * k, c, h, w).float().mul_(1.0 / 255.0)
        if h != img or w != img:
            xt = Fn.interpolate(xt, size=(img, img), mode="bilinear",
                                align_corners=False)
        mean = xt.new_tensor(norm[0] if norm else IM_MEAN).view(1, 3, 1, 1)
        std = xt.new_tensor(norm[1] if norm else IM_STD).view(1, 3, 1, 1)
        xt = (xt - mean) / std
        xt = xt.reshape(b, k, c, img, img)
        st = st.to(device)
        rt = rt.to(device)
        wt = wt.to(device)
        if fp16 and device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(xt, st, rt, wt)
        else:
            logits = model(xt, st, rt, wt)
        prob = torch.sigmoid(logits.float()).detach().cpu().numpy()
    if device == "cuda":
        torch.cuda.synchronize()
    return prob, time.perf_counter() - t0


def submit_cost_s(model_s, pack_s, workers: int = 4):
    """Steady seconds per study of the submit path: one scoring pass plus the pack time shared by the workers (an upper bound, it assumes
    no overlap). The parity cycle scores every batch twice (fp16 flag and fp32), so its wall time is not this number."""
    if model_s is None or pack_s is None:
        return None
    return float(model_s) + float(pack_s) / workers


def run_parity(id_file: str | None = None, comp: str | None = None,
               ckpt: str | None = None, out_csv: str | None = None) -> dict:
    """Same DICOM path as submission, on a list of training study ids.

    Does not read label columns. Does not write submission.csv.
    Spawn workers re-import this module. They must not start another pool.
    """
    import multiprocessing as mp
    if mp.current_process().name != "MainProcess":
        return {"ok": False, "error": "worker"}
    import json
    t_all = time.perf_counter()
    id_file = find_id_file(id_file)
    if id_file is None:
        return {"ok": False, "error": "parity_ids.txt not found"}
    refuse(id_file)
    uids = _read_ids(id_file)
    comp = find_comp(comp)
    if comp is None:
        return {"ok": False, "error": "competition root not found", "n_ids": len(uids)}
    refuse(comp)
    ckpt = find_ckpt(ckpt)
    if ckpt is None:
        return {"ok": False, "error": "loop_L2.pt not found"}
    series_path = os.path.join(comp, "train_series.csv")
    if not os.path.isfile(series_path):
        return {"ok": False, "error": "train_series.csv not found", "comp": comp}
    series = pd.read_csv(series_path, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    grouped = {k: v for k, v in series.groupby("StudyInstanceUID", sort=False)}
    img_root = find_train_root(comp, uids, series)
    meta = ckpt_meta(ckpt)
    lat_mm = float(meta["lat_mm"]) if meta["laterality"] else None
    geo = GEOMETRIES[meta["geometry"]]
    offsets = tuple(meta["tta_offsets"]) if meta.get("tta_offsets") and meta["geometry"] == "pack5" else (0.0,)
    sides: dict[str, int] = {}
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_load = time.perf_counter()
    model = _load_model(ckpt, device, meta["arch"])
    load_s = time.perf_counter() - t_load
    use_fp16 = device == "cuda"
    jobs = []
    rows_by = {}
    errors = []
    for uid in uids:
        group = grouped.get(uid)
        if img_root is None or group is None:
            errors.append({"uid": uid, "error": "no_series"})
            rows_by[uid] = [uid] + list(FALLBACK)
        else:
            jobs.append((uid, group.to_dict("records"), img_root, lat_mm, meta["geometry"]))
    code_dir = os.path.dirname(os.path.abspath(__file__))
    if __name__ != "pack_submit":
        code_dir = os.path.dirname(code_dir)
    os.environ["PYTHONPATH"] = code_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    spawn_note = "not started"
    decode_s, pack_s, model_s = [], [], []
    wall_s = []
    fp32_rows = {}
    syntax_n = {}
    n_headers = 0
    n_model = 0
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    ctx = mp.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as pool:
            probe = list(pool.map(_spawn_probe, [1, 2, 3, 4]))
            spawn_note = "ok" if probe == [2, 3, 4, 5] else f"probe {probe}"
            first_batch = True
            for batch in iter_job_batches(pool, jobs, fn=pack_study_job, size=4):
                t_cycle = time.perf_counter()
                good = [r for r in batch if r["error"] is None]
                gpu_s = 0.0
                probs = None
                probs32 = None
                if good:
                    try:
                        probs, gpu_s = score_chunks(
                            model, [r["pack"] for r in good], [r["mask"] for r in good],
                            device, meta.get("score_studies"), fp16=use_fp16, img=int(meta["img"]), k_eval=geo["k_eval"], bounds=geo["bounds"], offsets=offsets,
                            norm=norm_of(meta))
                        probs32, _ = score_chunks(
                            model, [r["pack"] for r in good], [r["mask"] for r in good],
                            device, meta.get("score_studies"), fp16=False, img=int(meta["img"]), k_eval=geo["k_eval"], bounds=geo["bounds"], offsets=offsets,
                            norm=norm_of(meta))
                    except Exception as exc:
                        for r in good:
                            r["error"] = f"score_{type(exc).__name__}"
                        probs = None
                        probs32 = None
                for r in good:
                    if r.get("side"):
                        sides[r["side"]] = sides.get(r["side"], 0) + 1
                cycle = time.perf_counter() - t_cycle
                if not first_batch:
                    wall_s.append(cycle / max(len(batch), 1))
                first_batch = False
                gi = 0
                for r in batch:
                    decode_s.append(float(r["decode_s"]))
                    pack_s.append(float(r["pack_s"]))
                    n_headers += int(r.get("n_headers") or 0)
                    for key, val in (r.get("syntax") or {}).items():
                        syntax_n[key] = syntax_n.get(key, 0) + int(val)
                    if r["error"] is None:
                        model_s.append(gpu_s / max(len(good), 1))
                        rows_by[r["uid"]] = [r["uid"]] + [float(v) for v in probs[gi]]
                        fp32_rows[r["uid"]] = [r["uid"]] + [float(v) for v in probs32[gi]]
                        n_model += 1
                        gi += 1
                    else:
                        errors.append({"uid": r["uid"], "error": r["error"]})
                        model_s.append(0.0)
                        rows_by[r["uid"]] = [r["uid"]] + list(FALLBACK)
                        fp32_rows[r["uid"]] = [r["uid"]] + list(FALLBACK)
                    r["pack"] = None
                    r["mask"] = None
    except Exception as exc:
        spawn_note = f"{type(exc).__name__}: {exc}"[:300]
    rows = []
    rows32 = []
    for uid in uids:
        rows.append(rows_by.get(uid, [uid] + list(FALLBACK)))
        rows32.append(fp32_rows.get(uid, [uid] + list(FALLBACK)))
    sub = pd.DataFrame(rows, columns=["StudyInstanceUID"] + FINDINGS)
    sub32 = pd.DataFrame(rows32, columns=["StudyInstanceUID"] + FINDINGS)
    if out_csv is None:
        out_csv = ("/kaggle/working/parity_preds.csv" if os.path.isdir("/kaggle/working")
                   else "parity_preds.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    sub.to_csv(out_csv, index=False)
    fp32_csv = os.path.join(os.path.dirname(out_csv) or ".", "parity_preds_fp32.csv")
    sub32.to_csv(fp32_csv, index=False)
    per = {
        "decode": _median(decode_s[1:]),
        "pack": _median(pack_s[1:]),
        "model": _median(model_s[1:]),
        "wall": _median(wall_s),
    }
    steady_sum = per["wall"]                      # one parity cycle: both scoring passes
    submit_s = submit_cost_s(per["model"], per["pack"])
    report = {
        "ok": n_model == len(uids) and not errors and spawn_note == "ok",
        "n": len(uids),
        "n_model": n_model,
        "n_errors": len(errors),
        "n_fallback": len(errors),
        "fallback": list(FALLBACK),
        "fallback_by_reason": {
            err: sum(1 for row in errors if row.get("error") == err) for err in
            {row.get("error") for row in errors}
        },
        "errors": errors[:20],
        "mode": "train-fast",
        "meta": meta,
        "sides": sides,
        "workers": 4,
        "spawn": spawn_note,
        "fp16": use_fp16,
        "submission_written": False,
        "preds": out_csv,
        "preds_fp32": fp32_csv,
        "runtime_s": round(time.perf_counter() - t_all, 3),
        "load_s": round(load_s, 3),
        "decode_s_per_study": _mean(decode_s),
        "pack_s_per_study": _mean(pack_s),
        "model_s_per_study": _mean(model_s),
        "steady_median_s": per,
        "steady_sum_s": steady_sum,
        "steady_submit_s": submit_s,
        "extrapolated_1300_s": None if submit_s is None else round(load_s + submit_s * 1300, 1),
        "extrapolated_1300_s_parity_cycle": None if steady_sum is None else round(load_s + steady_sum * 1300, 1),
        "device": device,
        "ckpt": ckpt,
        "comp": comp,
        "img_root": img_root,
        "id_file": id_file,
        "resize": "cv2.INTER_AREA",
        "k_eval": K_EVAL,
        "img": IMG,
        "syntax_files": syntax_n,
        "n_headers": n_headers,
        "gold": "not read",
    }
    report_path = os.path.join(os.path.dirname(out_csv) or ".", "parity_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    report["report"] = report_path
    return report


def run_submit(comp: str | None = None, ckpt: str | None = None,
               out_csv: str | None = None, img_root: str | None = None) -> dict:
    """Test studies, fast path. Writes submission.csv. Does not submit."""
    import json
    import multiprocessing as mp
    if mp.current_process().name != "MainProcess":
        return {"ok": False, "error": "worker"}
    t_all = time.perf_counter()
    comp = find_comp(comp)
    ckpt = find_ckpt(ckpt)
    if comp is None:
        return {"ok": False, "error": "test.csv not found"}
    refuse(comp)
    if ckpt is None:
        return {"ok": False, "error": "checkpoint not found", "ckpt_name": ckpt_name()}
    test = pd.read_csv(os.path.join(comp, "test.csv"), dtype={"StudyInstanceUID": str})
    series_path = os.path.join(comp, "test_series.csv")
    if not os.path.isfile(series_path):
        return {"ok": False, "error": "test_series.csv not found", "comp": comp}
    series = pd.read_csv(series_path, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    grouped = {k: v for k, v in series.groupby("StudyInstanceUID", sort=False)}
    uids = test["StudyInstanceUID"].astype(str).tolist()
    img_root = find_image_root(comp, uids, img_root)
    header = header_of(os.path.join(comp, "sample_submission.csv"))
    meta = ckpt_meta(ckpt)
    lat_mm = float(meta["lat_mm"]) if meta["laterality"] else None
    geo = GEOMETRIES[meta["geometry"]]
    offsets = tuple(meta["tta_offsets"]) if meta.get("tta_offsets") and meta["geometry"] == "pack5" else (0.0,)
    sides: dict[str, int] = {}
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_load = time.perf_counter()
    model = _load_model(ckpt, device, meta["arch"])
    load_s = time.perf_counter() - t_load
    jobs = []
    rows_by = {}
    errors = []
    for uid in uids:
        group = grouped.get(uid)
        if img_root is None or group is None:
            errors.append({"uid": uid, "error": "no_series"})
            rows_by[uid] = [uid] + list(FALLBACK)
        else:
            jobs.append((uid, group.to_dict("records"), img_root, lat_mm, meta["geometry"]))
    code_dir = os.path.dirname(os.path.abspath(__file__))
    if __name__ != "pack_submit":
        code_dir = os.path.dirname(code_dir)
    os.environ["PYTHONPATH"] = code_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    spawn_note = "not started"
    n_model = 0
    peak_rss = _tree_rss_gb()
    from concurrent.futures import ProcessPoolExecutor
    ctx = mp.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as pool:
            probe = list(pool.map(_spawn_probe, [1, 2, 3, 4]))
            spawn_note = "ok" if probe == [2, 3, 4, 5] else f"probe {probe}"
            # Bounded batches, same as the parity path: never hold every pack.
            for batch in iter_job_batches(pool, jobs):
                good = [r for r in batch if r["error"] is None]
                probs = None
                if good:
                    try:
                        probs, _gpu = score_chunks(
                            model, [r["pack"] for r in good], [r["mask"] for r in good],
                            device, meta.get("score_studies"), fp16=(device == "cuda"), img=int(meta["img"]), k_eval=geo["k_eval"], bounds=geo["bounds"], offsets=offsets,
                            norm=norm_of(meta))
                    except Exception as exc:
                        for r in good:
                            r["error"] = f"score_{type(exc).__name__}"
                for r in good:
                    if r.get("side"):
                        sides[r["side"]] = sides.get(r["side"], 0) + 1
                sample = _tree_rss_gb()
                if sample is not None:
                    peak_rss = sample if peak_rss is None else max(peak_rss, sample)
                gi = 0
                for r in batch:
                    if r["error"] is None:
                        rows_by[r["uid"]] = [r["uid"]] + [float(v) for v in probs[gi]]
                        n_model += 1
                        gi += 1
                    else:
                        errors.append({"uid": r["uid"], "error": r["error"]})
                        rows_by[r["uid"]] = [r["uid"]] + list(FALLBACK)
                    r["pack"] = None
                    r["mask"] = None
    except Exception as exc:
        spawn_note = f"{type(exc).__name__}: {exc}"[:300]
    rows = [rows_by.get(uid, [uid] + list(FALLBACK)) for uid in uids]
    sub = pd.DataFrame(rows, columns=["StudyInstanceUID"] + FINDINGS)
    sub = sub.reindex(columns=list(header))
    if out_csv is None:
        out_csv = "/kaggle/working/submission.csv" if os.path.isdir("/kaggle/working") else "submission.csv"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    sub.to_csv(out_csv, index=False)
    reasons = {row.get("error") for row in errors}
    report = {
        "ok": n_model == len(uids) and not errors and list(sub.columns) == list(header),
        "n": len(uids),
        "n_model": n_model,
        "n_fallback": len(errors),
        "fallback_by_reason": {err: sum(1 for row in errors if row.get("error") == err) for err in reasons},
        "fallback": list(FALLBACK),
        "header_match": list(sub.columns) == list(header),
        "columns": list(sub.columns),
        "n_columns": int(sub.shape[1]),
        "spawn": spawn_note,
        "fp16": device == "cuda",
        "runtime_s": round(time.perf_counter() - t_all, 3),
        "load_s": round(load_s, 3),
        "peak_rss_gb": None if peak_rss is None else round(peak_rss, 3),
        "submission": out_csv,
        "ckpt": ckpt,
        "ckpt_name": ckpt_name(),
        "comp": comp,
        "img_root": img_root,
        "gold": "not read",
        "mode": "submit-fast",
        "meta": meta,
        "sides": sides,
    }
    report_path = os.path.join(os.path.dirname(out_csv) or ".", "submit_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    report["report"] = report_path
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(run_submit(), indent=2))
