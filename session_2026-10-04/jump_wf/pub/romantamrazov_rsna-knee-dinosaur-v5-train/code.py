# %%%% CELL 0 [markdown] len=1324
# RSNA Knee | DINOsaur V5? train 🦖 

- ConvNeXt Trainer V3 FAST

This is the **V2 model/validation recipe with a faster data path**. It keeps:

- family-balanced public soft labels + Gold58 override;
- ConvNeXt-Tiny MRI adaptation on 4 views/study for 3 epochs;
- 150 mm physical crop, 2.5D triplets, up to 8 series and 5 clip quantiles;
- 5-fold Gold58 OOF and 2 independent head seeds per fold (10 heads total);
- the exact V2 output schema expected by `RSNA_Knee_DINOsaur_V5_ConvNeXt_RealArm_V2`.

Speed/stability changes:

1. DICOM decoding is moved completely out of the adaptation epoch loop into a bounded, threaded uint8 memmap precache.
2. Adaptation returns to physical batch 64 (effective batch unchanged) with pinned memmap workers and GPU-side normalization/augmentation.
3. DICOM ordering is cached for all selected series and header reads use only the required tags.
4. Feature extraction decodes triplets concurrently and writes one contiguous float16 feature memmap instead of thousands of compressed NPZ files.
5. Gold58 head training reads the contiguous feature bank directly, eliminating repeated `np.load(...npz)`/decompression overhead.

The goal is to remove the 30–40 s/step CPU bottleneck and keep a full Kaggle run within a practical multi-hour budget without intentionally weakening the V2 model.

# %%%% CELL 1 [code] len=1811
from __future__ import annotations
import os, gc, json, math, random, glob, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 20260921
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

TARGETS = [
    "ACL","MCL","Medial Meniscus","Lateral Meniscus",
    "Medial OA","Lateral OA","PF OA","Effusion","Synovitis",
    "Baker's","Contusion","Fracture",
]
N_TARGETS = len(TARGETS)

WORK = Path("/kaggle/working")
FEATURE_CACHE = WORK / "dinov5_convnext_features"
FEATURE_CACHE.mkdir(parents=True, exist_ok=True)

def find_comp_root():
    for p in (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ):
        if (p/"train.csv").is_file() and (p/"train_series.csv").is_file():
            return p
    raise FileNotFoundError("RSNA Knee competition input is not mounted")

ROOT = find_comp_root()
TRAIN = pd.read_csv(ROOT/"train.csv", dtype={"StudyInstanceUID":str})
SERIES = pd.read_csv(
    ROOT/"train_series.csv",
    dtype={"StudyInstanceUID":str, "SeriesInstanceUID":str},
)
TRAIN["StudyInstanceUID"] = TRAIN["StudyInstanceUID"].astype(str)
SERIES["StudyInstanceUID"] = SERIES["StudyInstanceUID"].astype(str)
SERIES["SeriesInstanceUID"] = SERIES["SeriesInstanceUID"].astype(str)
UIDS = TRAIN["StudyInstanceUID"].tolist()
UID_TO_INDEX = {u:i for i,u in enumerate(UIDS)}

print("studies:", len(TRAIN), "series:", len(SERIES))
print("GPU count:", torch.cuda.device_count())


# %%%% CELL 2 [code] len=4884
import hashlib as _cv_hashlib
import urllib.request as _cv_urllib
from pathlib import Path as _CvPath

_CONVNEXT_MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k"
_CONVNEXT_EXPECTED_SHA256 = "77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a"

def _cv_sha256(path, chunk=8 << 20):
    h = _cv_hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def _cv_candidate_files():
    exact = [
        _CvPath("/kaggle/input/models/ambrosm/convnext-tiny-fb-in22k-ft-in1k/pytorch/default/1/model.safetensors"),
        _CvPath("/kaggle/input/convnext-tiny-fb-in22k-ft-in1k/model.safetensors"),
    ]
    seen = set()
    out = []
    for p in exact:
        if p.is_file():
            rp = str(p.resolve())
            if rp not in seen:
                seen.add(rp)
                out.append(p)

    patterns = [
        "/kaggle/input/models/**/*convnext*/*model.safetensors",
        "/kaggle/input/models/**/*convnext*/*.safetensors",
        "/kaggle/input/**/*convnext*tiny*/**/*.safetensors",
        "/kaggle/input/**/*convnext*tiny*/**/*.pth",
        "/kaggle/input/**/*convnext*tiny*/**/*.pt",
        "/kaggle/input/**/*convnext*tiny*/**/pytorch_model.bin",
    ]
    for pattern in patterns:
        for raw in glob.glob(pattern, recursive=True):
            p = _CvPath(raw)
            if not p.is_file():
                continue
            low = str(p).lower()
            if "convnext" not in low or "tiny" not in low:
                continue
            rp = str(p.resolve())
            if rp not in seen:
                seen.add(rp)
                out.append(p)
    return out

def _cv_download_hf():
    target_dir = WORK / "convnext_tiny_pretrained"
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="timm/convnext_tiny.fb_in22k_ft_in1k",
            filename="model.safetensors",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        p = _CvPath(path)
        if p.is_file():
            return p
    except Exception as exc:
        print(
            f"[preflight] huggingface_hub download unavailable: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

    url = (
        "https://huggingface.co/timm/"
        "convnext_tiny.fb_in22k_ft_in1k/resolve/main/model.safetensors"
    )
    out = target_dir / "model.safetensors"
    try:
        _cv_urllib.urlretrieve(url, out)
        if out.is_file() and out.stat().st_size > 50_000_000:
            return out
    except Exception as exc:
        print(
            f"[preflight] direct download unavailable: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
    return None

def resolve_convnext_pretrained():
    candidates = _cv_candidate_files()
    if candidates:
        candidates = sorted(
            candidates,
            key=lambda p: (
                0 if p.name == "model.safetensors" else 1,
                0 if "fb-in22k-ft-in1k" in str(p).lower() else 1,
                len(str(p)),
                str(p),
            ),
        )
        chosen = candidates[0]
        print(f"[preflight] offline ConvNeXt: {chosen}", flush=True)
        return chosen

    print(
        "[preflight] ConvNeXt Kaggle Model input was not mounted; "
        "trying Hugging Face because this TRAIN notebook enables internet.",
        flush=True,
    )
    downloaded = _cv_download_hf()
    if downloaded is not None:
        digest = _cv_sha256(downloaded)
        if digest != _CONVNEXT_EXPECTED_SHA256:
            raise RuntimeError(
                "Downloaded ConvNeXt checkpoint SHA256 mismatch: "
                f"{digest}"
            )
        print(
            f"[preflight] downloaded verified ConvNeXt: {downloaded}",
            flush=True,
        )
        return downloaded

    mounted = []
    for root in (
        _CvPath("/kaggle/input/models"),
        _CvPath("/kaggle/input"),
    ):
        if root.exists():
            try:
                mounted.extend(
                    str(p) for p in list(root.iterdir())[:30]
                )
            except Exception:
                pass

    raise FileNotFoundError(
        "ConvNeXt pretrained weights are unavailable. "
        "Either enable Internet for this TRAIN notebook, or in Kaggle use "
        "Add Input -> Models -> ambrosm/convnext-tiny-fb-in22k-ft-in1k "
        "(PyTorch/default/1). This check now runs before expensive preprocessing. "
        f"Mounted input preview: {mounted[:20]}"
    )

CONVNEXT_PRETRAINED_PATH = resolve_convnext_pretrained()
print("ConvNeXt preflight OK:", CONVNEXT_PRETRAINED_PATH, flush=True)


# %%%% CELL 3 [code] len=4452
# ----------------------- public soft-label consensus --------------------------
ALIASES = {
    "acl":"ACL", "mcl":"MCL",
    "medial meniscus":"Medial Meniscus", "medial_meniscus":"Medial Meniscus",
    "lateral meniscus":"Lateral Meniscus", "lateral_meniscus":"Lateral Meniscus",
    "medial oa":"Medial OA", "medial_oa":"Medial OA",
    "lateral oa":"Lateral OA", "lateral_oa":"Lateral OA",
    "pf oa":"PF OA", "pf_oa":"PF OA", "patellofemoral oa":"PF OA",
    "effusion":"Effusion", "synovitis":"Synovitis",
    "baker's":"Baker's", "bakers":"Baker's", "baker cyst":"Baker's",
    "contusion":"Contusion", "fracture":"Fracture",
}

def target_name(col):
    raw = str(col).strip()
    if raw in TARGETS:
        return raw
    key = raw.lower().replace("-", " ").replace("__","_")
    return ALIASES.get(key)

def read_label_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    uid_col = next(
        (c for c in df.columns if str(c).strip().lower() in {
            "studyinstanceuid","study_instance_uid","studyuid","study_id","studyid"
        }),
        None,
    )
    if uid_col is None:
        return None
    rename = {uid_col:"StudyInstanceUID"}
    for c in df.columns:
        t = target_name(c)
        if t:
            rename[c] = t
    df = df.rename(columns=rename)
    have = [t for t in TARGETS if t in df.columns]
    if len(have) < 8:
        return None
    out = df[["StudyInstanceUID"] + have].copy()
    out["StudyInstanceUID"] = out["StudyInstanceUID"].astype(str)
    for t in have:
        out[t] = pd.to_numeric(out[t], errors="coerce")
        out.loc[~out[t].between(0,1), t] = np.nan
    return out

label_sources = []
source_paths = []
competition_train = str((ROOT/"train.csv").resolve())

for path in glob.glob("/kaggle/input/**/*.csv", recursive=True):
    low = path.lower()
    if not any(k in low for k in ("label","llm","pseudo","report","target","soft")):
        continue
    try:
        if str(Path(path).resolve()) == competition_train:
            continue
    except Exception:
        pass
    df = read_label_csv(path)
    if df is None:
        continue
    overlap = df["StudyInstanceUID"].isin(UIDS).mean()
    if overlap >= 0.50:
        label_sources.append(df)
        source_paths.append(path)

print("label sources discovered:", len(label_sources))
for p in source_paths:
    print("  ", p)

idx = pd.Index(UIDS, name="StudyInstanceUID")
pseudo = np.full((len(UIDS), N_TARGETS), np.nan, np.float32)
source_count = np.zeros((len(UIDS), N_TARGETS), np.int16)

for j,t in enumerate(TARGETS):
    columns = []
    for src in label_sources:
        if t in src.columns:
            s = src.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
            columns.append(s.reindex(idx)[t].to_numpy(np.float32))
    if columns:
        arr = np.stack(columns, axis=1)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="All-NaN slice encountered",
                category=RuntimeWarning,
            )
            pseudo[:,j] = np.nanmedian(arr, axis=1)
        source_count[:,j] = np.isfinite(arr).sum(axis=1)

gold = np.full_like(pseudo, np.nan)
for j,t in enumerate(TARGETS):
    if t in TRAIN.columns:
        gold[:,j] = pd.to_numeric(TRAIN[t], errors="coerce").to_numpy(np.float32)

gold_mask = np.isfinite(gold)
targets = np.where(gold_mask, gold, pseudo).astype(np.float32)

# Keep pseudo labels soft, but gently sharpen indecisive probabilities.
valid = np.isfinite(targets)
p = np.clip(np.where(valid, targets, 0.5), 1e-4, 1-1e-4)
logit = np.log(p/(1-p))
soft_sharp = 1/(1+np.exp(-logit/0.85))
targets = np.where(gold_mask, targets, soft_sharp).astype(np.float32)
targets[~valid] = np.nan

max_sources = max(len(label_sources), 1)
coverage = np.clip(source_count / max_sources, 0, 1)
certainty = np.clip(2*np.abs(np.nan_to_num(targets, nan=0.5)-0.5), 0, 1)
weights = (0.25 + 0.75*(0.55*coverage + 0.45*certainty)).astype(np.float32)
weights[~np.isfinite(targets)] = 0.0
weights[gold_mask] = 2.0

gold_study_mask = np.any(gold_mask, axis=1)
print("expert-labelled studies:", int(gold_study_mask.sum()))
print("pseudo-covered studies:", int(np.any(np.isfinite(pseudo), axis=1).sum()))

if np.any(np.isfinite(pseudo), axis=1).sum() < 3000:
    warnings.warn(
        "Pseudo-label coverage is low. Attach Steven/GPT public report-label datasets."
    )


# %%%% CELL 4 [code] len=23755
# ---------- V5 STREAMSAFE: bounded-RAM adaptation -> feature bank -------------
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import timm
import cv2

cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

PLANE_TO_ID = {"Sagittal":0, "Coronal":1, "Axial":2}
MAX_SERIES = 8
CLIP_Q = (0.15, 0.325, 0.50, 0.675, 0.85)
CROP_MM = 150.0
IMAGE_SIZE = 224
FEATURE_DIM = 768
ADAPT_REPEATS = 4
ADAPT_Q = (0.18,0.38,0.62,0.82)
ADAPT_EPOCHS = 3
ADAPT_BATCH = 64
CACHE_WORKERS = max(2, min(4, os.cpu_count() or 2))
TRAIN_WORKERS = 0

try:
    import ctypes as _d5_ctypes
    _d5_libc=_d5_ctypes.CDLL("libc.so.6")
    def _malloc_trim():
        try:
            _d5_libc.malloc_trim(0)
        except Exception:
            pass
except Exception:
    def _malloc_trim():
        pass

series_by_study = {k:v.copy() for k,v in SERIES.groupby("StudyInstanceUID")}

def choose_series(frame, max_series=MAX_SERIES):
    f = frame.copy()
    for c in ("Fluid_Sensitive","Fat_Suppression"):
        if c not in f.columns:
            f[c] = 0
        f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0).astype(int)
    f["Anatomical_Plane"] = f["Anatomical_Plane"].astype(str)
    chosen, used = [], set()
    for plane in ("Sagittal","Coronal","Axial"):
        z = f[f["Anatomical_Plane"].eq(plane)]
        if len(z):
            z = z.sort_values(["Fluid_Sensitive","Fat_Suppression"], ascending=False)
            r = z.iloc[0]
            chosen.append(r)
            used.add(str(r["SeriesInstanceUID"]))
    for fluid,fat in ((1,1),(1,0),(0,1),(0,0)):
        z = f[(f["Fluid_Sensitive"]==fluid) & (f["Fat_Suppression"]==fat)]
        for _,r in z.iterrows():
            sid = str(r["SeriesInstanceUID"])
            if sid not in used:
                chosen.append(r)
                used.add(sid)
                break
    for _,r in f.iterrows():
        if len(chosen) >= max_series:
            break
        sid = str(r["SeriesInstanceUID"])
        if sid not in used:
            chosen.append(r)
            used.add(sid)
    return pd.DataFrame(chosen[:max_series]) if chosen else f.head(0)

@lru_cache(maxsize=2048)
def ordered_files_cached(series_dir_string):
    values=[]
    for fp in Path(series_dir_string).glob("*.dcm"):
        try:
            h=pydicom.dcmread(
                str(fp),stop_before_pixels=True,force=True,
                specific_tags=["ImageOrientationPatient","ImagePositionPatient","InstanceNumber"],
            )
            iop=getattr(h,"ImageOrientationPatient",None)
            ipp=getattr(h,"ImagePositionPatient",None)
            if iop is not None and ipp is not None and len(iop)==6:
                row=np.asarray(iop[:3],np.float64)
                col=np.asarray(iop[3:6],np.float64)
                normal=np.cross(row,col)
                pos=float(np.dot(np.asarray(ipp[:3],np.float64),normal))
            else:
                pos=float(getattr(h,"InstanceNumber",0) or 0)
            values.append((pos,str(fp)))
        except Exception:
            pass
    values.sort(key=lambda x:x[0])
    return tuple(fp for _,fp in values)

def ordered_files(series_dir):
    return ordered_files_cached(str(series_dir))

def ordered_files_uncached(series_dir):
    return ordered_files_cached.__wrapped__(str(series_dir))

def read_slice_resized_uint8(fp):
    ds=pydicom.dcmread(str(fp),force=True)
    arr=ds.pixel_array.astype(np.float32,copy=False)
    slope=float(getattr(ds,"RescaleSlope",1.0) or 1.0)
    intercept=float(getattr(ds,"RescaleIntercept",0.0) or 0.0)
    if slope!=1.0 or intercept!=0.0:
        arr=arr*slope+intercept
    if str(getattr(ds,"PhotometricInterpretation",""))=="MONOCHROME1":
        arr=arr.max()-arr
    ps=getattr(ds,"PixelSpacing",None)
    spacing=float(ps[0]) if ps is not None else None
    if spacing and spacing>0:
        side=int(round(CROP_MM/spacing))
        side=max(16,min(side,arr.shape[0],arr.shape[1]))
        cy,cx=arr.shape[0]//2,arr.shape[1]//2
        y0=max(0,cy-side//2)
        x0=max(0,cx-side//2)
        arr=arr[y0:y0+side,x0:x0+side]
    probe=arr[::4,::4] if min(arr.shape)>=128 else arr
    lo,hi=np.percentile(probe,[1,99])
    arr=np.clip((arr-lo)/(hi-lo+1e-6),0,1)
    arr=cv2.resize(arr,(IMAGE_SIZE,IMAGE_SIZE),interpolation=cv2.INTER_LINEAR)
    out=np.clip(arr*255.0+0.5,0,255).astype(np.uint8)
    del ds,arr,probe
    return out

def triplet_uint8(files,center):
    n=len(files)
    ids=[max(0,min(n-1,center+d)) for d in (-1,0,1)]
    return np.stack([read_slice_resized_uint8(files[i]) for i in ids],axis=0)

def safe_float(v,default=0.0):
    z=pd.to_numeric(v,errors="coerce")
    return float(default if pd.isna(z) else z)

def meta_vector(row,q):
    plane=np.zeros(3,np.float32)
    plane[PLANE_TO_ID.get(str(row.get("Anatomical_Plane","")),2)]=1.0
    return np.array([
        *plane,
        safe_float(row.get("Fluid_Sensitive",0)),
        safe_float(row.get("Fat_Suppression",0)),
        float(q),
    ],np.float32)

_CONVNEXT_MODEL_NAME="convnext_tiny.fb_in22k_ft_in1k"
if not Path(CONVNEXT_PRETRAINED_PATH).is_file():
    raise FileNotFoundError(f"ConvNeXt preflight path disappeared: {CONVNEXT_PRETRAINED_PATH}")
print("offline ConvNeXt:",CONVNEXT_PRETRAINED_PATH)
try:
    encoder=timm.create_model(
        _CONVNEXT_MODEL_NAME,
        pretrained=True,
        pretrained_cfg_overlay={"file":str(CONVNEXT_PRETRAINED_PATH)},
        num_classes=0,
        global_pool="avg",
    )
except Exception as exc:
    raise RuntimeError(
        "timm could not load the resolved ConvNeXt checkpoint "
        f"{CONVNEXT_PRETRAINED_PATH}: {type(exc).__name__}: {exc}"
    ) from exc
FEATURE_DIM=int(encoder.num_features)
if FEATURE_DIM!=768:
    raise RuntimeError(f"Unexpected ConvNeXt feature dim: {FEATURE_DIM}")

# ------------------------ stage A: MRI domain adaptation ----------------------
adapt_indices=np.where(
    (~gold_study_mask)
    & (np.sum(np.isfinite(targets),axis=1)>=8)
    & (np.sum(weights>0,axis=1)>=8)
)[0]
if len(adapt_indices)<3000:
    raise RuntimeError(
        f"Only {len(adapt_indices)} non-gold pseudo-labelled studies are available. "
        "Attach the public LLM report-label datasets before training this arm."
    )

selected_rows={}
for ii in adapt_indices:
    uid=UIDS[int(ii)]
    f=series_by_study.get(uid)
    if f is None or not len(f):
        continue
    selected_rows[uid]=choose_series(f).to_dict("records")

# V5: no global 23k-series prewarm.  Build only the series actually sampled by
# the four adaptation views.  Specs are processed study-by-study for cache
# locality, but written to a deterministic random physical position so that
# contiguous training blocks are already well mixed.
N_ADAPT=len(adapt_indices)*ADAPT_REPEATS
ADAPT_BIN=WORK/"dinov5_adapt_triplets_uint8.dat"
ADAPT_MAP=WORK/"dinov5_adapt_sample_to_study.npy"
ADAPT_OK=WORK/"dinov5_adapt_valid.npy"
physical_pos=np.random.default_rng(SEED+77531).permutation(N_ADAPT)
sample_to_study=np.empty(N_ADAPT,np.int32)
valid_sample=np.zeros(N_ADAPT,np.uint8)
adapt_specs=[]
_k=0
for base_i in adapt_indices:
    for rep in range(ADAPT_REPEATS):
        out_i=int(physical_pos[_k]); _k+=1
        sample_to_study[out_i]=int(base_i)
        adapt_specs.append((out_i,int(base_i),rep))
del physical_pos,_k

mm=np.memmap(
    ADAPT_BIN,mode="w+",dtype=np.uint8,
    shape=(N_ADAPT,3,IMAGE_SIZE,IMAGE_SIZE),
)

def build_adapt_triplet(spec):
    out_i,base_i,rep=spec
    uid=UIDS[int(base_i)]
    rows=selected_rows.get(uid,[])
    rng=np.random.default_rng(SEED+int(base_i)*1009+int(rep)*9176)
    if rows:
        for ridx in rng.permutation(len(rows)):
            row=rows[int(ridx)]
            sid=str(row["SeriesInstanceUID"])
            files=ordered_files(ROOT/"train_series"/uid/sid)
            if not files:
                continue
            q=float(np.clip(ADAPT_Q[rep]+rng.uniform(-0.07,0.07),0.08,0.92))
            c=int(round(q*(len(files)-1)))
            try:
                return out_i,triplet_uint8(files,c),1
            except Exception:
                continue
    return out_i,np.zeros((3,IMAGE_SIZE,IMAGE_SIZE),np.uint8),0

print(
    "adapt precache lazy-order:",len(adapt_specs),
    "workers:",CACHE_WORKERS,
    "order_cache_max=2048",
    "disk_gb:",round(N_ADAPT*3*IMAGE_SIZE*IMAGE_SIZE/(1024**3),2),
    flush=True,
)
t0=time.time()
done=0
fallbacks=0
with ThreadPoolExecutor(max_workers=CACHE_WORKERS) as pool:
    for start_i in range(0,len(adapt_specs),256):
        block=adapt_specs[start_i:start_i+256]
        for out_i,arr,ok in pool.map(build_adapt_triplet,block):
            mm[out_i]=arr
            valid_sample[out_i]=ok
            fallbacks+=int(not ok)
            done+=1
        if done%1024<256 or done==len(adapt_specs):
            mm.flush()
            print(
                f"adapt precache {done}/{len(adapt_specs)} "
                f"fallback_triplets={fallbacks} "
                f"elapsed={(time.time()-t0)/60:.1f}m",
                flush=True,
            )
mm.flush()
try:
    mm._mmap.close()
except Exception:
    pass
del mm
np.save(ADAPT_MAP,sample_to_study)
np.save(ADAPT_OK,valid_sample)
valid_fraction=float(valid_sample.mean())
print(
    "adapt file ready:",ADAPT_BIN,
    "size_gb=",round(ADAPT_BIN.stat().st_size/(1024**3),2),
    "valid_fraction=",round(valid_fraction,4),flush=True,
)
if valid_fraction<0.97:
    raise RuntimeError(f"Too many adaptation triplet failures: valid_fraction={valid_fraction:.4f}")

# Drop all DICOM path tuples and thread-era allocations before gradient training.
ordered_files_cached.cache_clear()
adapt_specs.clear()
selected_rows.clear()
gc.collect()
_malloc_trim()

class AdaptModel(nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.encoder=encoder
        self.head=nn.Linear(FEATURE_DIM,N_TARGETS)
    def forward(self,x):
        return self.head(self.encoder(x))

for p in encoder.parameters():
    p.requires_grad=False
if not hasattr(encoder,"stages") or len(encoder.stages)<1:
    raise RuntimeError("Unexpected timm ConvNeXt structure: missing stages")
for p in encoder.stages[-1].parameters():
    p.requires_grad=True
for module_name in ("norm_pre","head"):
    module=getattr(encoder,module_name,None)
    if module is not None:
        for p in module.parameters():
            p.requires_grad=True

gpu_count=torch.cuda.device_count()
device=torch.device("cuda:0" if gpu_count else "cpu")
adapt_core=AdaptModel(encoder).to(device)
if device.type=="cuda":
    adapt_core=adapt_core.to(memory_format=torch.channels_last)
adapt_model=nn.DataParallel(adapt_core,device_ids=list(range(gpu_count))) if gpu_count>1 else adapt_core

trainable=[p for p in adapt_model.parameters() if p.requires_grad]
opt=torch.optim.AdamW(trainable,lr=8e-5,weight_decay=1e-4)
scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda"))
try:
    import psutil as _d5_psutil
    _d5_proc=_d5_psutil.Process(os.getpid())
    def _d5_rss_gb():
        return float(_d5_proc.memory_info().rss)/(1024**3)
except Exception:
    def _d5_rss_gb():
        return float("nan")

def weighted_bce(logits,y,w):
    valid=torch.isfinite(y)&(w>0)
    yy=torch.nan_to_num(y,nan=.5)
    loss=F.binary_cross_entropy_with_logits(logits,yy,reduction="none")
    ww=torch.where(valid,w,torch.zeros_like(w))
    return (loss*ww).sum()/ww.sum().clamp_min(1.0)

IM_MEAN_GPU=torch.tensor([0.485,0.456,0.406],device=device,dtype=torch.float32).view(1,3,1,1)
IM_STD_GPU=torch.tensor([0.229,0.224,0.225],device=device,dtype=torch.float32).view(1,3,1,1)

def prepare_uint8_batch(x,augment=False):
    x=x.to(device,non_blocking=False).float().mul_(1.0/255.0)
    if augment:
        b=x.shape[0]
        gamma=torch.empty((b,1,1,1),device=device).uniform_(0.88,1.12)
        gain=torch.empty((b,1,1,1),device=device).uniform_(0.92,1.08)
        bias=torch.empty((b,1,1,1),device=device).uniform_(-0.035,0.035)
        x=x.clamp_(0,1).pow(gamma)
        x=(x*gain+bias).clamp_(0,1)
    x=(x-IM_MEAN_GPU)/IM_STD_GPU
    if device.type=="cuda":
        x=x.contiguous(memory_format=torch.channels_last)
    return x

def _rss_parts_gb():
    out={"rss":float("nan"),"anon":float("nan"),"file":float("nan")}
    try:
        vals={}
        with open("/proc/self/status","r",encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(("VmRSS:","RssAnon:","RssFile:")):
                    k,v=line.split(":",1)
                    vals[k]=float(v.strip().split()[0])/(1024**2)
        out["rss"]=vals.get("VmRSS",out["rss"])
        out["anon"]=vals.get("RssAnon",out["anon"])
        out["file"]=vals.get("RssFile",out["file"])
    except Exception:
        out["rss"]=_d5_rss_gb()
    return out

def _gpu_mem_string():
    if not torch.cuda.is_available():
        return ""
    parts=[]
    for gi in range(torch.cuda.device_count()):
        a=torch.cuda.memory_allocated(gi)/(1024**3)
        r=torch.cuda.memory_reserved(gi)/(1024**3)
        parts.append(f"gpu{gi}={a:.2f}/{r:.2f}GB")
    return " ".join(parts)

# V5 STREAMSAFE: one reusable pinned host batch.  The training file is read
# with readinto() in contiguous blocks; no memmap is mapped and no ~10 MB
# NumPy/CPU allocation is created on every step.  The file itself was written
# in a random physical order, and only the block order is reshuffled per epoch.
_SAMPLE_BYTES=3*IMAGE_SIZE*IMAGE_SIZE
_N_BLOCKS=(N_ADAPT+ADAPT_BATCH-1)//ADAPT_BATCH
_PIN=(device.type=="cuda")
x_host=torch.empty(
    (ADAPT_BATCH,3,IMAGE_SIZE,IMAGE_SIZE),dtype=torch.uint8,pin_memory=_PIN
)
y_host=torch.empty((ADAPT_BATCH,N_TARGETS),dtype=torch.float32,pin_memory=_PIN)
w_host=torch.empty((ADAPT_BATCH,N_TARGETS),dtype=torch.float32,pin_memory=_PIN)
x_host_np=x_host.numpy()
y_host_np=y_host.numpy()
w_host_np=w_host.numpy()
x_host_bytes=memoryview(x_host_np).cast("B")

print(
    "MRI adaptation streaming:",
    "studies=",len(adapt_indices),
    "samples=",N_ADAPT,
    "physical_batch=",ADAPT_BATCH,
    "grad_accum=1",
    "gpus=",gpu_count,
    "reusable_pinned_buffer=",bool(_PIN),
    "block_shuffle=1",
    flush=True,
)

with open(ADAPT_BIN,"rb",buffering=0) as adapt_fh:
    for epoch in range(ADAPT_EPOCHS):
        adapt_model.train()
        opt.zero_grad(set_to_none=True)
        total_loss_gpu=torch.zeros((),device=device,dtype=torch.float32)
        window_loss_gpu=torch.zeros((),device=device,dtype=torch.float32)
        window_count=0
        t_epoch=time.time()
        block_order=np.random.default_rng(SEED+9109*epoch+313).permutation(_N_BLOCKS)

        for step,block_id in enumerate(block_order):
            start_i=int(block_id)*ADAPT_BATCH
            b=min(ADAPT_BATCH,N_ADAPT-start_i)
            nbytes=b*_SAMPLE_BYTES
            adapt_fh.seek(start_i*_SAMPLE_BYTES)
            got=adapt_fh.readinto(x_host_bytes[:nbytes])
            if got!=nbytes:
                raise IOError(
                    f"Short adaptation read at sample {start_i}: {got} != {nbytes}"
                )

            base_idx=sample_to_study[start_i:start_i+b]
            np.copyto(y_host_np[:b],targets[base_idx],casting="unsafe")
            np.copyto(w_host_np[:b],weights[base_idx],casting="unsafe")
            # Invalid precache entries remain physically present to preserve
            # contiguous IO, but contribute exactly zero training weight.
            if not np.all(valid_sample[start_i:start_i+b]):
                w_host_np[:b]*=valid_sample[start_i:start_i+b,None]

            x=prepare_uint8_batch(x_host[:b],augment=True)
            y=y_host[:b].to(device,non_blocking=_PIN)
            w=w_host[:b].to(device,non_blocking=_PIN)

            with torch.autocast(
                device_type=device.type,dtype=torch.float16,enabled=(device.type=="cuda")
            ):
                logits=adapt_model(x)
                loss=weighted_bce(logits,y,w)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(trainable,1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            detached=loss.detach().float()
            total_loss_gpu.add_(detached)
            window_loss_gpu.add_(detached)
            window_count+=1
            del x,y,w,logits,loss,detached,base_idx

            if (step+1)%100==0 or step+1==_N_BLOCKS:
                # One scalar sync per 100 steps also guarantees that the
                # asynchronous H2D copy no longer references x_host.
                recent=float((window_loss_gpu/max(1,window_count)).item())
                window_loss_gpu.zero_(); window_count=0
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gc.collect(); _malloc_trim()
                mem=_rss_parts_gb()
                print(
                    f"adapt epoch {epoch+1}/{ADAPT_EPOCHS} step {step+1}/{_N_BLOCKS} "
                    f"loss={recent:.4f} rss={mem['rss']:.2f}GB "
                    f"anon={mem['anon']:.2f}GB file={mem['file']:.2f}GB "
                    f"{_gpu_mem_string()} "
                    f"elapsed={(time.time()-t_epoch)/60:.1f}m",
                    flush=True,
                )

        mean_loss=float((total_loss_gpu/_N_BLOCKS).item())
        del total_loss_gpu,window_loss_gpu,block_order
        core=adapt_model.module if isinstance(adapt_model,nn.DataParallel) else adapt_model
        torch.save(
            {
                "epoch":epoch+1,
                "model":core.state_dict(),
                "mean_loss":mean_loss,
                "model_name":_CONVNEXT_MODEL_NAME,
            },
            WORK/f"dinov5_adapt_epoch{epoch+1}.pt",
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect(); _malloc_trim()
        mem=_rss_parts_gb()
        print(
            f"adapt epoch {epoch+1}/{ADAPT_EPOCHS} mean_loss={mean_loss:.4f} "
            f"rss={mem['rss']:.2f}GB anon={mem['anon']:.2f}GB "
            f"file={mem['file']:.2f}GB checkpoint_saved",
            flush=True,
        )

core=adapt_model.module if isinstance(adapt_model,nn.DataParallel) else adapt_model
encoder=core.encoder.cpu()
torch.save(encoder.state_dict(),WORK/"dinov5_convnext_backbone.pt")
del adapt_model,adapt_core,core,opt,scaler,trainable,IM_MEAN_GPU,IM_STD_GPU
del x_host,y_host,w_host,x_host_np,y_host_np,w_host_np,x_host_bytes
gc.collect(); _malloc_trim()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
try:
    ADAPT_BIN.unlink(missing_ok=True)
    ADAPT_MAP.unlink(missing_ok=True)
    ADAPT_OK.unlink(missing_ok=True)
except Exception as exc:
    print("temporary adaptation file cleanup warning:",repr(exc))

# ----------------------- stage B: contiguous study feature bank ---------------
encoder.eval()
for p in encoder.parameters():
    p.requires_grad=False
if device.type=="cuda":
    encoder=encoder.to(device,memory_format=torch.channels_last)
else:
    encoder=encoder.to(device)
if gpu_count>1:
    encoder=nn.DataParallel(encoder,device_ids=list(range(gpu_count)))

all_selected_rows={}
for uid in UIDS:
    frame=series_by_study.get(uid)
    if frame is None or not len(frame):
        all_selected_rows[uid]=[]
        continue
    all_selected_rows[uid]=choose_series(frame).to_dict("records")

print(
    "feature stage uses per-study uncached DICOM ordering; "
    "no global path cache is retained",
    flush=True,
)

MAX_TOKENS=MAX_SERIES*len(CLIP_Q)
FEATURE_BANK_BIN=WORK/"dinov5_feature_bank_f16.dat"
META_BANK_BIN=WORK/"dinov5_meta_bank_f16.dat"
FEATURE_LEN_PATH=WORK/"dinov5_feature_lengths.npy"
feature_mm=np.memmap(
    FEATURE_BANK_BIN,mode="w+",dtype=np.float16,
    shape=(len(UIDS),MAX_TOKENS,FEATURE_DIM),
)
meta_mm=np.memmap(
    META_BANK_BIN,mode="w+",dtype=np.float16,
    shape=(len(UIDS),MAX_TOKENS,6),
)
FEATURE_LENGTHS=np.zeros(len(UIDS),np.int16)

IM_MEAN_GPU=torch.tensor([0.485,0.456,0.406],device=device,dtype=torch.float32).view(1,3,1,1)
IM_STD_GPU=torch.tensor([0.229,0.224,0.225],device=device,dtype=torch.float32).view(1,3,1,1)

FEATURE_X_HOST=torch.empty(
    (MAX_TOKENS,3,IMAGE_SIZE,IMAGE_SIZE),dtype=torch.uint8,
    pin_memory=(device.type=="cuda"),
)
FEATURE_X_NP=FEATURE_X_HOST.numpy()

def normalize_feature_batch(x):
    x=x.to(device,non_blocking=(device.type=="cuda")).float().mul_(1.0/255.0)
    x=(x-IM_MEAN_GPU)/IM_STD_GPU
    if device.type=="cuda":
        x=x.contiguous(memory_format=torch.channels_last)
    return x

def decode_feature_spec(spec):
    files,center=spec
    try:
        return triplet_uint8(files,center)
    except Exception:
        return None

@torch.inference_mode()
def extract_study_fast(uid,pool):
    rows=all_selected_rows.get(uid,[])
    specs=[]
    metas=[]
    for row in rows:
        sid=str(row["SeriesInstanceUID"])
        files=ordered_files_uncached(ROOT/"train_series"/uid/sid)
        if not files:
            continue
        for q in CLIP_Q:
            center=int(round(q*(len(files)-1)))
            specs.append((files,center))
            metas.append(meta_vector(row,q))
    if not specs:
        return np.zeros((1,FEATURE_DIM),np.float16),np.zeros((1,6),np.float16)
    decoded=list(pool.map(decode_feature_spec,specs))
    images=[]
    good_meta=[]
    for arr,meta in zip(decoded,metas):
        if arr is not None:
            images.append(arr)
            good_meta.append(meta)
    if not images:
        return np.zeros((1,FEATURE_DIM),np.float16),np.zeros((1,6),np.float16)
    n_img=len(images)
    for jj,arr in enumerate(images):
        np.copyto(FEATURE_X_NP[jj],arr,casting="no")
    x=normalize_feature_batch(FEATURE_X_HOST[:n_img])
    with torch.autocast(
        device_type=device.type,dtype=torch.float16,enabled=(device.type=="cuda")
    ):
        z=encoder(x).float()
    feat=z.cpu().numpy().astype(np.float16,copy=False)
    meta=np.stack(good_meta,axis=0).astype(np.float16,copy=False)
    del x,z,images,decoded
    return feat,meta

t0=time.time()
feature_fallbacks=0
with ThreadPoolExecutor(max_workers=CACHE_WORKERS) as pixel_pool:
    for i,uid in enumerate(UIDS):
        try:
            feat,meta=extract_study_fast(uid,pixel_pool)
        except Exception as exc:
            feature_fallbacks+=1
            if feature_fallbacks<=10:
                print("feature fallback",uid,type(exc).__name__,str(exc)[:140])
            feat=np.zeros((1,FEATURE_DIM),np.float16)
            meta=np.zeros((1,6),np.float16)
        n=min(int(feat.shape[0]),MAX_TOKENS)
        feature_mm[i,:n]=feat[:n]
        meta_mm[i,:n]=meta[:n]
        FEATURE_LENGTHS[i]=max(1,n)
        if (i+1)%100==0 or i+1==len(UIDS):
            feature_mm.flush(); meta_mm.flush()
            gc.collect()
            _malloc_trim()
            print(
                f"feature bank {i+1}/{len(UIDS)} "
                f"fallbacks={feature_fallbacks} rss={_d5_rss_gb():.2f}GB "
                f"elapsed={(time.time()-t0)/60:.1f}m",
                flush=True,
            )
        del feat,meta
feature_mm.flush(); meta_mm.flush()
try:
    feature_mm._mmap.close(); meta_mm._mmap.close()
except Exception:
    pass
del feature_mm,meta_mm
np.save(FEATURE_LEN_PATH,FEATURE_LENGTHS)
ordered_files_cached.cache_clear()
del encoder,IM_MEAN_GPU,IM_STD_GPU,all_selected_rows,selected_rows,FEATURE_X_HOST,FEATURE_X_NP
gc.collect()
_malloc_trim()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(
    "feature extraction done; fallback studies:",feature_fallbacks,
    "feature_bank_gb=",round(FEATURE_BANK_BIN.stat().st_size/(1024**3),3),
    flush=True,
)

# %%%% CELL 5 [code] len=4666
# ----------------------- label-aware attention head ---------------------------
class FeatureDataset(Dataset):
    def __init__(self,indices,y,w):
        self.indices=list(map(int,indices))
        self.y=y
        self.w=w
        self._feat=None
        self._meta=None
    def _open(self):
        if self._feat is None:
            self._feat=np.memmap(
                FEATURE_BANK_BIN,mode="r",dtype=np.float16,
                shape=(len(UIDS),MAX_TOKENS,FEATURE_DIM),
            )
            self._meta=np.memmap(
                META_BANK_BIN,mode="r",dtype=np.float16,
                shape=(len(UIDS),MAX_TOKENS,6),
            )
        return self._feat,self._meta
    def __len__(self):
        return len(self.indices)
    def __getitem__(self,k):
        i=self.indices[k]
        feat_mm,meta_mm=self._open()
        n=max(1,int(FEATURE_LENGTHS[i]))
        feat=np.array(feat_mm[i,:n],dtype=np.float32,copy=True)
        meta=np.array(meta_mm[i,:n],dtype=np.float32,copy=True)
        return (
            torch.from_numpy(feat),
            torch.from_numpy(meta),
            torch.from_numpy(self.y[i].astype(np.float32)),
            torch.from_numpy(self.w[i].astype(np.float32)),
            i,
        )

def collate(batch):
    B=len(batch)
    S=max(item[0].shape[0] for item in batch)
    feat=torch.zeros(B,S,FEATURE_DIM)
    meta=torch.zeros(B,S,6)
    mask=torch.zeros(B,S,dtype=torch.bool)
    yy=[]; ww=[]; ids=[]
    for b,(f,m,y,w,i) in enumerate(batch):
        n=f.shape[0]
        feat[b,:n]=f
        meta[b,:n]=m
        mask[b,:n]=True
        yy.append(y); ww.append(w); ids.append(i)
    return feat,meta,mask,torch.stack(yy),torch.stack(ww),torch.tensor(ids)

class LabelAwareHead(nn.Module):
    def __init__(self,d=384):
        super().__init__()
        self.proj=nn.Sequential(
            nn.Linear(FEATURE_DIM,d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(.15),
        )
        self.meta=nn.Sequential(
            nn.Linear(6,64),nn.GELU(),nn.Linear(64,d)
        )
        self.query=nn.Parameter(torch.randn(N_TARGETS,d)*0.02)
        self.key=nn.Linear(d,d,bias=False)
        self.value=nn.Linear(d,d,bias=False)
        self.label_emb=nn.Parameter(torch.randn(N_TARGETS,d)*0.02)

        layer=nn.TransformerEncoderLayer(
            d_model=d,nhead=8,dim_feedforward=768,
            dropout=.12,activation="gelu",
            batch_first=True,norm_first=True,
        )
        self.cross_label=nn.TransformerEncoder(layer,num_layers=2)
        self.norm=nn.LayerNorm(d)
        self.h1=nn.Linear(d,192)
        self.out=nn.Parameter(torch.randn(N_TARGETS,192)*0.02)
        self.bias=nn.Parameter(torch.zeros(N_TARGETS))

    def forward(self,feat,meta,mask):
        h=self.proj(feat)+self.meta(meta)
        k=self.key(h)
        v=self.value(h)
        scores=torch.einsum("ld,bsd->bls",self.query,k)/math.sqrt(k.shape[-1])
        scores=scores.masked_fill(~mask[:,None,:],-1e4)
        attn=torch.softmax(scores,dim=-1)
        z=torch.einsum("bls,bsd->bld",attn,v)
        z=z+self.label_emb[None]
        z=self.cross_label(z)
        z=F.gelu(self.h1(self.norm(z)))
        return torch.einsum("blh,lh->bl",z,self.out)+self.bias[None]

def soft_bce(logits,y,w):
    valid=torch.isfinite(y)&(w>0)
    yy=torch.nan_to_num(y,nan=.5)
    loss=F.binary_cross_entropy_with_logits(logits,yy,reduction="none")
    ww=torch.where(valid,w,torch.zeros_like(w))
    return (loss*ww).sum()/ww.sum().clamp_min(1.0)

def pair_rank_loss(logits,y,w):
    pieces=[]
    for j in range(N_TARGETS):
        valid=torch.isfinite(y[:,j])&(w[:,j]>0)
        if valid.sum()<4:
            continue
        yy=y[valid,j]
        zz=logits[valid,j]
        pos=zz[yy>=.65]
        neg=zz[yy<=.35]
        if len(pos)==0 or len(neg)==0:
            continue
        n=min(64,max(len(pos),len(neg)))
        p=pos[torch.randint(len(pos),(n,),device=zz.device)]
        q=neg[torch.randint(len(neg),(n,),device=zz.device)]
        pieces.append(F.softplus(-(p-q)).mean())
    return torch.stack(pieces).mean() if pieces else logits.sum()*0

def macro_auc(y,p):
    vals=[]
    for j in range(N_TARGETS):
        m=np.isfinite(y[:,j])
        if m.sum()<3 or len(np.unique(y[m,j]))<2:
            continue
        vals.append(roc_auc_score(y[m,j],p[m,j]))
    return float(np.mean(vals)) if vals else float("nan")

def per_target_auc(y,p):
    result={}
    for j,t in enumerate(TARGETS):
        m=np.isfinite(y[:,j])
        if m.sum()>=3 and len(np.unique(y[m,j]))>=2:
            result[t]=float(roc_auc_score(y[m,j],p[m,j]))
        else:
            result[t]=None
    return result


# %%%% CELL 6 [code] len=6500
gold_indices=np.where(gold_study_mask)[0]
if len(gold_indices)<40:
    raise RuntimeError(f"Expected roughly 58 expert studies; found {len(gold_indices)}")

positive_count=np.nansum(gold[gold_indices],axis=1)
strata=np.clip(positive_count.astype(int),0,5)
vc=pd.Series(strata).value_counts()
for value in list(vc[vc<5].index):
    replacement=max(0,min(5,int(value)-1))
    strata[strata==value]=replacement

skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
fold_of=np.full(len(UIDS),-1,np.int8)
for fold,(_,va_local) in enumerate(skf.split(gold_indices,strata)):
    fold_of[gold_indices[va_local]]=fold

print(pd.Series(fold_of[gold_indices]).value_counts().sort_index())

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
non_gold=np.where(
    (~gold_study_mask)&np.any(np.isfinite(targets),axis=1)
)[0]
val_weights=np.where(np.isfinite(gold),1.0,0.0).astype(np.float32)

HEAD_SEEDS=(SEED+101,SEED+1009)
oof_sum=np.zeros((len(UIDS),N_TARGETS),np.float64)
oof_count=np.zeros(len(UIDS),np.float64)
fold_metrics={}
head_files={str(f):[] for f in range(5)}

for fold in range(5):
    val_idx=np.where(fold_of==fold)[0]
    train_gold=np.where(gold_study_mask&(fold_of!=fold))[0]
    base_train_idx=np.concatenate([non_gold,train_gold])
    seed_predictions=[]
    seed_scores={}

    for seed_slot,head_seed in enumerate(HEAD_SEEDS):
        random.seed(int(head_seed))
        np.random.seed(int(head_seed))
        torch.manual_seed(int(head_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(head_seed))

        train_idx=base_train_idx.copy()
        rng=np.random.default_rng(int(head_seed)+10007*fold)
        rng.shuffle(train_idx)

        train_ds=FeatureDataset(train_idx,targets,weights)
        val_ds=FeatureDataset(val_idx,gold,val_weights)

        generator=torch.Generator()
        generator.manual_seed(int(head_seed)+17*fold)

        train_dl=DataLoader(
            train_ds,batch_size=40,shuffle=True,num_workers=0,
            pin_memory=False,collate_fn=collate,generator=generator,
        )
        val_dl=DataLoader(
            val_ds,batch_size=32,shuffle=False,num_workers=0,
            pin_memory=False,collate_fn=collate,
        )

        model=LabelAwareHead().to(device)
        optimizer=torch.optim.AdamW(
            model.parameters(),lr=2e-4,weight_decay=2.5e-4
        )
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,T_max=16,eta_min=2e-5
        )

        best_score=-1.0
        best_state=None
        no_improve=0

        for epoch in range(16):
            model.train()
            train_loss=[]

            for feat,meta,mask,y,w,_ in train_dl:
                feat=feat.to(device,non_blocking=True)
                meta=meta.to(device,non_blocking=True)
                mask=mask.to(device,non_blocking=True)
                y=y.to(device,non_blocking=True)
                w=w.to(device,non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits=model(feat,meta,mask)
                loss=.86*soft_bce(logits,y,w)+.14*pair_rank_loss(logits,y,w)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
                train_loss.append(float(loss.detach().cpu()))

            scheduler.step()
            model.eval()
            pred=[]
            truth=[]

            with torch.inference_mode():
                for feat,meta,mask,y,w,_ in val_dl:
                    logits=model(
                        feat.to(device),
                        meta.to(device),
                        mask.to(device),
                    )
                    pred.append(torch.sigmoid(logits).cpu().numpy())
                    truth.append(y.numpy())

            pred=np.concatenate(pred)
            truth=np.concatenate(truth)
            score=macro_auc(truth,pred)

            print(
                f"fold={fold} seed={seed_slot} epoch={epoch+1:02d} "
                f"loss={np.mean(train_loss):.4f} GoldAUC={score:.4f}"
            )

            if np.isfinite(score) and score>best_score+1e-5:
                best_score=score
                no_improve=0
                best_state={
                    k:v.detach().cpu().clone()
                    for k,v in model.state_dict().items()
                }
            else:
                no_improve+=1

            if epoch>=7 and no_improve>=5:
                break

        if best_state is None:
            raise RuntimeError(
                f"fold {fold} seed {seed_slot}: no valid checkpoint"
            )

        model.load_state_dict(best_state)
        model.eval()
        pred=[]
        ids=[]

        with torch.inference_mode():
            for feat,meta,mask,y,w,ii in val_dl:
                logits=model(
                    feat.to(device),
                    meta.to(device),
                    mask.to(device),
                )
                pred.append(torch.sigmoid(logits).cpu().numpy())
                ids.append(ii.numpy())

        pred=np.concatenate(pred)
        ids=np.concatenate(ids)
        seed_predictions.append(pred)
        seed_scores[str(seed_slot)]=float(best_score)

        head_name=f"dinov5_convnext_head_fold{fold}_seed{seed_slot}.pt"
        torch.save(
            {
                "state":best_state,
                "fold":fold,
                "seed_slot":seed_slot,
                "seed":int(head_seed),
                "targets":TARGETS,
            },
            WORK/head_name,
        )
        head_files[str(fold)].append(head_name)

        del model,best_state,train_dl,val_dl,train_ds,val_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    bag_pred=np.mean(np.stack(seed_predictions,axis=0),axis=0)
    oof_sum[ids]+=bag_pred
    oof_count[ids]+=1.0

    fold_metrics[str(fold)]={
        "macro_auc":float(macro_auc(gold[ids],bag_pred)),
        "n_gold":int(len(val_idx)),
        "seed_macro_auc":seed_scores,
        "head_files":head_files[str(fold)],
    }

oof=np.full((len(UIDS),N_TARGETS),np.nan,np.float32)
covered=oof_count>0
oof[covered]=(oof_sum[covered]/oof_count[covered,None]).astype(np.float32)

gold_pred=oof[gold_indices]
gold_true=gold[gold_indices]
oof_macro=macro_auc(gold_true,gold_pred)
target_auc=per_target_auc(gold_true,gold_pred)

print("Gold58 OOF macro:",oof_macro)
print(json.dumps(target_auc,indent=2))


# %%%% CELL 7 [code] len=2578
# ----------------------- manifest + blend recommendation ----------------------
# Novelty is not enough; targets with weak Gold58 OOF get zero ensemble weight.
blend_beta={}
for t in TARGETS:
    auc=target_auc.get(t)
    if auc is None or not np.isfinite(auc) or auc<0.66:
        beta=0.0
    else:
        beta=float(np.clip(0.04+0.20*(auc-0.66),0.04,0.12))
    blend_beta[t]=beta

manifest={
    "schema":"dinov5_convnext_diversity_v2",
    "title":"RSNA Knee | DINOsaur V5? 🦖",
    "training_profile":"v3_fast_memmap_threaded",
    "seed":SEED,
    "targets":TARGETS,
    "backbone":"timm convnext_tiny.fb_in22k_ft_in1k -> non-gold soft-label MRI adaptation",
    "backbone_model_name":_CONVNEXT_MODEL_NAME,
    "offline_pretrained_source":str(CONVNEXT_PRETRAINED_PATH),
    "input":{
        "crop_mm":CROP_MM,
        "image_size":IMAGE_SIZE,
        "max_series":MAX_SERIES,
        "clip_quantiles":list(CLIP_Q),
        "order":"physical ImagePositionPatient projection",
    },
    "labels":{
        "sources":source_paths,
        "consensus":"median soft target",
        "gold_override":True,
        "pseudo_temperature":0.85,
    },
    "validation":{
        "gold_n":int(len(gold_indices)),
        "oof_macro_auc":float(oof_macro),
        "per_target_auc":target_auc,
        "fold_metrics":fold_metrics,
    },
    "recommended_main_beta":blend_beta,
    "weights":{
        "backbone":"dinov5_convnext_backbone.pt",
        "heads_by_fold":head_files,
        "heads":[name for fold in range(5) for name in head_files[str(fold)]],
    },
}

(WORK/"dinov5_convnext_manifest.json").write_text(
    json.dumps(manifest,indent=2,sort_keys=True)+"\n",
    encoding="utf-8",
)

oof_df=pd.DataFrame({"StudyInstanceUID":np.array(UIDS)[gold_indices]})
for j,t in enumerate(TARGETS):
    oof_df[t]=gold_pred[:,j]
oof_df.to_csv(WORK/"dinov5_convnext_gold58_oof.csv",index=False)

print(json.dumps(
    {"oof_macro_auc":oof_macro,"recommended_main_beta":blend_beta},
    indent=2,
))
print("\nSave this notebook version and attach its OUTPUT to the final V5 inference notebook.")


# FAST V3 training intermediates are intentionally not exported.
for _tmp in (FEATURE_BANK_BIN,META_BANK_BIN,FEATURE_LEN_PATH):
    try:
        Path(_tmp).unlink(missing_ok=True)
    except Exception as _exc:
        print("temporary feature-bank cleanup warning:",repr(_exc))

# Legacy feature-cache directory is also removed if it exists.
import shutil as _d5_shutil
_d5_shutil.rmtree(FEATURE_CACHE, ignore_errors=True)
print("Removed training-only feature cache:", FEATURE_CACHE)

