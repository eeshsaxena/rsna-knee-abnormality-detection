%%writefile /tmp/raptor_run.py
# Authorized scoring run: verify the selected checkpoint before inference.
RUN_INFERENCE = True
if not RUN_INFERENCE:
    raise SystemExit("Static notebook only. Attach the exact checkpoint and competition, enable a T4 GPU, and set RUN_INFERENCE = True to run inference.")

import hashlib
from pathlib import Path
import torch
if not torch.cuda.is_available():
    raise RuntimeError("This scored pipeline requires CUDA; CPU inference is not enabled.")
_weight_name = 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt'
_candidates = [
    Path('/kaggle/input/rsna-knee-publication-swa-weights-20261007') / _weight_name,
    Path('/kaggle/input/datasets/nartaa/rsna-knee-publication-swa-weights-20261007') / _weight_name,
    Path('/kaggle/input/rsna-knee-oai-original384-s42-pair') / _weight_name,
    Path('/kaggle/input/datasets/nartaa/rsna-knee-oai-original384-s42-pair') / _weight_name,
]
_weights = next((p for p in _candidates if p.is_file()), None)
if _weights is None:
    raise FileNotFoundError("Attach the exact selected weights; no alternative model is substituted.")
with _weights.open('rb') as _f:
    assert hashlib.file_digest(_f, 'sha256').hexdigest() == '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', "Checkpoint hash differs"


# %% [code]
#!/usr/bin/env python3
"""Knee MRI: twelve findings from a single model

This notebook takes a knee MRI study and scores twelve findings at once: ACL tear, MCL tear,
medial and lateral meniscus tears, osteoarthritis in the medial, lateral and patellofemoral
compartments, joint effusion, synovitis, a Baker's cyst, bone contusion and fracture. It scores
0.924 on the public leaderboard using one model, with no ensembling and no test-time augmentation.

This is the inference half of the work. The model was trained separately and its weights are
attached as a dataset, so this notebook only loads them and predicts:
https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense

Where the training labels came from

Worth saying up front, because it shapes everything else. The competition gives you 4,407 studies
but structured labels for only 58 of them. Every other study arrives with a free-text radiology
report and nothing more, so there is very little to train against out of the box.

The labels behind these weights were made by reading those reports with a language model and
turning each into twelve probabilities rather than twelve yes or no answers. A report that says a
tear is suspected becomes a number near 0.8, not a 1, which is a fairer target than forcing every
hedged sentence into a hard label. That yields 4,349 studies to train on. The 58 studies that came
with real labels were never trained on and are used to check the result honestly; the model reaches
0.9167 macro-AUC on them.

Building a fixed input from studies that are all shaped differently

The hard part of this competition is not the network, it is that no two studies look alike. A
study holds several DICOM series shot in different planes, the number of series varies, and the
number of slices in a series varies more. Anything that expects a fixed-size input has to be given
one.

The approach here is to fill five fixed slots per study, always in the same order, for a stack of
64 images:

  18 slices from a sagittal series, preferring a fluid-sensitive one
  14 slices from a second sagittal series, preferring one that is not fluid-sensitive
  12 slices from a coronal series, preferring a fluid-sensitive one
   8 slices from a second coronal series
  12 slices from an axial series

Preferring a fluid-sensitive series for some slots and not for others is deliberate. Fluid-
sensitive sequences show swelling, effusion and acute injury clearly, while the other sequences
show anatomy and cartilage better, and the twelve findings are split across both. If a study has
no series for a slot, the slot is left as zeros and the model is told to skip it rather than being
fed something misleading.

Within a series, slices are taken evenly across 6 to 94 percent of the stack rather than from the
middle. The outer slices are where the collateral ligaments and the lateral meniscus sit, and
cutting them was measurably costing accuracy on exactly those findings.

Every slice is cropped to a 140 mm box around the centre of the image using the pixel spacing from
the DICOM header, then resized to 336 pixels. Cropping by millimetres rather than by pixel count
matters: it means a knee occupies the same fraction of the frame whether the scan was acquired at
0.3 or 0.5 mm per pixel, so the model is not asked to learn scale differences that carry no medical
information.

How the model reads the stack

Three neighbouring slices are stacked into the three channels of one image. The network then sees
a little of what lies above and below the slice in the middle, which is most of the benefit of a 3D
model at the cost of a 2D one. Each of these three-slice windows is passed through a CoAtNet
backbone at 384 pixels.

The windows are combined with an attention layer that has separate weights for each of the twelve
findings. This is the part that matters most. A cruciate tear may be visible on two sagittal slices
while osteoarthritis is spread across many coronal ones, and a single pooled score forces those two
to share one notion of which slices are important. Giving each finding its own attention weights
lets each one draw on the slices that actually show it.

Running it

Scoring uses 42 windows per study. Inference runs in half precision and automatically retries a
study in full precision if it fails, so no study is ever dropped from the submission. The notebook
needs no internet: the backbone is loaded from the attached weights rather than downloaded.
"""
import os, sys, glob, time, json, gc
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading, queue
import torch, torch.nn as nn, torch.nn.functional as F
import timm
# T4 (Turing) cuDNN v9 has fp16/fp32 conv engines but NOT bf16 for these shapes
# ("GET was unable to find an engine..."); benchmark lets it pick a valid algo for
# the fixed (1,24,3,res,res) input.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ---- fixed config (must match training exactly) -----------------------------
INPUT_RES = 320
INPUT_MODE = "crop"
IMG = 384   # native 384 corpus: the model saw 384 px crops directly, no 336->384 upsample
CROP_MM = 140.0
# 96 slices per study at native 384 px. Must match the corpus the weights were trained on
# (cache/build_corpus96.py).
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18),
         ("Coronal", 0, 12), ("Axial", -1, 18)]
MAXS = sum(s[2] for s in SLOTS)                     # 96
K_EVAL = 94   # every window position of the 96-slice test volume (full coverage)
HDR_THREADS = 16   # per prep process: header reads are fuse-latency-bound, not CPU-bound
PX_THREADS = 8     # per prep process: pixel reads of the ~96 picked slices
QUEUE_MAX = 48     # studies in flight between the prep pool and the GPU consumers (RAM bound: ~14 MB each)
NORM = "imagenet"
FASTREAD_VARIANT = 'study_lut'   # kernel v3: research/fastread/readers.py, byte-identical to build_study
FR_HDR, FR_PX = 32, 16
_FASTREAD_B64 = "<B64>"
def _load_fastread():
    import base64 as _b, importlib.util as _iu
    p = '/kaggle/working/_fastread.py' if os.path.isdir('/kaggle/working') else os.path.join(os.path.dirname(os.path.abspath(__file__)), '_fastread.py')
    if not os.path.exists(p):
        open(p, 'wb').write(_b.b64decode(_FASTREAD_B64))
    spec = _iu.spec_from_file_location('_fastread', p); m = _iu.module_from_spec(spec); spec.loader.exec_module(m)
    return m
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Three arms: (weights filename, fallback arch, fallback res). ck carries arch+res too.
# Selected 2026-08-19 by greedy forward selection AND exhaustive subset search over a 7-arm
# panel on the 45-study gold set (phase2/blend_panel.py); both agree on this exact set.
# Singles: coatnet384 0.9025 | swinbase384 0.8825 | effv2l480 0.8716.
# Blend {coatnet+swin+effv2l} = 0.9068 (2-arm {coatnet+swin} = 0.9059, coatnet alone 0.9025).
# Dropped as redundant: cnn336 (0.8833, the former champion), cnbase384 (0.8754),
# cnlarge384 (0.8752), maxvit384 (0.8438).
#
# SINGLE ARM: coatnet_rmlp_2_rw_384 retrained on the EXPANDED 4,349-study corpus.
#
# Why one arm and not the 3-arm blend: on the live leaderboard CoAtNet alone scored 0.914 while
# every blend scored 0.914-0.915, so ensembling is worth ~+0.001 there -- the ~+0.010 it showed
# on the old 45-study gold set was gold-set noise. One arm is also 1/3 the kernel runtime.
#
# Corpus expansion: the corpus previously held 3,200 of the 4,349 labelled studies and only 45
# of the 58 gold studies. Rebuilt to 4,407 studies (+37.8% training data, 58-study gate).
#
# Measured on the 58-study gate (the incumbent re-scored on the SAME gate for a fair compare):
#   incumbent CoAtNet (3,155-study corpus) 0.8923
#   this model       (4,349-study corpus) 0.9054   (+0.0131, better in 92.7% of 2000 bootstraps)
# Biggest gains land on the findings that were capping us: Lateral Meniscus +0.071,
# Fracture +0.057, Lateral OA +0.048, Medial Meniscus +0.035, ACL +0.028.
ARMS = [   # our 5-fold CoAtNet (public recipe reproduced on our folds/labels), equal-weight rank-mean
    {"file": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0},
]


# ============================================================================
# Model -- verbatim from finetune_raptor.py
# ============================================================================
def build_backbone(arch, pretrained=False):
    # maxvit/maxxvit/coatnet are conv-attention hybrids: NO CLS token, NO interpolatable
    # pos-embed -> avg pool. The "vit" substring in "coatnet"/"maxvit" must NOT route them
    # down the ViT path (mirrors finetune_raptor.py exactly).
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3, img_size=INPUT_RES)
    if is_vit:
        kw.update(global_pool="token", dynamic_img_size=True)
    else:
        kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)


class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        # Bound backbone memory; the attention head still sees every window together.
        flat = x.flatten(0, 1)
        f = torch.cat([self.backbone(chunk) for chunk in flat.split(16)], dim=0)
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))


def load_model(pt_path, arch_default, res_default, device, ngpu=1):
    import hashlib
    with open(pt_path, "rb") as stream:
        observed_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    assert observed_sha == '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', "Bound checkpoint bytes differ"
    print(f"[weights] sha256 {observed_sha}", flush=True)
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", arch_default)
    assert int(ck.get("res", res_default)) == 384, "Expected original 384 checkpoint"
    ck_res = INPUT_RES
    bb = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device)
    # NOTE: DataParallel removed on purpose. On the full hidden test it drove a system-RAM OOM
    # (per-forward module replication over many studies); a single T4 handles K_EVAL=24 windows
    # fine. Arms are also run SEQUENTIALLY (see main) so peak RAM == one model, not two.
    del ck
    gc.collect()
    return model, ck_res


# ============================================================================
# Eval windowing -- verbatim from finetune_raptor.py StudyWindows (train=False)
# ============================================================================
def _eval_centers(mask, D, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


SLOT_SIZES = [26, 22, 18, 12, 18]
SAGITTAL_SLOTS = (0, 1)

def _slot_index(D):
    b = np.cumsum([0] + SLOT_SIZES)
    if b[-1] != D: raise RuntimeError(f'slot sizes sum to {b[-1]}, volume has {D} slices')
    s_ = np.searchsorted(b, np.arange(D), side='right') - 1
    rel = (np.arange(D) - b[s_]) / np.maximum(1, np.array(SLOT_SIZES)[s_] - 1)
    return s_, rel

def _mirror_tri(tri, cs, D):
    """final-g1-bundle MIRROR_TTA: true L/R mirror of (k,3,H,W) uint8 eval triplets (research/rnd4/mirror/mirror_f0.py). The slot of each
    window's CENTRE slice decides the axis: sagittal (slots 0,1) -> reverse the 3 slice-channels; coronal/axial (2,3,4) -> flip the width.
    Done on uint8 BEFORE the per-channel LUT, i.e. 'mirror the knee, then normalise'."""
    s_, _ = _slot_index(D)
    sag = torch.from_numpy(np.isin(s_[np.asarray(cs, dtype=np.int64)], SAGITTAL_SLOTS)).to(tri.device).view(-1, 1, 1, 1)
    return torch.where(sag, tri.flip(1), tri.flip(-1))

def infer_pair_probs(model, plain, mirrored, device):
    assert plain.shape==mirrored.shape==(94,3,320,320)
    assert all(not m.training for m in model.modules()), "TTA must not update BN or enable dropout"
    p = infer_probs(model, plain, device)
    q = infer_probs(model, mirrored, device)
    return 0.5 * (p + q)

def eval_windows(vol, mask, k, res, norm=NORM, mirror=False):
    D = vol.shape[0]
    cs = _eval_centers(mask, D, k)
    wins = np.empty((len(cs), 3, res, res), np.float32)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0)
        if mirror:
            tri = _mirror_tri(torch.from_numpy(tri)[None], [c], D)[0].numpy()
        tri = tri.astype(np.float32) / 255.0
        assert tri.shape[-2:] == (IMG, IMG) and res == INPUT_RES
        lo = (IMG - res) // 2
        tri = tri[:, lo:lo + res, lo:lo + res]
        t = torch.from_numpy(np.ascontiguousarray(tri))
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear",
                              align_corners=False)[0]
        wins[j] = t.numpy()
    x = torch.from_numpy(wins)
    if norm == "imagenet":
        x = (x - _MEAN) / _STD
    return x


_WLUT = {}
def gpu_windows(vol, mask, k, res, dev, norm=NORM, mirror=False):
    """Same windows as eval_windows, built on the GPU from the uint8 volume. Values come from a per-channel
    256-entry table computed on the CPU with eval_windows' own float32 ops, so they are bit-identical."""
    D = vol.shape[0]
    cs = np.array([max(1, min(c, D - 2)) for c in _eval_centers(mask, D, k)])
    tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(dev)   # (k,3,IMG,IMG) uint8
    if mirror:
        tri = _mirror_tri(tri, cs, D)
    assert tri.shape[-2:] == (IMG, IMG) and res == INPUT_RES
    lo = (IMG - res) // 2
    tri = tri[:, :, lo:lo + res, lo:lo + res]
    key = (str(dev), norm)
    if key not in _WLUT:
        v = torch.from_numpy((np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0))
        t = torch.stack([v, v, v], 0).view(3, 256, 1)
        if norm == 'imagenet':
            t = (t - _MEAN.view(3, 1, 1)) / _STD.view(3, 1, 1)
        _WLUT[key] = t.view(3, 256).contiguous().to(dev)
    L = _WLUT[key]
    x = torch.stack([L[c][tri[:, c].long()] for c in range(3)], 1)                   # exact float32 values
    if x.shape[-1] != res:
        if norm == 'imagenet':   # resize must happen BEFORE normalisation, as in eval_windows
            L0 = _WLUT.setdefault((str(dev), 'none'), torch.from_numpy(np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0).to(dev))
            x = L0[tri.long()]
            x = F.interpolate(x, size=(res, res), mode='bilinear', align_corners=False)
            x = (x - _MEAN.to(dev)) / _STD.to(dev)
        else:
            x = F.interpolate(x, size=(res, res), mode='bilinear', align_corners=False)
    return x


@torch.no_grad()
def infer_probs(model, xwins, device):
    assert str(device).startswith("cuda"), "Qualified inference requires CUDA/fp16"
    x = xwins.unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(x).float()
        assert torch.isfinite(logits).all(), "Nonfinite fp16 logits"
        o = torch.sigmoid(logits)
    return o[0].cpu().numpy()


def rankpct(x):                                   # per-column percentile rank in [0,1]
    order = x.argsort(0).argsort(0).astype(np.float64)
    return order / max(1, (x.shape[0] - 1))


# ============================================================================
# Preprocessing -- verbatim from kprep2/dino_preprocess.py, retargeted to TEST
# ============================================================================
def _make_reader():
    import pydicom, cv2
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def _hdr(f):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop = getattr(h, 'ImageOrientationPatient', None)
            ipp = getattr(h, 'ImagePositionPatient', None)
            if iop is not None and ipp is not None and len(iop) == 6:
                r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
            else:
                pos = float(getattr(h, 'InstanceNumber', 0) or 0)
            ps = getattr(h, 'PixelSpacing', None); ps = float(ps[0]) if ps is not None else 0.5
            return (pos, f, ps, True)
        except Exception:
            return (0.0, f, 0.5, False)

    def order_and_meta(sdir):
        # Header reads are latency-bound on Kaggle's mounted competition data (~240k opens on the hidden
        # set); a thread pool hides that latency. Records and sort are unchanged -> identical ordering.
        fs = glob.glob(sdir + "/*.dcm")
        with ThreadPoolExecutor(max_workers=HDR_THREADS) as ex:
            out = list(ex.map(_hdr, fs))
        recs = [(pos, f, ps) for pos, f, ps, _ in out]; ps_list = [ps for _, _, ps, ok in out if ok]
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return [(f, ps) for _, f, ps in recs], med_ps

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps):
        h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)

    return order_and_meta, read_px, mm_crop_resize


def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study(sid, ser_records, tsdir, reader):
    order_and_meta, read_px, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set()
    for plane, fluid, k in SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k; continue
        used.add(r['SeriesInstanceUID'])
        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k; continue
        # wide span: the collateral ligaments and lateral meniscus live in the
        # peripheral slices the old 0.15-0.85 crop threw away. Must match the corpus
        # the weights were trained on (build_corpus80.py, SPAN).
        n = len(files); lo, hi = int(n * 0.02), int(n * 0.98) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        sel = [files[min(p, n - 1)] for p in picks]
        def _px(fp_ps):
            fp, ps = fp_ps
            try: return read_px(fp), ps
            except Exception: return None, med_ps
        with ThreadPoolExecutor(max_workers=PX_THREADS) as ex:
            got = list(ex.map(_px, sel))                    # order preserved
        arrs = [a for a, _ in got]; pss = [ps for _, ps in got]
        valid = [a for a in arrs if a is not None]
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if idx >= MAXS: break
            if a is None: idx += 1; continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            vol[idx] = (aw * 255).astype(np.uint8); idx += 1
        if idx >= MAXS: break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask



# ---- parallel preprocessing helpers (module level so they pickle under fork) ----
_SER = None; _TSDIR = None; _READER = None
def _pre_init():
    global _READER
    _READER = _load_fastread().make_builder(FASTREAD_VARIANT, IMG, SLOTS, hdr_threads=FR_HDR, px_threads=FR_PX)
def _pre_one(sid):
    try:
        vol, mask = _READER(sid, _SER, _TSDIR)
    except Exception as e:
        print(f"  [pre] {sid[:16]} FAILED ({type(e).__name__}: {e})", flush=True)
        vol = np.zeros((MAXS, IMG, IMG), np.uint8); mask = np.zeros(MAXS, np.uint8)
    return sid, vol, mask

# ============================================================================
# Test-root discovery + weights + main
# ============================================================================
def find_test_root():
    cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
             "/kaggle/input/rsna-knee-abnormality-detection"]
    for b in cands:
        if os.path.exists(b + "/test.csv"):
            return b
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f and (os.path.isdir(d + "/test_series") or os.path.isdir(d + "/test_images")):
            return d
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f:
            return d
    raise RuntimeError("no test root under /kaggle/input")


def find_weight_file(fname):
    # direct dataset mounts first; NEVER recursive-glob the competitions DICOM tree.
    direct = [f"/kaggle/input/raptor-knee-arms/{fname}",
              f"/kaggle/input/raptor-knee-arms/1/{fname}",
              f"/kaggle/input/raptor-cnn336/{fname}"]
    for p in direct:
        if os.path.exists(p):
            return p
    for d in sorted(glob.glob("/kaggle/input/*/")):
        # skip the competition data wherever Kaggle mounts it (/kaggle/input/competitions/<slug> OR /kaggle/input/<slug>):
        # recursively globbing its DICOM tree hung a commit run for 20+ min (2026-09-26)
        if "competition" in d.lower() or "abnormality-detection" in d or any(
                os.path.exists(os.path.join(d, x)) for x in ("test_series", "train_images", "train.csv", "test.csv")):
            continue
        hits = glob.glob(os.path.join(d, "**", fname), recursive=True)
        if hits:
            return hits[0]
    raise RuntimeError(f"{fname} not found under /kaggle/input")


def main():
    import pandas as pd
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = torch.cuda.device_count()
    print(f"device {dev} | gpus {ngpu} | torch {torch.__version__}", flush=True)

    ROOT = find_test_root()
    tsdir = ROOT + "/test_series"
    if not os.path.isdir(tsdir):
        tsdir = ROOT + "/test_images"
    print("test root:", ROOT, "| series dir:", tsdir, flush=True)

    test = pd.read_csv(ROOT + "/test.csv"); test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
    test_ids = test["StudyInstanceUID"].tolist()
    tser = pd.read_csv(ROOT + "/test_series.csv")
    tser["StudyInstanceUID"] = tser["StudyInstanceUID"].astype(str)
    tser["SeriesInstanceUID"] = tser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID")}
    print(f"test studies {len(test_ids)} | test series {len(tser)}", flush=True)

    sub_cols = ["StudyInstanceUID"] + LAB
    ssub = os.path.join(ROOT, "sample_submission.csv")
    if os.path.exists(ssub):
        sub_cols = list(pd.read_csv(ssub, nrows=1).columns)

    N = len(test_ids); A = len(ARMS)
    arm_probs = [np.full((N, len(LAB)), 0.5, np.float32) for _ in range(A)]
    pos = {sid: i for i, sid in enumerate(test_ids)}

    # ---- 2. GPU consumers: one thread per GPU, every arm loaded on every GPU (per-study batch of 1,
    #         so the math is identical to the sequential kernel; only the wall-clock changes) ----
    xla = None
    try:
        import torch_xla; import torch_xla.core.xla_model as xm
        dev = torch_xla.device(); xla = xm
        print("device: XLA/TPU", dev, flush=True)
    except Exception as e:
        print("XLA unavailable:", type(e).__name__, str(e)[:80], "| device", dev, flush=True)
    devs = [f"cuda:{g}" for g in range(ngpu)] if (xla is None and ngpu > 0) else [dev]
    models = {}
    for d in devs:
        models[d] = []
        for a, arm in enumerate(ARMS):
            model, res = load_model(find_weight_file(arm["file"]), arm["arch"], arm["res"], d)
            models[d].append((model, res))
        print(f"[gpu {d}] loaded {A} arm(s) | {time.time()-t0:.0f}s", flush=True)
    q = queue.Queue(maxsize=QUEUE_MAX); done_n = [0]; lock = threading.Lock()

    def consumer(d):
        while True:
            item = q.get()
            if item is None: q.task_done(); return
            sid, vol, mask = item; i = pos[sid]
            try:
                for a, (model, res) in enumerate(models[d]):
                    xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM) if xla is not None else gpu_windows(vol, mask, K_EVAL, res, d)
                    assert xla is None, "Anatomical L/R TTA requires CUDA/fp16"
                    xm = gpu_windows(vol, mask, K_EVAL, res, d, norm=NORM, mirror=True)
                    arm_probs[a][i] = infer_pair_probs(model, xw, xm, d)
                    del xm
            except Exception as e:
                print(f"  [gpu {d}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)
            with lock:
                done_n[0] += 1
                if done_n[0] % 200 == 0 or done_n[0] == N:
                    print(f"  [gpu] {done_n[0]}/{N} | {time.time()-t0:.0f}s", flush=True)
            q.task_done()

    threads = [threading.Thread(target=consumer, args=(d,), daemon=True) for d in devs]
    for th in threads: th.start()

    # ---- 3. prep every study ONCE in a process pool and stream it to the GPUs as it lands ----
    import multiprocessing as mp
    global _SER, _TSDIR, _READER
    _SER, _TSDIR = SER, tsdir
    nproc = max(4, min(96, (os.cpu_count() or 8) - 8))
    t_pre = time.time()
    with mp.get_context("fork").Pool(nproc, initializer=_pre_init) as pool:
        for i, (sid, vol, mask) in enumerate(pool.imap_unordered(_pre_one, test_ids, chunksize=2)):
            q.put((sid, vol, mask))
            if (i + 1) % 200 == 0 or i + 1 == N:
                print(f"  [pre] {i+1}/{N} | {time.time()-t0:.0f}s", flush=True)
    print(f"[pre] done: {N} studies in {time.time()-t_pre:.0f}s (procs {nproc} x fastread {FASTREAD_VARIANT} {FR_HDR}/{FR_PX} threads)", flush=True)
    for _ in devs: q.put(None)
    q.join()
    for th in threads: th.join()
    print(f"[gpu] done: {done_n[0]}/{N} studies on {len(devs)} device(s) | {time.time()-t0:.0f}s", flush=True)
    _raw = pd.DataFrame(arm_probs[0].astype(np.float32), columns=LAB)   # [blend] raw probabilities; a failed study stays 0.5
    _raw.insert(0, "StudyInstanceUID", test_ids)
    _raw.to_csv("/kaggle/working/raptor_raw.csv", index=False)
    for d in devs:
        for m, _ in models[d]: del m
    models.clear(); gc.collect()

    # WEIGHTED rank-mean blend across the test set, per finding (the offline recipe).
    # Weights come from ARMS[*]["w"] and are normalised here, so dropping/adding an arm can
    # never silently change the scale. Falls back to equal weights if none are declared.
    _w = np.array([float(a.get("w", 1.0)) for a in ARMS], dtype=np.float64)
    _w = _w / _w.sum()
    print(f"[blend] weighted rank-mean w={dict(zip([a['file'] for a in ARMS], _w.round(4)))}", flush=True)
    ranks = np.tensordot(_w, np.stack([rankpct(np.clip(p, 0, 1)) for p in arm_probs]),
                         axes=(0, 0))                                          # (N,12) in [0,1]
    if not np.isfinite(ranks).all():
        ranks[~np.isfinite(ranks)] = 0.5

    sub = pd.DataFrame(ranks.astype(np.float32), columns=LAB)
    sub.insert(0, "StudyInstanceUID", test_ids)
    sub = sub[sub_cols]
    assert list(sub.columns) == sub_cols, "column order drift"
    assert sub["StudyInstanceUID"].tolist() == test_ids, "row identity drift"
    assert np.isfinite(sub[LAB].values).all()
    out = "/kaggle/working/raptor_sub.csv"   # [blend] Raptor stage output; the blend cell writes submission.csv
    sub.to_csv(out, index=False)
    print("wrote", out, "|", len(sub), "rows x", len(sub.columns), "cols", flush=True)
    print(sub.head().to_string(index=False), flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
