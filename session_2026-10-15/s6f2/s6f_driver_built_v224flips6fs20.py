#!/usr/bin/env python3
# s6f_driver.py [eeshsaxena V224-FLIP-S6F; not in any public notebook]. Written by build_v224s6f.py from s6f_driver_src.py.
# The B path ONLY of nartaa/rsna-knee-s6fall-solo (public, .950 solo): every "VERBATIM" block below is copied byte for byte from
# its version 2 source (sha256 6e87350d..., the run whose public commit output is our parity reference; its code tree, model and
# worker arguments are the ones version 1 scored with, each file checked against the same sha256 manifest). The builder proves
# each block equal to the pulled source (ast segment, byte for byte). NOT included: remove_submission(), the 8 h hang guard
# (_watchdog / arm_guard / faulthandler backstop), component A, the blend, the parity mode, and every "HARD ERROR" of a B failure.
# Our parts (marked [s6f]): paths, the argument parser, the environment-isolation probe, run_b_s6f (= nartaa's run_b with a
# deadline that cuts instead of raising, partial coverage allowed, a killed part read from its predictions.csv), main().
# This process never touches /kaggle/working/submission.csv: it writes only under --out (default /kaggle/working/s6f) and --tmp.
# The notebook cell that starts it owns the hard deadline and kills every S6F process group at it. No em dashes.
import argparse, hashlib, json, os, shutil, signal, subprocess, sys, tarfile, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd

# ---- [s6f] paths (nartaa: KROOT /kaggle, OUT /kaggle/working/k1, TMP /tmp/k1); set by configure() --------------------------
KROOT = Path("/kaggle")
INPUT, WORK = KROOT / "input", KROOT / "working"
OUT = WORK / "s6f"
TMP = Path("/tmp/s6f/work")
CODE_MANIFEST_SHA256 = "b66c574b5c0c13ccaa49d1219d41a8781431166186ef83e78cab6a9ba6b02d22"   # [s6f] code_manifest.json of the scored run (nartaa's commit receipt)
R7_ROOTS = ("/tmp/probe", "/kaggle/temp/probe")     # [s6f] R7's code / wheel dirs (V224 cell 6): never on our path

# ---- VERBATIM constants of nartaa's kernel (B path) -------------------------------------------------------------------------
COMP = "rsna-knee-abnormality-detection"
DS_SLUG = "rsna-knee-s6fall-weights"                         # public: our weights + Apache-2.0 code extras
HELIOS_SLUG = "rsna-orthofoundation-fold0-epoch7-20261004"   # public: Heliosli's anatomy.pt + cloud_code (attached, not copied)
EXTRAS_FILE = "s6_extras.bin"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
B_MODEL = {"file": "E_prior_s42_fall.pt", "source_file": "E_prior_s42_fall.pt", "sha256": "0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12", "run_config": "E_prior_s42_fall.run_config.json", "done": "E_prior_s42_fall.DONE.json", "arm": "E_prior", "expect_epochs": 16, "expect_train": 4263}
ANATOMY_FILE, ANATOMY_SHA256 = "anatomy.pt", "5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e"
PAYLOAD_FILE, PAYLOAD_MAGIC = "k1_code_payload.bin", b"K1PAYLOAD1\n"
SOURCE_SHA = {                                                 # the 0.954 notebook's SOURCE_SHA = S6F0/s6common.SOURCE_SHA
    "px_cache.py": "b13e38a7ad3bd496f8027e02209653e224b0926db4f0853eeb0d076a2d6e2632",
    "compact_train.py": "14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80",
    "compact_infer.py": "34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a",
    "probe_infer.py": "a6cc31448019c84be5868d0a2ff1d99719b23b0f86110efc03efff3ce6bd6980",
    "knee_anchor_bank_v2.py": "e564807a129dfcd3c984348917436ee6083066a4d685340587ebd1d3ea14f309",
    "side_common.py": "bbcc5d027834e6a6fd48c4a8cf228e25661b7522c792587c840fe4d7cd5c0566",
}
B_RECEIPT_SOURCES = ("px_cache.py", "compact_train.py", "compact_infer.py", "probe_infer.py", "knee_anchor_bank_v2.py")
CODE_SHA = {                                                   # the 0.954 notebook's CODE_SHA (cloud_code/training pins)
    "native_cache.py": "c53ce8985406b3b918d9e821d43518479e89d9f946daebd817f378e3949718b9",
    "native_data.py": "f59afe205c8e193369fa3ccd22edcce8df3b293205fc48ddb1dd45169ee43b36",
    "input_provenance.py": "b3a22500c7f976f6ee040239ada5cacecf2326861cb49af1c7d8cfdeeb55fcda",
    "intensity_mapping.py": "19be2d8cbd6f8971c1288d7512a8cca41d143470c2ba6e9b9096fde035fa9d41",
    "coronal_views.py": "841f216d2682ae2c0f8afd82fc902b9fa74924d7145b952069dea8b0bbd03c40",
    "exact_duplicates.py": "51b38db21ce9c5fe2b6278a835983414756b61ac6af81aba380b44b41b075897",
    "dicom_geometry.py": "12aa501f9704aef15b7a3bbd560951acb6d95b89b7de0141d6cf89fc3f051ab1",
    "common.py": "8283fa1f8e7ef8cfacbba52df1715636be3c9ad688e39912814e7d5bcf57105f",
}
B_GEOMETRY = {"backbone": "convnext_small", "size": 320, "crop": 288, "views": "v4", "slices": 3, "phys_mm": 3.5,
              "eval_stride": 1, "chunk": 192, "item_timeout": 900}             # = S6F0/oof_step.py, checked vs run_config.json
B_AMP = "fp16"
B_WORKERS = 2                    # DataLoader workers per GPU process (Heliosli's 0.954 notebook: 2; laptop OOF: 3, speed only)
EXIT_GRACE_S = 120               # outputs complete (A: DONE + submission.csv; B: WORKER_DONE + receipt + logits) but no exit yet
B_DEADLINE_MARGIN_S = 20 * 60    # compact_infer --deadline = B start + B budget - this (> --item-timeout 900 s: a worker hang at
B_RETRY_MIN_S = 10 * 60          # a fallback retry runs only if at least this much of the B budget is left
HEARTBEAT_S = 300
MODE = "submit"                  # [s6f] nartaa's MODE value (prepare_inputs reads it; parity mode is not ported)

T0 = time.time()
_LOG = None
_CHILDREN = {}                   # pgid -> name
_RECEIPT = {"tag": "s6f", "stages": {}, "gpu": {}}


# ---- VERBATIM helpers ---------------------------------------------------------------------------------------------------------
class K1Error(RuntimeError):
    pass


class K1Timeout(K1Error):
    """a stage hit its time cap, or (A attempt 1) its log stalled"""


def log(*a):
    msg = "%s +%6.0fs %s" % (time.strftime("%H:%M:%S", time.gmtime()), time.time() - T0, " ".join(str(x) for x in a))
    print(msg, flush=True)
    if _LOG is not None:
        try:
            _LOG.write(msg + "\n"); _LOG.flush()
        except Exception:
            pass


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str)); os.replace(tmp, path)


def tail(path, n=3000):
    try:
        return Path(path).read_text(errors="replace")[-n:]
    except Exception as e:
        return "(no log: %r)" % e


def gpu_snapshot():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        return [l.strip() for l in r.stdout.splitlines() if l.strip()] or [r.stderr.strip()[:200]]
    except Exception as e:
        return ["nvidia-smi failed: %r" % e]


def mem_available_mb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


def group_rss_mb(pgid):
    tot = 0
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            st = open("/proc/%s/stat" % d).read()
            f = st[st.rindex(")") + 2:].split()
            if int(f[2]) == pgid:
                tot += int(f[21]) * os.sysconf("SC_PAGE_SIZE")
        except Exception:
            continue
    return tot // (1 << 20)


def kill_group(pgid):
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    _CHILDREN.pop(pgid, None)


def kill_all_children():
    for pgid in list(_CHILDREN):
        kill_group(pgid)


def stage(name):
    _RECEIPT["stages"][name] = {"start_s": round(time.time() - T0, 1)}
    _RECEIPT["gpu"][name + ".start"] = gpu_snapshot()
    log("[stage] %s start | gpu %s | MemAvailable %d MB" % (name, _RECEIPT["gpu"][name + ".start"], mem_available_mb()))


def stage_done(name, **info):
    s = _RECEIPT["stages"][name]
    s["seconds"] = round(time.time() - T0 - s["start_s"], 1)
    s.update(info)
    _RECEIPT["gpu"][name + ".end"] = gpu_snapshot()
    log("[stage] %s done in %.0fs | gpu %s" % (name, s["seconds"], _RECEIPT["gpu"][name + ".end"]))


def _size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return -1


def run_group(specs, timeout_s, what):
    """Start every spec {name, cmd, env, cwd, log[, done_when, stall_s]} in its own session; poll until all exit, one fails, or
    timeout_s passes.  A failure, a stall or the timeout kills every group (DataLoader / Pool children included).
    done_when(): the child's outputs are complete; if it then has not exited within EXIT_GRACE_S (an interpreter-exit hang), its
    group is killed and the run counts as a success.  stall_s: the log has not grown for that long = K1Timeout.
    Returns [(name, rc, seconds, note)]."""
    procs, t_start, last_hb, last_check = [], time.time(), time.time(), 0.0
    try:
        for s in specs:
            fh = open(s["log"], "w")
            p = subprocess.Popen(s["cmd"], env=s["env"], cwd=s.get("cwd"), stdout=fh, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
            _CHILDREN[p.pid] = s["name"]
            procs.append({"s": s, "p": p, "fh": fh, "t1": time.time(), "size": -1, "changed": time.time(), "done_at": None})
            log("[%s] started %s pid %d" % (what, s["name"], p.pid))
        done = {}
        while len(done) < len(procs):
            now_ = time.time()
            check = now_ - last_check >= 5
            if check:
                last_check = now_
            for r in procs:
                s, p = r["s"], r["p"]
                if s["name"] in done:
                    continue
                rc, note = p.poll(), ""
                if rc is None and check and s.get("done_when") is not None:
                    try:
                        complete = bool(s["done_when"]())
                    except Exception:
                        complete = False
                    if complete and r["done_at"] is None:
                        r["done_at"] = now_
                    if r["done_at"] is not None and now_ - r["done_at"] > EXIT_GRACE_S and p.poll() is None:
                        kill_group(p.pid)
                        rc, note = 0, "outputs complete, no exit after %ds: group killed, accepted" % EXIT_GRACE_S
                        log("[%s] %s: %s" % (what, s["name"], note))
                if rc is not None:
                    done[s["name"]] = (rc, round(time.time() - r["t1"], 1), note)
                    r["fh"].close(); kill_group(p.pid)
                    log("[%s] %s exited rc %d after %.0fs" % (what, s["name"], rc, time.time() - r["t1"]))
                    if rc != 0:
                        for r2 in procs:
                            if r2["s"]["name"] not in done:
                                kill_group(r2["p"].pid); log("[%s] killed %s (a sibling failed)" % (what, r2["s"]["name"]))
                        raise K1Error("%s: %s exited %d; log tail:\n%s" % (what, s["name"], rc, tail(s["log"])))
                    continue
                if check and s.get("stall_s"):
                    sz = _size(s["log"])
                    if sz != r["size"]:
                        r["size"], r["changed"] = sz, now_
                    elif now_ - r["changed"] > s["stall_s"]:
                        for r2 in procs:
                            if r2["s"]["name"] not in done:
                                kill_group(r2["p"].pid)
                        raise K1Timeout("%s: %s stalled: log unchanged for %ds (at %d B); log tail:\n%s" % (
                            what, s["name"], s["stall_s"], sz, tail(s["log"], 1500)))
            if len(done) == len(procs):
                break
            if time.time() - t_start > timeout_s:
                for r in procs:
                    if r["s"]["name"] not in done:
                        kill_group(r["p"].pid)
                raise K1Timeout("%s: timeout after %ds; still running %s" % (
                    what, timeout_s, [r["s"]["name"] for r in procs if r["s"]["name"] not in done]))
            if time.time() - last_hb >= HEARTBEAT_S:
                last_hb = time.time()
                for r in procs:
                    if r["s"]["name"] in done:
                        continue
                    prog = [l for l in tail(r["s"]["log"], 4000).splitlines() if l.startswith("{") or "[gpu]" in l or "[pre]" in l][-1:]
                    log("[heartbeat] %s %s | %.0fs | rss %d MB | MemAvailable %d MB | gpu %s | %s" % (
                        what, r["s"]["name"], time.time() - r["t1"], group_rss_mb(r["p"].pid), mem_available_mb(), gpu_snapshot(),
                        prog[0][:300] if prog else "(no progress line yet)"))
            time.sleep(2)
        return [(r["s"]["name"],) + done[r["s"]["name"]] for r in procs]
    finally:
        for r in procs:
            if r["p"].poll() is None:
                kill_group(r["p"].pid)
            try:
                r["fh"].close()
            except Exception:
                pass



# ---- VERBATIM environment and inputs (prepare_inputs, probe_b_stack and their helpers) ---------------------------------------
PROBE = r'''
import json, sys, importlib.util
out = {"python": sys.version.split()[0]}
import torch
out.update(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
           cuda_available=torch.cuda.is_available(), n_gpu=torch.cuda.device_count())
if torch.cuda.is_available():
    out["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    out["capability"] = list(torch.cuda.get_device_capability(0))
    out["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
for m in ("numpy", "pandas", "timm", "pydicom", "cv2", "scipy", "safetensors"):
    try:
        out[m] = getattr(__import__(m), "__version__", "?")
    except Exception as e:
        out[m] = "MISSING " + type(e).__name__
for m in ("requests", "kagglesdk", "pylibjpeg", "gdcm"):
    out["has_" + m] = importlib.util.find_spec(m) is not None
print("K1_ENV " + json.dumps(out), flush=True)
'''
def probe_env():
    r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
    line = [l for l in r.stdout.splitlines() if l.startswith("K1_ENV ")]
    if r.returncode != 0 or not line:
        raise K1Error("environment probe failed rc %d: %s %s" % (r.returncode, r.stdout[-1500:], r.stderr[-1500:]))
    return json.loads(line[0][len("K1_ENV "):])


def find_mount(slug, marker):
    cands = [INPUT / slug, INPUT / "competitions" / slug]
    cands += list(INPUT.glob("datasets/*/" + slug)) + list(INPUT.glob("datasets/*/" + slug + "/*"))
    cands += list(INPUT.glob("*/" + slug)) + list(INPUT.glob(slug + "/*"))
    found = []
    for p in cands:
        if (p / marker).is_file() and p.resolve() not in [f.resolve() for f in found]:
            found.append(p)
    if len(found) != 1:
        raise K1Error("cannot resolve %s (marker %s): %s" % (slug, marker, found))
    return found[0]


def safe_rel(name):
    p = Path(name)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise K1Error("unsafe payload member %r" % name)
    return p


def extract_payload(path, target, members):
    """The code payload = PAYLOAD_MAGIC + an uncompressed tar (opaque to Kaggle's archive handling); exact member set, sizes, sha."""
    if target.exists():
        shutil.rmtree(target)
    with open(path, "rb") as f:
        if f.read(len(PAYLOAD_MAGIC)) != PAYLOAD_MAGIC:
            raise K1Error("payload magic mismatch")
        with tarfile.open(fileobj=f, mode="r:") as tar:
            ms = tar.getmembers()
            if {m.name for m in ms} != set(members) or len(ms) != len(members):
                raise K1Error("payload member set differs from MANIFEST")
            for m in ms:
                safe_rel(m.name)
                if not m.isfile() or m.size != members[m.name]["bytes"]:
                    raise K1Error("payload member %s: not a regular file or size differs" % m.name)
            for m in ms:
                dst = target / safe_rel(m.name)
                dst.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src, open(dst, "xb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
    for name, rec in members.items():
        if sha256_file(target / name) != rec["sha256"]:
            raise K1Error("payload member %s sha differs" % name)


LAUNCHER = r'''"""K1 component-B launcher = S6F0/infer_s6f0.py with Kaggle paths.  Runs Heliosli's compact_infer.main() UNCHANGED.
As on the laptop: sys.path = [cloud_code, heliosli, ...] and the training.download_data stub (only the chunked-archive reader
uses it; we read the DICOM directory), installed in the main process AND in every spawn worker (CStudies._setup).
Three K1 guards, none of which changes a prediction on the laptop's path (each acts only where that path would fail):
- distutils: the vendored MONAI 0.6.0 imports distutils.util.strtobool (Python >= 3.12 has no stdlib distutils; setuptools'
  shim provides it on the laptop).  Only if that import fails, a module holding the stdlib's strtobool is installed.
- setup failure (review M1): compact_infer's _setup sets self.native before it loads the SKM-TEA V-Net, so a failed load left a
  worker scoring every later study with field-centre views and no anchor, silently.  Here a failed _setup resets self.native
  (the next study re-runs it; probe_infer.install_native is memoised, the one step that is not re-entrant) and re-raises (this
  study becomes a fallback, retried); a scored item without an anchor is turned into a failed item (fallback, retried).
K1_PX_DUMP (parity mode only): every item's pixel / meta hashes are appended to <dir>/px_<pid>.jsonl; the item is not changed."""
import hashlib, json, os, sys, types
SRC = __SRC__
CODE = __CODE__
ANCHOR_VIEWS = ("v3k", "v4")


def _strtobool(val):
    """distutils.util.strtobool (CPython 3.11 / setuptools._distutils), unchanged semantics."""
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"invalid truth value {val!r}")


def install_distutils():
    if "distutils.util" in sys.modules:
        return
    try:
        import distutils.util  # noqa: F401  (setuptools' shim on Python >= 3.12)
    except Exception:
        d, u = types.ModuleType("distutils"), types.ModuleType("distutils.util")
        d.__path__ = []
        u.strtobool = _strtobool
        d.util = u
        sys.modules["distutils"], sys.modules["distutils.util"] = d, u
        os.environ["K1_DISTUTILS_SHIM"] = "1"


def install_paths():
    sys.dont_write_bytecode = True
    for p in (SRC, CODE):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "training.download_data" not in sys.modules:
        m = types.ModuleType("training.download_data")
        m.ChunkedArchive = None
        sys.modules["training.download_data"] = m
    install_distutils()


def memoise_install_native():
    import probe_infer as PI
    if getattr(PI, "_k1_memo", False):
        return
    orig, cache = PI.install_native, {}

    def install_native(lean=True):
        if lean not in cache:
            cache[lean] = orig(lean)
        return cache[lean]
    PI.install_native = install_native
    PI._k1_memo = True


install_paths()
import compact_infer as CI

assert os.path.dirname(os.path.abspath(CI.__file__)) == SRC, CI.__file__
PX_DUMP = os.environ.get("K1_PX_DUMP")


class LocalCStudies(CI.CStudies):
    def _setup(self):
        install_paths()
        memoise_install_native()
        try:
            super()._setup()
            if self.views in ANCHOR_VIEWS and self.bank is None:
                raise RuntimeError("K1: the SKM-TEA anchor bank did not load")
        except BaseException:
            self.native = None
            raise

    def __getitem__(self, i):
        out = super().__getitem__(i)
        if out.get("error") is None and self.views in ANCHOR_VIEWS and out.get("anchor") is None:
            out = dict(out, error="K1: item built without an SKM-TEA anchor", images=out["images"][:0], meta=out["meta"][:0])
        if PX_DUMP:
            try:
                img = out["images"].numpy()
                meta = out["meta"].numpy()
                rec = {"uid": out["uid"], "error": out["error"], "n": int(len(img)),
                       "img_sha256": hashlib.sha256(img.tobytes()).hexdigest(),
                       "meta_sha256": hashlib.sha256(meta.astype("float32").tobytes()).hexdigest(),
                       "win_sha1_10": [hashlib.sha1(w.tobytes()).hexdigest()[:10] for w in img],
                       "img_mean": float(img.mean()) if len(img) else None, "anchor": out.get("anchor"),
                       "seconds": out.get("seconds")}
            except Exception as exc:
                rec = {"uid": out.get("uid"), "hash_error": repr(exc)[:200]}
            with open(os.path.join(PX_DUMP, "px_%d.jsonl" % os.getpid()), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        return out


CI.CStudies = LocalCStudies

if __name__ == "__main__":
    CI.main()
'''
B_PROBE = r'''
import importlib.util, json, os, sys, time
launcher, anatomy, model, backbone, crop, slices = sys.argv[1:7]
t0 = time.time()
out = {}
try:
    import setuptools
    out["setuptools"] = setuptools.__version__
except Exception as e:
    out["setuptools"] = "MISSING " + type(e).__name__
out["has_distutils_hack"] = importlib.util.find_spec("_distutils_hack") is not None
spec = importlib.util.spec_from_file_location("k1_b_launcher_probe", launcher)
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)                      # the launcher's own path setup, stubs and distutils guard; imports compact_infer
import distutils.util, torch
out["distutils"] = "K1 shim" if os.environ.get("K1_DISTUTILS_SHIM") else str(getattr(sys.modules["distutils"], "__file__", "?"))
import knee_anchor_bank_v2 as KB
seg = KB.load_model(anatomy, "cpu")              # the vendored MONAI 0.6.0 V-Net + anatomy.pt, as every B worker loads it
import monai
out["monai"] = [monai.__version__, os.path.relpath(monai.__file__, L.CODE)]
with torch.no_grad():
    y = seg(torch.zeros(1, 1, 128, 128))
out["vnet_out"] = list(y.shape)
import timm, compact_train as CT
out["timm"] = timm.__version__
sd = torch.load(model, map_location="cpu", weights_only=True)
spatial = any(k == "sp_q" or k.endswith(".sp_q") for k in sd)
net = CT.Net(backbone, None, freeze_blocks=16, crop=int(crop), state_spec=None, in_chans=int(slices), spatial=spatial)
net.load_state_dict(sd, strict=True)            # = compact_infer.load_group for this model
out.update(spatial=spatial, params=int(sum(p.numel() for p in net.parameters())), seconds=round(time.time() - t0, 1))
print("K1_BPROBE " + json.dumps(out), flush=True)
'''
def probe_b_stack(I):
    """B's own stack before anything is spent on A: the launcher (stubs, distutils guard), compact_infer import, the vendored
    MONAI V-Net with anatomy.pt (+ one forward pass), compact_train.Net with the model's state dict (strict).  CPU only."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
    env.pop("K1_DISTUTILS_SHIM", None)
    g = B_GEOMETRY
    r = subprocess.run([sys.executable, "-c", B_PROBE, str(I.launcher), str(I.anatomy), str(I.models / B_MODEL["file"]),
                        g["backbone"], str(g["crop"]), str(g["slices"])],
                       capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL, env=env, cwd=str(TMP))
    line = [l for l in r.stdout.splitlines() if l.startswith("K1_BPROBE ")]
    if r.returncode != 0 or not line:
        raise K1Error("component-B stack probe failed rc %d: %s %s" % (r.returncode, r.stdout[-1500:], r.stderr[-2500:]))
    return json.loads(line[0][len("K1_BPROBE "):])


def write_launcher(src_dir, code_dir, path):
    text = LAUNCHER.replace("__SRC__", repr(str(src_dir))).replace("__CODE__", repr(str(code_dir)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class Inputs:
    pass


def prepare_inputs():
    I = Inputs()
    I.comp = find_mount(COMP, "test.csv")
    I.ds = find_mount(DS_SLUG, "checkpoint-manifest.json")
    I.helios = find_mount(HELIOS_SLUG, ANATOMY_FILE)
    log("competition root", I.comp, "| weights", I.ds, "| heliosli", I.helios)
    man_path = I.ds / "code_manifest.json"
    I.manifest_sha256 = sha256_file(man_path)
    man = json.loads(man_path.read_text())
    man["payload_members"] = man["members"]
    if sha256_file(I.ds / B_MODEL["file"]) != B_MODEL["sha256"]:
        raise K1Error("model sha differs from the kernel constant")
    if sha256_file(I.helios / ANATOMY_FILE) != ANATOMY_SHA256:
        raise K1Error("anatomy.pt differs from the S6 manifest anchor checkpoint")
    cfg = json.loads((I.ds / B_MODEL["run_config"]).read_text())
    want = {"size": B_GEOMETRY["size"], "crop": B_GEOMETRY["crop"], "views": B_GEOMETRY["views"], "slices": B_GEOMETRY["slices"],
            "phys_mm": B_GEOMETRY["phys_mm"], "backbone": B_GEOMETRY["backbone"], "spatial_attn": True, "arms": B_MODEL["arm"]}
    bad = {k: (cfg.get(k), v) for k, v in want.items() if cfg.get(k) != v}
    if bad:
        raise K1Error("run_config differs from the inference arguments (trained, inferred): %s" % bad)
    done = json.loads((I.ds / B_MODEL["done"]).read_text())
    if done.get("model_sha256") != B_MODEL["sha256"] or os.path.basename(done.get("model", "")) != B_MODEL.get("source_file", B_MODEL["file"]):
        raise K1Error("DONE marker does not record this model")
    for k in ("epochs", "train_studies"):
        if B_MODEL.get("expect_" + k) is not None and done.get(k) != B_MODEL["expect_" + k]:
            raise K1Error("DONE marker %s=%s, expected %s" % (k, done.get(k), B_MODEL["expect_" + k]))
    I.code_root = TMP / "k1src"
    # code = the scored run's exact file set: the extras from our dataset, everything else straight from Heliosli's; every
    # file is checked against the scored sha256/bytes, so the tree is byte-identical to the one that scored 0.950
    extras = {n: man["members"][n] for n in man["from_this_dataset"]}
    extract_payload(I.ds / EXTRAS_FILE, I.code_root, extras)
    for n, rec in man["members"].items():
        if n in extras:
            continue
        src, dst = I.helios / safe_rel(n), I.code_root / safe_rel(n)
        if not src.is_file() or src.stat().st_size != rec["bytes"]:
            raise K1Error("heliosli file %s missing or size differs" % n)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if sha256_file(dst) != rec["sha256"]:
            raise K1Error("heliosli file %s sha differs" % n)
    I.src = I.code_root / "heliosli"
    I.code = I.code_root / "cloud_code"
    for n, want_sha in SOURCE_SHA.items():
        if sha256_file(I.src / n) != want_sha:
            raise K1Error("heliosli/%s differs from SOURCE_SHA" % n)
    for n, want_sha in CODE_SHA.items():
        if sha256_file(I.code / "training" / n) != want_sha:
            raise K1Error("cloud_code/training/%s differs from CODE_SHA" % n)
    I.models = TMP / "k1models"
    if I.models.exists():
        shutil.rmtree(I.models)
    I.models.mkdir(parents=True)
    os.symlink(I.ds / B_MODEL["file"], I.models / B_MODEL["file"])        # the models folder holds exactly this one file
    I.anatomy = I.helios / ANATOMY_FILE
    I.launcher = write_launcher(I.src, I.code, TMP / "k1launch" / "k1_b_infer.py")
    test = pd.read_csv(I.comp / "test.csv", dtype={"StudyInstanceUID": str})
    I.test_ids = test["StudyInstanceUID"].astype(str).tolist()
    if not I.test_ids or len(I.test_ids) != len(set(I.test_ids)):
        raise K1Error("test.csv empty or has duplicate study ids")
    ss = I.comp / "sample_submission.csv"
    I.sub_cols = list(pd.read_csv(ss, nrows=1).columns) if ss.is_file() else ["StudyInstanceUID"] + LAB
    if I.sub_cols[0] != "StudyInstanceUID" or sorted(I.sub_cols[1:]) != sorted(LAB):
        raise K1Error("submission columns differ from the 12 findings: %s" % I.sub_cols)
    I.cost = {}
    for split in ("test", "train") if MODE == "parity" else ("test",):
        sc = pd.read_csv(I.comp / (split + "_series.csv"), dtype=str, usecols=["StudyInstanceUID"])
        I.cost.update(sc["StudyInstanceUID"].value_counts().to_dict())
    log("inputs ok: %d test studies | MANIFEST %s | model %s %s... | %d payload files" % (
        len(I.test_ids), I.manifest_sha256[:16], B_MODEL["file"], B_MODEL["sha256"][:16], len(man["payload_members"])))
    _RECEIPT["inputs"] = {"comp": str(I.comp), "ds": str(I.ds), "manifest_sha256": I.manifest_sha256, "model": B_MODEL,
                          "n_test": len(I.test_ids), "sub_cols": I.sub_cols}
    return I



# ---- VERBATIM component B helpers (run_b's building blocks) -------------------------------------------------------------------
def b_cmd(I, split, assignment, out_dir, tmp_dir, deadline):
    g = B_GEOMETRY
    return [sys.executable, "-u", str(I.launcher), "--data", str(I.comp), "--split", split, "--code", str(I.code),
            "--models", str(I.models), "--arm", B_MODEL["arm"], "--backbone", g["backbone"], "--size", str(g["size"]),
            "--crop", str(g["crop"]), "--views", g["views"], "--slices", str(g["slices"]), "--phys-mm", str(g["phys_mm"]),
            "--eval-stride", str(g["eval_stride"]), "--anchor-weights", str(I.anatomy), "--amp", B_AMP, "--chunk", str(g["chunk"]),
            "--workers", str(B_WORKERS), "--item-timeout", str(g["item_timeout"]), "--assignment", str(assignment),
            "--output", str(out_dir), "--tmp", str(tmp_dir), "--dump-logits", str(Path(out_dir) / "logits.json"),
            "--deadline", "%.0f" % deadline]


def split_parts(ids, cost, n):
    order = sorted(ids, key=lambda u: (-cost.get(u, 0), u))
    parts = [order[i::n] for i in range(n)]
    return [p[::2] + p[1::2] for p in parts if p]


def b_env(gi, px_dir=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gi), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MALLOC_ARENA_MAX="2", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
    env.pop("K1_PX_DUMP", None)
    if px_dir is not None:
        env["K1_PX_DUMP"] = str(px_dir)
    return env


def collect_part(pdir, part, I):
    rec = json.loads((pdir / "receipt.json").read_text())
    if rec.get("status") != "PASS" or rec.get("studies") != len(part):
        raise K1Error("%s: receipt status %s, studies %s (expected %d)" % (pdir.name, rec.get("status"), rec.get("studies"), len(part)))
    if rec.get("amp") != B_AMP or rec.get("models") != [B_MODEL["file"]] or rec.get("spatial_models") != 1:
        raise K1Error("%s: receipt amp %s models %s spatial %s" % (pdir.name, rec.get("amp"), rec.get("models"), rec.get("spatial_models")))
    if rec.get("anchor_weights_sha256") != ANATOMY_SHA256:
        raise K1Error("%s: anchor weights sha differs" % pdir.name)
    if not isinstance(rec.get("timed_out"), bool):           # compact_infer always states it (True / False)
        raise K1Error("%s: the receipt does not state timed_out (%r)" % (pdir.name, rec.get("timed_out")))
    shas = rec.get("sha256")
    if not isinstance(shas, dict) or not set(B_RECEIPT_SOURCES) <= set(shas):
        raise K1Error("%s: the receipt's source shas are incomplete: %s" % (pdir.name, sorted(shas) if isinstance(shas, dict) else shas))
    for n, s in shas.items():
        if SOURCE_SHA.get(n) != s:
            raise K1Error("%s: source %s sha differs in the receipt" % (pdir.name, n))
    n_anchor = sum(int(v) for v in (rec.get("anchor_methods") or {}).values())
    if n_anchor != rec.get("scored"):                  # every scored study must have had its SKM-TEA anchor record (review M1)
        raise K1Error("%s: %s scored studies but %d anchor records %s" % (pdir.name, rec.get("scored"), n_anchor, rec.get("anchor_methods")))
    logits = json.loads((pdir / "logits.json").read_text())
    pc = pd.read_csv(pdir / "predictions.csv", dtype={"StudyInstanceUID": str})
    if list(pc.columns) != ["StudyInstanceUID"] + LAB:
        raise K1Error("%s: predictions.csv columns %s" % (pdir.name, list(pc.columns)))
    pc = pc.set_index("StudyInstanceUID")
    z = {}
    for u in part:
        if u not in logits:
            continue
        a = np.asarray(logits[u], np.float64)
        if a.shape != (1, len(LAB)) or not np.isfinite(a).all():
            raise K1Error("%s: logits for %s have shape %s or are non-finite" % (pdir.name, u, a.shape))
        if u not in pc.index:
            raise K1Error("%s: %s has logits but no predictions.csv row" % (pdir.name, u))
        zm = a.mean(0)
        p = 1.0 / (1.0 + np.exp(-zm))
        d = float(np.abs(p - pc.loc[u, LAB].to_numpy(np.float64)).max())
        if d >= 1e-5:
            raise K1Error("%s: predictions.csv disagrees with the logits for %s (%.2e)" % (pdir.name, u, d))
        z[u] = zm
    extra = set(logits) - set(part)
    if extra:
        raise K1Error("%s: logits for studies outside the assignment: %s" % (pdir.name, sorted(extra)[:3]))
    fallback = dict(rec.get("fallback") or {})
    missing = [u for u in part if u not in z]
    for u in missing:
        fallback.setdefault(u, "no logits")
    return z, fallback, rec


def progress_lines(logp):
    out = []
    for l in tail(logp, 10 ** 7).splitlines():
        if l.startswith("{") and '"done"' in l:
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


def b_done(pdir):
    return (pdir / "receipt.json").is_file() and (pdir / "logits.json").is_file() and "WORKER_DONE" in tail(pdir / "log.txt", 3000)




# ---- [s6f] ours ---------------------------------------------------------------------------------------------------------------
def configure(kaggle_root=None, out=None, tmp=None):
    """[s6f] paths (tests point them at a temp dir)."""
    global KROOT, INPUT, WORK, OUT, TMP
    if kaggle_root is not None:
        KROOT = Path(kaggle_root)
        INPUT, WORK = KROOT / "input", KROOT / "working"
        OUT = WORK / "s6f"
    if out is not None:
        OUT = Path(out)
    if tmp is not None:
        TMP = Path(tmp)


ENV2 = r'''
import importlib, json, os, sys
out = {"executable": sys.executable, "sys_path": list(sys.path), "PYTHONPATH": os.environ.get("PYTHONPATH"),
       "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"), "MPLBACKEND": os.environ.get("MPLBACKEND"), "mods": {}}
for m in ("torch", "timm", "pydicom", "numpy", "pandas", "scipy", "cv2", "PIL", "safetensors", "setuptools",
          "pylibjpeg", "libjpeg", "openjpeg"):
    try:
        mod = importlib.import_module(m)
        out["mods"][m] = [str(getattr(mod, "__version__", "?")), os.path.dirname(os.path.dirname(os.path.abspath(mod.__file__)))]
    except Exception as e:
        out["mods"][m] = ["MISSING", type(e).__name__]
print("S6F_ENV2 " + json.dumps(out), flush=True)
'''


def env_isolation():
    """[s6f] what the S6F processes import, from where: a fresh interpreter with THIS process's environment (= what the B
    probe and the B workers inherit). Fails if an R7 dir (/tmp/probe/..) or /kaggle/working is on the import path."""
    r = subprocess.run([sys.executable, "-c", ENV2], capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL,
                       cwd=str(TMP))
    line = [l for l in r.stdout.splitlines() if l.startswith("S6F_ENV2 ")]
    if r.returncode != 0 or not line:
        raise K1Error("environment isolation probe failed rc %d: %s %s" % (r.returncode, r.stdout[-800:], r.stderr[-800:]))
    e = json.loads(line[0][len("S6F_ENV2 "):])
    bad = [p for p in e["sys_path"] + [sys.executable] if str(p).startswith(R7_ROOTS + (str(WORK),))]
    e["r7_or_working_on_path"] = bad
    e["driver_sys_path"] = list(sys.path)
    if bad or any(str(p).startswith(R7_ROOTS) for p in sys.path):
        raise K1Error("an R7 / working dir is on the S6F import path: %s" % bad)
    return e


def collect_part_csv(pdir, part):
    """[s6f] a part that has no receipt (its worker group was killed, or it crashed): the studies its compact_infer already wrote
    to predictions.csv (one row per scored study, flushed, '%.6f' probabilities) count as scored; z = logit(p) clipped to the
    6-decimal grid. Integrity: the 12 columns in label order, the row uid in the part, finite in [0, 1]."""
    z = {}
    p = pdir / "predictions.csv"
    if not p.is_file():
        return z
    try:
        pc = pd.read_csv(p, dtype={"StudyInstanceUID": str})
    except Exception:
        return z
    if list(pc.columns) != ["StudyInstanceUID"] + LAB:
        return z
    want = set(part)
    for _, r in pc.iterrows():
        u = r["StudyInstanceUID"]
        v = np.asarray([r[l] for l in LAB], np.float64)
        if u in want and np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all():
            v = np.clip(v, 5e-7, 1 - 5e-7)
            z[u] = np.log(v / (1 - v))
    return z


def run_b_s6f(I, tag, ids, split, stop_at):
    """[s6f] nartaa's run_b with these changes only (everything a scored study goes through is unchanged: split_parts, b_cmd,
    b_env, the launcher, compact_infer, collect_part's checks, z = mean logit, P = sigmoid(z), the B_probs.csv format):
    - budget: compact_infer --deadline = stop_at - B_DEADLINE_MARGIN_S (no NEW study after it), run_group timeout = stop_at - now;
    - a deadline, a failed / stalled / killed part, a retry that runs out of time: never an error. Studies without logits are
      'unscored' (NaN rows) and coverage = scored / N is reported; the notebook blend decides (>= 98 % or the S6F canary);
    - a part without a receipt: its predictions.csv rows (collect_part_csv);
    - the one retry of non-deadline fallbacks runs only if B_RETRY_MIN_S remain before stop_at;
    - no fail cap, no hard error."""
    def deadline_named(fb):
        return [u for u, v in fb.items() if "deadline" in str(v).lower()]
    root = OUT / tag
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    n_gpu = max(1, int(_RECEIPT["env"].get("n_gpu", 1)))
    parts = split_parts(ids, I.cost, n_gpu)
    t_start = time.time()
    deadline = stop_at - B_DEADLINE_MARGIN_S
    specs, pdirs = [], []
    for gi, part in enumerate(parts):
        pdir = root / ("part%d" % gi)
        pdir.mkdir()
        (pdir / "ids.json").write_text(json.dumps(part))
        specs.append({"name": "%s.part%d" % (tag, gi), "cmd": b_cmd(I, split, pdir / "ids.json", pdir, TMP / ("%s_tmp%d" % (tag, gi)), deadline),
                      "env": b_env(gi, None), "cwd": str(TMP), "log": pdir / "log.txt", "done_when": (lambda d=pdir: b_done(d))})
        pdirs.append(pdir)
    (root / "argv_part0.json").write_text(json.dumps(specs[0]["cmd"]))
    log("[%s] %d studies over %d GPU process(es) %s, stop_at +%ds, compact_infer deadline +%ds" % (
        tag, len(ids), len(parts), [len(p) for p in parts], stop_at - t_start, deadline - t_start))
    run_err = None
    try:
        runs = run_group(specs, max(30.0, stop_at - time.time()), tag)
    except K1Error as e:                               # K1Timeout included: the surviving parts' outputs are still read
        runs, run_err = None, repr(e)[:3000]
        log("[%s] run_group: %s" % (tag, repr(e)[:600]))
    z, fallback, receipts, sources = {}, {}, [], []
    for pdir, part in zip(pdirs, parts):
        if (pdir / "receipt.json").is_file():
            try:
                zz, fb, rec = collect_part(pdir, part, I)
                src = "logits"
            except Exception as e:
                zz, fb, rec, src = {}, {u: "collect: %r" % (e,) for u in part}, {}, "collect_failed: %r" % (e,)
        else:
            zz = collect_part_csv(pdir, part)
            fb, rec, src = {u: "no receipt (part killed or crashed)" for u in part if u not in zz}, {}, "predictions_csv"
        z.update(zz); fallback.update(fb)
        receipts.append({k: rec.get(k) for k in ("studies", "scored", "seconds", "amp", "gpu", "torch", "timm", "stats", "timed_out",
                                                 "anchor_methods", "per_study")})
        receipts[-1]["progress"] = progress_lines(pdir / "log.txt")
        sources.append(src)
    timed = [i for i, rc in enumerate(receipts) if rc.get("timed_out") is not False]
    retry = None
    redo = sorted(u for u in fallback if u not in z and u not in deadline_named(fallback))
    left = stop_at - time.time()
    if redo and left >= B_RETRY_MIN_S and not timed:
        log("[%s] %d fallback studies %s; %.0fs left before stop_at: one retry" % (tag, len(redo), dict(list(fallback.items())[:5]), left))
        rdir = root / "retry"
        rdir.mkdir()
        (rdir / "ids.json").write_text(json.dumps(redo))
        rdeadline = time.time() + max(left - B_DEADLINE_MARGIN_S, left * 0.5)
        rerr = None
        try:
            rruns = run_group([{"name": tag + ".retry", "cmd": b_cmd(I, split, rdir / "ids.json", rdir, TMP / (tag + "_tmp_retry"), rdeadline),
                                "env": b_env(0, None), "cwd": str(TMP), "log": rdir / "log.txt",
                                "done_when": lambda: b_done(rdir)}], max(30.0, stop_at - time.time()), tag)
        except K1Error as e:
            rruns, rerr = None, repr(e)[:2000]
        if (rdir / "receipt.json").is_file():
            try:
                zz, fb2, rec2 = collect_part(rdir, redo, I)
            except Exception as e:
                zz, fb2, rec2 = {}, {u: "collect: %r" % (e,) for u in redo}, {}
        else:
            zz = collect_part_csv(rdir, redo)
            fb2, rec2 = {u: "retry: no receipt" for u in redo if u not in zz}, {}
        z.update(zz)
        retry = {"ids": redo, "first_reasons": {u: fallback.get(u) for u in redo}, "still_failing": fb2, "seconds": rec2.get("seconds"),
                 "timed_out": rec2.get("timed_out"), "runs": rruns, "error": rerr}
        for u in redo:
            if u in zz:
                fallback.pop(u, None)
            else:
                fallback[u] = fb2.get(u, fallback.get(u))
    elif redo:
        log("[%s] %d fallback studies, %.0fs left before stop_at: no retry" % (tag, len(redo), left))
    missing = [u for u in ids if u not in z]
    failed = {u: str(fallback.get(u, "no logits"))[:300] for u in missing}
    nan12 = np.full(len(LAB), np.nan)
    Z = np.stack([z.get(u, nan12) for u in ids])
    P = 1.0 / (1.0 + np.exp(-Z))
    ok = np.array([u in z for u in ids])
    nonfinite = int((~np.isfinite(P[ok])).any(axis=1).sum()) if ok.any() else 0
    df = pd.DataFrame({"StudyInstanceUID": ids})
    for j, l in enumerate(LAB):
        df[l] = P[:, j]
    for j, l in enumerate(LAB):
        df["z|" + l] = Z[:, j]
    df.to_csv(root / ("%s_probs.csv" % tag), index=False, float_format="%.17g")
    secs = round(time.time() - t_start, 1)
    per_gpu = [rc.get("seconds") / max(1, rc.get("scored") or 0) for rc in receipts if rc.get("seconds") and rc.get("scored")]
    st = [rc.get("stats") or {} for rc in receipts]
    rows = sum(int(s_.get("rows") or 0) for s_ in st)
    info = {"seconds": secs, "stop_at": stop_at, "ci_deadline": deadline, "parts": [len(p) for p in parts], "runs": runs,
            "run_error": run_err, "sources": sources, "timed_out_parts": timed, "receipts": receipts, "retry": retry,
            "n": len(ids), "scored": int(ok.sum()), "coverage": float(ok.mean()) if len(ids) else 0.0,
            "nonfinite_scored_rows": nonfinite, "unscored": dict(list(failed.items())[:50]), "n_unscored": len(failed),
            "microbench": {"s_per_study_per_gpu_max": round(max(per_gpu), 3) if per_gpu else None,
                           "gpu_s_per_study": round(sum(float(s_.get("gpu_s") or 0) for s_ in st) / rows, 3) if rows else None,
                           "decode_s_per_study_per_worker": round(sum(float(s_.get("decode_s") or 0) for s_ in st) / rows, 3) if rows else None,
                           "windows_per_study": round(sum(float(s_.get("windows") or 0) for s_ in st) / rows, 1) if rows else None,
                           "wait_s_per_study": round(sum(float(s_.get("wait_s") or 0) for s_ in st) / rows, 3) if rows else None},
            "probs_sha256": sha256_file(root / ("%s_probs.csv" % tag))}
    save_json(root / ("%s_info.json" % tag), info)
    log("[%s] done: %d / %d studies scored (coverage %.4f), %.0fs, timed-out parts %s, retry %s, sources %s" % (
        tag, int(ok.sum()), len(ids), info["coverage"], secs, timed, None if retry is None else len(retry["ids"]), sources))
    return P, Z, info, failed


def main(argv=None):
    """[s6f] env probe -> inputs (nartaa's prepare_inputs + our code-manifest pin) -> B stack probe -> B. Exit 0 with
    s6f_driver_receipt.json {"ok": true} when B_probs.csv was written (any coverage); exit 3 with {"ok": false, "error"} else."""
    global _LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-at", type=float, required=True, help="epoch s: every S6F process ends by then (cell kills later)")
    ap.add_argument("--nb-t0", type=float, default=0.0, help="the notebook's NB_T0 (receipt ages)")
    ap.add_argument("--kaggle-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tmp", default=None)
    a = ap.parse_args(argv)
    configure(a.kaggle_root, a.out, a.tmp)
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    for stale in ("s6f_driver_receipt.json",):
        try:
            (OUT / stale).unlink()
        except FileNotFoundError:
            pass
    _LOG = open(OUT / "s6f_driver_log.txt", "a")
    age = lambda: round((time.time() - a.nb_t0) / 3600.0, 4) if a.nb_t0 else None
    _RECEIPT.update(ok=False, stop_at=a.stop_at, nb_t0=a.nb_t0, start_age_h=age(), pid=os.getpid(),
                    pgid=os.getpgrp() if hasattr(os, "getpgrp") else None, code_manifest_pin=CODE_MANIFEST_SHA256)
    rc = 0
    try:
        log("S6F_DRIVER start | python %s | stop_at in %.0fs | model %s %s... | out %s | tmp %s" % (
            sys.version.split()[0], a.stop_at - time.time(), B_MODEL["file"], B_MODEL["sha256"][:12], OUT, TMP))
        stage("env")
        env = probe_env()
        _RECEIPT["env"] = env
        _RECEIPT["env_isolation"] = env_isolation()
        log("S6F_ENV", json.dumps(env))
        log("S6F_ENV2", json.dumps({k: v for k, v in _RECEIPT["env_isolation"].items() if k in ("PYTHONPATH", "mods", "r7_or_working_on_path")}))
        if not env.get("cuda_available") or env.get("n_gpu", 0) < 1:
            raise K1Error("CUDA is not available")
        stage_done("env")
        stage("inputs")
        I = prepare_inputs()
        if I.manifest_sha256 != CODE_MANIFEST_SHA256:
            raise K1Error("code_manifest.json %s... is not the scored run's %s..." % (I.manifest_sha256[:16], CODE_MANIFEST_SHA256[:16]))
        stage_done("inputs")
        stage("B_probe")
        _RECEIPT["B_probe"] = probe_b_stack(I)
        log("K1_BPROBE", json.dumps(_RECEIPT["B_probe"]))
        stage_done("B_probe")
        stage("B")
        _P, _Z, b_info, b_failed = run_b_s6f(I, "B", I.test_ids, "test", a.stop_at)
        _RECEIPT["B"] = {k: v for k, v in b_info.items() if k != "receipts"}
        _RECEIPT["B"]["per_part"] = [{k: rc_.get(k) for k in ("studies", "scored", "seconds", "stats", "timed_out", "amp", "timm", "torch")}
                                     for rc_ in b_info["receipts"]]
        stage_done("B", coverage=b_info["coverage"])
        _RECEIPT["ok"] = True
    except BaseException as e:
        kill_all_children()
        _RECEIPT["ok"] = False
        _RECEIPT["error"] = repr(e)[:3000]
        _RECEIPT["error_trace"] = traceback.format_exc()[-4000:]          # receipt file only (kept out of the kernel log)
        log("S6F_DRIVER_FAILED %s" % repr(e)[:600])
        rc = 3
    finally:
        _RECEIPT["end_age_h"] = age()
        _RECEIPT["total_seconds"] = round(time.time() - T0, 1)
        save_json(OUT / "s6f_driver_receipt.json", _RECEIPT)
        log("S6F_DRIVER_DONE %s" % json.dumps({"ok": _RECEIPT["ok"], "seconds": _RECEIPT["total_seconds"],
                                               "coverage": (_RECEIPT.get("B") or {}).get("coverage"),
                                               "stages": {k: v.get("seconds") for k, v in _RECEIPT["stages"].items()}}))
    return rc


if __name__ == "__main__":
    sys.exit(main())
