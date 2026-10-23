# e2_app.py: Modal app "rsna-e2" for the E2 set = TTR288 frac75, seed 1046, folds 0-4.
# The argv is the box queue line "Q4 TTR288 s1045 f0 frac75" token for token with ONLY --seed 1046, --fold F and
# --out changed (meta/argv_template.txt, built and diff-checked by make_template.sh on the box).
# Volume rsna-e2 inputs: /code2 /novel/frac75 /rsna_train /hf /fleet_store (4407 competition npz) /meta.
#                outputs: /runs/fF_s1046_ttr288frac75/ + /runs/fF_s1046_ttr288frac75.log, /runs/_meta/*.json.
# No OAI data is ever uploaded (stage_and_upload.sh refuses any source under ~/oai_* or ~/union_store).
# Entry points (run on the box with ~/modalenv/bin/modal, profile eeshsaxena):
#   modal run e2_app.py::check            CPU only: md5 parity of every store/code/csv/HF file + parity.py vs the box
#   modal run e2_app.py::smoke            ~1000 training iterations on A100-40GB, L40S and H100 in parallel (timing)
#   modal run --detach e2_app.py::train   the folds in parallel on TRAIN_GPU (E2_FOLDS="0 1 2 3 4")
import os, sys, json, time, re, subprocess, threading, hashlib, shutil
import modal

APP_NAME = "rsna-e2"
VOL_NAME = "rsna-e2"
TRAIN_GPU = "L40S"             # smoke 10-09: L40S 0.176 s/it, H100 0.120, A100-40GB 0.216 -> L40S is cheapest per fold
TRAIN_TIMEOUT_S = 9540         # 2.65 h per-fold hard stop (fold MODEL 2.28 h on L40S) = the worst-case cost cap
TRAIN_CPU = (4.0, 8.0)         # (request, limit) cores; the L40S smoke used 4.4
TRAIN_MEM = (6144, 16384)      # (request, limit) MiB; the box trainer tree uses ~5.4 GB PSS
SEED = 1046
RUNTAG = "ttr288frac75"
V = "/vol"
R = "/root/rsna"               # local copy of the inputs (same layout as the box home)
CLOSE_RE = re.compile(r'^\{"raw": [^,]+, "ema": [^}]+\}$')

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME)
image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy==1.26.4", "pandas==3.0.5", "scikit-learn==1.9.0",
                 "scipy==1.17.1", "timm==1.0.29", "huggingface_hub==0.36.2", "safetensors==0.8.0",
                 "PyYAML==6.0.3", "pillow==12.3.0")
    # The box runs opencv-python-headless 4.14.0.94 on numpy 1.26.4 although its metadata asks for numpy 2.
    # Install it like the box and the Vast kit (no deps, pinned); check_inputs() proves identical decoding.
    .run_commands("python -m pip install --no-cache-dir --no-deps opencv-python-headless==4.14.0.94",
                  "python -m pip list --format=freeze > /image_pip_freeze.txt")
    .env({"PYTHONUNBUFFERED": "1"})
)


def log(*a):
    print("[e2 " + time.strftime("%H:%M:%S") + "]", *a, flush=True)


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _md5_many(paths, procs=8):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(procs) as ex:
        return dict(zip(paths, ex.map(_md5, paths, chunksize=16)))


def _store_manifest():
    m = {}
    for ln in open(V + "/meta/store_md5.txt"):
        h, n = ln.split()
        m[n.lstrip("*").replace("./", "", 1)] = h
    assert len(m) == 4407, "store manifest has %d entries" % len(m)
    return m


def _versions():
    keep = ("opencv-python-headless", "numpy", "pandas", "scikit-learn", "scipy", "timm", "huggingface-hub",
            "huggingface_hub", "safetensors", "pyyaml", "pillow", "torch", "torchvision")
    return sorted(l.strip() for l in open("/image_pip_freeze.txt") if l.split("==")[0].lower() in keep)


def _stage(local=True):
    """Copy the inputs from the volume to local disk (or use the volume in place), then verify every md5."""
    t0 = time.time()
    du = shutil.disk_usage("/root")
    log("local disk /root: free %.1f GB of %.1f GB" % (du.free / 1e9, du.total / 1e9))
    root = R if (local and du.free > 20e9) else V
    if local and root == V:
        log("NOTE: not enough local disk, the trainer reads the inputs from the volume in place")
    if root == R:
        from concurrent.futures import ThreadPoolExecutor
        for d in ("code2", "novel", "rsna_train", "hf"):
            shutil.copytree(V + "/" + d, R + "/" + d, dirs_exist_ok=True, symlinks=False)
        os.makedirs(R + "/fleet_store", exist_ok=True)
        names = sorted(os.listdir(V + "/fleet_store"))
        with ThreadPoolExecutor(32) as ex:
            list(ex.map(lambda n: shutil.copyfile(V + "/fleet_store/" + n, R + "/fleet_store/" + n), names))
    t1 = time.time()
    man = _store_manifest()
    names = sorted(os.listdir(root + "/fleet_store"))
    assert names == sorted(man), "store file list differs from the box manifest (%d files)" % len(names)
    got = _md5_many([root + "/fleet_store/" + n for n in names])
    bad = [n for n in names if got[root + "/fleet_store/" + n] != man[n]]
    assert not bad, "store md5 MISMATCH on %d files, e.g. %s" % (len(bad), bad[:3])
    small = {}
    for ln in open(V + "/meta/small_md5.txt"):
        h, n = ln.split(None, 1)
        small[n.strip()] = h
    sb = [n for n, h in small.items() if _md5(root + "/" + n) != h]
    assert not sb, "code/csv/HF md5 MISMATCH: %s" % sb
    t2 = time.time()
    log("inputs ready in %s: copy %.0fs, md5 %.0fs | store 4407/4407 md5 IDENTICAL to the box | %d code/csv/HF files "
        "md5 IDENTICAL" % (root, t1 - t0, t2 - t1, len(small)))
    return root, dict(copy_s=round(t1 - t0), md5_s=round(t2 - t1), root=root)


def _argv(root, fold, out):
    tpl = open(V + "/meta/argv_template.txt").read().strip()
    a = tpl.replace("@R@", root).replace("@F@", str(fold)).replace("@OUT@", out).split()
    assert len(a) == 40, "argv has %d tokens (want 40)" % len(a)
    g = dict(zip(a[::2], a[1::2]))
    assert g["--seed"] == str(SEED) and g["--fold"] == str(fold) and g["--out"] == out, g
    assert g["--epochs"] == "21" and g["--ttr_size"] == "288" and g["--grad_ckpt"] == "0" and g["--workers"] == "10", g
    assert g["--labels"].endswith("/novel/frac75/labels_frac75_logit.csv"), g
    assert g["--eval_labels"].endswith("/rsna_train/llm_labels_v4_blend.csv"), g
    return a


class _Res:
    """CPU-seconds of every process in the container (from /proc), summed RSS and GPU util, sampled every 30 s."""

    def __init__(self):
        self.samples, self.stop = [], False
        self.hz = os.sysconf("SC_CLK_TCK")
        threading.Thread(target=self._run, daemon=True).start()

    def _snap(self):
        cpu, rss = 0.0, 0
        for p in os.listdir("/proc"):
            if not p.isdigit():
                continue
            try:
                f = open("/proc/%s/stat" % p).read().rsplit(")", 1)[1].split()
                cpu += (int(f[11]) + int(f[12])) / self.hz
                rss += int(open("/proc/%s/statm" % p).read().split()[1]) * 4096
            except Exception:
                pass
        gu = ""
        try:
            gu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            pass
        return time.time(), cpu, rss, gu

    def _run(self):
        while not self.stop:
            self.samples.append(self._snap())
            time.sleep(30)

    def summary(self, since=0.0):
        s = [x for x in self.samples if x[0] >= since] or self.samples
        if len(s) < 2:
            return {}
        dt = s[-1][0] - s[0][0]
        gpu = [float(x[3].split(",")[0]) for x in s if x[3]]
        return dict(cpu_vcpu_avg=round((s[-1][1] - s[0][1]) / max(dt, 1), 2),
                    rss_gb_max=round(max(x[2] for x in s) / 1e9, 1),
                    gpu_util_avg=round(sum(gpu) / len(gpu), 1) if gpu else None, window_s=round(dt))


def _env(root):
    e = dict(os.environ, HF_HOME=root + "/hf", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    e.pop("CUDA_VISIBLE_DEVICES", None)
    return e


def _gpu_facts():
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,clocks.max.sm", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.stdout.strip()


def _cgroup_facts():
    out = {}
    for f in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            out[f] = open(f).read().strip()
        except Exception:
            pass
    return out


def _smoke(gpu_req, stop_it=1000, max_train_s=480):
    t_start = time.time()
    facts = _gpu_facts()
    log("SMOKE %s: %s | nproc %s | cgroup %s | versions %s" % (gpu_req, facts, os.cpu_count(), _cgroup_facts(), _versions()))
    root, st = _stage()
    out = "/root/smoke_out"
    argv = _argv(root, 0, out)
    res = _Res()
    t_launch = time.time()
    p = subprocess.Popen([sys.executable, "fleet_train.py"] + argv, cwd=root + "/code2", env=_env(root),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    marks, lines = {}, []
    for ln in p.stdout:
        ln = ln.rstrip()
        lines.append(ln)
        print(ln, flush=True)
        m = re.match(r"ep 0 it (\d+)/(\d+) loss \S+ (\S+)s/it", ln)
        if m:
            marks[int(m.group(1))] = (time.time(), float(m.group(3)), int(m.group(2)))
            if int(m.group(1)) >= stop_it:
                break
        if 0 in marks and time.time() - marks[0][0] > max_train_s:
            break
        if time.time() - t_start > 780:
            break
    p.kill()
    p.wait()
    its = sorted(marks)
    out_d = dict(gpu_req=gpu_req, gpu=facts, stage=st,
                 startup_to_it0_s=round(marks[0][0] - t_launch) if 0 in marks else None,
                 marks={k: [round(v[0] - t_launch, 1), v[1]] for k, v in marks.items()},
                 it_per_epoch=marks[its[-1]][2] if its else None,
                 res=res.summary(since=marks[200][0] if 200 in marks else 0.0), total_s=round(time.time() - t_start),
                 tail=lines[-4:])
    if len(its) >= 3 and its[1] >= 200:
        a, b = its[1], its[-1]
        out_d["steady_s_per_it"] = round((marks[b][0] - marks[a][0]) / (b - a), 4)
    res.stop = True
    log("SMOKE RESULT " + json.dumps(out_d))
    return out_d


SMOKE_KW = dict(image=image, volumes={V: vol}, timeout=14 * 60, retries=0, cpu=(4.0, 12.0), memory=(6144, 20480))


@app.function(gpu="A100-40GB", **SMOKE_KW)
def smoke_a100():
    return _smoke("A100-40GB")


@app.function(gpu="L40S", **SMOKE_KW)
def smoke_l40s():
    return _smoke("L40S")


@app.function(gpu="H100", **SMOKE_KW)
def smoke_h100():
    return _smoke("H100")


@app.function(image=image, volumes={V: vol}, timeout=30 * 60, retries=0, cpu=(4.0, 8.0), memory=(4096, 16384))
def check_inputs():
    """CPU only: md5 of every store/code/csv/HF file on the volume vs the box manifests, then parity.py (the Vast kit
    probe: contract, select_sha, jpeg_fp, model_sha, fold counts, 4-study decode probe, backbone weights) vs the box."""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    t0 = time.time()
    log("CHECK cgroup %s | nproc %s | versions %s" % (_cgroup_facts(), os.cpu_count(), _versions()))
    _, st = _stage(local=False)
    box = json.load(open(V + "/meta/box_parity.json"))
    r = subprocess.run([sys.executable, V + "/meta/parity.py", "--code", V + "/code2", "--store", V + "/fleet_store",
                        "--labels", V + "/novel/frac75/labels_frac75_logit.csv", "--folds_csv", V + "/rsna_train/folds5.csv",
                        "--exclude", V + "/rsna_train/gold74_eval_set.csv", "--folds", "0", "1", "2", "3", "4"],
                       capture_output=True, text=True,
                       env=dict(_env(V), TTR="288", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode == 0, "parity.py failed: " + r.stderr[-800:]
    me = json.loads(r.stdout.strip().splitlines()[-1])
    keys = ("contract", "select_sha", "jpeg_fp", "model_sha", "n", "fold_counts", "decode_probe", "backbone_sha",
            "state_keys", "pretrained_cfg")
    cmp = {k: ("IDENTICAL" if box.get(k) == me.get(k) else "DIFFERS box %s | modal %s" % (box.get(k), me.get(k)))
           for k in keys}
    ok = all(v == "IDENTICAL" for v in cmp.values())
    out = dict(ok=ok, parity=cmp, versions_box=box["versions"], versions_modal=me["versions"], image_pins=_versions(),
               stage=st, seconds=round(time.time() - t0))
    log("CHECK " + json.dumps(out))
    assert ok, "PARITY MISMATCH: do not train"
    return out


@app.function(image=image, volumes={V: vol}, gpu=TRAIN_GPU, timeout=TRAIN_TIMEOUT_S, retries=0, cpu=TRAIN_CPU,
              memory=TRAIN_MEM, max_containers=5)
def train_fold(fold: int):
    name = "f%d_s%d_%s" % (fold, SEED, RUNTAG)
    out, logf = V + "/runs/" + name, V + "/runs/" + name + ".log"
    os.makedirs(V + "/runs/_meta", exist_ok=True)
    t_start = time.time()
    facts = _gpu_facts()
    log("TRAIN %s on %s | timeout %ss | cgroup %s | versions %s" % (name, facts, TRAIN_TIMEOUT_S, _cgroup_facts(), _versions()))
    if os.path.exists(logf):
        last = open(logf).read().rstrip().splitlines()[-1:] or [""]
        if CLOSE_RE.match(last[0]):
            log("%s already COMPLETE on the volume: %s" % (name, last[0]))
            return dict(fold=fold, name=name, status="already_complete", closing=last[0])
    if os.path.exists(out) or os.path.exists(logf):
        a = V + "/runs/_partial/" + name + "_" + time.strftime("%m%d_%H%M%S")
        os.makedirs(a, exist_ok=True)
        for s in (out, logf):
            if os.path.exists(s):
                shutil.move(s, a + "/" + os.path.basename(s))
        log("partial run moved aside to " + a)
    root, st = _stage()
    argv = _argv(root, fold, out)
    log("LAUNCH %s: python fleet_train.py %s" % (name, " ".join(argv)))
    res = _Res()
    t_launch = time.time()
    with open(logf, "w", buffering=1) as lf:
        p = subprocess.Popen([sys.executable, "fleet_train.py"] + argv, cwd=root + "/code2", env=_env(root),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        last_commit = time.time()
        for ln in p.stdout:
            lf.write(ln)
            print(ln.rstrip(), flush=True)
            if time.time() - last_commit > 600:
                lf.flush()
                try:
                    vol.commit()
                except Exception as e:
                    log("vol.commit() warning: %s" % e)
                last_commit = time.time()
        rc = p.wait()
    lines = open(logf).read().rstrip().splitlines()
    closing = lines[-1] if lines else ""
    ok = rc == 0 and bool(CLOSE_RE.match(closing))
    summ = dict(fold=fold, name=name, status="complete" if ok else "FAILED", rc=rc, closing=closing, gpu=facts,
                stage=st, train_min=round((time.time() - t_launch) / 60, 1),
                total_min=round((time.time() - t_start) / 60, 1), res=res.summary(),
                finished=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    res.stop = True
    json.dump(summ, open(V + "/runs/_meta/" + name + "_summary.json", "w"), indent=1)
    vol.commit()
    log("TRAIN RESULT " + json.dumps(summ))
    if not ok:
        raise RuntimeError("%s FAILED rc %s: %s" % (name, rc, " | ".join(lines[-3:])))
    return summ


@app.function(image=image, volumes={V: vol}, timeout=600, retries=0, cpu=(1.0, 4.0), memory=(1024, 4096))
def manifest_fold(name: str):
    """size + md5 of what the box fetches for one fold: log, *_best.pt, top-level oof_*.csv, epochs/oof_*.csv."""
    vol.reload()
    base = V + "/runs/"
    want = [name + ".log"] + [name + "/" + f for f in sorted(os.listdir(base + name))
                              if f.endswith("_best.pt") or (f.startswith("oof_") and f.endswith(".csv"))]
    ep = base + name + "/epochs"
    if os.path.isdir(ep):
        want += [name + "/epochs/" + f for f in sorted(os.listdir(ep)) if f.startswith("oof_") and f.endswith(".csv")]
    return {w: [os.path.getsize(base + w), _md5(base + w)] for w in want}


@app.local_entrypoint()
def fetch(run: str, dest: str):
    name = run
    """Runs on the box: download one fold (no epochs/*.pt) into dest/, md5-verified against manifest_fold()."""
    import hashlib as _h
    man = manifest_fold.remote(name)
    assert man, "empty manifest for " + name
    t0, tot = time.time(), 0
    for rel, (size, md5) in man.items():
        p = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        h, n = _h.md5(), 0
        with open(p + ".part", "wb") as f:
            for chunk in vol.read_file("/runs/" + rel):
                f.write(chunk)
                h.update(chunk)
                n += len(chunk)
        assert n == size and h.hexdigest() == md5, "fetch MISMATCH %s: %d/%d bytes md5 %s/%s" % (rel, n, size, h.hexdigest(), md5)
        os.replace(p + ".part", p)
        tot += n
    json.dump(man, open(os.path.join(dest, name + ".modal_manifest.json"), "w"), indent=1)
    print("FETCH_OK %s: %d files, %.1f MB in %.0fs, every md5 identical to the volume" % (name, len(man), tot / 1e6, time.time() - t0))


@app.local_entrypoint()
def check():
    print("CHECK_RESULT " + json.dumps(check_inputs.remote()))


@app.local_entrypoint()
def smoke():
    calls = [f.spawn() for f in (smoke_a100, smoke_l40s, smoke_h100)]
    outs = []
    for c in calls:
        try:
            outs.append(c.get())
        except Exception as e:
            outs.append(dict(error=repr(e)[:500]))
    print("SMOKE_ALL " + json.dumps(outs))


@app.local_entrypoint()
def train():
    folds = [int(x) for x in os.environ.get("E2_FOLDS", "0 1 2 3 4").split()]
    print("[e2] launching folds %s on %s, timeout %ss each" % (folds, TRAIN_GPU, TRAIN_TIMEOUT_S), flush=True)
    calls = {f: train_fold.spawn(f) for f in folds}
    for f, c in calls.items():
        print("[e2] fold %d call %s" % (f, c.object_id), flush=True)
    for f, c in calls.items():
        try:
            print("[e2] fold %d RESULT %s" % (f, json.dumps(c.get())), flush=True)
        except Exception as e:
            print("[e2] fold %d ERROR %s" % (f, repr(e)[:600]), flush=True)
    print("[e2] all folds returned", flush=True)
