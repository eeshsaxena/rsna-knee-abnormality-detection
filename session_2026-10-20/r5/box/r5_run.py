"""r5_run.py (10-20 r5, Claude): r6g58_run.py with the data dir / split / id list / output as arguments (for the store-rebuilt
DICOM dirs of build_synth.py: RSNA gold58 approx and OAI V00 knees). Everything that defines the R6 serve (cell 7 / 8 shas,
bundle shas, WORKER_ARGS exec, worker command, env) is UNCHANGED from r6g58_run.py; only DATA, --split, the ids, --out, the
deadline and the inductor cache dir (~/r5_oai) differ. Original docstring follows.
R6 bundle fe055406 on the 58 gold train studies, served exactly as the c4 kernel (eeshsaxena/rsna-sub-v224r6a25c4j,
local copy session_2026-10-18/a18/kernel_v224r6a25c4j) serves it in cells 7-9 (10-18, Claude). No em dashes.

What is the same as the kernel (asserted here, not assumed):
  * the 7 worker sources are written from the kernel cell-7 SOURCES texts (byte-identical, sha256 == SOURCE_SHA), as cell 7 does;
  * the 8 mainline modules cell 7 pins (CODE_SHA) match the box code dir; anatomy.pt sha == manifest anchor_weights_sha256;
  * bundle: manifest bundle_id fe055406fad6a12f, every file sha256 == manifest['sha256'];
  * WORKER_ARGS are built by EXEC of the kernel's own cell-7 WORKER block and cell 8 (flip TTA forced) on this manifest;
  * worker command and env = probe_dual.py's compact-worker command (--amp auto, --workers 2, OMP/MKL/OPENBLAS 1, MALLOC_ARENA_MAX 2,
    PYTHONPATH = code dir), one GPU, studies in probe_dual's cost order.
Differences, all outside the worker code (the R7-OOF precedent, r7teacher/oof/cells/c3_controller.py): --split train (the gold58
studies are train studies), the controller is this script (probe_dual needs sample_submission ids), --dump-logits (read-out only),
deadline = start + 3 h, box env kmimic (py 3.11, torch 2.5.1, timm 1.0.29) instead of the Kaggle image, V100 instead of T4
(amp auto = fp16 on both), TORCHINDUCTOR_CACHE_DIR private (fresh compile, as on Kaggle).
Usage: python r6g58_run.py [--wait-gpu] [--limit N]
"""
import argparse, csv, hashlib, json, os, subprocess, sys, time
from pathlib import Path

H = Path.home()
R = H / "r6g58"
CELLS = R / "kernel_cells"
CELL7_SHA = "b415480651e138b2f62d3eccbdb985e581d34c02496c97a93ea0d681147c16b7"
CELL8_SHA = "ef420a8c066292d1cc05cbaf5ba2ac530a9b14b7a607c21dd216d88fef24ad05"
MANIFEST_SHA = "6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648"
CODE = H / "r7port/code"                 # mainline code (training package), same tree as ~/s6seed/code
CODE_SRC = H / "r7port/weights"          # holds anatomy.pt (the code dataset's SKM-TEA checkpoint)
BUNDLE = R / "bundle"
RUN_FLAG = H / "r5_oai" / "RUNNING_R6.txt"
SRC = H / "r5_oai" / "r6src"           # the worker sources are re-written from cell 7 here (sha-checked), not into ~/r6g58
PY = H / "miniconda3/envs/kmimic/bin/python"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gpu_apps():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout
    return [l.strip() for l in out.splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-gpu", action="store_true", help="wait until no compute process holds the GPU (2 checks 60 s apart)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="all checks + WORKER_ARGS, no GPU")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--ids", required=True, help="json list of study ids")
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--keep-order", action="store_true", help="score the ids in the given order (a deadline cut keeps a prefix of "
                    "the stratified OAI order) instead of probe_dual's cost order")
    # run A (10-18 23:36) hit box-only CUDA OOMs (compile check + 21 of 58 studies, "gpu:" fallbacks): the two loader processes held
    # ~7.4 GB of the V100 for the SKM-TEA anchor / side nets. Run B: one loader process + expandable segments. Neither changes what
    # is computed (DataLoader batch_size None, shuffle False: same studies, same order; allocator policy only).
    ap.add_argument("--loader-workers", type=int, default=2)
    ap.add_argument("--expandable", action="store_true", help="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    a = ap.parse_args()
    t0 = time.time()
    c7 = (CELLS / "cell7.py").read_text(encoding="utf-8")
    c8 = (CELLS / "cell8.py").read_text(encoding="utf-8")
    assert hashlib.sha256(c7.encode("utf-8")).hexdigest() == CELL7_SHA, "cell 7 text differs"
    assert hashlib.sha256(c8.encode("utf-8")).hexdigest() == CELL8_SHA, "cell 8 text differs"
    sep = "\nimport os, sys, json, shutil, subprocess, zipfile, time, hashlib\n"
    assert c7.count(sep) == 1
    head, rest = c7.split(sep)
    g = {}
    exec(head, g)                                   # ARM, BUNDLE_ID, CODE_SHA, SOURCE_SHA, SOURCES, ...
    i0, i1 = rest.index("WORKER = 'compact_infer.py'"), rest.index("print('ASSETS_READY'")
    wblock = rest[i0:i1]

    # ---- cell 7 checks (paths mapped to the box)
    for name, digest in g["CODE_SHA"].items():
        assert sha(CODE / "training" / name) == digest, ("mainline module differs", name)
    SRC.mkdir(parents=True, exist_ok=True)
    for name, text in g["SOURCES"].items():
        (SRC / name).write_bytes(text.encode("utf-8"))
        assert sha(SRC / name) == g["SOURCE_SHA"][name], name
    assert sorted(p.name for p in SRC.iterdir()) == sorted(g["SOURCES"]), sorted(p.name for p in SRC.iterdir())
    assert sha(BUNDLE / "manifest.json") == MANIFEST_SHA, "manifest file differs"
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    assert manifest["bundle_id"] == g["BUNDLE_ID"], (manifest["bundle_id"], g["BUNDLE_ID"])
    names = manifest["models"] + (manifest.get("models2") or []) + [manifest[k] for k in
             ("state_spec", "side_model", "fluid_stacker", "state_stacker") if manifest.get(k)]
    bundle_sha = {}
    for name in names:
        bundle_sha[name] = sha(BUNDLE / name)
        assert bundle_sha[name] == manifest["sha256"][name], name
    assert manifest["gold_in_training"] == 0
    ns = dict(manifest=manifest, BUNDLE=BUNDLE, SRC=SRC, CODE_SRC=CODE_SRC, SOURCE_SHA=g["SOURCE_SHA"], SOURCES=g["SOURCES"],
              hashlib=hashlib, json=json, find_mount=None, W_SLUG=g["W_SLUG"])
    exec(wblock, ns)                                # the kernel's WORKER_ARGS block, unchanged
    exec(c8, ns)                                    # cell 8: flip TTA forced on (asserts the sha 8401812c worker)
    WORKER, WORKER_ARGS = ns["WORKER"], ns["WORKER_ARGS"]
    assert WORKER == "compact_infer.py" and WORKER_ARGS[-1] == "--flip-tta" and ns["FLIP_TTA_FORCED_ON"]

    DATA = Path(a.data)
    ids = json.loads(Path(a.ids).read_text(encoding="utf-8"))
    assert len(ids) == len(set(ids)), len(ids)
    if a.limit:
        ids = ids[:a.limit]

    def cost(uid):                                  # probe_dual.py cost(): number of DICOM files
        n = 0
        for s in os.scandir(DATA / (a.split + "_series") / uid):
            if s.is_dir():
                n += sum(1 for _ in os.scandir(s.path))
        return n
    costs = {u: cost(u) for u in ids}
    assert all(costs[u] > 0 for u in ids), [u for u in ids if not costs[u]]
    if a.keep_order:
        part = list(ids)
    else:
        order = sorted(ids, key=lambda u: (-costs[u], u))
        part = order[::2] + order[1::2]

    out = Path(a.out)
    wdir = out / "worker0"
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / "ids.json").write_text(json.dumps(part), encoding="utf-8")
    print("R5RUN_READY", json.dumps({"worker_args": WORKER_ARGS, "studies": len(part), "dicom_files": sum(costs.values())}), flush=True)
    if a.dry:
        return

    if a.wait_gpu:
        ok = 0
        while ok < 2:
            ok = ok + 1 if not gpu_apps() else 0
            if ok < 2:
                time.sleep(60)
        print("GPU_FREE", time.strftime("%m-%d %H:%M:%S"), flush=True)
    elif gpu_apps():
        raise SystemExit("GPU busy (a compute process is running): refusing")
    RUN_FLAG.write_text("R6 r5 serve, pid %d, started %s\n" % (os.getpid(), time.strftime("%m-%d %H:%M:%S")))

    deadline = time.time() + a.hours * 3600
    cmd = [str(PY), "-u", str(SRC / WORKER), "--data", str(DATA), "--split", a.split, "--code", str(CODE), "--arm", g["ARM"],
           "--assignment", str(wdir / "ids.json"), "--output", str(wdir), "--amp", "auto", "--workers", str(a.loader_workers),
           "--deadline", str(deadline)]
    cmd += ["--models", str(BUNDLE)] + [str(x) for x in WORKER_ARGS]
    cmd += ["--dump-logits", str(wdir / "dump.json")]
    env = dict(os.environ, PYTHONPATH=str(CODE), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MALLOC_ARENA_MAX="2",
               TORCHINDUCTOR_CACHE_DIR=str(H / "r5_oai" / "inductor_cache"))
    if a.expandable:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    (out / "cmd.json").write_text(json.dumps(cmd, indent=1), encoding="utf-8")
    t1 = time.time()
    with open(wdir / "log.txt", "w", encoding="utf-8") as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    t2 = time.time()
    try:
        RUN_FLAG.unlink()
    except OSError:
        pass
    code_tree = hashlib.sha256()
    for p in sorted((CODE / "training").rglob("*.py")):
        code_tree.update(str(p.relative_to(CODE)).encode() + b"\0" + p.read_bytes())
    receipt = {"rc": rc, "worker_seconds": round(t2 - t1, 1), "total_seconds": round(t2 - t0, 1),
               "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)),
               "cell7_sha256": CELL7_SHA, "cell8_sha256": CELL8_SHA, "manifest_sha256": MANIFEST_SHA,
               "bundle_id": manifest["bundle_id"], "bundle_sha256": bundle_sha, "source_sha256": g["SOURCE_SHA"],
               "code_sha256_pinned": g["CODE_SHA"], "code_tree_py_sha256": code_tree.hexdigest(),
               "anatomy_sha256": sha(CODE_SRC / "anatomy.pt"), "worker_args": WORKER_ARGS, "cmd": cmd,
               "env_overrides": {k: env[k] for k in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                    "OPENBLAS_NUM_THREADS", "MALLOC_ARENA_MAX", "TORCHINDUCTOR_CACHE_DIR",
                                                    "PYTORCH_CUDA_ALLOC_CONF") if k in env},
               "loader_workers": a.loader_workers,
               "python": str(PY), "studies": len(part), "order": "given order (--keep-order), one GPU" if a.keep_order else "probe_dual cost order, one GPU"}
    (out / "r5_receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print("R5RUN_DONE", json.dumps({"rc": rc, "worker_seconds": receipt["worker_seconds"]}), flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
