"""px_build_s6.py v3 (10-15, BOX_PLAN_1015 S3; v3 = exit 4 when a worker dies, review S6R-2; v2 = px_build_s6.v2_8d94eff2.py): the S6fall training pixel cache from raw DICOMs, through the SAME code that
serves the model on Kaggle (Heliosli compact_infer 34f433fe CStudies, nartaa's S6F0 launcher semantics):
  lean native render -> geometry bank -> SKM-TEA knee anchors (anatomy.pt 5904dd16) -> px_cache.StudySlices.build (b13e38a7)
  S 320, views v4, NO side model (B_GEOMETRY of rsna-knee-s6fall-solo: size 320, crop 288, views v4, slices 3, phys 3.5,
  eval_stride 1; slices / phys / eval_stride are window parameters of the trainer and the server, not of the cache).
Output = compact_train 14767c7a --px format (heliosli px_cache layout): <out>/px.u8.npy [N, 320, 320] uint8, meta.f32.npy [N, 12],
index.json {done: uid -> [start, n_rows, n_blocks], blocks: uid -> [{series, plane, n_slices, views: [[view_id, row], ...]}], ...}.
Rows are written straight into the final files (no merge copy): a worker renders a study, takes a contiguous row range from a
shared cursor (flock), pwrites px + meta there, then records the study in its own part json. Every study's rows are contiguous.
Resumable: studies done in ANY part are skipped on restart (the proc count may change); errored studies are retried once per
restart; a crash between the row grab and the json write only leaves unreferenced rows. --finalize writes index.json and sets
the npy headers to the cursor (exact size); it can be re-run after a later resume.
CPU only (CUDA hidden by the caller). Writes only under --out and --tmp.
  build:    python px_build_s6.py --data D --split train --src SRC --code CODE --anchor anatomy.pt --out OUT --procs 8 --threads 2
  finalize: python px_build_s6.py --out OUT --finalize
"""
import argparse, fcntl, hashlib, json, os, resource, shutil, signal, sys, time, types
from pathlib import Path

S = 320
META_COLS = ["series_idx", "slot", "plane", "view_id", "rel_pos", "log_spacing", "extent_130", "fluid", "fatsat", "slice_order",
             "n_slices", "real_series_idx"]
GEOM = {"size": 320, "views": "v4", "slices": 3, "phys_mm": 3.5, "eval_stride": 1, "side_model": None}
PIN = {"compact_infer.py": "34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a",
       "compact_train.py": "14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80",
       "px_cache.py": "b13e38a7ad3bd496f8027e02209653e224b0926db4f0853eeb0d076a2d6e2632",
       "probe_infer.py": "a6cc31448019c84be5868d0a2ff1d99719b23b0f86110efc03efff3ce6bd6980",
       "knee_anchor_bank_v2.py": "e564807a129dfcd3c984348917436ee6083066a4d685340587ebd1d3ea14f309"}
CODE_PIN = {"native_cache.py": "c53ce8985406b3b918d9e821d43518479e89d9f946daebd817f378e3949718b9",
            "native_data.py": "f59afe205c8e193369fa3ccd22edcce8df3b293205fc48ddb1dd45169ee43b36",
            "input_provenance.py": "b3a22500c7f976f6ee040239ada5cacecf2326861cb49af1c7d8cfdeeb55fcda",
            "intensity_mapping.py": "19be2d8cbd6f8971c1288d7512a8cca41d143470c2ba6e9b9096fde035fa9d41",
            "coronal_views.py": "841f216d2682ae2c0f8afd82fc902b9fa74924d7145b952069dea8b0bbd03c40",
            "exact_duplicates.py": "51b38db21ce9c5fe2b6278a835983414756b61ac6af81aba380b44b41b075897",
            "dicom_geometry.py": "12aa501f9704aef15b7a3bbd560951acb6d95b89b7de0141d6cf89fc3f051ab1",
            "common.py": "8283fa1f8e7ef8cfacbba52df1715636be3c9ad688e39912814e7d5bcf57105f"}
ANATOMY_SHA = "5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e"
ROW_PX = S * S
ROW_META = len(META_COLS) * 4
NPY_HDR = 128            # header bytes of both npy files (fixed; finalize rewrites the shape in place)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_pins(a):
    bad = {n: sha(Path(a.src) / n)[:8] for n, s in PIN.items() if sha(Path(a.src) / n) != s}
    bad.update({"training/" + n: sha(Path(a.code) / "training" / n)[:8] for n, s in CODE_PIN.items()
                if sha(Path(a.code) / "training" / n) != s})
    if sha(a.anchor) != ANATOMY_SHA:
        bad["anatomy.pt"] = sha(a.anchor)[:8]
    if bad:
        raise SystemExit("PIN FAIL (sha prefix seen): %s" % bad)


def npy_header(shape, descr):
    import io
    import numpy as np
    fp = io.BytesIO()
    np.lib.format.write_array_header_1_0(fp, {"descr": descr, "fortran_order": False, "shape": tuple(shape)})
    h = fp.getvalue()
    if len(h) != NPY_HDR:
        raise RuntimeError("npy header is %d bytes, expected %d" % (len(h), NPY_HDR))
    return h


def init_files(out):
    """create px.u8.npy / meta.f32.npy (header only) and the row cursor, once."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "alloc.lock", "a"):
        pass
    with open(out / "alloc.lock") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            if not (out / "alloc.json").exists():
                for name, descr, tail in (("px.u8.npy", "|u1", (S, S)), ("meta.f32.npy", "<f4", (len(META_COLS),))):
                    p = out / name
                    if p.exists():
                        raise SystemExit("%s exists without alloc.json: use a new --out" % p)
                    with open(p, "wb") as fh:
                        fh.write(npy_header((0,) + tail, descr))
                json.dump({"cursor": 0, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, open(out / "alloc.json", "w"))
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def grab_rows(out, n):
    with open(out / "alloc.lock") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            st = json.load(open(out / "alloc.json"))
            cur = int(st["cursor"])
            st["cursor"] = cur + n
            tmp = out / "alloc.json.tmp"
            json.dump(st, open(tmp, "w"))
            os.replace(tmp, out / "alloc.json")
            return cur
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def install_paths(src, code):
    """nartaa's S6F0 launcher (rsna-knee-s6fall-solo LAUNCHER): sys.path [code, src, ...], the training.download_data stub, the
    distutils guard, a memoised probe_infer.install_native (its exec patch is not re-entrant)."""
    sys.dont_write_bytecode = True
    for p in (src, code):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "training.download_data" not in sys.modules:
        m = types.ModuleType("training.download_data")
        m.ChunkedArchive = None
        sys.modules["training.download_data"] = m
    try:
        import distutils.util  # noqa: F401
    except Exception:
        def _strtobool(val):
            val = val.lower()
            if val in ("y", "yes", "t", "true", "on", "1"):
                return True
            if val in ("n", "no", "f", "false", "off", "0"):
                return False
            raise ValueError("invalid truth value %r" % (val,))
        d, u = types.ModuleType("distutils"), types.ModuleType("distutils.util")
        d.__path__ = []
        u.strtobool = _strtobool
        d.util = u
        sys.modules["distutils"], sys.modules["distutils.util"] = d, u
    import probe_infer as PI
    if not getattr(PI, "_k1_memo", False):
        orig, cache = PI.install_native, {}

        def install_native(lean=True):
            if lean not in cache:
                cache[lean] = orig(lean)
            return cache[lean]
        PI.install_native = install_native
        PI._k1_memo = True


def rss_mb():
    hwm = cur = None
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                hwm = int(line.split()[1]) / 1024
            elif line.startswith("VmRSS:"):
                cur = int(line.split()[1]) / 1024
    except OSError:
        pass
    return cur, hwm


class StudyTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise StudyTimeout("study timeout")


def worker(a, wi, uids):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    install_paths(a.src, a.code)
    import numpy as np
    import torch
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(a.threads)
    import compact_infer as CI
    if os.path.dirname(os.path.abspath(CI.__file__)) != os.path.abspath(a.src):
        raise SystemExit("compact_infer imported from %s" % CI.__file__)
    out = Path(a.out)
    jp = out / ("part%03d.json" % wi)
    st = json.load(open(jp)) if jp.exists() else {"done": {}, "blocks": {}, "errors": {}, "anchor": {}, "sec": {}, "rss_mb": {}}
    tmp_root = Path(a.tmp) / ("w%03d_%d" % (wi, os.getpid()))
    ds = CI.CStudies(a.data, a.split, uids, str(tmp_root), a.code, S, eval_stride=GEOM["eval_stride"], views=GEOM["views"],
                     slices=GEOM["slices"], anchor_weights=a.anchor, phys_mm=GEOM["phys_mm"], side_model=None)
    ds._setup()
    if ds.bank is None or ds.side_net is not None:
        raise SystemExit("worker %d: anchor bank missing or a side model loaded" % wi)
    torch.set_num_threads(a.threads)          # CStudies._setup sets 2 threads on CPU; the build's own count wins
    fpx = os.open(out / "px.u8.npy", os.O_RDWR)
    fm = os.open(out / "meta.f32.npy", os.O_RDWR)
    signal.signal(signal.SIGALRM, _alarm)
    t0 = time.time()
    n_new = 0
    for uid in uids:
        if uid in st["done"]:
            continue
        key = hashlib.sha256(uid.encode()).hexdigest()[:24]
        work = tmp_root / ("s_" + key[:16])            # = CStudies.__getitem__'s work dir name
        ts = time.perf_counter()
        try:
            signal.alarm(int(a.study_timeout))
            # == CStudies.__getitem__ (compact_infer 34f433fe) up to builder.build; the window cut is the trainer's job
            ds._native_and_geometry(uid, work)
            ds.builder.cache = str(work)
            k2 = ds.KB.study_key(uid)
            with open(work / "studies" / (k2 + ".json"), encoding="utf-8") as fh:
                study = json.load(fh)
            rec = ds.bank.record(uid, study, geo_dir=str(work / "geometry"))
            (work / "anchors").mkdir(exist_ok=True)
            with open(work / "anchors" / (k2 + ".json"), "w", encoding="utf-8") as fh:
                json.dump(rec, fh)
            ds.builder.anchors = str(work / "anchors")
            item = ds.builder.build(uid)
            signal.alarm(0)
            if item.get("error"):
                raise ValueError(item["error"])
            px = np.ascontiguousarray(item["px"].numpy(), dtype=np.uint8)
            meta = np.ascontiguousarray(item["meta"].numpy(), dtype=np.float32)
            n = int(len(px))
            if n == 0:
                raise ValueError("no rows")
            if px.shape[1:] != (S, S) or meta.shape != (n, len(META_COLS)):
                raise ValueError("bad shapes %s %s" % (px.shape, meta.shape))
            cur = grab_rows(out, n)
            os.pwrite(fpx, px.tobytes(), NPY_HDR + cur * ROW_PX)
            os.pwrite(fm, meta.tobytes(), NPY_HDR + cur * ROW_META)
            blocks = []
            for b in item["blocks"]:
                blocks.append(dict(b, views=[[int(v), cur + int(off)] for v, off in b["views"]]))
            st["done"][uid] = [cur, n, len(blocks)]
            st["blocks"][uid] = blocks
            st["anchor"][uid] = rec.get("method")
            st["errors"].pop(uid, None)
        except BaseException as exc:
            signal.alarm(0)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            st["errors"][uid] = repr(exc)[:400]
        finally:
            shutil.rmtree(work, ignore_errors=True)
        dt = time.perf_counter() - ts
        st["sec"][uid] = round(dt, 2)
        cur_mb, hwm_mb = rss_mb()
        st["rss_mb"][uid] = [round(cur_mb or 0), round(hwm_mb or 0)]
        n_new += 1
        tmpj = str(jp) + ".tmp"
        json.dump(st, open(tmpj, "w"))
        os.replace(tmpj, jp)
        if n_new % a.log_every == 0:
            el = time.time() - t0
            print("worker %d: %d new, %d done, %d errors, %.1f s/study, rss %.0f MB, peak %.0f MB" % (
                wi, n_new, len(st["done"]), len(st["errors"]), el / n_new, cur_mb or 0, hwm_mb or 0), flush=True)
    os.close(fpx)
    os.close(fm)
    shutil.rmtree(tmp_root, ignore_errors=True)
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    st["final"] = {"pid": os.getpid(), "ru_maxrss_mb": round(ru), "new": n_new, "seconds": round(time.time() - t0, 1)}
    tmpj = str(jp) + ".tmp"
    json.dump(st, open(tmpj, "w"))
    os.replace(tmpj, jp)
    print("worker %d finished: %d new, %d done, %d errors, peak rss %.0f MB" % (wi, n_new, len(st["done"]), len(st["errors"]), ru),
          flush=True)


def load_parts(out):
    parts = sorted(out.glob("part*.json"))
    return [(p, json.load(open(p))) for p in parts]


def finalize(a):
    import numpy as np
    out = Path(a.out)
    st = json.load(open(out / "alloc.json"))
    N = int(st["cursor"])
    done, blocks, errors, anchor, sec, rss = {}, {}, {}, {}, {}, {}
    for p, s in load_parts(out):
        for u, v in s["done"].items():
            if u in done:
                continue                       # a study rendered twice (should not happen): the first record wins
            done[u] = v
            blocks[u] = s["blocks"][u]
            anchor[u] = s["anchor"].get(u)
        for u, e in s["errors"].items():
            errors.setdefault(u, e)
        sec.update(s.get("sec", {}))
        rss.update({u: v for u, v in s.get("rss_mb", {}).items()})
    errors = {u: e for u, e in errors.items() if u not in done}
    for u, (st_, n, nb) in done.items():
        if st_ + n > N:
            raise SystemExit("row range of %s beyond the cursor" % u)
    for name, descr, tail, rowb in (("px.u8.npy", "|u1", (S, S), ROW_PX), ("meta.f32.npy", "<f4", (len(META_COLS),), ROW_META)):
        p = out / name
        fd = os.open(p, os.O_RDWR)
        try:
            os.pwrite(fd, npy_header((N,) + tail, descr), 0)
            os.ftruncate(fd, NPY_HDR + N * rowb)
            os.fsync(fd)
        finally:
            os.close(fd)
    px = np.load(out / "px.u8.npy", mmap_mode="r")
    meta = np.load(out / "meta.f32.npy", mmap_mode="r")
    assert px.shape == (N, S, S) and meta.shape == (N, len(META_COLS)), (px.shape, meta.shape)
    order = sorted(done, key=lambda u: done[u][0])
    cfg = {"size": S, "local_mm": 100.0, "policy": "series_companded", "views": "v4", "view_recipe": "v4",
           "anchors": "SKM-TEA anchors (anatomy.pt %s) rebuilt per study via compact_infer 34f433fe CStudies" % ANATOMY_SHA[:8],
           "sidefix": "none (no side model; = rsna-knee-s6fall-solo B stage)", "duplicates": "collapse_exact",
           "rows": "one per (series, view, slice), physical order; each study's rows contiguous", "gold_first": 0,
           "builder": "px_build_s6.py", "builder_sha256": sha(__file__), "geometry": GEOM}
    index = {"order": order, "n_upper": N, "done": done, "blocks": blocks, "cursor": N, "errors": errors, "removed_series": 0,
             "meta_cols": META_COLS, "config": cfg}
    tmp = out / "index.json.tmp"
    json.dump(index, open(tmp, "w"))
    os.replace(tmp, out / "index.json")
    secs = sorted(sec[u] for u in done if u in sec)
    peaks = [v[1] for v in rss.values() if v]
    summ = {"studies_done": len(done), "errors": len(errors), "rows": N, "px_GB": round(N * ROW_PX / 1e9, 2),
            "rows_per_study": round(N / max(len(done), 1), 1),
            "s_per_study_median": secs[len(secs) // 2] if secs else None, "s_per_study_mean": round(sum(secs) / len(secs), 2) if secs else None,
            "s_per_study_p90": secs[int(len(secs) * .9)] if secs else None, "peak_rss_mb_max": max(peaks) if peaks else None,
            "anchor_methods": {m: sum(1 for v in anchor.values() if v == m) for m in sorted(set(map(str, anchor.values())))}}
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print("FINALIZED " + json.dumps(summ), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--code")
    ap.add_argument("--src")
    ap.add_argument("--anchor")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=None, help="per-study scratch (native render); default /dev/shm/px_build_s6_<out name>")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--uids", default=None, help="file with one study uid per line, or a csv with StudyInstanceUID (default: <data>/<split>.csv)")
    ap.add_argument("--study-timeout", type=float, default=900.)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--no-retry-errors", action="store_true")
    ap.add_argument("--mem-cap-gb", type=float, default=60., help="stop every worker (resumable) when worker RSS + tmpfs scratch exceed this")
    ap.add_argument("--progress-s", type=float, default=600.)
    ap.add_argument("--finalize", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if a.finalize:
        return finalize(a)
    for k in ("data", "code", "src", "anchor"):
        if not getattr(a, k):
            raise SystemExit("--%s is required for a build" % k)
    a.src, a.code = os.path.abspath(a.src), os.path.abspath(a.code)
    check_pins(a)
    a.tmp = a.tmp or ("/dev/shm/px_build_s6_" + out.name)
    import csv
    if a.uids:
        txt = Path(a.uids).read_text().splitlines()
        uids = [r["StudyInstanceUID"] for r in csv.DictReader(txt)] if txt and "StudyInstanceUID" in txt[0] else [l.strip() for l in txt if l.strip()]
    else:
        with open(Path(a.data) / (a.split + ".csv"), encoding="utf-8") as fh:
            uids = [r["StudyInstanceUID"] for r in csv.DictReader(fh)]
    uids = sorted(set(uids))
    init_files(out)
    run_lock = open(out / "run.lock", "a")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)      # one build per --out at a time (held until exit)
    except OSError:
        raise SystemExit("another px_build_s6 run holds %s" % (out / "run.lock"))
    prev = load_parts(out)
    done = set().union(*[set(s["done"]) for _, s in prev]) if prev else set()
    errs = set().union(*[set(s["errors"]) for _, s in prev]) if prev else set()
    todo = [u for u in uids if u not in done and (not a.no_retry_errors or u not in errs)]
    print("px_build_s6: %d studies listed, %d already done, %d to build (%d previous errors %s), procs %d x threads %d, out %s, tmp %s" % (
        len(uids), len(done & set(uids)), len(todo), len(errs - done), "skipped" if a.no_retry_errors else "retried", a.procs,
        a.threads, out, a.tmp), flush=True)
    if not todo:
        return
    shutil.rmtree(a.tmp, ignore_errors=True)          # scratch of a crashed earlier run (tmpfs = RAM); no other run holds run.lock
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = max(1, min(a.procs, len(todo)))
    ps = [ctx.Process(target=worker, args=(a, i, todo[i::procs])) for i in range(procs)]
    for p in ps:
        p.start()
    # parent: memory guard (worker RSS + tmpfs scratch) and a progress line every --progress-s seconds
    t0, last, n0 = time.time(), 0.0, len(done & set(uids))
    capped = False
    while any(p.is_alive() for p in ps):
        time.sleep(10)
        rss = 0
        for p in ps:
            try:
                for line in open("/proc/%d/status" % p.pid):
                    if line.startswith("VmRSS:"):
                        rss += int(line.split()[1]) * 1024
            except OSError:
                pass
        scratch = 0
        for root, _, files in os.walk(a.tmp):
            for f in files:
                try:
                    scratch += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        if a.mem_cap_gb and rss + scratch > a.mem_cap_gb * 2 ** 30:
            print("MEM_CAP %.1f GB > %.0f GB (rss %.1f, scratch %.1f): stopping the workers; the build is resumable" % (
                (rss + scratch) / 2 ** 30, a.mem_cap_gb, rss / 2 ** 30, scratch / 2 ** 30), flush=True)
            for p in ps:
                p.terminate()
            capped = True
            break
        if time.time() - last >= a.progress_s:
            last = time.time()
            n = 0
            for _, s in load_parts(out):
                n += len(s["done"])
            el = time.time() - t0
            rate = (n - n0) / max(el, 1e-6)
            print("PROGRESS %s done %d/%d (+%d this run), %.2f studies/min, ETA %.1f h, rss %.1f GB, scratch %.2f GB, rows %d" % (
                time.strftime("%H:%M:%S"), n, len(uids), n - n0, rate * 60, (len(uids) - n) / max(rate, 1e-9) / 3600,
                rss / 2 ** 30, scratch / 2 ** 30, json.load(open(out / "alloc.json"))["cursor"]), flush=True)
    for p in ps:
        p.join()
    print("all workers exited:", [p.exitcode for p in ps], flush=True)
    shutil.rmtree(a.tmp, ignore_errors=True)
    if capped:
        sys.exit(3)
    # v3 (10-15 review S6R-2): a worker that died (OOM killer, import error, crash outside the per-study try) left its share of
    # the studies neither done nor errored; v2 returned 0 here and the caller finalized a partial cache
    bad = [(i, p.exitcode) for i, p in enumerate(ps) if p.exitcode != 0]
    if bad:
        print("WORKER_FAILURE (worker, exit code) %s: the build is INCOMPLETE; re-run the same command (it resumes)" % bad, flush=True)
        sys.exit(4)
    n_done = n_err = 0
    for _, s_ in load_parts(out):
        n_done += len(s_["done"])
        n_err += len([u for u in s_["errors"] if u not in s_["done"]])
    print("BUILD_END listed %d done %d errors %d" % (len(uids), n_done, n_err), flush=True)


if __name__ == "__main__":
    main()
