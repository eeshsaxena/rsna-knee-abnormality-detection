"""unzip_parts.py (10-15, BOX_PLAN_1015 S5): parallel, resumable extraction of the competition zip, read either from one archive or
directly from its ordered byte-range part files (part.0 part.1 ... = the archive cut at increasing offsets), so the 265 GB parts
never need a concatenation copy. Python zipfile (ZIP64, deflate / stored; CRC-32 of every member checked on read).
Each process extracts a contiguous run of members in archive order (one sequential read stream per process).
A member is written to <dest>.part then renamed; a member already present with the right size is skipped (resume).
Member names with '..' or an absolute path are refused.
  python unzip_parts.py --parts part.0 part.1 ... part.7 --out DIR --procs 6 [--include REGEX] [--list-only]"""
import argparse, bisect, io, json, os, re, shutil, sys, time, zipfile


class MultiFile(io.RawIOBase):
    """read-only, seekable concatenation of files (os.pread, no copy)."""

    def __init__(self, paths):
        self.paths = list(paths)
        self.sizes = [os.path.getsize(p) for p in self.paths]
        self.starts = [0]
        for s in self.sizes:
            self.starts.append(self.starts[-1] + s)
        self.total = self.starts[-1]
        self.fds = [os.open(p, os.O_RDONLY) for p in self.paths]
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        elif whence == 2:
            self.pos = self.total + off
        if self.pos < 0:
            raise ValueError("negative seek")
        return self.pos

    def readinto(self, b):
        if self.pos >= self.total:
            return 0
        i = bisect.bisect_right(self.starts, self.pos) - 1
        rel = self.pos - self.starts[i]
        n = min(len(b), self.sizes[i] - rel)
        data = os.pread(self.fds[i], n, rel)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)

    def close(self):
        for fd in self.fds:
            try:
                os.close(fd)
            except OSError:
                pass
        super().close()


def open_zip(paths):
    return zipfile.ZipFile(io.BufferedReader(MultiFile(paths), buffer_size=8 << 20))


def safe_dest(out, name):
    if name.startswith("/") or "\\" in name or any(p == ".." for p in name.split("/")):
        raise ValueError("unsafe member name %r" % name)
    return os.path.join(out, name)


def worker(a, wi, lo, hi, q):
    zf = open_zip(a.parts)
    infos = sorted((i for i in zf.infolist() if not i.is_dir() and (not a.include or re.search(a.include, i.filename))),
                   key=lambda i: i.header_offset)[lo:hi]
    t0, nb, nf, ns, errs = time.time(), 0, 0, 0, []
    for k, info in enumerate(infos):
        try:
            dest = safe_dest(a.out, info.filename)
            if os.path.exists(dest) and os.path.getsize(dest) == info.file_size:
                ns += 1
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            tmp = dest + ".part"
            with zf.open(info) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 4 << 20)       # zipfile raises BadZipFile on a CRC mismatch at the member's end
            os.replace(tmp, dest)
            nb += info.file_size
            nf += 1
        except Exception as exc:
            errs.append([info.filename, repr(exc)[:200]])
        if (k + 1) % 2000 == 0:
            el = time.time() - t0
            print("w%d: %d/%d members, %d skipped, %.1f GB written, %.0f MB/s, %d errors" % (
                wi, k + 1, len(infos), ns, nb / 1e9, nb / 1e6 / max(el, 1e-6), len(errs)), flush=True)
    q.put({"worker": wi, "members": len(infos), "written": nf, "skipped": ns, "bytes": nb, "errors": errs,
           "seconds": round(time.time() - t0, 1)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True, help="the archive, or its byte-range parts IN ORDER")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--include", default=None, help="regex on member names (default: every member)")
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    zf = open_zip(a.parts)
    infos = sorted((i for i in zf.infolist() if not i.is_dir() and (not a.include or re.search(a.include, i.filename))),
                   key=lambda i: i.header_offset)
    tot_c = sum(i.compress_size for i in infos)
    tot_u = sum(i.file_size for i in infos)
    methods = {}
    for i in infos:
        methods[i.compress_type] = methods.get(i.compress_type, 0) + 1
    tops = {}
    for i in infos:
        t = i.filename.split("/")[0]
        tops[t] = tops.get(t, 0) + 1
    print("archive %.1f GB in %d part(s); %d members selected, %.1f GB compressed -> %.1f GB; methods %s; top level %s; "
          "central directory read in %.1f s" % (MultiFile(a.parts).total / 1e9, len(a.parts), len(infos), tot_c / 1e9, tot_u / 1e9,
                                                methods, dict(sorted(tops.items(), key=lambda kv: -kv[1])[:8]), time.time() - t0), flush=True)
    for i in infos[:20000]:
        safe_dest(a.out, i.filename)
    if a.list_only:
        return
    os.makedirs(a.out, exist_ok=True)
    # contiguous runs balanced by compressed bytes
    cuts, acc, P = [0], 0, max(1, a.procs)
    for k, i in enumerate(infos):
        acc += i.compress_size
        if len(cuts) < P and acc >= tot_c * len(cuts) / P:
            cuts.append(k + 1)
    cuts.append(len(infos))
    if len(cuts) == 2:                       # one process: extract inline
        import queue
        q = queue.Queue()
        worker(a, 0, 0, len(infos), q)
        res, codes = [q.get()], [0]
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        q = ctx.Queue()
        ps = [ctx.Process(target=worker, args=(a, w, cuts[w], cuts[w + 1], q)) for w in range(len(cuts) - 1)]
        for p in ps:
            p.start()
        res = [q.get() for _ in ps]
        for p in ps:
            p.join()
        codes = [p.exitcode for p in ps]
    errs = [e for r in res for e in r["errors"]]
    rec = {"parts": a.parts, "out": a.out, "members": len(infos), "written": sum(r["written"] for r in res),
           "skipped": sum(r["skipped"] for r in res), "bytes_written": sum(r["bytes"] for r in res), "errors": len(errs),
           "error_sample": errs[:20], "seconds": round(time.time() - t0, 1), "workers": res, "exitcodes": codes}
    json.dump(rec, open(os.path.join(a.out, "_unzip_receipt.json"), "w"), indent=1)
    print("UNZIP %s: %d members, %d written, %d skipped, %.1f GB, %d errors, %.0f s" % (
        "OK" if not errs and all(c == 0 for c in codes) else "FAIL", rec["members"], rec["written"], rec["skipped"],
        rec["bytes_written"] / 1e9, len(errs), rec["seconds"]), flush=True)
    sys.exit(0 if not errs else 1)


if __name__ == "__main__":
    main()
