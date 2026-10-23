"""PX_PARITY reference (10-15): Heliosli compact_infer 34f433fe main() UNCHANGED, run on the box CPU through nartaa's S6F0 launcher
semantics (rsna-knee-s6fall-solo LAUNCHER: sys.path [code, src], training.download_data stub, distutils guard, memoised
install_native, LocalCStudies = no scored item without an SKM-TEA anchor, K1_PX_DUMP window hashes).
The only change: compact_infer.main() hard-codes torch.device("cuda"); here the module's `torch` name is a proxy whose device("cuda")
is the CPU and whose cuda.get_device_capability / get_device_name answer for a missing GPU. Pass --amp fp32 (no autocast).
Env: S6_SRC (heliosli sources), S6_CODE (cloud_code with the training package). CUDA must be hidden by the caller."""
import hashlib, json, os, sys, types
SRC = os.path.abspath(os.environ["S6_SRC"])
CODE = os.path.abspath(os.environ["S6_CODE"])
ANCHOR_VIEWS = ("v3k", "v4")


def _strtobool(val):
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
        import distutils.util  # noqa: F401
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
import torch
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


class _CpuCuda:
    def __getattr__(self, k):
        return getattr(torch.cuda, k)

    @staticmethod
    def get_device_capability(*a, **k):
        return (7, 0)

    @staticmethod
    def get_device_name(*a, **k):
        return "cpu (PX_PARITY reference)"


class _TorchCpu:
    """compact_infer's `torch`: every attribute is the real torch except device('cuda') -> cpu and the cuda queries above."""
    cuda = _CpuCuda()

    def __getattr__(self, k):
        return getattr(torch, k)

    @staticmethod
    def device(d, *a, **k):
        return torch.device("cpu") if str(d).startswith("cuda") else torch.device(d, *a, **k)


if __name__ == "__main__":
    if torch.cuda.is_available():
        raise SystemExit("PX_PARITY reference must run with CUDA hidden (CUDA_VISIBLE_DEVICES=)")
    if "--amp" not in sys.argv or sys.argv[sys.argv.index("--amp") + 1] != "fp32":
        raise SystemExit("PX_PARITY reference needs --amp fp32 on CPU")
    CI.torch = _TorchCpu()
    CI.main()
