"""Single-process px_build_s6 run (the worker function called inline, no spawn), for byte tests of builder variants:
  python inline_build.py BUILDER.py --data D --src S --code C --anchor A --out OUT --uids LIST --threads T
then finalize with the same builder. Writes only under --out and its tmp."""
import argparse, importlib.util, os, sys
builder = sys.argv[1]
spec = importlib.util.spec_from_file_location("pxb", builder)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
ap = argparse.ArgumentParser()
for k in ("--data", "--src", "--code", "--anchor", "--out", "--uids"):
    ap.add_argument(k, required=True)
ap.add_argument("--split", default="train")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--study-timeout", type=float, default=900.)
ap.add_argument("--log-every", type=int, default=1)
a = ap.parse_args(sys.argv[2:])
a.src, a.code = os.path.abspath(a.src), os.path.abspath(a.code)
a.tmp = "/dev/shm/px_inline_" + os.path.basename(a.out)
B.check_pins(a)
uids = [l.split(",")[0] for l in open(a.uids).read().splitlines() if l and not l.startswith("StudyInstanceUID")]
B.init_files(B.Path(a.out))
B.worker(a, 0, uids)
B.finalize(argparse.Namespace(out=a.out))
