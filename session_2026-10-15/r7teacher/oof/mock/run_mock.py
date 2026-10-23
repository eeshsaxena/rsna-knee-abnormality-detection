"""Logic test of the R7-OOF notebook cells 1, 3, 4 with fake_worker.py instead of compact_infer.py (box CPU, no GPU).
usage: python run_mock.py NOTEBOOK.ipynb MOCKROOT DATA_DIR(train.csv, train_series.csv) R7_SRC_DIR(compact_train.py) BUNDLE_SRC(manifest + jsons)
The only edit to the cells: the compact_infer sha256 assert (the fake worker is not compact_infer.py)."""
import json, os, shutil, stat, sys, time
from pathlib import Path

nb_path, root, data_src, r7src, bsrc = map(Path, sys.argv[1:6])
cfg_over = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
if root.exists():
    shutil.rmtree(root)
for d in ("working", "tmp", "data", "src", "bundle", "code", "bin"):
    (root / d).mkdir(parents=True)
for f in ("train.csv", "train_series.csv"):
    os.symlink(data_src / f, root / "data" / f)
shutil.copy(Path(__file__).with_name("fake_worker.py"), root / "src" / "compact_infer.py")
for f in ("manifest.json", "state_stacker.json", "fluid_stacker.json", "state_spec.json"):
    shutil.copy(bsrc / f, root / "bundle" / f)
for k in range(4):
    (root / "bundle" / ("E_lrprior_s0_f%d.pt" % k)).write_bytes(b"fake")
smi = root / "bin" / "nvidia-smi"
smi.write_text("#!/bin/sh\necho 'Tesla T4'\necho 'Tesla T4'\n")
smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
os.environ["PATH"] = str(root / "bin") + os.pathsep + os.environ["PATH"]
for k, v in cfg_over.items():
    os.environ["R7OOF_" + k.upper()] = json.dumps(v)

nb = json.loads(nb_path.read_text(encoding="utf-8"))
src = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"]]
G = {"__name__": "__main__"}
exec(compile(src[1], "cell1", "exec"), G)
manifest = json.loads((root / "bundle" / "manifest.json").read_text())
G.update(WORK=root / "working", TMP=root / "tmp", DATA=root / "data", CODE=root / "code", SRC=root / "src", WORKER="compact_infer.py",
         ARM="E_lrprior", BUNDLE=root / "bundle", manifest=manifest,
         SOURCES={"compact_train.py": (r7src / "compact_train.py").read_text(encoding="utf-8")},
         WORKER_ARGS=["--backbone", "convnext_small", "--size", "384", "--crop", "352", "--eval-stride", "1", "--views", "v4",
                      "--state-spec", str(root / "bundle" / "state_spec.json"), "--slices", "3", "--phys-mm", "3.5", "--compile",
                      "--fluid-stacker", str(root / "bundle" / "fluid_stacker.json"), "--state-stacker", str(root / "bundle" / "state_stacker.json")])
ctl = src[3]
line = "assert hashlib.sha256((SRC / WORKER).read_bytes()).hexdigest() == manifest['src_sha256']['compact_infer.py']"
assert ctl.count(line) == 1
ctl = ctl.replace(line, "pass  # MOCK: fake worker")
t = time.time()
exec(compile(ctl, "cell3", "exec"), G)
exec(compile(src[4], "cell4", "exec"), G)
print("MOCK_DONE %.1fs" % (time.time() - t))
