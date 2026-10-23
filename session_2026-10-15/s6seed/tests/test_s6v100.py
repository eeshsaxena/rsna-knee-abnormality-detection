"""CPU unit tests: ct_s6v100.py (patched copy) vs the ORIGINAL compact_train.py 14767c7a (BOX_PLAN_1015 S2).  Box CPU only, the GPU
is never touched (CUDA_VISIBLE_DEVICES must be empty).  The original file is imported read-only; the test redirects its
torch.device("cuda") to the CPU through a module-local torch proxy (the file itself is never edited).

  T1 READER     windows byte-equal (img, meta, index) on every fixture study: train mode (uniform + cover sampling, slice jitter 0 /
                .5, 3 epochs, 2 seeds) and eval mode (stride 1 / 3), for S6_PX_GAP in {16 (default), 0, 4, -1, 10**6}
  T2 FWD-BWD    one fp32 forward + backward on the first fold-0 training batch (2 studies x 64 windows, crop 288, recipe model:
                convnext_small + spatial attention + grad ckpt): loss and every parameter gradient equal, same input bytes
  T3 STEP       one full optimizer step (clip 5, AdamW, LambdaLR): parameters after the step equal for original / patched (scaler
                off) / patched with an ENABLED GradScaler (S6_SCALER=force, scale 2**16; the unscale is exact for a power of 2)
  T4 ALL-DATA   S6_ALL_DATA=1 CLI run, 2 epochs x 2 steps on a 4-study pool (+1 gold): training set = whole pool, steps =
                epochs x pool//batch, models/E_prior_s43_fall.pt + .DONE.json, file matches compact_infer's <arm>_s*_f*.pt glob and
                loads STRICT into the ORIGINAL Net; the original Net + original reader reproduce the run's gold logits; a run
                interrupted after epoch 0 (--stop-after-epoch 0) and resumed gives the same final weights (incl. scaler state)
  T5 SMOKE      S6_MAX_STEPS=2: exit 0, an S6_SMOKE receipt line, no model written
  T6 SOURCE     the only bf16 left in the patched file is the S6_AMP=bf16 option; unified diff written for review
Prints one line per check and a final  S6V100_TESTS PASS n/n  (or FAIL)."""
import argparse, copy, difflib, fnmatch, hashlib, importlib.util, json, os, re, shutil, subprocess, sys, time, types
import numpy as np

assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "run with CUDA_VISIBLE_DEVICES= (empty): the GPU is off limits"
os.environ["S6_DEVICE"] = "cpu"
os.environ["S6_AMP"] = "fp32"
os.environ["S6_ALL_DATA"] = "0"
import torch

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print("%s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class _TorchCPU(types.ModuleType):
    """torch, except torch.device('cuda...') -> cpu (only inside the original module's globals)."""

    def __getattr__(self, k):
        return getattr(torch, k)


def cpu_proxy():
    p = _TorchCPU("torch")
    p.device = lambda *a, **k: torch.device("cpu") if (a and str(a[0]).startswith("cuda")) else torch.device(*a, **k)
    return p


class StopStep(Exception):
    pass


def instrument(mod):
    """Record the Net instances, the loss values and the encoder input bytes of a module's main()."""
    rec = {"nets": [], "loss": [], "in_sha": []}
    Base = mod._BaseNet                       # always wrap the module's own classes (never a previous wrapper)

    class RecNet(Base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            rec["nets"].append(self)

        def forward(self, img, meta, mask, chunk=0, return_aux=False):
            if self.training:
                rec["in_sha"].append(hashlib.sha256(img.detach().cpu().numpy().tobytes() + meta.cpu().numpy().tobytes()).hexdigest())
            return super().forward(img, meta, mask, chunk=chunk, return_aux=return_aux)

    mod.Net = RecNet
    bce = mod._BaseBce

    def bce_rec(z, t, w):
        v = bce(z, t, w)
        rec["loss"].append(v.detach().clone())
        return v

    mod.bce_soft = bce_rec
    return rec, Base


def run_main(mod, argv, stop_after_first_step):
    """mod.main() with argv; optionally raise StopStep right after the first optimizer step (2nd LambdaLR.step call)."""
    LR = torch.optim.lr_scheduler.LambdaLR
    orig_step = LR.step
    calls = {"n": 0}

    def step(self, *a, **k):
        r = orig_step(self, *a, **k)
        calls["n"] += 1
        if stop_after_first_step and calls["n"] >= 2:            # call 1 = the constructor's initial step
            raise StopStep()
        return r

    LR.step = step
    old = sys.argv
    sys.argv = ["ct"] + argv
    torch.manual_seed(0)
    np.random.seed(0)
    try:
        mod.main()
        return "returned"
    except StopStep:
        return "stopped"
    finally:
        sys.argv = old
        LR.step = orig_step


def params(net):
    return {n: p.detach().clone() for n, p in net.named_parameters()}


def grads(net):
    return {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in net.named_parameters()}


def maxdiff(da, db):
    if da.keys() != db.keys():
        return float("inf"), "key sets differ"
    worst, wk, nonbit = 0.0, None, 0
    for k in da:
        a, b = da[k], db[k]
        if (a is None) != (b is None):
            return float("inf"), k + " None mismatch"
        if a is None:
            continue
        d = float((a.double() - b.double()).abs().max()) if a.numel() else 0.0
        nonbit += int(not torch.equal(a, b))
        if d > worst:
            worst, wk = d, k
    return worst, "%d of %d tensors not bitwise equal; worst %s" % (nonbit, len(da), wk)


def t1_reader(CO, CN, px, uids):
    gaps = [16, 0, 4, -1, 10 ** 6]
    n_cmp = n_bad = 0
    io = {}
    variants = []
    for seed in (9000 + 17 * 43, 12345):
        for sampling in ("uniform", "cover"):
            for jit in (0.0, 0.5):
                variants.append(dict(mode="train", seed=seed, train_sampling=sampling, slice_jitter=jit, epochs=(0, 1, 7)))
    for es in (1, 3):
        variants.append(dict(mode="eval", seed=0, eval_stride=es, epochs=(0,)))
    for v in variants:
        kw = dict(eval_stride=v.get("eval_stride", 3), train_sampling=v.get("train_sampling", "uniform"), cover_stride=3, slices=3,
                  slice_jitter=v.get("slice_jitter", 0.0), phys_mm=3.5)
        do = CO.Studies(px, uids, 64, v["mode"], v["seed"], **kw)
        dns = {}
        for g in gaps:                       # one patched dataset per gap (S6_PX_GAP is read at call time)
            dns[g] = CN.Studies(px, uids, 64, v["mode"], v["seed"], **kw)
        for ep in v["epochs"]:
            do.epoch = ep
            for i in range(len(uids)):
                a_img, a_meta, a_i = do[i]
                for g in gaps:
                    CN.S6_PX_GAP = g
                    dn = dns[g]
                    dn.epoch = ep
                    b_img, b_meta, b_i = dn[i]
                    same = (a_i == b_i and a_img.dtype == b_img.dtype and a_img.shape == b_img.shape and torch.equal(a_img, b_img)
                            and a_meta.shape == b_meta.shape and a_meta.numpy().tobytes() == b_meta.numpy().tobytes())
                    n_cmp += 1
                    n_bad += int(not same)
        if v["mode"] == "train" and v["train_sampling"] == "uniform" and v["slice_jitter"] == 0.0 and v["seed"] == 9000 + 17 * 43:
            for g in gaps:
                io[g] = [dns[g].read_stats[0] / (len(uids) * len(v["epochs"])), dns[g].read_stats[1] / 1e6 / (len(uids) * len(v["epochs"]))]
    CN.S6_PX_GAP = 16
    # negative control: a reader that is off by one row must be caught by the same comparison
    kw = dict(eval_stride=3, train_sampling="uniform", cover_stride=3, slices=3, slice_jitter=0.0, phys_mm=3.5)
    do = CO.Studies(px, uids, 64, "train", 9731, **kw)
    dn = CN.Studies(px, uids, 64, "train", 9731, **kw)
    good = CN.Studies._px_rows
    dn._px_rows = types.MethodType(lambda self, f, idx: good(self, f, np.minimum(idx + 1, self.px[f].shape[0] - 1)), dn)
    caught = not torch.equal(do[0][0], dn[0][0])
    check("T1_NEGATIVE_CONTROL", caught, "an off-by-one-row reader differs: %s" % caught)
    check("T1_READER_BYTE_EQUAL", n_bad == 0 and n_cmp > 0, "%d study-windows sets compared (%d studies x %d variants x %d gaps x epochs), %d differ" % (
        n_cmp, len(uids), len(variants), len(gaps), n_bad))
    print("T1 io per study (train uniform, 64 windows): " + "; ".join("gap %s: %.1f preads, %.1f MB" % (g, v[0], v[1]) for g, v in io.items()), flush=True)
    return io


def base_argv(px, targets, out, init, extra=()):
    return ["--px", px, "--targets", targets, "--out", out, "--arms", "E_prior", "--backbone", "convnext_small", "--init", init,
            "--seeds", "1", "--seed-base", "43", "--folds", "5", "--only-folds", "0", "--epochs", "16", "--batch", "2",
            "--train-windows", "64", "--crop", "288", "--slices", "3", "--slice-jitter", "0", "--ema", "0", "--phys-mm", "3.5",
            "--spatial-attn", "--grad-ckpt", "--lr", "1e-4", "--head-lr", "1e-3", "--wd", "0.05", "--workers", "1",
            "--eval-workers", "1", "--train-sampling", "uniform", "--aug", "crop", "--ckpt-every", "1", "--eval-stride", "3"] + list(extra)


def t2_t3(CO, CN, px, targets, init, out):
    # T2: forward + backward only
    ro, _ = instrument(CO)
    r = run_main(CO, base_argv(px, targets, os.path.join(out, "t2_orig"), init, ["--forward-only"]), False)
    go, lo, so = grads(ro["nets"][-1]), ro["loss"][-1], ro["in_sha"][-1]
    rn, _ = instrument(CN)
    r2 = run_main(CN, base_argv(px, targets, os.path.join(out, "t2_new"), init, ["--forward-only"]), False)
    gn, ln, sn = grads(rn["nets"][-1]), rn["loss"][-1], rn["in_sha"][-1]
    d, why = maxdiff(go, gn)
    check("T2_SAME_INPUT_BATCH", so == sn, "input sha %s / %s" % (so[:12], sn[:12]))
    check("T2_LOSS_EQUAL_FP32", torch.equal(lo, ln), "loss orig %.10f new %.10f" % (float(lo), float(ln)))
    check("T2_GRADS_EQUAL_FP32", d == 0.0, "max |dgrad| %.3g (%s); %d params with grad" % (d, why, sum(v is not None for v in go.values())))
    # T3: one full optimizer step
    res = {}
    for tag, mod, env in (("orig", CO, {}), ("new", CN, {"S6_SCALER": "auto"}), ("new_scaler", CN, {"S6_SCALER": "force"})):
        if mod is CN:
            CN.S6_SCALER = env["S6_SCALER"]
        rr, _ = instrument(mod)
        how = run_main(mod, base_argv(px, targets, os.path.join(out, "t3_" + tag), init, ["--i-have-user-authorisation-to-finetune"]), True)
        net = rr["nets"][-1]
        res[tag] = (how, params(net), grads(net), rr["loss"][-1], rr["in_sha"][-1])
    CN.S6_SCALER = "auto"
    ok_how = all(v[0] == "stopped" for v in res.values())
    check("T3_ONE_STEP_RAN", ok_how, str({k: v[0] for k, v in res.items()}))
    check("T3_SAME_INPUT_BATCH", len({v[4] for v in res.values()}) == 1, "")
    dp1, w1 = maxdiff(res["orig"][1], res["new"][1])
    dp2, w2 = maxdiff(res["orig"][1], res["new_scaler"][1])
    dg2, wg2 = maxdiff(res["orig"][2], res["new_scaler"][2])
    check("T3_PARAMS_EQUAL_SCALER_OFF", dp1 == 0.0, "max |dparam| %.3g (%s)" % (dp1, w1))
    check("T3_GRADS_EQUAL_SCALER_ON_AFTER_UNSCALE", dg2 == 0.0, "max |dgrad| %.3g (%s)" % (dg2, wg2))
    check("T3_PARAMS_EQUAL_SCALER_ON", dp2 == 0.0, "max |dparam| %.3g (%s)" % (dp2, w2))
    return res


def cli(script, argv, env_extra, log):
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "S6_DEVICE": "cpu", "S6_AMP": "fp32"})
    env.update(env_extra)
    with open(log, "w") as fh:
        p = subprocess.run([sys.executable, "-u", script] + argv, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return p.returncode, open(log).read()


def t4_t5(CO, script, px, fixture_targets, init, out):
    import csv
    rows = list(csv.DictReader(open(fixture_targets, encoding="utf-8")))
    idx = json.load(open(os.path.join(px, "index.json")))
    done = {u for u, v in idx["done"].items() if v[1] > 0}
    pool = [r for r in rows if r["is_gold"] == "0" and r["StudyInstanceUID"] in done][:4]
    gold = [r for r in rows if r["is_gold"] == "1" and r["StudyInstanceUID"] in done][:1]
    t4t = os.path.join(out, "targets_t4.csv")
    with open(t4t, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(pool + gold)
    small = ["--epochs", "2", "--train-windows", "8", "--eval-stride", "9", "--resume", "--i-have-user-authorisation-to-finetune"]

    def argv(o, extra=()):
        a = base_argv(px, t4t, o, init)
        for k in ("--epochs", "--train-windows", "--eval-stride"):          # drop the base values that `small` overrides
            i = a.index(k)
            del a[i:i + 2]
        return a + small + list(extra)

    env = {"S6_ALL_DATA": "1", "S6_SCALER": "force", "S6_LOG_EVERY": "1"}
    o1 = os.path.join(out, "t4_full")
    rc, log = cli(script, argv(o1), env, os.path.join(out, "t4_full.log"))
    mp = os.path.join(o1, "models", "E_prior_s43_fall.pt")
    dj = os.path.join(o1, "models", "E_prior_s43_fall.DONE.json")
    check("T4_RUN_OK", rc == 0 and "ALL DONE" in log, "rc %d" % rc)
    m = re.search(r"train studies (\d+), holdout studies (\d+), steps per epoch (\d+), total steps (\d+)", log)
    check("T4_ALL_DATA_TRAIN_SET", m is not None and m.groups() == ("4", "0", "2", "4"), "log: %s" % (m.group(0) if m else None))
    ok = os.path.exists(mp) and os.path.exists(dj)
    done_rec = json.load(open(dj)) if ok else {}
    sha = hashlib.sha256(open(mp, "rb").read()).hexdigest() if ok else None
    check("T4_DONE_JSON", ok and done_rec.get("model_sha256") == sha and done_rec.get("train_studies") == 4 and done_rec.get("optimizer_steps") == 4
          and done_rec.get("all_data") is True and done_rec.get("holdout_studies") == 0 and done_rec.get("file_tag") == "fall",
          json.dumps({k: done_rec.get(k) for k in ("model", "train_studies", "optimizer_steps", "all_data", "file_tag")}))
    check("T4_COMPACT_INFER_GLOB", fnmatch.fnmatch("E_prior_s43_fall.pt", "E_prior_s*_f*.pt") and fnmatch.fnmatch("E_prior_s42_fall.pt", "E_prior_s*_f*.pt"), "")
    # strict load into the ORIGINAL Net + reproduce the run's gold logits with the original reader / predict maths (fp32)
    sd = torch.load(mp, map_location="cpu", weights_only=True)
    net = CO._BaseNet("convnext_small", None, crop=288, in_chans=3, spatial=True)
    try:
        net.load_state_dict(sd, strict=True)
        strict_ok = True
    except Exception as exc:
        strict_ok = False
        print(exc)
    check("T4_STRICT_LOAD_ORIGINAL_NET", strict_ok, "%d tensors" % len(sd))
    preds = json.load(open(os.path.join(o1, "partial_preds_E_prior_s43.json")))
    net = net.eval().to(memory_format=torch.channels_last)
    ds = CO.Studies(px, preds["gold_uids"], 8, "eval", 43, eval_stride=9, train_sampling="uniform", cover_stride=3, slices=3, phys_mm=3.5)
    zz = []
    with torch.no_grad():
        for i in range(len(ds)):
            img, meta, _ = ds[i]
            img = img[None]
            o = (img.shape[-1] - 288) // 2
            mask = torch.ones((1, img.shape[1]), dtype=torch.bool)
            zz.append(net(img[..., o:o + 288, o:o + 288], meta[None], mask, chunk=512)[0].float().numpy())
    gl = np.array(preds["gold_logits"], np.float64)
    dz = float(np.abs(np.array(zz, np.float64) - gl).max()) if len(zz) else float("inf")
    check("T4_ORIGINAL_NET_REPRODUCES_GOLD_LOGITS", len(zz) == len(gl) == len(gold) and dz <= 2e-5, "max |dz| %.2g over %d gold x 12 (json rounds to 1e-5); gold logits not /K" % (dz, len(zz)))
    # resume: stop after epoch 0, then resume -> same final weights as the uninterrupted run
    o2 = os.path.join(out, "t4_resume")
    rc_a, log_a = cli(script, argv(o2, ["--stop-after-epoch", "0"]), env, os.path.join(out, "t4_resume_a.log"))
    ck = os.path.join(o2, "ckpt", "E_prior_s43_fall.pt")
    ck_ok = os.path.exists(ck)
    ck_scaler = torch.load(ck, map_location="cpu", weights_only=False).get("scaler") if ck_ok else None
    rc_b, log_b = cli(script, argv(o2), env, os.path.join(out, "t4_resume_b.log"))
    m2 = os.path.join(o2, "models", "E_prior_s43_fall.pt")
    same = False
    if os.path.exists(m2):
        sd2 = torch.load(m2, map_location="cpu", weights_only=True)
        same = sd.keys() == sd2.keys() and all(torch.equal(sd[k], sd2[k]) for k in sd)
    check("T4_RESUME_EQUALS_UNINTERRUPTED", rc_a == 3 and ck_ok and bool(ck_scaler) and rc_b == 0 and "resumed" in log_b and same,
          "stop rc %d, ckpt %s (scaler state %s), resume rc %d, weights equal %s" % (rc_a, ck_ok, bool(ck_scaler), rc_b, same))
    # T5 smoke path
    o3 = os.path.join(out, "t5_smoke")
    rc5, log5 = cli(script, argv(o3), {"S6_ALL_DATA": "1", "S6_MAX_STEPS": "2", "S6_LOG_EVERY": "1"}, os.path.join(out, "t5_smoke.log"))
    sm = [l for l in log5.splitlines() if l.startswith("S6_SMOKE ")]
    rec5 = json.loads(sm[-1][9:]) if sm else {}
    check("T5_SMOKE_RECEIPT", rc5 == 0 and rec5.get("steps") == 2 and not os.path.exists(os.path.join(o3, "models")),
          json.dumps({k: rec5.get(k) for k in ("steps", "s_per_step_median", "nonfinite_losses", "train_studies", "steps_per_epoch")}))


def t6_source(orig_path, new_path, out):
    src = open(new_path, encoding="utf-8").read()
    lines = [l for l in src.splitlines() if "bfloat16" in l]
    check("T6_ONLY_BF16_IS_THE_OPTION", len(lines) == 1 and "S6_AMP == \"fp16\" else torch.bfloat16" in lines[0], "%d bf16 lines" % len(lines))
    a = open(orig_path, encoding="utf-8").read().splitlines(keepends=True)
    b = src.splitlines(keepends=True)
    diff = list(difflib.unified_diff(a, b, "compact_train.py (14767c7a)", "ct_s6v100.py"))
    open(os.path.join(out, "ct_s6v100.diff"), "w", encoding="utf-8").writelines(diff)
    removed = [l for l in diff if l.startswith("-") and not l.startswith("---")]
    check("T6_DIFF_WRITTEN", len(removed) <= 25, "%d original lines replaced; diff at %s" % (len(removed), os.path.join(out, "ct_s6v100.diff")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--px", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--only", default="T1,T2,T4,T6")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert sha(a.orig).startswith("14767c7a"), "the original must be compact_train 14767c7a"
    print("orig %s  new %s  torch %s threads %d" % (sha(a.orig)[:16], sha(a.new)[:16], torch.__version__, a.threads), flush=True)
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out)
    CO = load_module("ct_orig", a.orig)
    CO.torch = cpu_proxy()
    CN = load_module("ct_new", a.new)
    for M in (CO, CN):
        M._BaseNet, M._BaseBce = M.Net, M.bce_soft
    assert CN.S6_DEVICE == "cpu" and CN.S6_AMP == "fp32" and not CN.S6_ALL_DATA
    idx = json.load(open(os.path.join(a.px, "index.json")))
    uids = [u for u in idx["order"] if idx["done"][u][1] > 0]
    only = set(a.only.split(","))
    t0 = time.time()
    if "T1" in only:
        t1_reader(CO, CN, a.px, uids)
        print("T1 %.0f s" % (time.time() - t0), flush=True)
    if "T2" in only:
        try:
            t2_t3(CO, CN, a.px, a.targets, a.init, a.out)
        except Exception:
            import traceback
            traceback.print_exc()
            check("T2_T3_NO_EXCEPTION", False)
        for M in (CO, CN):
            M.Net, M.bce_soft = M._BaseNet, M._BaseBce
        print("T2/T3 %.0f s" % (time.time() - t0), flush=True)
    if "T4" in only:
        try:
            t4_t5(CO, a.new, a.px, a.targets, a.init, a.out)
        except Exception:
            import traceback
            traceback.print_exc()
            check("T4_T5_NO_EXCEPTION", False)
        print("T4/T5 %.0f s" % (time.time() - t0), flush=True)
    if "T6" in only:
        t6_source(a.orig, a.new, a.out)
    n_ok = sum(ok for _, ok in RESULTS)
    print("S6V100_TESTS %s %d/%d" % ("PASS" if n_ok == len(RESULTS) and RESULTS else "FAIL", n_ok, len(RESULTS)), flush=True)


if __name__ == "__main__":
    main()
