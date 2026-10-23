"""CPU tests for the ct_s6v100 v2 fixes (10-15 review S6R-5 / S6R-4 / S6R-1).  Box CPU only (CUDA_VISIBLE_DEVICES must be
empty), GradScaler forced on (S6_SCALER=force) on an fp32 CPU run so the skip / scale accounting really executes.  Runs the SAME
checks against any trainer file (--trainer), so v1 (7f10854c) gives the fail-before and v2 the pass-after.

  N1 NONFINITE_LOSS   loss of step 2 forced to +inf (a forward overflow): the loss scale and the parameters must stay as they
                      were, the step counts as non-finite only (not also as scaler-skipped), the LR schedule still advances
  N2 INF_GRAD         step 4 has a finite loss but an inf gradient: the scaler skips the update and halves the scale (both
                      versions), counted as scaler-skipped only
  N3 SCALE_FLOOR      S6_MIN_SCALE = 2**15.5: one inf-gradient step (scale 2**16 -> 2**15) must ABORT the run at once
  N4 LOADER_TIMEOUT   every DataLoader with workers gets timeout = S6_LOADER_TIMEOUT (600 s)
  N5 HEAD_SEED        S6_HEAD_SEED=42 with --seed-base 43: Net init == the seed-42 init (head differs from seed 43's), while
                      the first training batch (data / sampler / augment streams) == the seed-43 run's
  N6 SMOKE_RECEIPT    the S6_SMOKE line carries loss_scale_min and nonfinite_steps
Prints one line per check and  S6V100_V2_TESTS PASS n/n  (or FAIL)."""
import argparse, contextlib, hashlib, importlib.util, io, json, os, sys, time
import numpy as np

assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "run with CUDA_VISIBLE_DEVICES= (empty): the GPU is off limits"
os.environ.update({"S6_DEVICE": "cpu", "S6_AMP": "fp32", "S6_ALL_DATA": "1", "S6_SCALER": "force", "S6_LOG_EVERY": "1"})
import torch
import timm, timm.data  # noqa: F401  (imported before any DataLoader patch)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print("%s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)


class InfGrad(torch.autograd.Function):
    """identity forward, +inf gradient backward (a finite loss whose gradient overflows)."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * float("inf")


class Stop(Exception):
    pass


def load(path, tag):
    spec = importlib.util.spec_from_file_location("ct_%s_%d" % (tag, time.time_ns()), path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    m._BaseNet, m._BaseBce = m.Net, m.bce_soft
    return m


def argv(px, targets, out, init, seed_base=43):
    return ["--px", px, "--targets", targets, "--out", out, "--arms", "E_prior", "--backbone", "convnext_small", "--init", init,
            "--seeds", "1", "--seed-base", str(seed_base), "--folds", "5", "--only-folds", "0", "--epochs", "16", "--batch", "2",
            "--train-windows", "4", "--crop", "288", "--slices", "3", "--slice-jitter", "0", "--ema", "0", "--phys-mm", "3.5",
            "--spatial-attn", "--lr", "1e-4", "--head-lr", "1e-3", "--wd", "0.05", "--workers", "1", "--eval-workers", "1",
            "--train-sampling", "uniform", "--aug", "crop", "--ckpt-every", "1", "--eval-stride", "3",
            "--i-have-user-authorisation-to-finetune"]


def run(mod, av, inject=None, env=None, stop_at_first_forward=False):
    """mod.main() in-process.  inject: {step_no (1-based): 'inf_loss' | 'inf_grad'}.  Returns a record."""
    rec = {"scale": [], "bias": [], "lr": [], "loaders": [], "init_sd": None, "in_sha": None, "exit": None, "out": ""}
    env = env or {}
    for k, v in env.items():
        setattr(mod, k, v)
    calls = {"n": 0}
    bce = mod._BaseBce

    def bce_inj(z, t, w):
        calls["n"] += 1
        v = bce(z, t, w)
        what = (inject or {}).get(calls["n"])
        if what == "inf_loss":
            return v * float("inf")
        if what == "inf_grad":
            return InfGrad.apply(v)
        return v

    mod.bce_soft = bce_inj
    Base = mod._BaseNet

    class RecNet(Base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            if rec["init_sd"] is None:
                rec["init_sd"] = {n: p.detach().clone() for n, p in self.state_dict().items()}
            rec["net"] = self

        def forward(self, img, meta, mask, chunk=0, return_aux=False):
            if self.training and rec["in_sha"] is None:
                rec["in_sha"] = hashlib.sha256(img.detach().numpy().tobytes() + meta.numpy().tobytes()).hexdigest()
                if stop_at_first_forward:
                    raise Stop()
            return super().forward(img, meta, mask, chunk=chunk, return_aux=return_aux)

    mod.Net = RecNet
    scalers = []
    mk = mod.s6_scaler

    def s6_scaler(dev):
        s = mk(dev)
        scalers.append(s)
        return s

    mod.s6_scaler = s6_scaler
    DL = torch.utils.data.DataLoader

    class DLrec(DL):                                    # a subclass (timm subclasses DataLoader at import time)
        def __init__(self, *a, **k):
            rec["loaders"].append((k.get("num_workers", 0), k.get("timeout", 0)))
            super().__init__(*a, **k)

    torch.utils.data.DataLoader = DLrec
    LR = torch.optim.lr_scheduler.LambdaLR
    orig_step = LR.step

    def step(self, *a, **k):
        r = orig_step(self, *a, **k)
        if scalers and "net" in rec and self.last_epoch >= 1:          # call 0 = the constructor's initial step
            s = scalers[-1]
            rec["scale"].append(s.get_scale() if s.is_enabled() else None)
            rec["bias"].append(rec["net"].bias.detach().clone())
            rec["lr"].append(self.get_last_lr()[0])
        return r

    LR.step = step
    old = sys.argv
    sys.argv = ["ct"] + av
    torch.manual_seed(0)
    np.random.seed(0)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            mod.main()
        rec["exit"] = "returned"
    except Stop:
        rec["exit"] = "stopped"
    except SystemExit as e:
        rec["exit"] = "exit %s" % (e.code,)
    finally:
        sys.argv = old
        LR.step = orig_step
        torch.utils.data.DataLoader = DL
        mod.Net, mod.bce_soft, mod.s6_scaler = Base, bce, mk
    rec["out"] = buf.getvalue()
    rec["state"] = {k: mod.S6_STATE.get(k) for k in ("steps", "skipped", "nonfinite", "scale_min")}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--px", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sha = hashlib.sha256(open(a.trainer, "rb").read()).hexdigest()
    print("trainer %s sha %s torch %s" % (a.trainer, sha[:16], torch.__version__), flush=True)
    os.makedirs(a.out, exist_ok=True)

    # N1 + N2 + N4 + N6: one smoke-style run, 5 steps; step 2 inf loss, step 4 inf gradient
    M = load(a.trainer, "n1")
    r = run(M, argv(a.px, a.targets, os.path.join(a.out, "n1"), a.init), inject={2: "inf_loss", 4: "inf_grad"},
            env={"S6_MAX_STEPS": 5, "S6_LOG_EVERY": 1})
    sc, b, lr = r["scale"], r["bias"], r["lr"]
    print("N1/N2 exit %s, scale per step %s, state %s" % (r["exit"], sc, r["state"]), flush=True)
    ok_len = len(sc) >= 5
    if ok_len:
        same_b2 = torch.equal(b[1], b[0])                 # step 2 (inf loss): no update
        moved_b3 = not torch.equal(b[2], b[1])           # step 3: a normal update
        same_b4 = torch.equal(b[3], b[2])                 # step 4 (inf grad): scaler skip, no update
        check("N1_NONFINITE_LOSS_KEEPS_SCALE", sc[1] == sc[0] == 2.0 ** 16, "scale after step 1 %s, after the inf-loss step 2 %s" % (sc[0], sc[1]))
        check("N1_NONFINITE_LOSS_NO_UPDATE", same_b2 and moved_b3, "params unchanged at step 2 %s, updated at step 3 %s" % (same_b2, moved_b3))
        check("N1_LR_SCHEDULE_ADVANCES", len(set(lr[:5])) == 5 and all(lr[i] < lr[i + 1] for i in range(4)), "lr %s (warm-up)" % ["%.3g" % x for x in lr[:5]])
        check("N2_INF_GRAD_SKIPS_AND_HALVES", same_b4 and sc[3] == sc[2] / 2, "params unchanged at step 4 %s, scale %s -> %s" % (same_b4, sc[2], sc[3]))
        st = r["state"]
        check("N1N2_COUNTS_NO_DOUBLE", st.get("steps") == 5 and st.get("skipped") == 1 and st.get("nonfinite") == 1,
              "steps %s skipped %s non-finite %s (want 5 / 1 / 1)" % (st.get("steps"), st.get("skipped"), st.get("nonfinite")))
    else:
        for n in ("N1_NONFINITE_LOSS_KEEPS_SCALE", "N1_NONFINITE_LOSS_NO_UPDATE", "N1_LR_SCHEDULE_ADVANCES", "N2_INF_GRAD_SKIPS_AND_HALVES", "N1N2_COUNTS_NO_DOUBLE"):
            check(n, False, "run gave %d steps (exit %s)" % (len(sc), r["exit"]))
    lw = [t for w, t in r["loaders"] if w > 0]
    check("N4_LOADER_TIMEOUT", bool(lw) and all(t == 600 for t in lw), "loaders (workers, timeout): %s" % r["loaders"])
    sm = [l for l in r["out"].splitlines() if l.startswith("S6_SMOKE ")]
    rec = json.loads(sm[-1][9:]) if sm else {}
    check("N6_SMOKE_RECEIPT_SCALE_FIELDS", r["exit"] == "exit 0" and rec.get("loss_scale_min") == 2.0 ** 15 and rec.get("nonfinite_steps") == 1,
          "exit %s, loss_scale %s, loss_scale_min %s, nonfinite_steps %s, scaler_skipped %s" % (
              r["exit"], rec.get("loss_scale"), rec.get("loss_scale_min"), rec.get("nonfinite_steps"), rec.get("scaler_skipped")))

    # N3: scale floor 2**15.5 -> the first gradient overflow (step 2) must abort
    M = load(a.trainer, "n3")
    r = run(M, argv(a.px, a.targets, os.path.join(a.out, "n3"), a.init), inject={2: "inf_grad"},
            env={"S6_MAX_STEPS": 4, "S6_LOG_EVERY": 1, "S6_MIN_SCALE": 2.0 ** 15.5})
    check("N3_SCALE_FLOOR_ABORTS", r["exit"].startswith("exit S6V100 ABORT: loss scale") and r["state"].get("steps") == 2,
          "exit %r after %s steps" % (r["exit"][:90], r["state"].get("steps")))

    # N5: head-init seed hook
    recs = {}
    for tag, sb, env in (("s43", 43, {}), ("s43_head42", 43, {"S6_HEAD_SEED": 42}), ("s42", 42, {})):
        M = load(a.trainer, "n5" + tag)
        recs[tag] = run(M, argv(a.px, a.targets, os.path.join(a.out, "n5_" + tag), a.init, seed_base=sb), env=env,
                        stop_at_first_forward=True)

    def eq(x, y, keys=None):
        if x is None or y is None or x.keys() != y.keys():
            return False
        return all(torch.equal(x[k], y[k]) for k in (keys if keys is not None else x))

    i43, ih, i42 = recs["s43"]["init_sd"], recs["s43_head42"]["init_sd"], recs["s42"]["init_sd"]
    head_keys = [k for k in (i43 or {}) if not k.startswith("enc.")]
    heads_differ = bool(head_keys) and not eq(i43, i42, head_keys)
    check("N5_HEAD_SEED_INIT_EQUALS_SEED42", heads_differ and eq(ih, i42),
          "seed-43 vs seed-42 heads differ %s (%d head tensors); head42 run init == seed-42 init (all %d tensors) %s" % (
              heads_differ, len(head_keys), len(i42 or {}), eq(ih, i42)))
    check("N5_HEAD_SEED_KEEPS_DATA_STREAM", recs["s43_head42"]["in_sha"] == recs["s43"]["in_sha"] != recs["s42"]["in_sha"],
          "first batch sha s43 %s head42 %s s42 %s" % tuple((recs[t]["in_sha"] or "None")[:12] for t in ("s43", "s43_head42", "s42")))
    n_ok = sum(ok for _, ok in RESULTS)
    print("S6V100_V2_TESTS %s %d/%d  (trainer sha %s)" % ("PASS" if n_ok == len(RESULTS) else "FAIL", n_ok, len(RESULTS), sha[:16]), flush=True)


if __name__ == "__main__":
    main()
