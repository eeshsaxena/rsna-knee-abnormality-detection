"""Recipe / launcher / targets consistency (10-15 review S6R-6). CPU, seconds, reads files only.
  R1 the launcher's trainer argv (S6_PRINT_RECIPE=1) == recipe_s6s43.json argv (paths compared by basename; --out / --resume are
     the launcher's own) and the recipe env is a subset of the launcher env (S6_EXPECT_TRAIN excepted)
  R2 the recipe does not call seed 43 'the one intended change' and lists every known difference vs the scored S6fall (seed,
     labels, pool, steps, precision), the gold58 in-sample caveat, and the step count 16 x floor(4,349 / 2) = 34,784
  R3 the targets builder really uses blendjev_sol (the label claim matches the file that trains the member)
  R4 the smoke gate in the recipe has the loss-scale bars the v2 launcher enforces
Usage: python test_recipe_consistency.py --recipe R --launcher L --targets-builder B --px PX  ->  RECIPE_TESTS PASS n/n"""
import argparse, json, os, subprocess, sys

RES = []


def check(n, ok, d=""):
    RES.append(bool(ok))
    print("%s %s %s" % ("PASS" if ok else "FAIL", n, d), flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--recipe", required=True)
ap.add_argument("--launcher", required=True)
ap.add_argument("--targets-builder", required=True)
ap.add_argument("--px", required=True)
a = ap.parse_args()
r = json.load(open(a.recipe))
env = dict(os.environ, S6_PRINT_RECIPE="1", S6_DRYRUN="1", S6_QUEUE_DIR="/tmp/none_queue", S6_KIT="/tmp/s6_recipe_kit")
p = subprocess.run(["bash", a.launcher, "--px", a.px, "--now"], env=env, capture_output=True, text=True)
out = p.stdout.strip().splitlines()
if "--ENV--" in out:
    i = out.index("--ENV--")
    largv, lenv = out[:i], dict(l.split("=", 1) for l in out[i + 1:])
else:
    largv, lenv = [], {}


def norm(av, drop=("--out", "--resume")):
    o, k = [], 0
    while k < len(av):
        x = av[k]
        if x in drop:
            k += 1 if x == "--resume" else 2
            continue
        if x in ("--px", "--targets", "--init"):
            o += [x, os.path.basename(av[k + 1]) if x != "--px" else "<px>"]
            k += 2
            continue
        o.append(x)
        k += 1
    return o


ra, la = norm(r["argv"]), norm(largv)
check("R1_ARGV_EQUAL", bool(la) and ra == la, "recipe %d tokens, launcher %d tokens%s" % (len(ra), len(la), "" if ra == la else
      "; first diff %s" % str(next(((x, y) for x, y in zip(ra, la) if x != y), (len(ra), len(la))))))
renv = {k: v for k, v in r.get("env", {}).items() if k != "S6_EXPECT_TRAIN"}
miss = {k: v for k, v in renv.items() if lenv.get(k) != v}
check("R1_ENV_SUBSET", bool(lenv) and not miss, "recipe env not in the launcher: %s" % miss)
txt = json.dumps(r)
d = r.get("differs_from_scored_S6fall") or {}
claim = "the one intended change" in txt.replace("said 'seed 43 = the one intended change'", "")
check("R2_NOT_SEED_ONLY", not claim, "recipe still calls seed 43 the one intended change" if claim else "")
check("R2_DIFFERENCES_LISTED", all(k in d for k in ("seed", "labels", "pool", "steps", "precision")) and "34,784" in d.get("steps", "")
      and "sol56" in d.get("labels", "") and 16 * (4349 // 2) == 34784, "keys %s" % sorted(d))
check("R2_GOLD58_CAVEAT", "in-sample" in r.get("gold58_caveat", "") and "RELATIVE" in r.get("gold58_caveat", ""), "")
tb = open(a.targets_builder).read()
check("R3_TARGETS_USE_BLENDJEV_SOL", "labels_raptor_rsna_blendjev_sol.csv" in tb, "")
sg = r.get("smoke_gate_A7", {})
lt = open(a.launcher).read()
check("R4_SMOKE_SCALE_BARS", sg.get("loss_scale_final_min") == 1024 and sg.get("loss_scale_min_min") == 256 and 'MIN_SCALE_END="1024"' in lt
      and 'MIN_SCALE_MIN="256"' in lt, "recipe %s" % {k: sg.get(k) for k in ("loss_scale_final_min", "loss_scale_min_min")})
print("RECIPE_TESTS %s %d/%d" % ("PASS" if all(RES) else "FAIL", sum(RES), len(RES)))
