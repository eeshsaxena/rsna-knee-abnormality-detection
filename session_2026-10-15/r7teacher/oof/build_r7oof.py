"""Build the private Kaggle notebook eeshsaxena/rsna-r7-oof (R7 out-of-fold teacher on RSNA train studies).

  python build_r7oof.py --mode commit                      -> kernel_r7oof_commit/  (8 studies, 4 proof studies, 12.9 min budget)
  python build_r7oof.py --mode full --shard-index 0 --shard-count 2 [--skip-from r7_oof.csv ...] [--budget-h 8.5]
                                                           -> kernel_r7oof_full_s0of2/  (NOT to be pushed without Eesh's OK)

Cells: 0 markdown | 1 config (baked) | 2 the R7 notebook's asset cell VERBATIM (heliosli/0-957-latest-dataset-score-in-one-click
cell 4 = varshitha V224 cell 6; text sha256 asserted here) | 3 controller (cells/c3_controller.py) | 4 merge + checks (cells/c4_merge.py).
No em dashes.
"""
import argparse, csv, hashlib, json, os, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R7_NB = HERE / "src" / "0-957-latest-dataset-score-in-one-click.ipynb"
V224_NB = ROOT / "session_2026-10-14" / "v224" / "kernel_v224" / "rsna_sub_v224.ipynb"

COMMIT_UIDS = {   # 2 non-gold studies per R7 hash fold, 5-6 series, md5('r7oof-commit-' + uid) order (prove_fold_map.py table)
    0: ["1.2.826.0.1.3680043.8.498.38303307720827621886420004896097661262", "1.2.826.0.1.3680043.8.498.97156518925820521351080132062400795929"],
    1: ["1.2.826.0.1.3680043.8.498.64029768638431636860053484202080135316", "1.2.826.0.1.3680043.8.498.99155175553749543768821558247962844052"],
    2: ["1.2.826.0.1.3680043.8.498.32910372391370203605482772577821971133", "1.2.826.0.1.3680043.8.498.11190292536746235063218740991532759863"],
    3: ["1.2.826.0.1.3680043.8.498.51509214978951980397275258374070843967", "1.2.826.0.1.3680043.8.498.66993076561387452114335652021593805727"],
}


def cell_text(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def r7_fold(u):
    return int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["commit", "proof", "full"], required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--skip-from", nargs="*", default=[], help="r7_oof.csv files of earlier sessions: their studies are skipped")
    ap.add_argument("--budget-h", type=float, default=None)
    ap.add_argument("--chunk", type=int, default=None, help="studies per worker run (default: 100 commit / proof, 600 full: one run per fold per 2-shard session)")
    ap.add_argument("--loader-workers", type=int, default=2)
    ap.add_argument("--proof-n", type=int, default=0, help="full mode: this many extra studies also run through all 4 fold models")
    ap.add_argument("--out", default=None)
    ap.add_argument("--slug", default="rsna-r7-oof", help="kernel slug under eeshsaxena/ (a second slug, e.g. rsna-r7-oof-b, lets two shards run at once)")
    a = ap.parse_args()
    if a.chunk is None:
        a.chunk = 600 if a.mode == "full" else 100

    r7 = json.loads(R7_NB.read_text(encoding="utf-8"))
    asset = cell_text(r7["cells"][4])
    assert asset.startswith("ARM = 'E_lrprior'\nBUNDLE_ID = '8c7b666e27dc0e8f'"), "unexpected R7 cell 4"
    v224 = json.loads(V224_NB.read_text(encoding="utf-8"))
    assert cell_text(v224["cells"][6]) == asset, "R7 asset cell differs from the V224 copy"
    asset_sha = hashlib.sha256(asset.encode("utf-8")).hexdigest()
    assert asset_sha == "cb168cfe82f5a04114894f6c07bbd6daf0373ad2eb0dfcee34df3ecf625986af", asset_sha   # R7 cell 4 text, 10-13 pull
    assert "T0 = NB_T0" in asset and "PROBE_CLASS = None" in asset

    skip = set()
    for f in a.skip_from:
        with open(f, encoding="utf-8", newline="") as fh:
            skip |= {r["StudyInstanceUID"] for r in csv.DictReader(fh) if r.get("precision") == "full"}
    if a.mode == "commit":
        uids = [u for k in range(4) for u in COMMIT_UIDS[k]]
        assert all(r7_fold(u) == k for k in range(4) for u in COMMIT_UIDS[k])
        proof = [COMMIT_UIDS[k][0] for k in range(4)]
        cfg = {"mode": "commit", "shard_index": 0, "shard_count": 1, "uids": uids, "skip": [], "time_budget_h": a.budget_h or 0.215,
               "min_job_start_min": 2.0, "chunk": a.chunk, "loader_workers": a.loader_workers, "dump_state": True,
               "proof_uids": proof, "io_bench": True, "io_bench_before_min": 13.2}
    elif a.mode == "proof":
        # proof-only commit: the unchanged 4-model R7 path on 32 non-gold studies (8 per hash fold) + 4 gold controls (never trained)
        rows = list(csv.DictReader(open(HERE / "r7_fold_map.csv", encoding="utf-8")))
        used = {u for k in range(4) for u in COMMIT_UIDS[k]}
        c = [r for r in rows if r["n_series"] in ("5", "6") and r["StudyInstanceUID"] not in used]
        c.sort(key=lambda r: hashlib.md5(("r7oof-proof2-" + r["StudyInstanceUID"]).encode()).hexdigest())
        per = {k: [] for k in range(4)}
        gold = []
        for r in c:
            if r["is_gold"] == "1":
                if len(gold) < 4:
                    gold.append(r["StudyInstanceUID"])
            elif len(per[int(r["r7_fold"])]) < 8:
                per[int(r["r7_fold"])].append(r["StudyInstanceUID"])
        proof = [per[k][i] for i in range(8) for k in range(4)] + gold
        cfg = {"mode": "proof", "shard_index": 0, "shard_count": 1, "uids": [], "skip": [], "time_budget_h": a.budget_h or 0.215,
               "min_job_start_min": 2.0, "chunk": a.chunk, "loader_workers": a.loader_workers, "dump_state": True,
               "proof_uids": proof, "proof_gold_controls": gold, "io_bench": False, "io_bench_before_min": 0}
    else:
        assert 0 <= a.shard_index < a.shard_count
        proof = []
        if a.proof_n:
            rows = list(csv.DictReader(open(HERE / "r7_fold_map.csv", encoding="utf-8")))
            c = [r["StudyInstanceUID"] for r in rows if r["is_gold"] == "0"
                 and int(hashlib.md5(r["StudyInstanceUID"].encode()).hexdigest()[:8], 16) % a.shard_count == a.shard_index]
            c.sort(key=lambda u: hashlib.md5(("r7oof-proof-" + u).encode()).hexdigest())
            proof = c[:a.proof_n]
        cfg = {"mode": "full", "shard_index": a.shard_index, "shard_count": a.shard_count, "uids": None, "skip": sorted(skip),
               "time_budget_h": a.budget_h or 8.5, "min_job_start_min": 12.0, "chunk": a.chunk, "loader_workers": a.loader_workers,
               "dump_state": True, "proof_uids": proof, "io_bench": False, "io_bench_before_min": 0}
    cfg["built"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cfg["r7_asset_cell_sha256"] = asset_sha

    md = ("# R7 out-of-fold teacher on RSNA train studies (private, not a submission)\n\n"
          "Every RSNA train study is scored only by the R7 fold model that did not train on it. heliosli's R7 trainer "
          "(compact_train.py v9.8, pinned by the bundle manifest) folds the pool by `sha1(uid)[:8] %% 4` and trains fold k's model on "
          "the other three folds; gold studies are never trained. Cell 2 is the R7 notebook's asset cell verbatim (sha256 "
          "`%s`). Cell 3 runs the R7 worker `compact_infer.py` (unchanged) with the R7 worker arguments (unchanged, plain pass, no flip) "
          "on `--split train`, one fold model per run. Outputs: `r7_oof.csv`, `r7_oof_status.csv`, `r7_oof_receipt.json`, `r7_proof.csv`.\n\n"
          "Credits: heliosli (R7 bundle `heliosli/rsna-knee-compact-r7cnxfsss`, code `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`)."
          % asset_sha)
    cfg_cell = ("# R7-OOF configuration [eeshsaxena], baked by build_r7oof.py; R7OOF_<KEY> environment variables (json) override in an "
                "interactive session\nimport os, json, time\nNB_T0 = time.time()        # the R7 asset cell below counts its clock from NB_T0 "
                "(the R7 notebook sets it in its Raptor stage)\nR7OOF = json.loads(%r)\nfor _k in list(R7OOF):\n"
                "    _e = os.environ.get('R7OOF_' + _k.upper())\n    if _e is not None:\n        R7OOF[_k] = json.loads(_e)\n"
                "print('R7OOF_CONFIG', json.dumps({k: (v if not isinstance(v, list) or len(v) <= 8 else '%%d items' %% len(v)) "
                "for k, v in R7OOF.items()}), flush=True)\n" % json.dumps(cfg))
    ctl = (HERE / "cells" / "c3_controller.py").read_text(encoding="utf-8")
    mrg = (HERE / "cells" / "c4_merge.py").read_text(encoding="utf-8")

    def code(src):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": md}, code(cfg_cell), code(asset), code(ctl), code(mrg)],
          "metadata": r7["metadata"], "nbformat": r7["nbformat"], "nbformat_minor": r7["nbformat_minor"]}
    out = Path(a.out) if a.out else HERE / ({"commit": "kernel_r7oof_commit", "proof": "kernel_r7oof_proof"}.get(a.mode) or
                                            "kernel_r7oof_full_s%dof%d" % (a.shard_index, a.shard_count))
    out.mkdir(parents=True, exist_ok=True)
    (out / "rsna_r7_oof.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    assert a.slug.startswith("rsna-r7-oof") and a.slug == a.slug.lower()
    meta = {"id": "eeshsaxena/" + a.slug, "title": a.slug.replace("-", " "), "code_file": "rsna_r7_oof.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": ["heliosli/rsna-knee-compact-r7cnxfsss", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
            "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
            "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
            "machine_shape": "NvidiaTeslaT4"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    nb_sha = hashlib.sha256((out / "rsna_r7_oof.ipynb").read_bytes()).hexdigest()
    print("BUILT", out, "mode", a.mode, "studies", len(cfg["uids"]) if cfg["uids"] is not None else "shard %d/%d" % (a.shard_index, a.shard_count),
          "skip", len(cfg["skip"]), "proof", len(cfg["proof_uids"]), "budget_h", cfg["time_budget_h"], "asset_sha", asset_sha[:16], "nb_sha", nb_sha[:16])


if __name__ == "__main__":
    main()
