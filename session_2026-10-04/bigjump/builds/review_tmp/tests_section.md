All tests were re-run after the fixes, 06:24-06:47. Every test passes (laptop 7/7 rc 0; box BOXCHECK, BOXPARITY and
BOXGP0 PASS).

**Laptop CPU** (`BLD/tests/run_all.sh`; 64 px synthetic store, convnext_atto, cv2 PIL shim; ~11 min; transcript
`BLD/tests/transcript.txt`, pre-fix copy `BLD/tests_pre_fix/transcript.txt`):

- **t_b1_model**
  - Contract files are identical in all 4 trees.
  - The model_sha registry is identical in worker, pack and tta_bench: f1 `fa35880fca963652`, f2 `ba3e1f85e38128bf`,
    **TTR `f141aeed136dfc01`**.
  - TTR with ttr_size == input size == KneeFleetNet: eval, train logits and 144 grads all max diff 0; the state_dict
    keys are the same plus `ttr_size_t`.
  - encode == manual interpolate + norm + cnn (diff 0). A 320 px input reaches the backbone at 224. Antialias is
    active.
  - **(e) TTR-6**: `ttr_size_t` stays int64 224 in raw and EMA after 5 real `AveragedModel` updates. Strict load into
    KneeFleetNet is refused; TTR -> TTR strict load works.
- **t_b2_default_parity**: with no opt-in flag the run is bit-identical to round3/fleet (`~/fleet`) AND fleet2
  (`~/fleet2_code`). Cases: fold-0 2 ep, all-data stop_epoch, T6 `--train_slices 16`, `--f2 1` (resnet18). Compared:
  ckpt names, all state_dict tensors `torch.equal`, ckpt args, cv/epoch, OOF bytes and metric log lines.
- **t_b2_heavy_equiv**:
  - The heavy path with pixel aug off == default `mirror_views(full)[:, pos]` on 480 cases, K 16/24.
  - View drop forced on 120 cases. Sagittal mirror reverses slot depth and channels.
  - Aug is deterministic; holes are identical on all 48 slice-channels.
- **t_b2_ttr_train**:
  - TTR fold-0 3 ep and all-data 3 ep at **seed 1042**: names `f0_s1042_{raw,ema}_ttr224-64x24_*` and
    `f-1_s1042_*`, plus `epochs/` snapshots and per-epoch OOFs. Per-epoch ckpt cv == log; top-level best == its
    epoch snapshot; args right (sha `f141aeed136dfc01`).
  - Guards refuse `--model ttr --f2 1`, `--save_epochs 3-1` and **`--model ttr --seed 42` (BJ-5)**.
  - **TTR-6** on real trainer ckpts: int64 224, strict load into KneeFleetNet refused.
  - Pack `--tag f0_s1042ttr,=,=,=`; the manifest `sources` have the FIX-5 keys only.
  - The bigjump worker serves 2 TTR + 2 F1 members; F1 predictions == K2 worker (max diff 0). The K2 worker refuses a
    TTR member.
- **t_b4_pack_tag**:
  - Same-name pair: the K2 pack tool and the bigjump pack without `--tag` refuse it; `--tag =,f0_s42v4` packs both
    (bytes == sources).
  - **FIX-5**: `sources` = original / run_dir / sha256[:16] (== the bytes) / model_mod / model_sha, with no ':' or '/'.
  - All 6 bad re-keys are refused.
  - **K2 notebook k2_check PASS 22/22 on 24 studies and on the 3-study commit cut, n_keep [2, 2]**. K2 worker swap:
    fleetsolo byte-identical. Real pair: `build_k2.py ... =f0_s42v4` gives 2 members, 225.3 MB / 12 files (local only).
- **t_b5_tta_bench**:
  - Pass A == bigjump worker P (max diff 0, TTR and F1 members).
  - Pass B == independent `mirror_views(sample_triplets(+HALF))` (0).
  - All variants + gate.
  - Trainer OOF vs pass A at chunk 32 and 48: 0.
  - **(6) HALF**: n=24 P/N centres 0..23 (exact .5 gives `0,2,2,4,4,..`); every other n 1-32 is the same as .5,
    except the N pass's last slot on even n.
  - **(7) `--passes A,B`**: none / A+B / A+B lesion-max + raw+EMA A+B + 2 gate lines, all == the 5-pass run.
  - **(8)** a `--max_minutes` cut does not crash and is flagged PARTIAL.
- **t_gt2_saveep**:
  - `--save_epochs` does not change training (tensors, cv, OOF bytes; args differ only in `save_epochs`).
  - gt2_read score == recomputed; verdict == rule; exit 3 PROVISIONAL without a closing line.
- **ttr_preflight CPU smoke** (real parity npz, convnext_atto, `--no_pretrained --bs 1 --steps 3`): default case `ttr`
  only, finite losses. An unknown case is refused. The GPU guard lines were not executed (no GPU in this pass).

**Box CPU** (`bash ~/bigjump_build/box/run_box_tests.sh 3`, `CUDA_VISIBLE_DEVICES=`, nice 10, real cv2 4.14;
06:36-06:46; transcript `BLD/tests/out/box_run2.txt`). All 16 staged files have md5 == laptop.

- **box_check PASS**:
  - Contract files == `~/fleet` and `~/fleet2_code`.
  - model_sha in the box env: f1 `fa35880fca963652`, f2 `ba3e1f85e38128bf`, **TTR `f141aeed136dfc01`** (pinned).
  - **B3**: `~/rsna_train/labels_gated75_logit.csv` md5 `a813b986aa09b6fc4736ff493fbaa918` (== laptop); 4407 ids;
    finite, in [0,1]; report-ADDRESSED cells == steven_v4 exactly (38,427 cells); report-UNK 27.3 %, all changed
    (mean |diff| .135); fold-0 train 3466 / val 867.
- **box_parity PASS**: the 7 default-path comparisons are bit-identical in the box env. The TTR path runs there (seed
  1042, sha `f141aeed136dfc01`).
- **box_gp0 (G-P0) PASS**:
  - Setup: 4 real K2 parity studies (`~/fleet_store`, steven_v4, 2 train / 2 val), REAL pretrained ConvNeXt-T, the
    exact TTR flags + `--seed 1042`, 1 epoch CPU fp32 (50 s).
  - Ckpt `f0_s1042_{raw,ema}_ttr224-320x24_*` with sha `f141aeed136dfc01` and `ttr_size_t` 224.
  - **Trainer path vs worker path (tta_bench pass A): max |diff| 0.0 for raw and EMA at chunk 32 AND 48.**
  - Mirror check PASS; aug PNGs in `BLD/tests/out/box_aug/`.
  - Real-data tta_bench smoke on the interim family-1 member (3 val + 3 gold58 x 5 passes): pass A vs its GPU fp16
    trainer OOF max |diff| 2.1e-4 (fp16 noise).
- **gt2_read on the LIVE logs** (pre-fix run, format dry run): parses both real logs; OOF macro AUC == the log
  values; exit 3 PROVISIONAL. gt2_read changed only in comments since then.
