## AMENDMENT 6 (KC1, revised with the adopted BOX_PLAN_CRITIQUE_1015). Written 10-15 <HH:MM> IST, before any KC GPU launch and before any KC model number (the only KC numbers that exist are CPU smoke runs of a 2-step model on 458 studies)

**Family.** KC1 = ConvNeXt-S KC-TTR, trained by `fleet_train_kc.py` through the `--kc` dispatch shim (PREREG_TEACHER A3).
- Backbone convnext_small.fb_in22k_ft_in1k, init `~/r7port/pretrained/convnext_small_in22k_ft_in1k.safetensors` (sha256 2457488b; asserted at load).
- Input: 6 laterality-canonical knee-centred views x 12 source slices at 256 px (lossless PNG), cut from the d96 corpus (verbatim reader sha 30aa72fd). Canonical frame = a left knee.
  - sag_lat: SagFS block, canonical slots 0-55 %, 110 mm box. sag_med: SagFS, 45-100 %, 110 mm.
  - sag_pd: SagPD, 15-85 %, 120 mm. cor_med / cor_lat: CorFS 10-90 %, 90 mm boxes at x = -22 / +22 mm. ax: Axial 5-95 %, 140 mm.
  - Model input = 2.5D triplets (c-1, c, c+1) of the 12 slices; training uses 8 of 12 slots per view.
- Store contract hash 48a46a0d63ecfc59 (kc_common.py md5 45255ae0).
- Side rule: mean_x sign if |mean_x| > 30 mm, else the DICOM laterality tag, else the sign if > 5 mm, else U (left frame, no flip).
- Flips come from the geometry of the series that fills each d96 slot, XOR the side:
  - sagittal blocks are reversed iff (side R) XOR (normal_x > 0);
  - coronal / axial are flipped left-right iff (side R) XOR (row_x < 0).
- Measured on the train side table (main's kernel, 4,407 rows):
  - expected geometry on 4,407/4,407 slot series;
  - the kc_common flips equal the table's flips 17,628/17,628;
  - U = 19 (0.43 %);
  - weak zone (rule other than mean_x) = 180 (4.1 %).
- present[view] = the d96 block is non-empty (the corpus mask; SagPD is absent in 119/4,407), identical on train and serve.
- No TE / TR in d96, so contrast_emb is 0 and the stored contrast class is the constant 3, on train and serve.
- Mirror aug OFF: no per-study coin, heavy-aug mirror p 0.
- Anatomical vbias init of +1.0, 0 elsewhere:
  - MCL, Medial Meniscus, Medial OA -> sag_med and cor_med;
  - Lateral Meniscus, Lateral OA -> sag_lat and cor_lat;
  - ACL -> sag_pd;
  - PF OA, Baker's, Effusion -> ax.
- Schedule:
  - 13 epochs, bs 1 x accum 8, lr 1e-4 / head 1e-3, wd .05, drop_path .3, dropout .3, EMA .9995;
  - heavy aug (fleet_aug.aug_view_heavy, verbatim) with mirror p 0;
  - grad_ckpt 0, with the runner's OOM fallback to 1.
- KC applies to all 12 labels, like the public recipe. This is fixed now and never changed after G2.

**Members (fixed now; in queue order).** gold74 is never trained on (`--exclude`).
1. **f0 s2001**, the gate model: T_b (labels_r7t_b), folds5 fold 0, `--save_epochs 3-13`.
   - E* = the raw epoch with the best val AUC vs T_b inside epochs 5-13; on a tie, the later epoch.
2. **AD s2003**: dread soft labels, all-data, epoch E*. It runs right after f0.
   - The labels are dreaddevelopment/rsna-knee-labels: source md5 96e6723f, built by make_dread_labels.py into ~/kc/labels/labels_dread_fleet.csv.
   - Expected table md5 032668fe (4,349 rows, pool 4,333/4,333 covered, gold58 absent; MEASURED on a test build).
3. **AD s2002**: its label set is decided by the S read (57218523), mechanically:
   - T_b if S is a best-id move (positive under R1);
   - otherwise a second dread seed (dread2), whether the read is negative or UNREAD.
   - Main writes ~/kc/gate/s_read.verdict (`PASS S positive ...` or `FAIL S not positive ...`). kc_gate_watch.sh turns it into the two s2002 line gates. Exactly one s2002 line runs, and only if f0 passed G1.
4. **Speculative, train-only** (Amendment 8 stub below), each gated on f0 G1 PASS:
   - AD s2004 (dread seed 3);
   - KC2 f0 s2011 (EffNetV2-S via the f2 frozen-BN path, T_b, gate file ~/kc/gate/kc2_f0.verdict).
   - Neither enters any LB entry before a KC1 positive read and an Amendment 8 entry text.
- All-data members run 13 epochs, save epochs 5-13, and are served at E*.

**Leg, blend, entry.**
- Leg = the per-label mean of the sigmoid probabilities of the members that passed G1.
- Entry = per label, .8 rank(team-best base output) + .2 rank(KC leg).
- The weight is a prior, never fitted. No weight variants and no member re-mixes go to the LB.
- KC1 ships as soon as f0 and the first all-data member that passes G1 exist (>= 2 passing members).
  - Later members (the s2002 line, s2004, KC2) enter only in the Amendment 8 form "KC-all vs KC1", one change on the best.
- Serve, one fp16 pass, never hflip or mirror TTA:
  - a weak-zone study (side rule other than mean_x) or one whose geometry is not OK = the mean of the KC probabilities under both canons (side L and side R, same slot geometry);
  - every other study = its own canon.

**G0 = KC_READY** (`~/kc/kc_ready.sh` writes `PASS kc_ready ...` only if every item holds; trimmed per the critique):
- the side table: 4,407 rows with per-slot geometry;
- the full store build PASS:
  - coverage >= 99 % of 4,407;
  - U <= 2 %;
  - contract 48a46a0d;
  - the side-table sha recorded;
  - the test458 subset array-equal;
- C2b, the visual canon check of the contact sheets, PASS by main (~/kc/contact/C2B_OK);
- the C4 CPU smoke receipt, with md5s equal to the installed copies;
- the C7 refs frozen (~/kc/gate/refs.json);
- the sha256 of this amendment;
- the shim installed (trainer md5 8d12515f) and shim regression (a) + (b) + (c) PASS;
- inputs RSNA-only. The OAI-string scan still runs before any dataset.

C3 serve parity and C5 are not in G0. C3 must PASS before the G3 pack. C5 is enforced by the s2003 line's missing-input DEFER.

**G1.**
- **f0, automatic** (the fleet_train_kc.py end hook writes ~/kc/gate/kc_f0.verdict, a single PASS / FAIL line). PASS iff all of:
  - all 13 epochs ran;
  - the last-epoch mean loss is finite (non-finite iterations <= 1 %);
  - the fold-0 **view ablation** at E* PASSes:
    - setup: full val set (~867 studies), labels T_b thresholded at .5;
    - drop_X(l) = AUC_full(l) - AUC(l with views X removed). "Removed" means present = False; the view softmax masks them, so this equals removing the views;
    - med_margin = mean over {Medial Meniscus, Medial OA} of (drop_med - drop_lat) must be > 0;
    - lat_margin = mean over {Lateral Meniscus, Lateral OA} of (drop_lat - drop_med) must be > 0;
  - **gold74** macro AUC at E* >= .895 (secondary check).
  - Any crash, exception, SIGTERM, or SKIP of the f0 line writes FAIL (atexit path + kc_gate_watch.sh). A FAIL #SKIPs every AD / speculative line.
  - The gold74 lateral swap test is reported only.
- **Early information** (not a gate): the same ablation on the epoch-3 ckpt.
  - It runs on the box CPU, 400 random fold-0 studies, ~50 min (MEASURED 7.6 s/study), starting ~1.2 h into f0.
  - On a clear FAIL (both margins < 0), main may stop f0. A stopped f0 counts as a G1 FAIL.
- **All-data members:**
  - completed with a finite loss;
  - gold74 macro AUC at E* >= .895 (T_b) or >= .885 (dread); the trainer writes epochs/gold_* per saved epoch.
  - A failing member is dropped. The leg ships only with >= 2 passing members, else there is no entry.

**G2, a family sanity floor only.** The refs were frozen 10-15 12:49 in ~/kc/gate/refs.json by kc_oof_kit.py md5 62218d1c, the same code G2 runs. Reference values (MEASURED):
- gold74: R7 .9158, S .9123, D .9136;
- gains at .2: S +.0049, D +.0054;
- rho(T_b f0 OOF, R7 OOF), fold 0, n 867 = .8170.

Bars:
- leg gold74 macro AUC >= .9036;
- gold74 gain of .8 rank(R7 OOF) + .2 rank(KC leg) over R7 OOF alone >= 0 (changed from +.0024, per the critique);
- per-label-mean Spearman(KC f0 OOF at E*, R7 OOF) on fold 0 <= .8370.

**G3, kernel (before submit).** As in BOX_PLAN_1015 sec 4.1 G3, with these parity bars:
- C3 serve parity on the 458 box train studies:
  - side / rule / flips / presence equal on 458/458;
  - KC crop max |diff| <= 2 LSB and per-study mean |diff| < .05.
  - MEASURED 10-15 by the kc_worker test run: d96 vol max 1 LSB (on 5 inspected studies), KC crop max 2 LSB on 223/458 (cv2 uint8 resize rounding of 1-LSB vol differences), mean <= .047 (median .0056). The plan's crop max <= 1 is not attainable by construction.
- C9 probability parity on 20 eda_data studies (including one SagPD-absent and one weak-zone study), DICOM path vs store path, same ckpt: max |dp| <= 1e-3.
- The 3 visible studies' commit dump vs a box CPU run of the same ckpts: max |dp| <= 1e-3. The side matches the header rule.
- Base rows identical with KC removed; the blend recompute <= 1e-6.
- The KC stage runs in its own subprocess and process group, with start guard G_KC, a hard wall at NB_T0 + 8.2 h, and coverage >= 98 %. Otherwise KC is dropped with the visible canary Baker's = 0.5 on every row; unscored studies take rank(base).
- Runtime review p99 <= 8.0 h.

**Read (R1) and final picks.** Unchanged from BOX_PLAN_1015 sec 4.1:
- A best-id move = KC positive.
- A non-move on an unchanged base = negative: KC dropped, KC2 cancelled.
- A non-move while the base changed = UNREAD: retest once as current best + KC.
- A canary = UNREAD.
- KC is eligible for final picks only after a positive read. A J-form KC skip falls back to base rows.

**Amendment 8 stub (KC2 / extra seeds), written now.**
- s2004 (dread seed 3) and KC2 f0 s2011 train now, only after f0 G1 PASS. KC2 is EffNetV2-S tf_efficientnetv2_s.in21k_ft_in1k via `--f2 1`:
  - frozen BN, fp16 watchdog F2-T5;
  - 13 epochs, lr 1e-4 / 1e-3, drop_path .2 (GUESS recipe).
  - Its G1 is the same hook, written to ~/kc/gate/kc2_f0.verdict.
- They enter no LB entry unless KC1 reads positive AND the full Amendment 8 (entry form "KC-all vs KC1", one change on the best, same G1-G3) is written before that entry.

**Code freeze (md5, box ~/kc/code == installed ~/fast_build/code2 copies via install_c8.sh; manifest ~/kc/code/MANIFEST_1015.txt):**
- trainer path: fleet_train_kc.py 5a22d1eb, kc_common.py 45255ae0, kc_g1.py b9c27f2e;
- store and gates: kc_build_store.py 4b836891, kc_oof_kit.py 62218d1c (refs), kc_ready.py 64168c83, kc_gate_watch.sh 0f898531;
- G1 early and serve: kc_g1_ablation.py 913e474c, kc_worker.py 6605ee06 (reader text d96_reader_src.py sha256 30aa72fd), kc_parity.py 7e4dfa47, kc_prob_parity.py ca9cd0ca;
- labels: make_dread_labels.py d6edd8a4;
- trainer fleet_train.py 8d12515f (shimmed, PREREG_TEACHER A3).
- Any later change needs an amendment before the affected line launches.
