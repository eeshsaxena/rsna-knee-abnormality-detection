# PROBE5_PREREG: a DIFFERENT backbone (Swin-T) on the T123 OAI recipe, probe f0

Written 2026-10-12 ~12:55 IST by Claude, BEFORE any Swin training (only a CPU forward test of the architecture was run).
Nothing below may be edited after the probe starts; results go to PROBE5_RESULTS.md. No em dashes.
Box copy: ~/oai_build/PROBE5_PREREG.md (same bytes; sha256 in PROBE5_PREREG.sha256).

## Why
- The public H blend (.954) = CoAtNet (Raptor) + ConvNeXt-small (S6). Our fleet is ConvNeXt-T, partly the same family as S6.
  More ConvNeXt seeds add ~.0001 each. The lever left is a DIFFERENT family that is still strong (Eesh 10-12: "yes").
- History: EfficientNetV2-S (BatchNorm family) was decorrelated (Spearman .835) but weak (.8635) and diverged in fp16 with
  the TTR recipe; DINOv2 frozen was weak. A LayerNorm transformer with ImageNet norm runs on the unchanged round3 / TTR path.
- CPU check 12:50: `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k` (timm 1.0.29) builds in KneeFleetNetTTR, no BatchNorm,
  ImageNet mean / std, 768 features (= ConvNeXt-T), encode(4 x 3 x 320 x 320 uint8) -> (4, 768). ViT-S/16 is refused by the
  trainer (mean .5 = non-ImageNet norm), so not used.

## Pair (one change only)
- **Probe SW f0** = the T123 probe argv EXACTLY (TTR224, frac75 + OAI T123v00 union, 17 ep, save 15-17, gc1, seed 1043,
  fold 0) + `--backbone swin_tiny_patch4_window7_224.ms_in22k_ft_in1k`, out `f0_s1043_ttr_frac75oaiT123swinT`.
- **Reference**: DT f0 (`f0_s1043_ttr_frac75oaiT123v00`, OOF .887660) and D f0 (`f0_s1043_ttr_frac75`, OOF .883777).

## Gate (gate_sw.py; fold-0 RSNA val rows, OOF vs llm_labels_v4_blend, best raw ckpts; gold58 = pass A CPU preds)
| gate | rule |
|---|---|
| G1 strength | OOF12(SW) >= OOF12(DT f0) - .006 = **.881660** |
| G2 diversity value | OOF12(mean prob {DT, SW}) - OOF12(mean prob {DT, D}) >= **+.0010** (a Swin partner beats a ConvNeXt partner) |
| G3 gold58 guard | gold58(mean {DT, SW}) >= gold58(mean {DT, D}) - .005 |
PASS = G1 and G2 and G3 -> verdict `~/oai_build/out/sw.verdict`; SW folds 1-4 (21 ep, gc0, seed 1043) run next (AHEAD of the
PROBE4 folds). FAIL -> no Swin folds. Report only: Spearman SW~DT and SW~D (OOF), per-label OOF, gold74.
A 5-fold SW set must still pass its own pre-registered fleet / HF gate before any submission. OAI-trained -> N pick only.
