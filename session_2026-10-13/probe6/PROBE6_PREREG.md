# PROBE6_PREREG: a different TYPE on the box, RadImageNet ResNet-50 (medical pretraining, BatchNorm CNN, 320 px) on the
# T123 OAI recipe, probe f0

Written 2026-10-13 ~01:55 IST by Claude (Eesh: "start a diff type on uni box"), BEFORE any training of this config (only a
CPU forward test). Nothing below may be edited after the probe starts; results go to PROBE6_RESULTS.md. No em dashes.
Box copy: ~/oai_build/PROBE6_PREREG.md (sha256 in PROBE6_PREREG.sha256).

## Why this type
- Swin-T (PROBE5) FAILED on diversity (blend gain -.0001): an ImageNet backbone swap on the same recipe does not decorrelate.
- What differs MORE: the PRETRAINING domain and the input path. RadImageNet ResNet-50 = medical-image pretraining (the crop
  pipeline's champion encoder on 10-03: +0.13 over natural-image init), BatchNorm family (frozen BN), no TTR resize (native 320 px),
  norm .5. Our ImageNet ResNet-50 probe (10-07, mix50, lr 3e-4) was weak (.8710); this changes the init, the labels (T123 OAI)
  and the backbone lr (1e-4, gentler, to keep the medical features).
- CPU check 01:50: KneeFleetNetF2(resnet50.a1_in1k, init_ckpt ~/rsna_train/ResNet50_radimagenet.pt, freeze_bn, channels_last)
  builds; encode(4 x 3 x 320 x 320) -> (4, 2048).

## Probe RR f0 (one config)
`--f2 1 --backbone resnet50.a1_in1k --init_ckpt ~/rsna_train/ResNet50_radimagenet.pt --freeze_bn 1 --channels_last 1 --lr 1e-4
--aug heavy --train_slices 16 --drop_path 0.1 --dropout 0.3 --ema 0.9995 --epochs 17 --save_epochs 15-17 --workers 10 --seed 1043`
+ the T123 union (`--store ~/union_store_t123v00 --labels .../labels_frac75oai_T123v00.csv --folds .../folds5_oai.csv`), fold 0,
gc1, out `f0_s1043_r50rad_frac75oaiT123`.

## Gate (gate_rr.py = gate_sw.py logic with RR in place of SW; same bars, same references)
| gate | rule |
|---|---|
| G1 strength | OOF12(RR) >= OOF12(DT f0) - .006 = **.881660** |
| G2 diversity | OOF12(mean {DT, RR}) - OOF12(mean {DT, D}) >= **+.0010** |
| G3 gold58 guard | gold58(mean {DT, RR}) >= gold58(mean {DT, D}) - .005 |
PASS -> RR folds 1-4 (21 ep, gc1) queued ahead of the remaining PROBE4 folds; then its own pre-registered HF / fleet gate before
any submission. FAIL -> the "different type" line closes. OAI-trained -> N pick only.
