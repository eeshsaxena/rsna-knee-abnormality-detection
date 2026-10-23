# PROBE6 RESULTS (RadImageNet ResNet-50 on T123 OAI, probe f0), recorded 2026-10-13 10:20 IST

**STOPPED EARLY at Eesh's request (10-13 ~09:55: "stop it and run the PROBE4 folds"); verdict set FAIL by hand.**
Deviation from PROBE6_PREREG.md (sha 6ff96ada), which had no early-stop rule: the trainer (pid 441452, child of the queue
runner) was stopped with SIGTERM at epoch 7 of 17 (rc 143, 4.6 h of run). `~/oai_build/out/rr.verdict` = FAIL,
`gate_rr.txt` records the reason. The queue skipped the 4 RR fold lines and launched PROBE4 (PL) f2 at 10:17.

Trajectory seen (raw OOF vs llm_labels_v4_blend, fold 0): ep 6 best .8466 (T123 DT f0 at ep 6 .8761; D f0 .8709); 0.60 s/it
(~30 min/epoch). The G1 bar .881660 was out of reach on this trajectory (the ImageNet R50 probe of 10-07 ended at .8710).
Conclusion: the "different type" line (Swin PROBE5, RadImageNet R50 PROBE6) is CLOSED.
