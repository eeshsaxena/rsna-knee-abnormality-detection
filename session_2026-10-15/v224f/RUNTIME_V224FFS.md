# Runtime review V224-FF-S (eeshsaxena/rsna-sub-v224ffs20 v1), main session 10-15 03:40 IST

- Commit (MEASURED, out_v224ffs20_v1): COMPLETE; fleet stage 2.63 min; 25 members (K14 20 + S 5) clean on both workers.
  - Microbench s/study/GPU 11.39 / 10.97, against 8.83-10.18 for K14 20 under the R7 wheels.
  - S members 0.309-0.317 s/study/GPU each (both TTA passes), under the 0.40 criterion (preflight note). The other members average 0.47-0.49.
  - S total ~1.56 s/study/GPU, == the RUNTIME_FFS_1015 microbench (1.44-1.62).
- wall_model.py (MODEL): fleet 25 on hidden 1300 studies takes 2.08 h central, 2.39 h slow, 3.10 h slow session.
- With flip 4.9 h (== FF20) and fleet start 5.1 h (MIN_START_MIN 192), RUNTIME_FFS_1015 MC applies unchanged (MODEL): median 6.61 h, p99 7.67 h, slow corner 8.22 h vs the 8.3 h wall, P(fleet skipped) ~.001, P(> 9 h) 0.
- Gate: V224FFS_CHECK PASS 14/14 mode clean (gate_v224ffs20_v1.txt). END-TO-END recompute max diff 2.5e-8.
- **Verdict: SUBMIT-able on runtime.** Submit only under PREREG_10-15 Amendment 2 a / Amendment 3 (FF20 must become the team best first), with PREREG_10-14 Amendment 4 written before the submit.
