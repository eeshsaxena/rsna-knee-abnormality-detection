
### AMENDMENT 20 (10-20 ~00:30 IST): E-OA, OA decorrelation on the final c4 output (Eesh 10-19 ~23:40: "do offline validation ... then push the 2 if they are strong")
- Mechanism: report-trained models couple Medial / Lateral / PF OA (model rho ~.75; report-label phi .53-.61); expert graders score compartments more independently (gold58/74 phi .24-.40; OAI MOAKS ~.05). Hidden labels are expert-regime (host 733491).
- Change (one, cell 18 S6F path, after the blanked-column fill): for t in the OA trio, out_t = pct(pct(f_t) - a (pct(f_o1) + pct(f_o2)) / 2). All 9 other columns byte-identical to c4.
- Entries: **eoa20** (a .2, rsna-sub-v224r6c4eoa20j, sha d40f43a4) and **eoa10** (a .1, rsna-sub-v224r6c4eoa10j, sha c1242740). Base c4 57343435 (sha b47c29c3).
- Evidence used to choose them (NOT the public LB, NOT a gold58 sweep): OAI G1 (pre-registered rule, SHIP_A .2): OA-mean AUC +.0062 / +.0115, P>0 1.000, 3/3 proxies, 5/5 folds, a -.1 control negative; replication per A20_VALIDATION_PREREG.md (see VALIDATION RESULT below). gold58 is a code gate only (+.00142 / +.00247).
- Gates: offline A20_OFFLINE_GATE PASS (a20/gate_a20_offline.log; eoa00 control byte-identical to c4); commit gate a20/probe_check_eoa.py PASS 17/17.
- Submit: tick RSNA_A20_tick after 05:31 IST 10-20, only with a20/A20_GO.txt.
- Read: R1 vs c4 57343435. Parallel entries reveal the max. A best-id move to either: A = that entry, B stays R6 57248084 (B = R6 + E-OA is a separate later decision). No move: E-OA closed (no a .3, no stacking).
