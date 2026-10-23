# PREREG 10-12b: HF with the K14 fleet (written 2026-10-12 ~12:05 IST, BEFORE the HF20 / HF30 reads)

Seen before writing: HV (57093094) = .954, hidden wall ~3.3 h. K14 gate PASS, verdict R (C + DT + E + E2 replaces FN11;
dOOF +.00106, d58 -.0007). HF20 (57097813) / HF30 (57097814) submitted 11:45, NOT read yet. No em dashes.

## Kernels (built by hf/build_hf.py --fleet k14; H cells unchanged; F inserted after Raptor, before S6)
- **HF-R(w)** = H + F_R, F_R = the K14 R fleet (C + DT + E + E2, 20 members, datasets k11w + k14w merged by the K12-MERGE
  cell), final (1 - w) x rank(H) + w x rank(F_R).
- **HF-DT(w)** = same with F = DT only (5 x TTR224 OAI-T123), for a runtime fallback only.
- Runtime: HV wall 3.3 h + F_R slow ~2.7 h (FN11 fleet model) = ~6.0 h <= 8 h -> HF-R allowed. S6 keeps its NB_T0 + 8.3 h
  deadline, so the notebook cannot overrun; a failed F -> H + HF_FALLBACK.
- Commit gate: h_check.py --kind hf with the member count (20 or 5) PASS, plus the K12-MERGE cell's own checks (k14w manifest
  == build, sha256 of every DT member).

## Lineup rule (decided now, applied after the HF20 / HF30 reads)
Let w* = the wF of the better of HF20 / HF30 on the 3-dp display (tie -> .20).
1. If max(HF20, HF30) > .954 (F helps on top of H): submit **HF-R(w*)**; next day also HF-R at the other w if slots allow.
2. If both <= .954: submit ONE **HF-R(.20)** as the last test of whether a stronger F adds anything. If it is also <= .954,
   the HF line stops (no more same-family fleet work for the LB).
3. HF-DT is submitted only if an HF-R commit run fails for a reason tied to the 15 k11w members (runtime or merge).
Final picks (10-22) are a separate decision; OAI exposure: H already contains OAI-trained Raptor, DT is OAI-trained.
