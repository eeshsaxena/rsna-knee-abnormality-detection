## PREREG_TEACHER Amendment A3: KC dispatch shim in the pinned trainer. Written 10-15 <HH:MM> IST, BEFORE the shim is installed and before any KC GPU launch

**Filing (important).** Keep this text as a SEPARATE file next to PREREG_TEACHER.md, for example ~/r7t_labels/PREREG_TEACHER_A3_SHIM.md with its own .sha256. Do NOT append it to PREREG_TEACHER.md and do NOT change ~/r7t_labels/PREREG_TEACHER.sha256.
- Reason (MEASURED, labels_gate.sh lines 16-25): `labels_gate.sh` PASSes a table only if the label receipt's prereg_sha256 equals the first token of ~/r7t_labels/PREREG_TEACHER.sha256 (9e987880, the receipt value).
- Editing PREREG_TEACHER.md together with its .sha256 would make every re-run of labels_gate.sh FAIL. Today the T_a f4 gate is not affected, because its verdict file already says PASS. A future E3 T_b s1049 line with a labels gate would be.

**What changes.** ~/fast_build/code2/fleet_train.py (pinned in sec 2 at md5 3e62434a) gets ONE 5-line block, placed after the header comments and before its first import:
```python
import sys as _kc_sys, os as _kc_os, runpy as _kc_runpy
if '--kc' in _kc_sys.argv:   # 10-15 KC dispatch shim (BOX_PLAN_1015 sec 3.3, PREREG_TEACHER A3); argv without --kc runs the unchanged trainer
    _kc_sys.argv.remove('--kc')
    _kc_runpy.run_path(_kc_os.path.join(_kc_os.path.dirname(_kc_os.path.abspath(__file__)), 'fleet_train_kc.py'), run_name='__main__')
    _kc_sys.exit(0)
```
- The trainer md5 goes from 3e62434adea4fd8971af7c96e0465773 to **8d12515f237d204ad63ca93bd74b36b9**.
- Patch: ~/kc/code/shim/fleet_train_shim.patch (+5 lines, -0).
- Full text: ~/kc/code/shim/ft_shimmed.py.txt.
- Backup: ~/kc/code/fleet_train.py.bak_3e62434a.
- Rollback: ~/kc/code/shim/uninstall_shim.sh.
- Installed by ~/kc/code/shim/install_shim.sh. It refuses unless the target is 3e62434a and the KC files are present; it writes via a temp file + mv in the same directory.
- A running trainer is unaffected, because Python compiled its source at launch.

**Why.** The queue runner only ever starts `python fleet_train.py <argv>` from ~/fast_build/code2 (MEASURED, queue_runner.sh). The KC1 / KC2 lines (PREREG_10-15 Amendment 6) enter through `--kc`. Without `--kc` the code path is the pinned trainer.

**Proof that argv without --kc behaves exactly as 3e62434a** (regression, box CPU cores 28-39, 12 torch threads, GPU hidden):
- **(a)** ast.parse of the installed file is OK, and its line diff vs the backup is exactly the 5-line block (no line removed).
- **(b)** The A1 T_a dry-run form:
  - argv = sheets/queue_r7ta_SIMCHECK.txt line 2 (md5 300faa2d) + `--store ~/novel/frac75/dry/store --device cpu --min_cover 0 --time_epochs 1` (31 studies, 1 epoch);
  - run through the SHIMMED trainer;
  - it must give ckpt args IDENTICAL to the pre-shim run of 10-14 14:42 (trainer 3e62434a, ~/r7t_labels/dry/f1_s1048_ttr_r7tadry).
  - Sandbox pre-check, MEASURED 10-15 13:0x: the shimmed bytes run under another file name. Args identical (37 keys, 0 differing); the same closing line {"raw": 0.4831018518518519, "ema": 0.5060185185185185} as the pre-shim run; max |dW| 4.5e-6 (thread count only); 294 s.
- **(c)** The KC C4 smoke argv with `--kc` prepended, run through fleet_train.py: rc 0, the closing line, the SMOKE_TEST_ONLY verdict written, ckpt args model_mod kc_ttr and mirror 0.
  - Sandbox pre-check MEASURED: PASS in 199 s.
- The official (a)-(c) run is `bash ~/kc/code/shim/kc_shim_regress.sh post` by main, after the install and before 15:20. It writes ~/kc/regress/REGRESS.txt, ending `REGRESS PASS trainer_md5=8d12515f...`, which KC_READY requires.

**Scope.** T_a f4 (and any later E3 T_b s1049 member) launches on the shimmed trainer. Their argv contains no `--kc`, so (b) shows they get the pinned trainer's behaviour. The runner's LAUNCH line will log md5 8d12515f instead of 3e62434a; this amendment is the record of that change. Tables, gates, seeds, recipe and every bar of PREREG_TEACHER are unchanged.
