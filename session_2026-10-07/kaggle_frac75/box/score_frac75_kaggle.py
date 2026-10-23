# score_frac75_kaggle.py (kaggle_frac75, 2026-10-07, Claude): the PRE-REGISTERED frac75 gate scorer for a fold-0 probe
# that was trained on Kaggle (notebook rsna-train-ttr-frac75-f01) instead of on the V100. Box copy: ~/kt_frac75/box/.
# It runs ~/novel/frac75/score_frac75.py ITSELF (sha256 pinned below; bars G1-G3, J1 refs, conventions, outputs and exit
# codes unchanged) with exactly two source patches, both asserted to apply once:
#   1. pairing: the probe's ckpt args may differ from J1's in train_labels (as before) AND in `epochs` / `save_epochs`
#      (the Kaggle notebook sizes E to its time budget: E ~17 instead of 24, --save_epochs = its last 3 epochs). Every
#      other arg must still equal J1's (seed, fold, contract, decode_probe, amp, model_sha, aug, ...). Extra integrity
#      checks whenever epochs/save_epochs differ from J1's: 14 <= E <= 24 and save_epochs == f'{E-2}-{E}' (the
#      notebook's rule). A probe with J1's own schedule (the registered box probe) is scored exactly as the original.
#   2. ROOT may be redirected by KT_F75_ROOT (tests only; default ~/novel/frac75, so the gate json lands where
#      ~/queue/gates/frac75_gate.sh and launch_frac75.sh look).
# After a scorable verdict (exit 0 / 1) it appends REPORT-ONLY lines (never change the verdict or the exit code):
#   - the Kaggle addendum read (PREREG_KAGGLE_ADDENDUM.md): PASS / FAIL / INCONCLUSIVE-NARROW (every failing gate within
#     .003 of its bar: the E<24 schedule is a plausible cause, so the registered 24-epoch box probe may break the tie);
#   - schedule-end read: mean of the LAST 3 epochs' OOF vs v4 (probe epochs E-2..E vs J1 epochs 22-24, selection-free,
#     both at the end of their own cosine) and the trainer-logged gold74/gold58/val curves at matched schedule fraction;
#   - an internal control on gold74: the Fracture delta minus the 11-label non-Fracture delta (a schedule/platform shift
#     moves all labels; the frac75 change moves Fracture only).
# usage (same CLI as score_frac75.py; frac75_gate.sh can run it via F75_SCORER=~/kt_frac75/box/score_frac75_kaggle.py):
#   CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/rsna/bin/python ~/kt_frac75/box/score_frac75_kaggle.py \
#       [--run ~/fleet_runs/f0_s1043_ttr_frac75] [--gold_preds CSV] [--expect_labels NAME] [--dry] [--tag NAME]
# env: F75_KAGGLE_RECEIPT (default ~/fleet_runs/kaggle_frac75_f01_run_receipt.json, printed if present),
#      F75_ORIG_SCORER (tests only), KT_F75_ROOT (tests only).
# exit: as score_frac75.py (0 PASS, 1 FAIL, 2 integrity error, 3 not scorable). Never uses the GPU.
import os, sys, re, json, hashlib

H = os.path.expanduser
ORIG = H(os.environ.get('F75_ORIG_SCORER', '~/novel/frac75/score_frac75.py'))
ORIG_SHA256 = 'f19ddd8c94bfe918273f15aea55e747915d205d0cd9b6ac60f765ca51c0e7ed4'     # read 2026-10-07 14:20 IST
ALLOW = ('epochs', 'save_epochs')
NARROW = 0.003          # = the G1/G3 tolerance itself; E 21 vs 24 measured ~0 OOF, E ~17 GUESS 0 to -.003
E_MIN = 14              # K5_MIN_EPOCHS (Eesh 10-06); below it the run is not read as the probe
J1_LAST3 = (22, 23, 24)

PATCH_PAIR_OLD = ("diff = sorted(k for k in set(a) | set(aj) if k not in ('train_labels',) and "
                  "repr(a.get(k)) != repr(aj.get(k)))")
PATCH_PAIR_NEW = ("diff = sorted(k for k in set(a) | set(aj) if k not in ('train_labels',) + KAGGLE_ALLOW and "
                  "repr(a.get(k)) != repr(aj.get(k)))\n"
                  "KAGGLE_DIFF = {k: (a.get(k), aj.get(k)) for k in KAGGLE_ALLOW if repr(a.get(k)) != repr(aj.get(k))}\n"
                  "P(f'KAGGLE ADDENDUM: schedule args allowed to differ from J1 (probe, J1): {KAGGLE_DIFF}')\n"
                  "_E = int(a.get('epochs', -1))\n"
                  "if KAGGLE_DIFF:     # a Kaggle-scheduled probe: the notebook's rules (J1's own schedule needs none)\n"
                  "    check(KAGGLE_E_MIN <= _E <= 24, f'probe epochs {_E} outside [{KAGGLE_E_MIN}, 24]: not read as the frac75 probe')\n"
                  "    check(a.get('save_epochs') == f'{max(1, _E - 2)}-{_E}', f'save_epochs {a.get(\"save_epochs\")} is not the "
                  "Kaggle notebook rule {max(1, _E - 2)}-{_E}')")
PATCH_ROOT_OLD = "ROOT = H('~/novel/frac75')"
PATCH_ROOT_NEW = "ROOT = H(os.environ.get('KT_F75_ROOT', '~/novel/frac75'))"


def kaggle_read(vals, bars, E, narrow=NARROW):
    """vals/bars: {'G1': x, 'G2': x, 'G3': x}. -> (read, detail). PASS: all >= bar. FAIL: some gate below its bar by
    >= narrow. INCONCLUSIVE-NARROW: every failing gate within narrow of its bar."""
    fails = {k: round(bars[k] - vals[k], 4) for k in ('G1', 'G2', 'G3') if not vals[k] >= bars[k]}
    weak = f' (E {E} < 16: shorter schedule than the kit expects, read with extra care)' if E < 16 else ''
    if not fails:
        return 'PASS', 'all three registered gates pass despite the shorter schedule' + weak
    if all(v < narrow for v in fails.values()):
        return 'INCONCLUSIVE-NARROW', (f'only {sorted(fails)} fail, each by < {narrow} (shortfall {fails}): within the '
                                       f'plausible E {E} vs 24 schedule effect; tie-break = the registered 24-epoch box '
                                       f'probe (RUNBOOK sec 7c), or accept FAIL' + weak)
    return 'FAIL', f'shortfall {fails} (>= {narrow} on at least one gate): FAIL as pre-registered' + weak


def _auc_block(ns, run, eps):
    """mean over epochs `eps` of the OOF-vs-v4 per-label AUCs of run/epochs/oof_f0_s1043_raw_ep{k}.csv"""
    import numpy as np
    rows = []
    for k in eps:
        p = f'{run}/epochs/oof_f0_s1043_raw_ep{k}.csv'
        if not os.path.exists(p):
            return None, f'missing {p}'
        o = ns['rd'](p)[ns['T']].astype(float)
        Y = ns['v4'].loc[o.index, ns['T']].to_numpy(np.float64)
        rows.append(ns['per_label'](o.to_numpy(np.float64), Y))
    M = np.vstack(rows)
    FR = ns['FR']
    return dict(macro12=float(np.mean(np.nanmean(M, 1))), non_frac11=float(np.mean(np.nanmean(np.delete(M, FR, 1), 1))),
                fracture=float(np.nanmean(M[:, FR])), epochs=list(eps)), ''


def _log_curve(lines):
    pat = re.compile(r'^ep (\d+) raw \| gold74 ([0-9.]+) \(\d+ labels\) \| gold58 ([0-9.]+) \(\d+ labels\) \| val AUC vs \S+ '
                     r'([0-9.]+)')
    out = {}
    for l in lines:
        m = pat.match(l)
        if m:
            out[int(m.group(1)) + 1] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))   # 1-based epoch
    return out


def report(ns, code):
    import numpy as np
    P, a = ns['P'], ns['a']
    E = int(a.get('epochs', 24))
    bars = ns['BAR']
    vals = dict(G1=ns['g1'], G2=ns['g2'], G3=ns['g3'])
    P('')
    P('== KAGGLE ADDENDUM (report only; the registered verdict and exit code above are unchanged) ==')
    rp = H(os.environ.get('F75_KAGGLE_RECEIPT', '~/fleet_runs/kaggle_frac75_f01_run_receipt.json'))
    rec_info = None
    if os.path.exists(rp):
        r = json.load(open(rp))
        pl = r.get('plan') or {}
        rec_info = dict(status=r.get('status'), E=pl.get('E'), save_epochs=pl.get('save_epochs'), s_it=pl.get('s_it'),
                        elapsed_h=r.get('elapsed_h'), warnings=r.get('warnings'))
        P(f'kaggle receipt {rp}: {rec_info}')
        if pl.get('E') is not None and int(pl['E']) != E:
            P(f'  WARN: receipt plan E {pl.get("E")} != ckpt args epochs {E}')
    else:
        P(f'kaggle receipt {rp}: not present (copy it per the RUNBOOK; report only)')
    P(f'schedule: probe cosine over E {E} epochs vs J1 24 (same seed, folds, data order; T4 x2, 3 workers vs V100, 10)')
    last3 = tuple(range(max(1, E - 2), E + 1))
    pb, why_p = _auc_block(ns, ns['RUN'], last3)
    jb, why_j = _auc_block(ns, ns['J1_RUN'], J1_LAST3)
    s3 = None
    if pb and jb:
        s3 = {k: dict(probe=pb[k], J1=jb[k], d=pb[k] - jb[k]) for k in ('macro12', 'non_frac11', 'fracture')}
        P(f'schedule-end read (selection-free, mean of the last 3 epochs; probe ep {last3} vs J1 ep {J1_LAST3}), OOF vs v4: '
          + ' | '.join(f'{k} {v["probe"]:.4f} vs {v["J1"]:.4f} (d {v["d"]:+.4f})' for k, v in s3.items()))
    else:
        P(f'schedule-end read: n/a ({why_p or why_j})')
    cp = _log_curve(ns['lines'])
    cj = _log_curve(open(f'{ns["J1_RUN"]}.log', errors='replace').read().splitlines())
    tab = []
    for k in range(max(1, E - 5), E + 1):
        j = int(round(k * 24 / E))
        if k in cp and j in cj:
            tab.append((k, j, cp[k], cj[j]))
            P(f'  matched schedule fraction {k / E:.2f}: probe ep {k:2d} gold74 {cp[k][0]:.4f} gold58 {cp[k][1]:.4f} val '
              f'{cp[k][2]:.4f} | J1 ep {j:2d} gold74 {cj[j][0]:.4f} gold58 {cj[j][1]:.4f} val {cj[j][2]:.4f}')
    l3 = None
    if all(k in cp for k in last3) and all(j in cj for j in J1_LAST3):
        l3 = {nm: (float(np.mean([cp[k][i] for k in last3])), float(np.mean([cj[j][i] for j in J1_LAST3])))
              for i, nm in enumerate(('gold74', 'gold58', 'val_v4'))}
        P('trainer-logged last-3 means (12-label macro, fp16 GPU eval): ' +
          ' | '.join(f'{nm} {p:.4f} vs J1 {q:.4f} (d {p - q:+.4f})' for nm, (p, q) in l3.items()))
    a74, a74j, FR = ns['a74'], ns['a74j'], ns['FR']
    dF = float(a74[FR] - a74j[FR])
    dO = float(np.nanmean(np.delete(a74, FR)) - np.nanmean(np.delete(a74j, FR)))
    P(f'internal control, gold74 best raw (pass A): Fracture d {dF:+.4f} | 11 non-Fracture macro d {dO:+.4f} | '
      f'Fracture-specific (d Frac - d other) {dF - dO:+.4f}  (OOF 11 non-Fracture d {ns["g1"] - ns["g1j"]:+.4f})')
    read, detail = kaggle_read(vals, bars, E)
    if code == 0 and read != 'PASS' or code == 1 and read == 'PASS':
        P(f'  WARN: addendum read {read} disagrees with the scorer exit {code} (bug: trust the exit code)')
    P(f'KAGGLE ADDENDUM READ: {read}: {detail}' + ('  [DRY RUN, NOT A DECISION]' if ns['A'].dry else ''))
    return dict(kaggle_read=read, kaggle_detail=detail, kaggle_E=E, kaggle_diff=ns.get('KAGGLE_DIFF'),
                kaggle_receipt=rec_info, schedule_end_last3=s3, logged_last3=l3, matched_fraction=tab,
                gold74_internal_control=dict(d_frac=dF, d_other11=dO, frac_specific=dF - dO),
                kaggle_scorer=os.path.abspath(__file__), orig_scorer=ORIG, orig_scorer_sha256=ORIG_SHA256)


def main():
    b = open(ORIG, 'rb').read()
    if hashlib.sha256(b).hexdigest() != ORIG_SHA256:
        print(f'INTEGRITY ERROR: {ORIG} sha256 {hashlib.sha256(b).hexdigest()} != pinned {ORIG_SHA256} (the '
              f'pre-registered scorer changed; re-review before scoring)', flush=True)
        sys.exit(2)
    src = b.decode()
    for old, new in ((PATCH_PAIR_OLD, PATCH_PAIR_NEW), (PATCH_ROOT_OLD, PATCH_ROOT_NEW)):
        if src.count(old) != 1:
            print(f'INTEGRITY ERROR: patch anchor found {src.count(old)} times: {old[:60]}', flush=True)
            sys.exit(2)
        src = src.replace(old, new)
    ns = dict(__name__='__score_frac75__', __file__=ORIG, KAGGLE_ALLOW=ALLOW, KAGGLE_E_MIN=E_MIN)
    sys.argv[0] = ORIG
    try:
        exec(compile(src, ORIG, 'exec'), ns)
        code = 0
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    if code not in (0, 1):
        sys.exit(code)
    extra = report(ns, code)
    root, tag = ns['ROOT'], ns['TAG']
    open(f'{root}/out/gate_{tag}.txt', 'w').write('\n'.join(ns['L']) + '\n')
    jp = f'{root}/out/gate_{tag}.json'
    j = json.load(open(jp))
    j.update(extra)
    json.dump(j, open(jp, 'w'), indent=1, default=str)
    sys.exit(code)


if __name__ == '__main__':
    main()
