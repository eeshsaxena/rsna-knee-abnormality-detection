#!/usr/bin/env python3
"""d1_pick.py (box, read-only): apply the pre-registered D1 label rule + the tag rule + G1/G1b to the two fold-0 runs
and print THE checkpoint to ship in K2 (one member: the D1 loser never ships; one tag per run).

  ~/miniconda3/envs/kmimic/bin/python ~/k2_test/k2/box/d1_pick.py [--final_epochs 10]

Inputs (all read-only): ~/fleet_runs/f0_s42.log (mix50_logit labels), ~/fleet_runs/f0_s42_v4.log (pure steven_v4),
their *_best.pt names (cv = val AUC vs steven_v4 at the best epoch), their oof_*.csv (867 fold-0 val studies) and the
parent's real Raptor v8 arm ~/fleet/teacher_v2_native384_v8_shard0.csv.

Rules (ENDGAME_V3 sec 4 D1 with its [verify] correction, sec 5; fleet2/SCHEDULE 4c):
  tag   per run: EMA unless the raw best file's cv is higher than the EMA best file's cv (ties -> EMA)
  solo  per run: gold58 macro AUC, mean of raw AND ema over the last 3 logged epochs (cuts checkpoint noise)
  rho   per run: mean per-label Spearman(OOF of the chosen tag, v8 arm) on the 867 fold-0 val studies
  lift  per run: closed-form binormal best-W lift over the parent at .943 (fleet_g1b.best_lift)
  D1    mix50 wins only if lift(mix50) - lift(v4) > .001 AND gold58 edge (mix50 - v4) > GOLD_EDGE AND rho(mix50) <=
        rho(v4); if both rho > .85: v4; else v4 (measured decorrelation prior). GOLD_EDGE = .012 stands in for
        1.5 x the paired-bootstrap SD of a 58-study AUC difference (~.008), which needs per-study gold58 predictions
        the logs do not have (k2/RUNBOOK.md gap 3)
  G1    the shipped tag needs val AUC vs steven_v4 >= .87 AND gold58 >= .87 at its best epoch; if the D1 winner fails
        and the other run passes, ship the other; if both fail: no K2 (EV2S f0 is the family switch)
  G1b   rho <= .85 informational here (it steers later runs to v4 labels, it does not block K2)
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd
from scipy.stats import norm

H = os.path.expanduser('~')
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
       'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
RUNS = {'mix50': f'{H}/fleet_runs/f0_s42', 'v4': f'{H}/fleet_runs/f0_s42_v4'}
LINE = re.compile(r'^ep (\d+) (raw|ema) \| gold74 ([\d.]+).*?\| gold58 ([\d.]+).*?\| val AUC vs \S+ ([\d.]+)')
GOLD_EDGE, LIFT_EDGE, RHO_BOTH = 0.012, 0.001, 0.85


def best_lift(a1, a2, rho):                               # == ~/fleet/fleet_g1b.py best_lift
    mu = lambda a: np.sqrt(2) * norm.ppf(a)
    ws = np.linspace(0, 1, 101); m = (1 - ws) * mu(a1) + ws * mu(a2); v = (1 - ws) ** 2 + ws ** 2 + 2 * ws * (1 - ws) * rho
    b = norm.cdf(m / np.sqrt(2 * v)); i = b.argmax(); return float(ws[i]), float(b[i] - a1)


def run_info(name, d, arm, final_epochs):
    rows = []
    for l in open(d + '.log', errors='replace'):
        m = LINE.match(l.strip())
        if m:
            rows.append(dict(ep=int(m[1]), tag=m[2], gold74=float(m[3]), gold58=float(m[4]), val=float(m[5])))
    df = pd.DataFrame(rows)
    assert len(df), f'{name}: no epoch lines in {d}.log'
    last_ep = int(df.ep.max())
    best = {}
    for tag in ('raw', 'ema'):
        fs = sorted(glob.glob(f'{d}/f0_s42_{tag}_*_best.pt'))
        assert len(fs) == 1, (name, tag, fs)
        cv = int(re.search(r'_cv(\d+)_best', fs[0])[1]) / 1e4
        sub = df[df.tag == tag]
        row = sub.iloc[(sub.val - cv).abs().argmin()]                     # the epoch that produced the best file
        best[tag] = dict(path=fs[0], cv=cv, ep=int(row.ep), gold58=float(row.gold58), val=float(row.val))
    tag = 'raw' if best['raw']['cv'] > best['ema']['cv'] else 'ema'
    last3 = df[df.ep > last_ep - 3]
    solo = float(last3.gold58.mean())
    o = pd.read_csv(f'{d}/oof_f0_s42_{tag}.csv', index_col='StudyInstanceUID')
    j = o.index.intersection(arm.index)
    rho = float(np.mean([o.loc[j, l].corr(arm.loc[j, l], method='spearman') for l in LAB]))
    w, lift = best_lift(0.943, min(solo, 0.999), rho)
    g1 = best[tag]['val'] >= 0.87 and best[tag]['gold58'] >= 0.87
    return dict(name=name, epochs_logged=last_ep + 1, final=last_ep + 1 >= final_epochs, tag=tag, ship=best[tag],
                other=best['ema' if tag == 'raw' else 'raw'], solo_gold58_last3=solo, rho_v8=rho, n_rho=len(j),
                best_w=w, lift=lift, g1=g1, g1b=rho <= RHO_BOTH)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--final_epochs', type=int, default=10)
    a = ap.parse_args()
    arm = pd.read_csv(f'{H}/fleet/teacher_v2_native384_v8_shard0.csv', index_col='StudyInstanceUID')
    R = {k: run_info(k, d, arm, a.final_epochs) for k, d in RUNS.items()}
    for k, r in R.items():
        print(f'{k:5s} epochs {r["epochs_logged"]}{"" if r["final"] else " (NOT final)"} | tag {r["tag"]} '
              f'(raw cv {r["ship"]["cv"] if r["tag"] == "raw" else r["other"]["cv"]:.4f}, ema cv '
              f'{r["ship"]["cv"] if r["tag"] == "ema" else r["other"]["cv"]:.4f}) best ep {r["ship"]["ep"] + 1} '
              f'gold58 {r["ship"]["gold58"]:.4f} val {r["ship"]["val"]:.4f} | solo(last3 raw+ema) {r["solo_gold58_last3"]:.4f} '
              f'| rho_v8 {r["rho_v8"]:.3f} (n {r["n_rho"]}) | best W {r["best_w"]:.2f} lift {r["lift"]:+.4f} '
              f'| G1 {"PASS" if r["g1"] else "FAIL"} G1b {"PASS" if r["g1b"] else "FAIL (later runs on v4)"}')
    m, v = R['mix50'], R['v4']
    edge = m['solo_gold58_last3'] - v['solo_gold58_last3']
    if m['rho_v8'] > RHO_BOTH and v['rho_v8'] > RHO_BOTH:
        win, why = 'v4', 'both rho > .85 -> v4'
    elif m['lift'] - v['lift'] > LIFT_EDGE and edge > GOLD_EDGE and m['rho_v8'] <= v['rho_v8']:
        win, why = 'mix50', f'lift edge {m["lift"] - v["lift"]:+.4f} > {LIFT_EDGE}, gold58 edge {edge:+.4f} > {GOLD_EDGE}, rho not higher'
    else:
        win, why = 'v4', (f'default v4 (lift edge {m["lift"] - v["lift"]:+.4f}, gold58 edge {edge:+.4f}, '
                          f'rho mix50 {m["rho_v8"]:.3f} vs v4 {v["rho_v8"]:.3f})')
    pick = R[win]
    if not pick['g1']:
        other = R['v4' if win == 'mix50' else 'mix50']
        if other['g1']:
            why += f' | {win} FAILS G1 -> ship {other["name"]}'; pick = other
        else:
            print(f'D1 {win} ({why}) | BOTH runs fail G1: NO K2 (ENDGAME G1: EV2S f0 is the family switch)')
            return 2
    not_final = [k for k, r in R.items() if not r['final']]
    print(f'D1 winner {win}: {why}')
    print(f'SHIP {pick["ship"]["path"]}  (run {pick["name"]}, tag {pick["tag"]}, best epoch {pick["ship"]["ep"] + 1})'
          + (f'  WARNING provisional: {not_final} not at {a.final_epochs} epochs' if not_final else ''))
    print(f'laptop: scp -o BatchMode=yes training@192.168.10.30:{pick["ship"]["path"].replace(H + "/", "")} "$SP/k2/ckpt/"')
    return 0


if __name__ == '__main__':
    sys.exit(main())
