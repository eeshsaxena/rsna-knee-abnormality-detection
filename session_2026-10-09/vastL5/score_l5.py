# score_l5.py: the PRE-REGISTERED L5 gate (RESEARCH_TO_TOP_10-08.md L5), written 2026-10-09 ~11:30 IST BEFORE the
# L5 run started. Laptop side, read only: python score_l5.py CONTROL.log L5.log [--json]
#   CONTROL = ~/fleet_runs/f0_s1045_ttr288frac75.log (Q4 f0: 16 train slots), L5 = ~/fleet_runs/f0_s1045_ttr288frac75s24.log
# "val" = the trainer's "val AUC vs llm_labels_v4_blend.csv" on the "ep N raw" lines (same field g320_gate.py reads).
# Gate: PASS iff
#   (1) last-3 raw val mean (epochs 19-21) of L5 >= that of Q4 f0 + .002, AND
#   (2) gold58 not lower by more than .003. PRIMARY reading = gold58 at the best-raw-val epoch (the convention of
#       ~/queue/gates/g320_gate.py, and the epoch whose *_best.pt ships). SECONDARY reading = last-3 gold58 mean,
#       reported only; if the two readings disagree on clause (2) the result is flagged "reading-sensitive".
# Both logs must have 21 raw epoch lines and the closing {"raw": .., "ema": ..} line, else the verdict is INVALID.
import json, re, sys
RE = re.compile(r'^ep (\d+) (raw|ema) \| gold74 ([0-9.]+) \(12 labels\) \| gold58 ([0-9.]+) \(12 labels\) \| '
                r'val AUC vs llm_labels_v4_blend\.csv ([0-9.]+)')
BAR_LAST3, TOL_GOLD58, EPOCHS = .002, .003, 21


def read(path):
    L = open(path, encoding='utf-8', errors='replace').read().replace('\r', '').splitlines()
    ev = {'raw': [], 'ema': []}
    for l in L:
        m = RE.match(l)
        if m:
            ev[m.group(2)].append(dict(ep=int(m.group(1)) + 1, gold74=float(m.group(3)), gold58=float(m.group(4)), val=float(m.group(5))))
    closing = L[-1] if L and L[-1].startswith('{"raw"') else None
    raw = ev['raw']
    best = max(raw, key=lambda e: e['val']) if raw else None
    tmin = [float(x) for x in re.findall(r'TRAIN-ONLY ([0-9.]+) min', '\n'.join(L))]
    return dict(path=path, n_raw=len(raw), closing=closing, raw=raw, ema=ev['ema'],
                last3_val=sum(e['val'] for e in raw[-3:]) / 3 if len(raw) >= 3 else None,
                last3_gold58=sum(e['gold58'] for e in raw[-3:]) / 3 if len(raw) >= 3 else None,
                last3_gold74=sum(e['gold74'] for e in raw[-3:]) / 3 if len(raw) >= 3 else None,
                best_val=best['val'] if best else None, best_ep=best['ep'] if best else None,
                gold58_at_best=best['gold58'] if best else None, gold74_at_best=best['gold74'] if best else None,
                ema_last3_val=sum(e['val'] for e in ev['ema'][-3:]) / 3 if len(ev['ema']) >= 3 else None,
                train_min_per_ep=sum(tmin) / len(tmin) if tmin else None)


def score(c, t):
    valid = all(r['n_raw'] == EPOCHS and r['closing'] for r in (c, t))
    d_last3 = t['last3_val'] - c['last3_val']
    d_g58_best = t['gold58_at_best'] - c['gold58_at_best']
    d_g58_last3 = t['last3_gold58'] - c['last3_gold58']
    c1 = d_last3 >= BAR_LAST3 - 1e-12
    c2 = d_g58_best >= -TOL_GOLD58 - 1e-12
    c2b = d_g58_last3 >= -TOL_GOLD58 - 1e-12
    verdict = 'INVALID' if not valid else ('PASS' if (c1 and c2) else 'FAIL')
    return dict(verdict=verdict, valid=valid, clause1_last3=c1, clause2_gold58_at_best=c2, clause2_secondary_last3=c2b,
                reading_sensitive=(c2 != c2b), d_last3_val=d_last3, d_gold58_at_best=d_g58_best, d_gold58_last3=d_g58_last3,
                bar_last3=c['last3_val'] + BAR_LAST3, floor_gold58=c['gold58_at_best'] - TOL_GOLD58)


if __name__ == '__main__':
    c, t = read(sys.argv[1]), read(sys.argv[2])
    s = score(c, t)
    if '--json' in sys.argv:
        print(json.dumps(dict(control={k: v for k, v in c.items() if k not in ('raw', 'ema')},
                              l5={k: v for k, v in t.items() if k not in ('raw', 'ema')}, gate=s), indent=1))
        sys.exit(0)
    for name, r in (('Q4 f0 (16 slots)', c), ('L5 f0 (24 slots)', t)):
        print(f"{name}: raw epochs {r['n_raw']} | last-3 raw val {r['last3_val']:.4f} | best raw val {r['best_val']:.4f} @ep{r['best_ep']} "
              f"| gold58 @best {r['gold58_at_best']:.4f} | last-3 gold58 {r['last3_gold58']:.4f} | gold74 @best {r['gold74_at_best']:.4f} "
              f"| ema last-3 val {r['ema_last3_val']:.4f} | closing {r['closing']}")
    print(f"clause 1: d last-3 raw val {s['d_last3_val']:+.4f} (bar +{BAR_LAST3:.3f}) -> {'OK' if s['clause1_last3'] else 'NOT MET'}")
    print(f"clause 2: d gold58 @best {s['d_gold58_at_best']:+.4f} (floor -{TOL_GOLD58:.3f}) -> {'OK' if s['clause2_gold58_at_best'] else 'NOT MET'}"
          f" | secondary d last-3 gold58 {s['d_gold58_last3']:+.4f} -> {'OK' if s['clause2_secondary_last3'] else 'NOT MET'}"
          f"{' | READING-SENSITIVE' if s['reading_sensitive'] else ''}")
    print(f"VERDICT: {s['verdict']}")
