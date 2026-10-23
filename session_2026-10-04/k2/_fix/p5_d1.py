import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_lib import patch
K2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
patch(f'{K2}/box/d1_pick.py', [
("""  ~/miniconda3/envs/kmimic/bin/python ~/k2_test/k2/box/d1_pick.py [--final_epochs 10]
""",
"""  CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/kmimic/bin/python ~/k2_test/k2/box/d1_pick.py [--final_epochs 10]

Exit 0 = final pick (SHIP line), 2 = both runs fail G1 (no K2), 3 = provisional (a trainer has not printed its closing
{"raw": .., "ema": ..} line yet, or a best file is mid-save): never act on exit 3, re-run later.
FINAL means: the run's log holds its closing json line, which fleet_train.py prints only AFTER the last save(); the
last EMA eval + save land ~4.6 min after the last 'ep 9 raw' line (box 10-04: raw save 03:23:07, EMA save 03:27:41), and
save() deletes the old best file before writing the new one, so '10 epochs logged' alone is not final.
The best file's epoch and exact cv are read from the checkpoint itself (ck['epoch'], ck['cv']), never matched back
from the truncated cv in its file name.
"""),
("""import argparse, glob, os, re, sys
""",
"""import argparse, glob, json, os, re, sys
"""),
("""def run_info(name, d, arm, final_epochs):
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
""",
"""class Provisional(Exception):
    \"\"\"The run is still training or saving: nothing may be shipped from this call.\"\"\"


def run_info(name, d, arm, final_epochs):
    import torch                                                  # kmimic env; CPU only (CUDA_VISIBLE_DEVICES=)
    rows, done = [], False
    for l in open(d + '.log', errors='replace'):
        l = l.strip()
        if l.startswith('{"raw"'):                                # json.dumps(best): printed only after the last save()
            try:
                done = isinstance(json.loads(l), dict)
            except ValueError:
                pass
        m = LINE.match(l)
        if m:
            rows.append(dict(ep=int(m[1]), tag=m[2], gold74=float(m[3]), gold58=float(m[4]), val=float(m[5])))
    df = pd.DataFrame(rows)
    assert len(df), f'{name}: no epoch lines in {d}.log'
    last_ep = int(df.ep.max())
    best = {}
    for tag in ('raw', 'ema'):
        fs = sorted(glob.glob(f'{d}/f0_s42_{tag}_*_best.pt'))
        try:
            assert len(fs) == 1, f'{len(fs)} {tag} best files {fs}'
            ck = torch.load(fs[0], map_location='cpu', weights_only=False)
            cv, ep_ck = float(ck['cv']), int(ck['epoch']); del ck   # exact cv + the epoch that saved this file
        except Exception as e:
            if done:
                raise
            raise Provisional(f'{name} {tag}: best file not readable while training/saving ({type(e).__name__}: '
                              f'{str(e)[:120]}); re-run in a few minutes')
        sub = df[(df.tag == tag) & (df.ep == ep_ck)]
        assert len(sub), (name, tag, f'no log line for epoch {ep_ck} of {fs[0]}')
        row = sub.iloc[-1]
        assert abs(float(row.val) - cv) < 6e-5, (name, tag, 'log val (.4f) != ckpt cv', float(row.val), cv, ep_ck)
        best[tag] = dict(path=fs[0], cv=cv, ep=ep_ck, gold58=float(row.gold58), val=float(row.val))
"""),
("""    return dict(name=name, epochs_logged=last_ep + 1, final=last_ep + 1 >= final_epochs, tag=tag, ship=best[tag],
""",
"""    return dict(name=name, epochs_logged=last_ep + 1, final=done and last_ep + 1 >= final_epochs, done=done, tag=tag, ship=best[tag],
"""),
("""    R = {k: run_info(k, d, arm, a.final_epochs) for k, d in RUNS.items()}
    for k, r in R.items():
        print(f'{k:5s} epochs {r["epochs_logged"]}{"" if r["final"] else " (NOT final)"} | tag {r["tag"]} '
""",
"""    try:
        R = {k: run_info(k, d, arm, a.final_epochs) for k, d in RUNS.items()}
    except Provisional as e:
        print(f'PROVISIONAL (exit 3): {e}')
        return 3
    for k, r in R.items():
        print(f'{k:5s} epochs {r["epochs_logged"]}{"" if r["final"] else " (NOT final: closing json line " + ("present" if r["done"] else "missing") + ")"} | tag {r["tag"]} '
"""),
("""    print(f'D1 winner {win}: {why}')
    print(f'SHIP {pick["ship"]["path"]}  (run {pick["name"]}, tag {pick["tag"]}, best epoch {pick["ship"]["ep"] + 1})'
          + (f'  WARNING provisional: {not_final} not at {a.final_epochs} epochs' if not_final else ''))
    print(f'laptop: scp -o BatchMode=yes training@192.168.10.30:{pick["ship"]["path"].replace(H + "/", "")} "$SP/k2/ckpt/"')
    return 0
""",
"""    print(f'D1 winner {win}: {why}')
    if not_final:                                                 # provisional picks never print SHIP, never exit 0
        print(f'PROVISIONAL (exit 3): {not_final} not final (closing json line + {a.final_epochs} epochs); would ship '
              f'{pick["ship"]["path"]} (run {pick["name"]}, tag {pick["tag"]}, best epoch {pick["ship"]["ep"] + 1}). '
              f'Do not act on it.')
        return 3
    print(f'SHIP {pick["ship"]["path"]}  (run {pick["name"]}, tag {pick["tag"]}, best epoch {pick["ship"]["ep"] + 1}, '
          f'cv {pick["ship"]["cv"]:.6f})')
    print(f'laptop: mkdir -p "$SP/k2/ckpt" && scp -o BatchMode=yes training@192.168.10.30:'
          f'{pick["ship"]["path"].replace(H + "/", "")} "$SP/k2/ckpt/"')
    return 0
"""),
])
