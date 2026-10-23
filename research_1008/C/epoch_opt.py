# epoch_opt.py: gold58 epoch-selection optimism analog from our own training logs (macro). Light CPU.
import re, glob, numpy as np
OLD = 'C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad'
R = 'C:/Users/eeshs/Downloads/rsna-knee'
logs = sorted(set(glob.glob(OLD + '/k7/kaggle_s1044/out_*/*.log') + glob.glob(OLD + '/kaggle_frac75/out_v1/*.log') +
                  glob.glob(OLD + '/kaggle_train/out_v1/*.log') + glob.glob(R + '/session_2026-10-07/kaggle_frac75b/out_*/*.log')))
pat = re.compile(r'^ep (\d+) raw .*gold74 ([0-9.]+).*gold58 ([0-9.]+).*vs llm_labels_v4_blend.csv ([0-9.]+)')
rows = []
for L in logs:
    ep = []
    for ln in open(L, encoding='utf-8', errors='ignore'):
        m = pat.search(ln)
        if m: ep.append(tuple(float(x) for x in m.groups()))
    if len(ep) < 4: continue
    ep = np.array(ep); n = len(ep)
    late = ep[ep[:, 0] >= max(ep[:, 0].max() * 0.5, 0)]          # second half of training (where any pick would be)
    jv = int(np.argmax(late[:, 3])); jg = int(np.argmax(late[:, 2]))
    opt = late[jg, 2] - late[jv, 2]
    rows.append(opt)
    print(f'{L.split("/")[-1]:28s} epochs {n:2d} late {len(late):2d} | val-best ep {int(late[jv,0]):2d} gold58 {late[jv,2]:.4f} | '
          f'gold58-best ep {int(late[jg,0]):2d} gold58 {late[jg,2]:.4f} | optimism {opt:+.4f} | late gold58 SD {late[:,2].std():.4f}')
r = np.array(rows)
print(f'n {len(r)} runs: macro gold58 optimism of picking the epoch on gold58 (vs on val) mean {r.mean():+.4f} SD {r.std():.4f} max {r.max():+.4f}')
