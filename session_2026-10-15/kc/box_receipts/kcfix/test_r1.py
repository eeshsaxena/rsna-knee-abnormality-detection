import os, re, sys, subprocess, pandas as pd, numpy as np
H = os.path.expanduser('~'); R = f'{H}/kc/kcfix/r1'
src = open(f'{H}/kc/code/fleet_train_kc.py').read()
line = [l for l in src.splitlines() if l.startswith('gold_ids = [s for s in gold_tab.index')]
assert len(line) == 1; line = line[0]; print('trainer line:', line)
gold_tab = pd.read_csv(f'{H}/rsna_train/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
have = set(pd.read_csv(f'{H}/raptor_store/rsna_index.csv', dtype={'id': str}).id)      # the full store = all 4,407 index ids
L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
r7 = pd.read_csv(f'{H}/r7t_labels/real/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
fo = pd.read_csv(f'{H}/rsna_train/folds5.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
for tag in ('old', 'new'):
    lab = pd.read_csv(f'{R}/{tag}.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    ns = dict(gold_tab=gold_tab, have=have, lab=lab); exec(line, ns); gids = ns['gold_ids']
    nan_rows = lab.index[lab[L].isna().all(1)]
    print(f'{tag}: rows {len(lab)} | trainer gold_ids {len(gids)}/74 | all-NaN rows {len(nan_rows)} (all fold<0: {bool((fo.loc[nan_rows] < 0).all())}) '
          f'| NaN rows in folds5 pool {int((fo.reindex(nan_rows) >= 0).sum())}')
    # stand-in member gold csv exactly as KC-9 writes it (index = gold_ids, 12 label columns)
    pd.DataFrame(r7.loc[gids, ['p|' + l for l in L]].values, columns=L, index=pd.Index(gids, name='StudyInstanceUID')).to_csv(f'{R}/gold_{tag}.csv')
    cmd = [sys.executable, f'{H}/kc/code/kc_oof_kit.py', '--g2', '--refs', f'{H}/kc/gate/refs.json', '--kc_gold', f'{R}/gold_{tag}.csv', f'{R}/gold_{tag}.csv',
           '--kc_oof', f'{H}/fleet_runs/f0_s1047_ttr_r7tb/oof_f0_s1047_raw.csv', '--out', f'{R}/g2_{tag}.txt']
    p = subprocess.run(cmd, capture_output=True, text=True)
    last = (p.stdout.strip().splitlines() or [''])[0] if p.returncode == 0 else p.stderr.strip().splitlines()[-1]
    print(f'  kc_oof_kit --g2 on a {len(gids)}-row member gold csv: rc {p.returncode} | {last[:170]}')
