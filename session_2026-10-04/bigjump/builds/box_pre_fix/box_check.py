# box_check.py (box, CPU, read-only on everything outside ~/bigjump_build): environment, contract files, model_sha
# registry in the box env, and B3 (the staged gated label file vs steven_v4 / report verdicts / store / folds).
import os, sys, ast, json, hashlib
H = os.path.expanduser('~'); BJ = f'{H}/bigjump_build'; CODE = f'{BJ}/code'
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, torch, timm, cv2
print(f'env: python {sys.version.split()[0]} ({sys.executable}) torch {torch.__version__} timm {timm.__version__} '
      f'cv2 {cv2.__version__} numpy {np.__version__} | cuda visible {torch.cuda.is_available()} | nproc {os.cpu_count()}')
ok = True
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
for f in ('fleet_common.py', 'fleet_model.py', 'fleet_model_f2.py'):
    refs = {'bigjump': f'{CODE}/{f}', '~/fleet2_code': f'{H}/fleet2_code/{f}'}
    if f != 'fleet_model_f2.py':
        refs['~/fleet'] = f'{H}/fleet/{f}'
    h = {k: md5(p) for k, p in refs.items()}
    same = len(set(h.values())) == 1
    ok &= same
    print(f'{"PASS" if same else "FAIL"} contract file {f} identical: {h}')
sys.path.insert(0, CODE)
tree = ast.parse(open(f'{CODE}/fleet_worker.py', encoding='utf-8').read())
node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_model_sha'][0]
ns = {}; exec(compile(ast.Module(body=[node], type_ignores=[]), 'fleet_worker._model_sha', 'exec'), ns)
shas = {m: ns['_model_sha'](m) for m in ('fleet_model', 'fleet_model_f2', 'fleet_model_ttr')}
good = shas['fleet_model'] == 'fa35880fca963652' and len(set(shas.values())) == 3
ok &= good
print(f'{"PASS" if good else "FAIL"} model_sha (box env, bigjump worker registry): {shas} (family 1 must be fa35880fca963652)')
# B3
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
       "Baker's", 'Contusion', 'Fracture']
g = pd.read_csv(f'{H}/rsna_train/labels_gated75_logit.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
v4 = pd.read_csv(f'{H}/rsna_train/llm_labels_v4_blend.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
r2 = pd.read_csv(f'{H}/rsna_train/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
folds = pd.read_csv(f'{H}/rsna_train/folds5.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
gold = set(pd.read_csv(f'{H}/rsna_train/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID)
c1 = list(g.columns) == LAB and set(g.index) == set(v4.index) and len(g) == len(v4)
G = g[LAB].to_numpy(float); V = v4.reindex(g.index)[LAB].to_numpy(float)
c2 = np.isfinite(G).all() and (G >= 0).all() and (G <= 1).all()
verd = np.stack([r2.reindex(g.index)[f'{l}__verdict'].fillna('UNK').to_numpy() for l in LAB], 1)
unk = verd == 'UNK'
d = np.abs(G - V)
c3 = float(d[~unk].max()) < 1e-9
frac_unk = float(unk.mean()); frac_chg_unk = float((d[unk] > 1e-9).mean())
have = {s for s in g.index if os.path.exists(f'{H}/fleet_store/{s}.npz')}
held = set(folds.index[folds < 0]) | gold
va = [s for s in g.index if s in folds.index and s in have and s not in held and folds[s] == 0]
tr = [s for s in g.index if s in folds.index and s in have and s not in held and folds[s] != 0]
c4 = len(va) == 867 and len(tr) == 3466
for c, msg in ((c1, f'columns == StudyInstanceUID + 12 targets, same {len(g)} ids as llm_labels_v4_blend.csv'),
               (c2, 'all finite, in [0, 1]'),
               (c3, f'report-ADDRESSED cells == steven_v4 exactly (max |diff| {float(d[~unk].max()):.2e} over {int((~unk).sum())} cells)'),
               (frac_chg_unk > 0.9, f'report-UNK cells {frac_unk:.1%} of all cells; {frac_chg_unk:.1%} of them differ from v4 '
                                    f'(mean |diff| {float(d[unk].mean()):.3f})'),
               (c4, f'fold-0 contract with these labels: train {len(tr)} / val {len(va)} (expect 3466 / 867), store has {len(have)}')):
    ok &= bool(c)
    print(f'{"PASS" if c else "FAIL"} B3 labels_gated75_logit.csv: {msg}')
print(f'BOXCHECK {"PASS" if ok else "FAIL"}')
sys.exit(0 if ok else 1)
