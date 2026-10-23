# build_labels_raptor.py (box CPU, seconds): RSNA targets for our Raptor seed = nartaa's "blendjev" structure with lixin's
# GPT-5.6 read (labels_llm_gpt56sol.csv) in place of their non-public Gemini 3 Flash read (decision 10-13: diversity).
#   blend    = mean(sol56, v4_blend); where v4 == 0.25 exactly ("not addressed") -> sol56 alone; sol56 missing -> v4
#   blendjev = mean(blend, JEV HYBRID) where JEV is finite, else blend
# Output ~/raptor2/labels_raptor_rsna_blendjev_sol.csv (4,407 rows x 12, all finite). gold58 rows are kept in the file and
# removed by the trainer (by id). Prints the gold58 macro AUC of the table (diagnostic; JEV is in-sample on gold58).
import hashlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
H = '/home/training'
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
SRC = dict(v4=f'{H}/rsna_train/llm_labels_v4_blend.csv', sol=f'{H}/rsna_train/labels_llm_gpt56sol.csv',
           jev=f'{H}/jump2/labels_in/jev_hybrid_scores.csv')
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T].astype(float)
for k, p in SRC.items():
    print(f'input {k:4s} md5 {md5(p)} {p}')
v4 = rd(SRC['v4']); ids = v4.index
assert len(ids) == 4407 and ids.is_unique and np.isfinite(v4.values).all()
sol = rd(SRC['sol']).reindex(ids); jev = rd(SRC['jev']).reindex(ids)
b = (sol + v4) / 2
m = (v4 == 0.25) & sol.notna(); b[m] = sol[m]
b[sol.isna()] = v4[sol.isna()]
out = (b + jev) / 2
out[jev.isna()] = b[jev.isna()]
assert np.isfinite(out.values).all() and (out.values >= 0).all() and (out.values <= 1).all()
print(f'v4 == .25 cells {(v4 == 0.25).values.mean():.3f} | sol missing cells {sol.isna().values.mean():.4f} | jev missing cells {jev.isna().values.mean():.4f}')
g = pd.read_csv(f'{H}/rsna_train/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g = g[g.source == 'original_gold58'].set_index('StudyInstanceUID')
a = [roc_auc_score(g[t], out.reindex(g.index)[t]) for t in T]
print(f'gold58 macro {np.mean(a):.4f} | ' + ' '.join(f'{x:.3f}' for x in a))
p = f'{H}/raptor2/labels_raptor_rsna_blendjev_sol.csv'
out.reset_index().to_csv(p, index=False, float_format='%.8g')
print('wrote', p, 'md5', md5(p), '| mean target per label', ' '.join(f'{x:.3f}' for x in out.drop(index=g.index).mean()))
