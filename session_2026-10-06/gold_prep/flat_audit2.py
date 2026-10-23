"""Pick-2 shapes vs pick 1 (= parent map, W .50): flat .60 at W .50 / .60, map at W .60. Both brackets, 2 fleet proxies.
Imports flat_audit.py's objects by running it (prints suppressed)."""
import io, contextlib, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import flat_audit as fa
from inject_common import final_blend, k2_global, mauc, rank_pct
import numpy as np
out = {}
for b, Tb in fa.TB.items():
    for fn, F in fa.FLEETS.items():
        p_map = final_blend(Tb, fa.side, fa.MAPS['map']); p_flat = final_blend(Tb, fa.side, fa.MAPS['flat60'])
        pick1 = k2_global(p_map, F, 0.50)
        cands = {'flat_W50': k2_global(p_flat, F, 0.50), 'flat_W60': k2_global(p_flat, F, 0.60),
                 'map_W60': k2_global(p_map, F, 0.60), 'flat_W40': k2_global(p_flat, F, 0.40)}
        r = {'pick1_auc': round(mauc(fa.Y, pick1), 4)}
        for k, v in cands.items():
            s = fa.summarize(v, pick1)
            rho_all = float(np.mean([np.corrcoef(rank_pct(v)[:, j], rank_pct(pick1)[:, j])[0, 1] for j in range(12)]))
            r[k] = {'d_macro': s['d_macro'], 'ci90': s['ci90'], 'P_gt0': s['P_gt0'], 'mars': s['mars'],
                    'rho_mean12_vs_pick1': round(rho_all, 4), 'sd390': s['scaled_sd_public390']}
        out[f'{b}/{fn}'] = r
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flat_audit2_out.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
