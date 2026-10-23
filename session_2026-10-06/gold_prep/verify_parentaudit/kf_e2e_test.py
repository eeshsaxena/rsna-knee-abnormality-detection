"""Verifier's end-to-end test of the KF hedge through the REAL K-kit cell (CPU only, no GPU, no network).
A stub fleet dataset (fleet_manifest.json + a stub fleet_worker.py that serves a fixed prediction table for every
member, with the K5 TTA report) and a stub `torch` module (1 fake device, never touches a GPU) let _k2_entry_inner()
run to a CLEAN stage. Checks: clean, every wXX == rank((1-W) rank(P_flat) + W rank(F)), submission.csv == w_main,
_k2_parent.csv == true parent bytes, status/receipt carry outer_flat, and the gold58 AUC of submission.csv.
Usage: python kf_e2e_test.py <cell31.py> <kf_patch_block.py> <parent_work_dir> <parent_csv_name> <fleet_csv> [labels_csv]
"""
import os, sys, json, shutil, tempfile, types, io, contextlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kf_integration_test import splice  # noqa: E402  (same splice as the integration test; its cases run on import)

c31_p, kf_p, src_work, par_name, fleet_csv = sys.argv[1:6]
labels_csv = sys.argv[6] if len(sys.argv) > 6 else None
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
WORKER = r'''
import os, json, numpy as np, pandas as pd
out = os.environ['FLEET_OUT']; work = os.path.dirname(out); fd = os.environ['FLEET_DIR']
g, ng = int(os.environ['FLEET_SHARD']), int(os.environ['FLEET_NSHARD'])
man = json.load(open(os.path.join(fd, 'fleet_manifest.json'))); names = man['checkpoints']
ids = pd.read_csv(os.path.join(work, '_k2_parent.csv'), dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()[g::ng]
F = pd.read_csv(os.path.join(fd, 'stub_fleet.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
P = np.stack([F.loc[ids].to_numpy(np.float32) for _ in names])
modes = {n: man['sources'][n].get('tta', 'A') for n in names}; rows = len(ids)
fwd = {n: {'A': rows, 'B': rows if modes[n] == 'AB' else 0} for n in names}
np.savez(out, names=np.array(names), n_keep=len(names), hard_stopped=False, minutes=0.0, loop_min=0.0, n_fail=0,
         n_blank=0, parity=json.dumps({}), stats=json.dumps({'decode_s': 0.0}), P=P, NV=np.full(rows, 4),
         sids=np.array(ids), tta=json.dumps({'modes': modes, 'half': 0.5 - 1e-3, 'rows': rows, 'fwd': fwd}))
'''


def main():
    C31 = open(c31_p, encoding='utf-8').read(); KF = open(kf_p, encoding='utf-8').read()
    import kf_integration_test as kit
    kit.KF_BLOCK = KF[:KF.index('# ---- KF call lines')]
    cfg = eval(C31.split('\n', 1)[0].split('=', 1)[1].split('#')[0])
    names = cfg['MEMBERS']; tta = cfg.get('TTA') or {}
    res = {}
    for flat in (None, 0.60):
        d = tempfile.mkdtemp(prefix='kfe2e_'); fd = tempfile.mkdtemp(prefix='kfe2e_fleet_')
        os.makedirs(f'{d}/diagnostics')
        for f in ('btk_v32_rest_input.npz', '_raptor.csv', 'diagnostics/btkd_parent_after_rad.npz'):
            shutil.copyfile(f'{src_work}/{f}', f'{d}/{f}')
        shutil.copyfile(f'{src_work}/{par_name}', f'{d}/submission.csv')
        par = pd.read_csv(f'{d}/submission.csv', dtype={'StudyInstanceUID': str})
        fl = pd.read_csv(fleet_csv, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        fl.loc[par.StudyInstanceUID, T].reset_index().to_csv(f'{fd}/stub_fleet.csv', index=False)
        json.dump({'checkpoints': names, 'sources': {n: {'tta': tta.get(n, 'A'), 'sha256': 'stub'} for n in names}},
                  open(f'{fd}/fleet_manifest.json', 'w'))
        open(f'{fd}/fleet_worker.py', 'w').write(WORKER)
        for n in names:
            open(f'{fd}/{n}', 'w').write('stub')
        s = kit.splice(C31, flat).replace("_K2_WORK = '/kaggle/working'", f'_K2_WORK = {d!r}', 1)
        s = s.replace("for _c in ('/kaggle/input/datasets/eeshsaxena/rsna-knee-cnxt-fleet', '/kaggle/input/rsna-knee-cnxt-fleet'):",
                      f'for _c in ({fd!r},):', 1)
        s = s.replace('_k2_receipt = _k2_entry()', '_k2_receipt = None', 1)
        fake = types.ModuleType('torch')
        fake.cuda = types.SimpleNamespace(device_count=lambda: 1, is_available=lambda: True, empty_cache=lambda: None)
        real = sys.modules.get('torch'); sys.modules['torch'] = fake
        G = {'__name__': '__kfe2e__', '_coatnet_weight': dict({t: 0.60 for t in T}, **{
            'ACL': 0.75, 'Medial Meniscus': 0.80, 'Lateral Meniscus': 1.00, 'Lateral OA': 0.75, 'Fracture': 0.75})}
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(s, 'c31_kf', 'exec'), G)
                rec = G['_k2_entry_inner']()
        finally:
            if real is None:
                sys.modules.pop('torch', None)
            else:
                sys.modules['torch'] = real
        rk = lambda a: pd.DataFrame(np.asarray(a, np.float64)).rank(method='average', pct=True).to_numpy()
        P0 = par[T].to_numpy(np.float64)
        Pf = G['_kf_outer_flat'](par, flat, {}) if flat is not None else P0
        F = fl.loc[par.StudyInstanceUID, T].to_numpy(np.float32).astype(np.float64)
        st = json.load(open(f'{d}/_k2_stage_fleet.json'))
        r = {'clean': rec.get('clean'), 'why': rec.get('why')[:160], 'status_outer_flat': st.get('outer_flat'),
             'parent_file_is_true_parent': bool(open(f'{d}/_k2_parent.csv', 'rb').read() ==
                                                open(f'{src_work}/{par_name}', 'rb').read()),
             'receipt_outer_flat': (rec.get('outer_flat') or {}).get('rebuild_max_abs')}
        if rec.get('clean'):
            for w in [cfg['W_MAIN']] + list(cfg['W_VARIANTS']):
                fn = f'submission_w{int(round(w * 100)):02d}.csv'
                got = pd.read_csv(f'{d}/{fn}', dtype={'StudyInstanceUID': str})[T].to_numpy(np.float64)
                r[f'{fn}_max_abs_vs_recompute'] = float(np.abs(got - rk((1 - w) * rk(Pf) + w * rk(F))).max())
            sub = pd.read_csv(f'{d}/submission.csv', dtype={'StudyInstanceUID': str})
            wm = pd.read_csv(f"{d}/submission_w{int(round(cfg['W_MAIN'] * 100)):02d}.csv", dtype={'StudyInstanceUID': str})
            r['submission_eq_wmain'] = bool(sub.equals(wm))
            if labels_csv:
                from sklearn.metrics import roc_auc_score
                Y = pd.read_csv(labels_csv, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[
                    sub.StudyInstanceUID, T].to_numpy(np.float64)
                r['auc_submission'] = round(float(np.mean([roc_auc_score(Y[:, j], sub[T].to_numpy()[:, j])
                                                           for j in range(12)])), 5)
        res[str(flat)] = r
        shutil.rmtree(d, ignore_errors=True); shutil.rmtree(fd, ignore_errors=True)
    print('E2E', json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
