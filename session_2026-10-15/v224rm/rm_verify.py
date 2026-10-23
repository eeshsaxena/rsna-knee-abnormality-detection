# rm_verify.py (V224-RM): is the r224 mirror TTA the r384 serving code's mirror TTA? Prints R224_MIRROR_VERIFIED <json>.
#   usage: python rm_verify.py R384_SCRIPT R224_PATCHED_SCRIPT     (CPU only; touches no GPU, no weights, no data)
import sys, ast, json, threading, time, types
import numpy as np, torch

S384 = open(sys.argv[1], encoding='utf-8').read()
S224 = open(sys.argv[2], encoding='utf-8').read()


def top_defs(s):
    out = {}
    for node in ast.parse(s).body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = ast.get_source_segment(s, node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = ast.get_source_segment(s, node)
    return out


D384, D224 = top_defs(S384), top_defs(S224)
same = {k: D384.get(k) is not None and D384.get(k) == D224.get(k) for k in ('SLOT_SIZES', 'SAGITTAL_SLOTS', '_slot_index', '_mirror_tri', 'LAB')}
assert all(same.values()), ('r224 mirror pieces differ from r384', same)
TRI = '    tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(dev)   # (k,3,IMG,IMG) uint8\n'
ML = '    if mirror:\n        tri = _mirror_tri(tri, cs, D)\n'
for nm, D in (('r384', D384), ('r224', D224)):
    assert D['gpu_windows'].startswith('def gpu_windows(vol, mask, k, res, dev, norm=NORM, mirror=False):'), nm
    assert (TRI + ML) in D['gpu_windows'] + '\n', (nm, 'gpu_windows mirrors the uint8 triplets first')
for line in ('assert xla is None, "Anatomical L/R TTA requires CUDA/fp16"',
             'xm = gpu_windows(vol, mask, K_EVAL, res, d, norm=NORM, mirror=True)',
             'arm_probs[a][i] = infer_pair_probs(model, xw, xm, d)'):
    assert S384.count(line) == 1 and S224.count(line) == 1, line

# _mirror_tri: functional equality (random uint8 triplets, every centre slot) + its definition (sagittal: channel reverse)
ns384 = {'np': np, 'torch': torch}; ns224 = {'np': np, 'torch': torch}
for k in ('SLOT_SIZES', 'SAGITTAL_SLOTS', '_slot_index', '_mirror_tri'):
    exec(D384[k], ns384); exec(D224[k], ns224)
g = torch.Generator().manual_seed(0)
D = sum(ns384['SLOT_SIZES'])
cs = list(range(1, D - 1))
tri = torch.randint(0, 256, (len(cs), 3, 9, 7), dtype=torch.uint8, generator=g)
m384, m224 = ns384['_mirror_tri'](tri, cs, D), ns224['_mirror_tri'](tri, cs, D)
sag = [c < ns384['SLOT_SIZES'][0] + ns384['SLOT_SIZES'][1] for c in cs]
exp = torch.stack([tri[j].flip(0) if s else tri[j].flip(-1) for j, s in enumerate(sag)])
assert torch.equal(m384, m224) and torch.equal(m224, exp) and torch.equal(ns224['_mirror_tri'](m224, cs, D), tri)

# infer_pair_probs: DERIVE the class map and the combine of each implementation from stub passes
LAB = ast.literal_eval(D384['LAB'].split('=', 1)[1].strip())


def derive(Dx, shape, extra):
    ns = {'np': np, 'torch': torch, 'threading': threading, 'time': time}
    if extra:
        exec(Dx['_PAIR_LAST'], ns)
    exec(Dx['infer_pair_probs'], ns)
    res = {}
    for case, (p, q) in {'id': (0.01 + np.arange(12) / 100.0, 0.5 + np.arange(12) / 100.0),
                         'space': (np.full(12, 0.9), np.full(12, 0.6))}.items():
        p, q = p.astype(np.float32), q.astype(np.float32)
        calls = [p.copy(), q.copy()]
        ns['infer_probs'] = lambda model, x, device: calls.pop(0)
        model = types.SimpleNamespace(modules=lambda: [types.SimpleNamespace(training=False)])
        x = types.SimpleNamespace(shape=shape)
        out = np.asarray(ns['infer_pair_probs'](model, x, x, 'cuda:0'), dtype=np.float64)
        assert not calls, 'both passes must be scored'
        res[case] = (p.astype(np.float64), q.astype(np.float64), out)
        if extra:
            assert np.array_equal(ns['_PAIR_LAST'].pq[0], p) and np.array_equal(ns['_PAIR_LAST'].pq[1], q)
    p, q, out = res['id']
    perm = [int(round((2 * out[j] - p[j] - 0.5) * 100)) for j in range(12)]
    assert np.allclose(out, 0.5 * (p + q[perm]), atol=1e-6), 'not a re-indexed mean'
    p, q, out = res['space']
    lg = lambda v: np.log(v / (1 - v))
    space = ('probability' if np.allclose(out, 0.5 * (p + q), atol=1e-6) else
             'logit' if np.allclose(out, 1 / (1 + np.exp(-0.5 * (lg(p) + lg(q)))), atol=1e-6) else 'other')
    return perm, space


perm384, space384 = derive(D384, (94, 3, 320, 320), False)
perm224, space224 = derive(D224, (94, 3, 224, 224), True)
assert perm384 == perm224 == list(range(12)), (perm384, perm224)
assert space384 == space224 == 'probability', (space384, space224)
print('R224_MIRROR_VERIFIED', json.dumps({'perm_r384': perm384, 'perm_r224': perm224, 'perm_named': {LAB[j]: LAB[perm224[j]] for j in range(12)},
                                          'combine': '0.5 * (p + q), ' + space224 + ' space (both)', 'identical_source': same,
                                          'mirror_tri_functional_equal': True, 'windows_checked': len(cs)}), flush=True)
