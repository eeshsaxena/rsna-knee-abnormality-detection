#!/usr/bin/env python
# build_v224rm.py (2026-10-14, Claude subagent): V224-RM = a V224-family notebook + nartaa's r224 checkpoint scored with the
# r384 serving code's anatomical mirror TTA (RESEARCH_960 lever 4). ONE change vs its base. No em dashes.
#
# The change (identical on every base): 2 inserted cells, every base cell kept byte-identical and in order.
#   RM     right after varshitha's r224 %%writefile cell (her cell 9), before her stage 2b. It
#          1. asserts /tmp/raptor224_run.py == her cell 9 and /tmp/raptor_run.py == her cell 3 (r384), sha256 of the bytes;
#          2. applies 6 literal patches to the r224 script (each anchor exactly once), porting the r384 mirror TTA:
#             SLOT_SIZES / SAGITTAL_SLOTS / _slot_index / _mirror_tri copied verbatim from the r384 script; infer_pair_probs
#             copied with only the window-shape assert changed (94,3,320,320) -> (94,3,224,224) (the two passes are also kept
#             for a raw dump); gpu_windows(..., mirror=False) with r384's two mirror lines (the mirror acts on the uint8
#             triplets BEFORE the 0.8 crop, the resize and the normalisation, as r384 mirrors before its 320 crop); r384's
#             consumer lines (plain + mirrored windows -> infer_pair_probs). Plus raw dumps (raptor224_raw.csv = the mirror-
#             averaged probabilities stage 3 ranks; _plain / _mirror = each pass) + raptor224_tta_receipt.json; more than 3 %
#             failed studies -> exit 4 (her stage 2b: ok=False -> blend without r224 -> the V224 canary blanks Fracture);
#          3. runs rm_verify.py (subprocess, CPU): the class permutation and the combine of BOTH infer_pair_probs (r384's from
#             /tmp/raptor_run.py and the patched r224 one) are DERIVED from stub passes and must be the identity map (r384
#             re-indexes nothing: a true L/R mirror keeps every Medial / Lateral label; R7 is different because it works in a
#             side-canonical frame, so its flip swaps them) and 0.5 * (p + q) in probability space; SLOT_SIZES / SAGITTAL_SLOTS
#             / _slot_index / _mirror_tri / LAB byte-identical to r384's, _mirror_tri functionally equal; then installs the
#             patched script (sha256 asserted) and prints R224_MIRROR_PATCHED. Any failure raises (never a silent plain r224).
#   RM_GUARD  last cell: r224 used but no valid mirror receipt, or the r224 csv stage 3 ranked is not the ranks of the
#          mirror-averaged raw -> Fracture = 0.5 for every study (the same visible canary column as "r224 not used").
#
# Bases (parametric in w_f, one flag, same gates):
#   --wf 0           V224-exact  session_2026-10-14/v224/kernel_v224 (sha 4259ec97; submitted 10-14, ref 57171955)  -> V224-RM
#   --wf 0 --flip    V224-FLIP   session_2026-10-14/v224/kernel_v224flip (sha 73a9d562; always-on R7 flip)          -> V224-FLIP-RM
#   --wf W           V224-F{W}   the sibling builder session_2026-10-15/v224f/build_v224f.py is RUN into base/ (fleet stage
#                    at the end, start budget 5.5 h, Synovitis fleet canary); asserted byte-identical to v224f/kernel_v224f{W}
#                    when that exists (the kernel the sibling commit-tested)                                     -> V224-F{W}-RM
#   --wf W --flip    V224-FF{W}  the sibling builder v224ff/build_v224ff.py RUN into base/ (time-aware R7 flip); asserted
#                    byte-identical to v224ff/kernel_v224ff{W} when that exists                                -> V224-FF{W}-RM
#   --base NB        any other V224-family notebook that holds her 13 cells verbatim.
# Her 13 cells must appear verbatim and in order in the base; no cell after her stage 3 may re-assign a name RM_GUARD reads.
#   usage: python build_v224rm.py --wf 0.20 [--flip] [--slug rsna-sub-v224rm] [--out DIR] [--base NB.ipynb]
#          [--start-budget-h H --wall-h H --flip-budget-h H: passed through to the sibling builders, only when given]
import os, sys, ast, json, copy, hashlib, argparse, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
V224 = os.path.join(ROOT, 'session_2026-10-14', 'v224')
HER_NB, HER_SHA = os.path.join(V224, 'src', 'varshitha_v224.ipynb'), '6406e92300011e48f0534ce74a91f5b7a0629b3771168ab7275f5584f884f36d'
EXACT_NB, EXACT_SHA = (os.path.join(V224, 'kernel_v224', 'rsna_sub_v224.ipynb'),
                       '4259ec9774167bb280b794574b73e1ba73203e8a1995e97116388011c4226877')
EXACT_META = os.path.join(V224, 'kernel_v224', 'kernel-metadata.json')
FLIP_NB, FLIP_SHA = (os.path.join(V224, 'kernel_v224flip', 'rsna_sub_v224flip.ipynb'),
                     '73a9d5620d0a637ed171ecec13bcf243bd0af67992b5b4cb8ad7f35bf72b1b1b')
V224F_BUILDER = os.path.join(ROOT, 'session_2026-10-15', 'v224f', 'build_v224f.py')      # sibling: V224-F (fleet at the end)
V224FF_BUILDER = os.path.join(ROOT, 'session_2026-10-15', 'v224ff', 'build_v224ff.py')   # sibling: V224-F + time-aware R7 flip
GUARD_READS = ('r224', 'R224_OK', 'labels', 'test_ids', 'n', 'WORK')   # her stage-3 globals RM_GUARD reads


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def sha_f(p):
    return sha_b(open(p, 'rb').read())


def dumps(nb):
    # her file's exact serialisation (build_v224.py verified: compact separators, ensure_ascii False == her bytes)
    return json.dumps(nb, ensure_ascii=False, separators=(',', ':'))


def src(cell):
    s = cell['source']
    return ''.join(s) if isinstance(s, list) else s


def code_cell(text):
    return {'cell_type': 'code', 'metadata': {}, 'source': text, 'outputs': [], 'execution_count': None}


def body(cell_text):
    # what IPython's %%writefile writes: the cell after its first line (MEASURED: the v224diag kernel printed the sha of
    # this text after its two replacements, 507f881b9995f30c, and the offline recompute of that sha matches)
    first, rest = cell_text.split('\n', 1)
    assert first.startswith('%%writefile '), first
    return rest


# ---------------------------------------------------------------------------------------------------------------------
# her notebook + the r384 / r224 sources
assert sha_f(HER_NB) == HER_SHA, 'varshitha notebook copy changed'
HER_RAW = open(HER_NB, 'rb').read()
HER = json.loads(HER_RAW.decode('utf-8'))
assert dumps(HER).encode('utf-8') == HER_RAW
HC = HER['cells']
assert len(HC) == 13
C3, C9, C12 = src(HC[3]), src(HC[9]), src(HC[12])
assert C3.startswith('%%writefile /tmp/raptor_run.py\n') and C9.startswith('%%writefile /tmp/raptor224_run.py\n')
assert src(HC[4]).startswith('# Stage 1: public Raptor model') and src(HC[6]).startswith("ARM = 'E_lrprior'\n")
assert src(HC[10]).startswith('# Stage 2b [s5]') and C12.startswith('# Stage 3: per-finding rank blend')
R384, R224 = body(C3), body(C9)
R384_SHA, R224_SHA = sha_b(R384.encode('utf-8')), sha_b(R224.encode('utf-8'))
assert "'7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'" in R384
assert "'3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6'" in R224

# ---- the r384 pieces, taken from her cell 3 text (so the port is verbatim by construction) -------------------------
_a = R384.index('SLOT_SIZES = [26, 22, 18, 12, 18]\n')
_b = R384.index('def infer_pair_probs(model, plain, mirrored, device):\n')
_c = R384.index('def eval_windows(vol, mask, k, res, norm=NORM, mirror=False):\n')
assert _a < _b < _c and R384.count('SLOT_SIZES = [26, 22, 18, 12, 18]\n') == 1
MIRROR_DEFS = R384[_a:_b]                      # SLOT_SIZES, SAGITTAL_SLOTS, _slot_index, _mirror_tri (verbatim)
PAIR384 = R384[_b:_c]                          # r384 infer_pair_probs (verbatim)
assert 'def _mirror_tri(tri, cs, D):' in MIRROR_DEFS and MIRROR_DEFS.endswith('\n\n')
assert PAIR384 == ('def infer_pair_probs(model, plain, mirrored, device):\n'
                   '    assert plain.shape==mirrored.shape==(94,3,320,320)\n'
                   '    assert all(not m.training for m in model.modules()), "TTA must not update BN or enable dropout"\n'
                   '    p = infer_probs(model, plain, device)\n'
                   '    q = infer_probs(model, mirrored, device)\n'
                   '    return 0.5 * (p + q)\n\n'), repr(PAIR384)
TRI_LINE = '    tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(dev)   # (k,3,IMG,IMG) uint8\n'
MIRROR_LINES = '    if mirror:\n        tri = _mirror_tri(tri, cs, D)\n'
assert R384.count(TRI_LINE + MIRROR_LINES) == 1, 'r384 gpu_windows mirror lines moved'
R384_CONSUMER = ('                    xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM) if xla is not None else gpu_windows(vol, mask, K_EVAL, res, d)\n'
                 '                    assert xla is None, "Anatomical L/R TTA requires CUDA/fp16"\n'
                 '                    xm = gpu_windows(vol, mask, K_EVAL, res, d, norm=NORM, mirror=True)\n'
                 '                    arm_probs[a][i] = infer_pair_probs(model, xw, xm, d)\n'
                 '                    del xm\n')
assert R384.count(R384_CONSUMER) == 1, 'r384 consumer lines moved'
assert R384.count('def gpu_windows(vol, mask, k, res, dev, norm=NORM, mirror=False):\n') == 1

PAIR224 = ('_PAIR_LAST = threading.local()   # [V224-RM] per consumer thread: the last (plain, mirrored) pass and their seconds\n'
           + PAIR384.replace('    assert plain.shape==mirrored.shape==(94,3,320,320)\n',
                             '    assert plain.shape==mirrored.shape==(94,3,224,224)\n')
                    .replace('    p = infer_probs(model, plain, device)\n    q = infer_probs(model, mirrored, device)\n',
                             '    _t0 = time.time()\n'
                             '    p = infer_probs(model, plain, device)\n'
                             '    _t1 = time.time()\n'
                             '    q = infer_probs(model, mirrored, device)\n'
                             '    _PAIR_LAST.pq, _PAIR_LAST.t = (p, q), (_t1 - _t0, time.time() - _t1)   # [V224-RM] for the raw dump\n'))
assert PAIR224.count('(94,3,224,224)') == 1 and PAIR224.count('_PAIR_LAST.pq') == 1 and PAIR224.endswith('    return 0.5 * (p + q)\n\n')

# ---- the 6 patches of her r224 script (old -> new; every old must occur exactly once) ---------------------------------
P_EVAL = 'def eval_windows(vol, mask, k, res, norm=NORM):\n'
P_GPU = 'def gpu_windows(vol, mask, k, res, dev, norm=NORM):\n'
P_ALLOC = '    arm_probs = [np.full((N, len(LAB)), 0.5, np.float32) for _ in range(A)]\n'
P_CONS_OLD = ('                    xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM) if xla is not None else gpu_windows(vol, mask, K_EVAL, res, d)\n'
              '                    if xla is not None:\n'
              '                        with torch.no_grad():\n'
              '                            o = torch.sigmoid(model(xw.unsqueeze(0).to(d)).float()); xla.mark_step()\n'
              '                            arm_probs[a][i] = o[0].cpu().numpy()\n'
              '                    else:\n'
              '                        arm_probs[a][i] = infer_probs(model, xw, d)\n')
P_CONS_NEW = ('                    _rm_t = time.time()\n'
              + R384_CONSUMER.replace('                    del xm\n', '')
              + '                    del xm\n'
              '                    arm_plain[a][i], arm_mirror[a][i] = _PAIR_LAST.pq   # [V224-RM] each pass alone (raw dump only)\n'
              '                    arm_done[a][i] = True\n'
              '                    with lock:\n'
              '                        _rm_times.append((time.time() - _rm_t,) + tuple(_PAIR_LAST.t))\n')
P_DONE = '    print(f"[gpu] done: {done_n[0]}/{N} studies on {len(devs)} device(s) | {time.time()-t0:.0f}s", flush=True)\n'
RM_ALLOC = ('    arm_plain = [np.full((N, len(LAB)), 0.5, np.float32) for _ in range(A)]    # [V224-RM] plain pass alone\n'
            '    arm_mirror = [np.full((N, len(LAB)), 0.5, np.float32) for _ in range(A)]   # [V224-RM] mirrored pass alone\n'
            '    arm_done = [np.zeros(N, dtype=bool) for _ in range(A)]                      # [V224-RM] scored by both passes\n'
            '    _rm_times = []                                                               # [V224-RM] per study: total, plain, mirror s\n')
RM_DUMP = r'''    # [V224-RM] raw dumps + mirror-TTA receipt. More than 3 % failed studies (her r384 stage-3 rule) -> exit 4: stage 2b then
    # reports rc=4 ok=False, stage 3 blends without r224 and the canary blanks Fracture (never a silent 0.5-filled r224 under
    # the RM name). A failed study keeps 0.5 in every column, exactly as in her r224 script.
    _ok = arm_done[0]
    _nf = int((~_ok).sum())
    for _nm, _arr in (("raptor224_raw.csv", arm_probs[0]), ("raptor224_raw_plain.csv", arm_plain[0]),
                      ("raptor224_raw_mirror.csv", arm_mirror[0])):
        _df = pd.DataFrame(_arr.astype(np.float32), columns=LAB)
        _df.insert(0, "StudyInstanceUID", test_ids)
        _df.to_csv("/kaggle/working/" + _nm, index=False, float_format="%.9g")
    _tt = np.asarray(_rm_times, dtype=np.float64).reshape(-1, 3)
    _d = np.abs(arm_plain[0][_ok].astype(np.float64) - arm_mirror[0][_ok].astype(np.float64))
    _rec = {"mirror": True, "perm": list(range(len(LAB))), "labels": LAB,
            "transform": "r384 _mirror_tri on the uint8 triplets (sagittal slots 0,1: reverse the 3 slice channels; coronal / axial: flip the width), before the 0.8 crop, the resize and the normalisation",
            "combine": "0.5 * (p + q), probability space (r384 infer_pair_probs)",
            "n": int(N), "n_pairs": int(_ok.sum()), "n_fail": _nf, "devices": len(devs),
            "per_study_s": {"total": float(_tt[:, 0].mean()) if len(_tt) else None, "plain": float(_tt[:, 1].mean()) if len(_tt) else None,
                            "mirror": float(_tt[:, 2].mean()) if len(_tt) else None},
            "per_study_s_list": [[round(float(v), 4) for v in r] for r in _tt[:64]],
            "sum_s": {"total": float(_tt[:, 0].sum()), "plain": float(_tt[:, 1].sum()), "mirror": float(_tt[:, 2].sum())},
            "plain_vs_mirror": {"max_abs": float(_d.max()) if _d.size else None, "mean_abs": float(_d.mean()) if _d.size else None,
                                "per_label_max": {c: float(_d[:, j].max()) for j, c in enumerate(LAB)} if _d.size else None},
            "seconds": round(time.time() - t0, 1)}
    with open("/kaggle/working/raptor224_tta_receipt.json", "w") as _fh:
        json.dump(_rec, _fh, indent=1)
    print("R224_MIRROR_TTA", json.dumps({k: _rec[k] for k in ("n", "n_pairs", "n_fail", "per_study_s", "sum_s", "seconds")}), flush=True)
    if _nf > 0.03 * N:
        print(f"R224_MIRROR_FAILED {_nf} of {N} studies failed in the r224 mirror stage (more than 3 %): exit 4", flush=True)
        sys.exit(4)
'''
PATCHES = [
    (P_EVAL, MIRROR_DEFS + PAIR224 + P_EVAL),
    (P_GPU, 'def gpu_windows(vol, mask, k, res, dev, norm=NORM, mirror=False):\n'),
    (TRI_LINE, TRI_LINE + MIRROR_LINES),
    (P_ALLOC, P_ALLOC + RM_ALLOC),
    (P_CONS_OLD, P_CONS_NEW),
    (P_DONE, P_DONE + RM_DUMP),
]


def apply_patches(text):
    for old, new in PATCHES:
        assert text.count(old) == 1, ('patch anchor not unique', old[:80])
        text = text.replace(old, new)
    return text


R224_RM = apply_patches(R224)
R224_RM_SHA = sha_b(R224_RM.encode('utf-8'))
compile(R224_RM, '/tmp/raptor224_run.py', 'exec')
assert R224_RM.count(TRI_LINE + MIRROR_LINES) == 1 and R224_RM.count(R384_CONSUMER.replace('                    del xm\n', '')) == 1

VERIFY = r'''# rm_verify.py (V224-RM): is the r224 mirror TTA the r384 serving code's mirror TTA? Prints R224_MIRROR_VERIFIED <json>.
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
'''

RM_CELL = r'''# [eeshsaxena V224-RM; the ONE change vs its base] nartaa's r224 checkpoint (.945, 0.8 crop) scored with the r384 serving
# code's anatomical mirror TTA (RESEARCH_960 lever 4). Her r224 script above is patched in place by 6 literal edits (each
# anchor exactly once): r384's SLOT_SIZES / SAGITTAL_SLOTS / _slot_index / _mirror_tri copied verbatim, r384's
# infer_pair_probs with the 224 window shape, gpu_windows(..., mirror) mirroring the uint8 triplets before the 0.8 crop /
# resize / normalisation (as r384 does before its 320 crop), r384's consumer lines, raw dumps + raptor224_tta_receipt.json,
# more than 3 % failed studies -> exit 4 (canary). rm_verify.py derives the class map and the combine of BOTH
# infer_pair_probs (r384 from /tmp/raptor_run.py, written by the r384 cell above) on stub passes: identity map (r384
# re-indexes nothing: a true L/R mirror keeps every Medial / Lateral label) and 0.5 * (p + q) in probability space.
# Any mismatch raises here (the kernel stops; never a silent plain r224 under the RM name). Nothing is fitted.
import os, sys, json, hashlib, subprocess
_RM_R224_HER_SHA = '__R224_SHA__'
_RM_R384_HER_SHA = '__R384_SHA__'
_RM_R224_RM_SHA = '__R224_RM_SHA__'
_RM_PATCHES = __PATCHES__
_RM_VERIFY = __VERIFY__
_rm_sha = lambda t: hashlib.sha256(t.encode('utf-8')).hexdigest()
_rm_txt = open('/tmp/raptor224_run.py', encoding='utf-8').read()
assert _rm_sha(_rm_txt) == _RM_R224_HER_SHA, 'the r224 script is not the public cell (sha256)'
assert _rm_sha(open('/tmp/raptor_run.py', encoding='utf-8').read()) == _RM_R384_HER_SHA, 'the r384 script is not the public cell (sha256)'
for _rm_old, _rm_new in _RM_PATCHES:
    assert _rm_txt.count(_rm_old) == 1, ('patch anchor not unique', _rm_old[:80])
    _rm_txt = _rm_txt.replace(_rm_old, _rm_new)
assert _rm_sha(_rm_txt) == _RM_R224_RM_SHA, 'patched r224 script differs from the build'
compile(_rm_txt, '/tmp/raptor224_run.py', 'exec')
with open('/tmp/raptor224_rm_candidate.py', 'w', encoding='utf-8', newline='') as _rm_f:
    _rm_f.write(_rm_txt)
with open('/tmp/rm_verify.py', 'w', encoding='utf-8', newline='') as _rm_f:
    _rm_f.write(_RM_VERIFY)
_rm_v = subprocess.run([sys.executable, '/tmp/rm_verify.py', '/tmp/raptor_run.py', '/tmp/raptor224_rm_candidate.py'],
                       capture_output=True, text=True, timeout=900, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
print(_rm_v.stdout.strip(), flush=True)
if _rm_v.returncode != 0 or 'R224_MIRROR_VERIFIED' not in _rm_v.stdout:
    print(_rm_v.stderr[-3000:], flush=True)
    raise RuntimeError('V224-RM: the r224 mirror port does not match the r384 serving code (rm_verify rc %d)' % _rm_v.returncode)
os.replace('/tmp/raptor224_rm_candidate.py', '/tmp/raptor224_run.py')   # stage 2b below runs the patched script
assert _rm_sha(open('/tmp/raptor224_run.py', encoding='utf-8').read()) == _RM_R224_RM_SHA
RM_ON = True
print('R224_MIRROR_PATCHED', json.dumps({'r224_her_sha': _RM_R224_HER_SHA[:16], 'r224_rm_sha': _RM_R224_RM_SHA[:16],
                                         'patches': len(_RM_PATCHES)}), flush=True)
'''

RM_GUARD = r'''# [eeshsaxena V224-RM GUARD; not in the public notebook] The canary above blanks Fracture when r224 was not used at all.
# This cell covers the one new silent path of V224-RM: r224 used, but not the mirror-TTA r224 (mirror receipt missing or
# inconsistent, or the r224 csv stage 3 ranked is not the ranks of the mirror-averaged raw). Then Fracture = 0.5 for every
# study (the same visible ~ -.03 display). Clean path: nothing changes. It edits submission.csv as text, so every other
# cell keeps its bytes.
import json as _gj, os as _go
import numpy as _gnp, pandas as _gpd
_GW = '/kaggle/working'
_g = {'r224_used': r224 is not None, 'mirror_ok': None, 'why': None}
if r224 is not None:
    try:
        _gt = _gj.load(open(_GW + '/raptor224_tta_receipt.json'))
        assert _gt.get('mirror') is True and _gt.get('perm') == list(range(len(labels))) and _gt.get('labels') == labels, 'receipt'
        assert _gt.get('n') == n and _gt.get('n_pairs', -1) + _gt.get('n_fail', -1) == n and _gt['n_fail'] <= 0.03 * n, 'counts'
        _graw = _gpd.read_csv(_GW + '/raptor224_raw.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]
        assert _gnp.isfinite(_graw.values).all(), 'raw not finite'
        _grho = {}
        for _gc in labels:
            _ga, _gb = _gpd.Series(_graw[_gc].values), _gpd.Series(r224[_gc].values)
            if n >= 3 and _ga.nunique() > 1 and _gb.nunique() > 1:
                _grho[_gc] = round(float(_ga.corr(_gb, method='spearman')), 4)
            elif _ga.nunique() != _gb.nunique() and n >= 3:
                _grho[_gc] = -1.0
        assert not _grho or min(_grho.values()) >= 0.99, ('r224 csv is not the ranks of the mirror-averaged raw', _grho)
        _g.update(mirror_ok=True, n_pairs=_gt['n_pairs'], n_fail=_gt['n_fail'], seconds=_gt.get('seconds'),
                  per_study_s=_gt.get('per_study_s'), plain_vs_mirror_max=(_gt.get('plain_vs_mirror') or {}).get('max_abs'),
                  spearman_raw_vs_sub_min=min(_grho.values()) if _grho else None)
    except Exception as _ge:
        _g.update(mirror_ok=False, why=repr(_ge)[:400])
_gcr = {}
try:
    _gcr = _gj.load(open(_GW + '/canary_receipt.json'))
except Exception as _ge:
    _gcr = {'canary_receipt_error': repr(_ge)[:200], 'fired': []}
if r224 is not None and not _g['mirror_ok']:
    _gs = _gpd.read_csv(_GW + '/submission.csv', dtype=str, keep_default_na=False)
    assert _gs['StudyInstanceUID'].tolist() == test_ids and 'Fracture' in _gs.columns
    _gs['Fracture'] = '0.5'
    _gs.to_csv(_GW + '/submission.csv.tmp', index=False)
    _go.replace(_GW + '/submission.csv.tmp', _GW + '/submission.csv')
    _gchk = _gpd.read_csv(_GW + '/submission.csv', dtype={'StudyInstanceUID': str})
    assert _gchk['StudyInstanceUID'].tolist() == test_ids and _gnp.isfinite(_gchk[labels].values).all()
    _gcr.setdefault('fired', []).append('r224 mirror->Fracture')
    print('RM_GUARD_FIRED r224 used without a valid mirror receipt: Fracture set to 0.5 for all %d rows' % n, _gj.dumps(_g), flush=True)
else:
    print('RM_GUARD_CLEAN', _gj.dumps(_g), flush=True)
_gcr['rm_guard'] = _g
_gcr['rm_on'] = bool(globals().get('RM_ON', False))
_gcr['w_f_global'] = globals().get('W_F')
(WORK / 'canary_receipt.json').write_text(_gj.dumps(_gcr, indent=1))
print('V224RM_DONE', _gj.dumps({k: _gcr.get(k) for k in ('fired', 'r224_used', 'flip_tta_forced', 'rm_on', 'w_f_global')}), flush=True)
'''

def local_verify(r384_text, r224_text):
    with tempfile.TemporaryDirectory() as td:
        a, b, v = os.path.join(td, 'raptor_run.py'), os.path.join(td, 'raptor224_run.py'), os.path.join(td, 'rm_verify.py')
        for p, t in ((a, r384_text), (b, r224_text), (v, VERIFY)):
            open(p, 'w', encoding='utf-8', newline='').write(t)
        r = subprocess.run([sys.executable, '-E', '-P', v, a, b], capture_output=True, text=True, timeout=600)
    return r.returncode, r.stdout.strip(), r.stderr.strip()[-3000:]


def rm_cell_text():
    t = RM_CELL
    for k, v in (('__R224_SHA__', R224_SHA), ('__R384_SHA__', R384_SHA), ('__R224_RM_SHA__', R224_RM_SHA)):
        assert t.count(k) == 1
        t = t.replace(k, v)
    assert t.count('__PATCHES__') == 1 and t.count('__VERIFY__') == 1
    t = t.replace('__PATCHES__', repr(PATCHES)).replace('__VERIFY__', repr(VERIFY))
    compile(t, '<rm_cell>', 'exec')
    return t


def top_stores(text):
    # names a cell binds at module level (not inside functions / classes / lambdas / comprehensions)
    out = set()

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
            return
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.add(node.id)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                out.add(a.asname or a.name.split('.')[0])
        if isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
        for ch in ast.iter_child_nodes(node):
            visit(ch)

    for st in ast.parse(text).body:
        visit(st)
    return out


def get_base(A):
    """-> (raw bytes, kernel-metadata dict, base name, provenance dict)"""
    if A.base:
        raw = open(A.base, 'rb').read()
        meta = json.load(open(os.path.join(os.path.dirname(os.path.abspath(A.base)), 'kernel-metadata.json'), encoding='utf-8'))
        return raw, meta, 'base:' + os.path.basename(A.base), {'base_path': A.base, 'base_sha256': sha_b(raw)}
    if A.wf == 0:
        p, h, nm = (FLIP_NB, FLIP_SHA, 'V224-FLIP') if A.flip else (EXACT_NB, EXACT_SHA, 'V224-exact')
        assert sha_f(p) == h, (nm, 'build changed')
        meta = json.load(open(os.path.join(os.path.dirname(p), 'kernel-metadata.json'), encoding='utf-8'))
        return open(p, 'rb').read(), meta, nm, {'base_path': p, 'base_sha256': h}
    wtag = '%02d' % int(round(A.wf * 100))
    assert abs(A.wf * 100 - round(A.wf * 100)) < 1e-9, A.wf
    tag = ('v224ff' if A.flip else 'v224f') + wtag
    builder = V224FF_BUILDER if A.flip else V224F_BUILDER
    bout = os.path.join(HERE, 'base')
    cmd = [sys.executable, builder, '--wf', repr(A.wf), '--out', bout]
    for k in ('start_budget_h', 'wall_h') + (('flip_budget_h',) if A.flip else ()):
        if getattr(A, k) is not None:
            cmd += ['--' + k.replace('_', '-'), repr(getattr(A, k))]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    print('BASE_BUILD', ' '.join(os.path.basename(c) if os.sep in c or '/' in c else c for c in cmd[1:]), 'rc', r.returncode)
    print('\n'.join('    | ' + ln for ln in (r.stdout.strip().splitlines()[-4:])))
    assert r.returncode == 0, r.stderr[-3000:]
    nbp = os.path.join(bout, 'kernel_' + tag, 'rsna_sub_%s.ipynb' % tag)
    raw = open(nbp, 'rb').read()
    sib = os.path.join(os.path.dirname(builder), 'kernel_' + tag, 'rsna_sub_%s.ipynb' % tag)
    same = None
    if os.path.isfile(sib):
        same = open(sib, 'rb').read() == raw
        assert same, ('the fresh base differs from the sibling build (the kernel it commit-tested)', sib)
    meta = json.load(open(os.path.join(bout, 'kernel_' + tag, 'kernel-metadata.json'), encoding='utf-8'))
    nm = ('V224-FF' if A.flip else 'V224-F') + wtag
    return raw, meta, nm, {'base_path': nbp, 'base_sha256': sha_b(raw), 'base_builder': builder, 'sibling_kernel': sib,
                           'identical_to_sibling_kernel': same, 'sibling_slug': 'eeshsaxena/rsna-sub-' + tag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wf', type=float, required=True, help='fleet weight w_f; 0 = no fleet (V224-exact / V224-FLIP base)')
    ap.add_argument('--flip', action='store_true', help='flip base: V224-FLIP (wf 0) or the sibling V224-FF{w} (wf > 0)')
    ap.add_argument('--slug', default='rsna-sub-v224rm')
    ap.add_argument('--base', default=None, help='apply RM to this V224-family notebook instead')
    ap.add_argument('--start-budget-h', type=float, default=None)
    ap.add_argument('--wall-h', type=float, default=None)
    ap.add_argument('--flip-budget-h', type=float, default=None)
    ap.add_argument('--out', default=None)
    A = ap.parse_args()
    assert 0.0 <= A.wf < 1.0
    assert not (A.base and (A.wf > 0 or A.flip)), '--base carries its own fleet / flip'

    rc, vout, verr = local_verify(R384, R224_RM)
    print('LOCAL_VERIFY rc', rc, vout, verr[-500:] if rc else '')
    assert rc == 0 and 'R224_MIRROR_VERIFIED' in vout, 'local rm_verify failed'
    # negative controls: a swapped Medial / Lateral map, a logit-space combine and swapped mirror axes must FAIL the verify
    tail = '    return 0.5 * (p + q)\n\ndef eval_windows(vol, mask, k, res, norm=NORM):'
    bads = {'swap M/L': R224_RM.replace(tail, tail.replace('(p + q)', '(p + q[[0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 10, 11]])')),
            'logit combine': R224_RM.replace(tail, tail.replace('return 0.5 * (p + q)', 'return 1 / (1 + np.exp(-0.5 * ('
                                                                'np.log(p / (1 - p)) + np.log(q / (1 - q)))))')),
            'axes swapped': R224_RM.replace('    return torch.where(sag, tri.flip(1), tri.flip(-1))',
                                            '    return torch.where(sag, tri.flip(-1), tri.flip(1))')}
    for nm, bad in bads.items():
        assert bad != R224_RM, nm
        assert local_verify(R384, bad)[0] != 0, ('negative control passed the verify', nm)
    print('LOCAL_VERIFY negative controls FAIL as they should:', ', '.join(bads))

    raw, meta, base_name, prov = get_base(A)
    base = json.loads(raw.decode('utf-8'))
    assert dumps(base).encode('utf-8') == raw, 'base is not in her serialisation'
    bc = base['cells']
    idx, j = [], 0
    for i, c in enumerate(bc):
        if j < 13 and c['cell_type'] == HC[j]['cell_type'] and src(c) == src(HC[j]):
            idx.append(i); j += 1
    assert j == 13, 'base does not hold her 13 cells verbatim and in order'
    i9, i10, i12 = idx[9], idx[10], idx[12]
    assert i10 == i9 + 1, 'her stage 2b must follow her r224 writefile directly'
    for k, c in enumerate(bc):
        if k not in (i9, i10):
            assert '/tmp/raptor224_run.py' not in src(c), ('another base cell touches the r224 script', k)
    for k in range(i12 + 1, len(bc)):
        if bc[k]['cell_type'] == 'code':
            clash = top_stores(src(bc[k])) & set(GUARD_READS)
            assert not clash, ('a base cell after her stage 3 re-binds what RM_GUARD reads', k, clash)
    assert not any(src(c).startswith('# [eeshsaxena V224-RM') for c in bc), 'base already has RM'
    rm, guard = code_cell(rm_cell_text()), code_cell(RM_GUARD)
    compile(RM_GUARD, '<guard>', 'exec')
    cells = [copy.deepcopy(c) for c in bc[:i9 + 1]] + [rm] + [copy.deepcopy(c) for c in bc[i9 + 1:]] + [guard]
    ins = {i9 + 1, len(cells) - 1}
    nb = copy.deepcopy(base); nb['cells'] = cells
    back = copy.deepcopy(nb); back['cells'] = [c for k, c in enumerate(cells) if k not in ins]
    assert dumps(back).encode('utf-8') == raw, 'base cells are not byte-identical'
    her_back = [c for k, c in enumerate(bc) if k in idx]
    assert [src(c) for c in her_back] == [src(c) for c in HC]

    name = base_name.replace('V224-exact', 'V224') + '-RM'
    code_file = A.slug.replace('-', '_') + '.ipynb'
    out = A.out or os.path.join(HERE, 'kernel_' + A.slug.replace('rsna-sub-', ''))
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, code_file)
    with open(p, 'w', encoding='utf-8', newline='') as f:
        f.write(dumps(nb))
    meta = {k: v for k, v in meta.items() if k not in ('id_no', 'keywords')}
    meta.update(id='eeshsaxena/' + A.slug, title=A.slug.replace('-', ' '), code_file=code_file, is_private=True,
                enable_gpu=True, enable_tpu=False, enable_internet=False)
    with open(os.path.join(out, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    rec = dict(name=name, base=base_name, wf=A.wf, flip=A.flip, slug=A.slug, notebook=p, notebook_sha256=sha_f(p),
               cells=len(cells), inserted=sorted(ins), datasets=meta['dataset_sources'], r224_her_sha=R224_SHA,
               r384_her_sha=R384_SHA, r224_rm_sha=R224_RM_SHA, local_verify=vout, **prov)
    # review copies live next to the builder, so the kernel dir holds only the notebook + kernel-metadata.json (what is pushed)
    tag = A.slug.replace('rsna-sub-', '')
    with open(os.path.join(HERE, 'build_receipt_%s.json' % tag), 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=1)
    with open(os.path.join(HERE, 'raptor224_rm.py'), 'w', encoding='utf-8', newline='') as f:
        f.write(R224_RM)              # review copy only (the kernel builds it from her cell + the patches)
    with open(os.path.join(HERE, 'rm_verify.py'), 'w', encoding='utf-8', newline='') as f:
        f.write(VERIFY)
    with open(os.path.join(HERE, 'celldiff_%s.txt' % tag), 'w', encoding='utf-8', newline='\n') as f:
        f.write('# %s (%s): base %s sha %s, %d base cells byte-identical; inserted cells %s (full text below)\n'
                % (name, A.slug, base_name, prov['base_sha256'], len(bc), sorted(ins)))
        for k in sorted(ins):
            f.write('\n===== inserted cell %d =====\n%s\n' % (k, src(cells[k])))
    print('BUILT', json.dumps({k: rec[k] for k in ('name', 'slug', 'notebook_sha256', 'cells', 'inserted', 'base_sha256',
                                                   'datasets', 'r224_rm_sha')}))


if __name__ == '__main__':
    main()
