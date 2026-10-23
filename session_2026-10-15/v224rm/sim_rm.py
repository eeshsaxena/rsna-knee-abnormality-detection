#!/usr/bin/env python
# sim_rm.py (2026-10-14, Claude subagent): offline CPU simulations of V224-RM before the one commit. No GPU, no weights, no
# competition data: synthetic volumes and a tiny stand-in model. Prints SIM_RM PASS/FAIL n/m.
#   S1  the RM cell itself (paths /tmp -> a temp dir): sha asserts, the 6 patches, rm_verify in a subprocess, install;
#       negative: a 1-byte change of the r224 script must make the cell raise.
#   S2  her r224 script vs the patched one, executed end to end (main()) on CPU with a stand-in model / reader / pool:
#       plain pass of the patched run == her run (raw identical, ranks identical); avg == 0.5 (plain + mirror) exactly;
#       raptor224_sub == rankpct(avg); mirror != plain; receipt counts; > 3 % failed studies -> exit 4 and no sub csv;
#       <= 3 % failed -> sub written, the failed row stays 0.5.
#   S3  RM_GUARD on fixtures (her stage-3 globals): clean -> bytes unchanged; missing receipt, wrong perm, raw
#       inconsistent with the r224 csv -> Fracture 0.5 only, the other 11 columns byte-equal; r224 unused -> no-op.
import os, sys, json, shutil, tempfile, types, hashlib, contextlib, io, multiprocessing
import numpy as np, pandas as pd, torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_v224rm as B   # module level only: shas, patches, cell texts (main() is not run)

R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print(f'  {"OK  " if ok else "FAIL"} {name} {info}', flush=True)


# ---------------------------------------------------------------------------------------------------- S1 RM cell offline
print('S1 RM cell offline', flush=True)
T = tempfile.mkdtemp(prefix='rm_s1_').replace('\\', '/')
open(T + '/raptor_run.py', 'w', encoding='utf-8', newline='').write(B.R384)
open(T + '/raptor224_run.py', 'w', encoding='utf-8', newline='').write(B.R224)
cell = B.rm_cell_text().replace("'/tmp/", "'" + T + '/')
assert "'/tmp/" not in cell
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(cell, '<rm_cell>', 'exec'), {'__name__': 'rm_s1'})
out = buf.getvalue()
chk('RM cell: R224_MIRROR_VERIFIED + R224_MIRROR_PATCHED printed', 'R224_MIRROR_VERIFIED' in out and 'R224_MIRROR_PATCHED' in out,
    [ln[:120] for ln in out.splitlines() if 'R224_MIRROR_PATCHED' in ln])
chk('RM cell: /tmp/raptor224_run.py now == the patched build', open(T + '/raptor224_run.py', encoding='utf-8', newline='').read() == B.R224_RM)
open(T + '/raptor224_run.py', 'w', encoding='utf-8', newline='').write(B.R224.replace('K_EVAL = 94', 'K_EVAL = 95', 1))
try:
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(cell, '<rm_cell>', 'exec'), {'__name__': 'rm_s1'})
    chk('RM cell raises on a changed r224 script', False)
except AssertionError as e:
    chk('RM cell raises on a changed r224 script', 'not the public cell' in str(e), str(e)[:80])
shutil.rmtree(T, ignore_errors=True)

# ---------------------------------------------------------------------------------------------------- S2 end-to-end CPU
print('S2 her r224 vs patched r224, main() on CPU', flush=True)
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
       "Baker's", "Contusion", "Fracture"]


class Tiny(nn.Module):
    # stand-in for the CoAtNet MIL head: input (1, 94, 3, 224, 224) -> 12 logits; NOT mirror invariant
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.W = nn.Parameter(torch.randn(3, 224, 224, generator=g), requires_grad=False)
        self.M = nn.Parameter(torch.randn(94, 12, generator=g) * 3, requires_grad=False)

    def forward(self, x):
        assert x.shape[1:] == (94, 3, 224, 224), x.shape
        f = (x * self.W).mean(dim=(-1, -2, -3))          # (1, 94)
        return f @ self.M                                # (1, 12)


class FakePool:
    def __init__(self, *a, **k):
        if k.get('initializer'):
            k['initializer']()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def imap_unordered(self, f, it, chunksize=1):
        return map(f, it)


class FakeCtx:
    Pool = FakePool


def make_root(T, ids):
    os.makedirs(T + '/data/test_series', exist_ok=True)
    pd.DataFrame({'StudyInstanceUID': ids}).to_csv(T + '/data/test.csv', index=False)
    pd.DataFrame({'StudyInstanceUID': ids, 'SeriesInstanceUID': [s + '.1' for s in ids], 'Anatomical_Plane': 'Sagittal',
                  'Fluid_Sensitive': 1}).to_csv(T + '/data/test_series.csv', index=False)
    ss = pd.DataFrame(np.full((len(ids), 12), 0.5), columns=LAB); ss.insert(0, 'StudyInstanceUID', ids)
    ss.to_csv(T + '/data/sample_submission.csv', index=False)
    return T + '/data'


def vol_for(sid, bad):
    rs = np.random.RandomState(int(hashlib.md5(sid.encode()).hexdigest()[:8], 16))
    D = 95 if sid in bad else 96          # a 95-slice volume fails only in the mirror pass (_slot_index), on purpose
    base = rs.randint(0, 256, (D, 48, 48)).astype(np.float32)
    vol = np.kron(base, np.ones((8, 8), np.float32)).astype(np.uint8)          # (D, 384, 384), blocky
    vol = np.ascontiguousarray(vol[:, :, ::-1] if rs.rand() < .5 else vol)      # break any symmetry
    mask = np.ones(D, np.uint8); mask[:2] = 0
    return vol, mask


def run_script(text, T, ids, bad=()):
    os.makedirs(T + '/work', exist_ok=True)
    code = text[text.index('# %% [code]'):]                  # skip the checkpoint-hash header (no weights here)
    code = code.replace('/kaggle/working/', T + '/work/').replace("os.path.isdir('/kaggle/working')", "True")
    ns = {'__name__': 'rm_sim'}
    exec(compile(code, '<r224>', 'exec'), ns)
    root = make_root(T, ids)
    ns['find_test_root'] = lambda: root
    ns['find_weight_file'] = lambda f: 'stand-in'
    ns['load_model'] = lambda p, arch, res, d, ngpu=1: (Tiny().eval(), 224)
    ns['_pre_init'] = lambda: None
    ns['_pre_one'] = lambda sid: (sid, *vol_for(sid, set(bad)))
    gc0 = multiprocessing.get_context
    multiprocessing.get_context = lambda *a, **k: FakeCtx
    buf = io.StringIO(); rc = 0
    try:
        with contextlib.redirect_stdout(buf):
            ns['main']()
    except SystemExit as e:
        rc = e.code
    finally:
        multiprocessing.get_context = gc0
    return rc, buf.getvalue(), ns


def rd(p):
    return pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')


def rankpct(x):   # her r224 script's rankpct
    order = x.argsort(0).argsort(0).astype(np.float64)
    return order / max(1, (x.shape[0] - 1))


ids = ['1.2.3.%d' % k for k in range(8)]
TH, TM = tempfile.mkdtemp(prefix='rm_her_').replace('\\', '/'), tempfile.mkdtemp(prefix='rm_rm_').replace('\\', '/')
rc_h, log_h, ns_h = run_script(B.R224, TH, ids)
rc_m, log_m, ns_m = run_script(B.R224_RM, TM, ids)
chk('her script and the patched one both exit 0', rc_h in (0, None) and rc_m in (0, None), (rc_h, rc_m))
chk('no FALLBACK in either log', 'FALLBACK' not in log_h and 'FALLBACK' not in log_m)
sh, sm = rd(TH + '/work/raptor224_sub.csv'), rd(TM + '/work/raptor224_sub.csv')
plain, mirr, avg = (rd(TM + '/work/raptor224_raw_plain.csv').loc[ids, LAB].values.astype(np.float32),
                    rd(TM + '/work/raptor224_raw_mirror.csv').loc[ids, LAB].values.astype(np.float32),
                    rd(TM + '/work/raptor224_raw.csv').loc[ids, LAB].values.astype(np.float32))
# her raw: recompute with HER functions directly (her script dumps no raw)
her_raw = np.stack([ns_h['infer_probs'](ns_h['load_model']('', '', 224, 'cpu')[0],
                                        ns_h['gpu_windows'](*vol_for(s, set()), ns_h['K_EVAL'], 224, 'cpu'), 'cpu') for s in ids])
chk('patched plain pass == her probabilities (CPU, exact)', np.array_equal(plain, her_raw.astype(np.float32)),
    'max diff %.3g' % float(np.abs(plain - her_raw).max()))
chk('her raptor224_sub == rankpct(patched plain)', np.allclose(sh.loc[ids, LAB].values, rankpct(np.clip(plain, 0, 1)), atol=1e-7))
chk('avg raw == 0.5 * (plain + mirror) in float32, exact', np.array_equal(avg, np.float32(0.5) * (plain + mirr)))
chk('patched raptor224_sub == rankpct(avg raw)', np.allclose(sm.loc[ids, LAB].values, rankpct(np.clip(avg, 0, 1)), atol=1e-7))
d = float(np.abs(plain - mirr).max())
chk('mirror pass differs from plain', d > 1e-3, 'max |plain - mirror| %.4f' % d)
chk('mirror changes the r224 ranks somewhere (8 studies)', not np.allclose(sm.loc[ids, LAB].values, sh.loc[ids, LAB].values), '')
rec = json.load(open(TM + '/work/raptor224_tta_receipt.json'))
chk('receipt: mirror, identity perm, n 8, n_pairs 8, n_fail 0', rec['mirror'] is True and rec['perm'] == list(range(12)) and
    (rec['n'], rec['n_pairs'], rec['n_fail']) == (8, 8, 0) and rec['labels'] == LAB, {k: rec[k] for k in ('n', 'n_pairs', 'n_fail')})
chk('R224_MIRROR_TTA printed', 'R224_MIRROR_TTA' in log_m)
# mirror windows: the patched gpu_windows(mirror=True) == her pipeline applied to r384-mirrored uint8 triplets
vol, mask = vol_for(ids[0], set())
xm = ns_m['gpu_windows'](vol, mask, 94, 224, 'cpu', mirror=True)
xp = ns_m['gpu_windows'](vol, mask, 94, 224, 'cpu')
chk('patched gpu_windows(mirror=False) == her gpu_windows', torch.equal(xp, ns_h['gpu_windows'](vol, mask, 94, 224, 'cpu')))
D = vol.shape[0]
cs = np.array([max(1, min(c, D - 2)) for c in ns_m['_eval_centers'](mask, D, 94)])
tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)]))
sag = torch.from_numpy(cs < 48).view(-1, 1, 1, 1)
trim = torch.where(sag, tri.flip(1), tri.flip(-1))
L0 = torch.from_numpy(np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0)
x = ns_m['_crop'](L0[trim.long()])
x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
x = (x - ns_m['_MEAN']) / ns_m['_STD']
chk('mirrored windows == mirror(uint8 triplets) -> 0.8 crop -> resize -> normalise', torch.equal(xm, x))
shutil.rmtree(TH, ignore_errors=True); shutil.rmtree(TM, ignore_errors=True)

T2 = tempfile.mkdtemp(prefix='rm_bad_').replace('\\', '/')
rc_b, log_b, _ = run_script(B.R224_RM, T2, ids, bad=[ids[3]])          # 1 of 8 = 12.5 % > 3 %
chk('> 3 % failed studies: exit 4, no raptor224_sub.csv, R224_MIRROR_FAILED',
    rc_b == 4 and not os.path.isfile(T2 + '/work/raptor224_sub.csv') and 'R224_MIRROR_FAILED' in log_b, (rc_b,))
shutil.rmtree(T2, ignore_errors=True)
ids40 = ['1.2.4.%d' % k for k in range(40)]
T3 = tempfile.mkdtemp(prefix='rm_ok1_').replace('\\', '/')
rc_c, log_c, _ = run_script(B.R224_RM, T3, ids40, bad=[ids40[5]])       # 1 of 40 = 2.5 % <= 3 %
rec = json.load(open(T3 + '/work/raptor224_tta_receipt.json')) if os.path.isfile(T3 + '/work/raptor224_tta_receipt.json') else {}
avg40 = rd(T3 + '/work/raptor224_raw.csv').loc[ids40, LAB] if os.path.isfile(T3 + '/work/raptor224_raw.csv') else None
chk('<= 3 % failed: exit 0, sub written, n_fail 1, failed row 0.5 (her rule)',
    rc_c in (0, None) and os.path.isfile(T3 + '/work/raptor224_sub.csv') and rec.get('n_fail') == 1 and avg40 is not None
    and bool((avg40.loc[ids40[5]].values == 0.5).all()), (rc_c, rec.get('n_fail')))
shutil.rmtree(T3, ignore_errors=True)

# ---------------------------------------------------------------------------------------------------- S3 guard + canary
print('S3 RM_GUARD fixtures', flush=True)
EX = os.path.join(B.V224, 'out_v224_v1')
sub0 = open(os.path.join(EX, 'submission.csv'), 'rb').read()
r224sub = rd(os.path.join(EX, 'raptor224_sub.csv'))
tids = pd.read_csv(os.path.join(EX, 'submission.csv'), dtype={'StudyInstanceUID': str})['StudyInstanceUID'].tolist()
labels = LAB
from pathlib import Path


def fixture(T, receipt='ok', raw='ok', hf=None, canary=True):
    os.makedirs(T, exist_ok=True)
    open(T + '/submission.csv', 'wb').write(sub0)
    if canary:
        json.dump({'r224_used': True, 'R224_OK': True, 'use_f': False, 'flip_tta_forced': False, 'fired': []}, open(T + '/canary_receipt.json', 'w'))
    rawv = r224sub.loc[tids, labels].values * 0.5 + 0.1
    if raw == 'reversed':
        rawv = 1 - rawv
    df = pd.DataFrame(rawv.astype(np.float32), columns=labels); df.insert(0, 'StudyInstanceUID', tids)
    df.to_csv(T + '/raptor224_raw.csv', index=False, float_format='%.9g')
    rec = {'mirror': True, 'perm': list(range(12)), 'labels': labels, 'n': 3, 'n_pairs': 3, 'n_fail': 0,
           'per_study_s': {}, 'plain_vs_mirror': {'max_abs': 0.01}, 'seconds': 15}
    if receipt == 'perm':
        rec['perm'] = [0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 10, 11]
    if receipt != 'missing':
        json.dump(rec, open(T + '/raptor224_tta_receipt.json', 'w'))
    if hf is not None:
        json.dump(hf, open(T + '/hf_receipt.json', 'w'))


def run_cell(text, T, used=True, extra=None):
    g = {'r224': r224sub.loc[tids, labels] if used else None, 'R224_OK': used, 'labels': labels, 'test_ids': tids,
         'n': len(tids), 'WORK': Path(T), 'RM_ON': True, 'W_F': 0.2}
    g.update(extra or {})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(text.replace("'/kaggle/working'", repr(T)), '<cell>', 'exec'), g)
    return buf.getvalue(), open(T + '/submission.csv', 'rb').read(), json.load(open(T + '/canary_receipt.json'))


def cols_equal_except(b1, b2, blanked):
    a = pd.read_csv(io.BytesIO(b1), dtype=str, keep_default_na=False)
    b = pd.read_csv(io.BytesIO(b2), dtype=str, keep_default_na=False)
    keep = [c for c in a.columns if c not in blanked]
    return a[keep].equals(b[keep]) and all((b[c] == '0.5').all() for c in blanked)


base = tempfile.mkdtemp(prefix='rm_s3_').replace('\\', '/')
fixture(base + '/a'); o, s, cr = run_cell(B.RM_GUARD, base + '/a')
chk('guard clean: RM_GUARD_CLEAN, submission bytes unchanged', 'RM_GUARD_CLEAN' in o and s == sub0 and cr['fired'] == [] and cr['rm_guard']['mirror_ok'])
for case, kw in (('missing receipt', dict(receipt='missing')), ('wrong perm', dict(receipt='perm')),
                 ('raw not the r224 csv ranks', dict(raw='reversed'))):
    T = base + '/' + case.replace(' ', '_')
    fixture(T, **kw); o, s, cr = run_cell(B.RM_GUARD, T)
    chk(f'guard {case}: RM_GUARD_FIRED, Fracture 0.5, other 11 columns byte-equal', 'RM_GUARD_FIRED' in o and
        cols_equal_except(sub0, s, ['Fracture']) and cr['fired'] == ['r224 mirror->Fracture'], cr['rm_guard'].get('why', '')[:70])
T = base + '/unused'; fixture(T, receipt='missing'); o, s, cr = run_cell(B.RM_GUARD, T, used=False)
chk('guard: r224 not used -> no-op (the canary owns that path)', 'RM_GUARD_CLEAN' in o and s == sub0)
shutil.rmtree(base, ignore_errors=True)

print(f'SIM_RM {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}', flush=True)
sys.exit(0 if all(R) else 1)
