#!/usr/bin/env python
# build_hr7rs.py (2026-10-13, Claude): "our Raptor seeds" variants of HR7-RR v2 and HFR7-RR v1.
#   HR7-RS  = HR7-RR v2 (kernel_hr7rr, sha 18cb5a27) + stage 1c (our seeds) + stage-3 Raptor_mix with a third member.
#   HFR7-RS = HFR7-RR v1 (kernel_hfr7rr, sha ca3da2fb) + the same two changes; fleet / HF cells byte-identical.
# New cells, inserted right after the stage-1b runner (so stage 1c also runs BEFORE the R7 assets cell that pip-installs
# timm 1.0.25 over the image's 1.0.29, i.e. in the same stock environment as stages 1 and 1b):
#   2c = %%writefile /tmp/raptor_ours_run.py: the r384 scored source (cell "%%writefile /tmp/raptor_run.py") with ONLY
#        (a) the checkpoint preamble: our seed files + their sha256 (seeds.json) instead of nartaa's r384, (b) ARMS = one arm
#        per seed, (c) the per-file sha256 in load_model, (d) the raw dump = MEAN of the seeds' probabilities (full float32
#        precision, failed study stays 0.5) to raptor_ours_raw.csv and the rank file to raptor_ours_sub.csv. Same serve path:
#        d96 fastread study_lut, 320 crop, 94 windows, anatomical mirror, fp16 autocast.
#   3c = stage-1c runner: own process, RS_TIMEOUT_S cap, RAPTOR_SEEDS_FALLBACK (never raises).
# Edited: stage 3: after the [rr] block, if stage 1c is OK and its outputs pass the checks, Raptor_mix =
#   per-label average-tie percentile ranks r384 W3['r384'] / r224 W3['r224'] / ours W3['ours'] (weights fixed in
#   PREREG_RAPTOR_KERNELS.md before any public read), renormalised over the members available in a row (r224 or ours failed
#   on that study); r384-failed rows stay NaN exactly as in the parent. Any stage-1c failure -> RAPTOR_SEEDS_FALLBACK = the
#   parent's Raptor_mix, i.e. the parent submission. The receipt gains a 'raptor_seeds' block.
# usage: python build_hr7rs.py --seeds seeds.json [--target hr7rs|hfr7rs|all] [--rs_timeout_min 60]
import os, json, hashlib, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
RR = os.path.join(HERE, '..', '..', 'hr7rr')
BASES = {'hr7rs': (os.path.join(RR, 'kernel_hr7rr', 'rsna_sub_hr7rr.ipynb'), '18cb5a27f39f7a6b1bc0f45be10450f596d137f3a10d5bc4d49b5c103f76ef8d',
                   os.path.join(RR, 'kernel_hr7rr', 'kernel-metadata.json')),
         'hfr7rs': (os.path.join(RR, 'kernel_hfr7rr', 'rsna_sub_hfr7rr.ipynb'), 'ca3da2fbfbf35816f2a34b8d3c8e137f0a5fada89b2ac3555fc55e47571f9b11',
                    os.path.join(RR, 'kernel_hfr7rr', 'kernel-metadata.json'))}
DS = 'eeshsaxena/rsna-knee-raptor-seeds'
W3 = {'r384': 0.45, 'r224': 0.25, 'ours': 0.30}
ap = argparse.ArgumentParser(); ap.add_argument('--seeds', required=True); ap.add_argument('--target', default='all')
ap.add_argument('--rs_timeout_min', type=int, default=60); ap.add_argument('--out', default=HERE)
A = ap.parse_args()
SEEDS = json.load(open(A.seeds))                                # [{"file": "raptor_rsna_s43_swa.pt", "sha256": "..."}, ...]
assert 2 <= len(SEEDS) <= 6 and all(len(s['sha256']) == 64 and s['file'].endswith('.pt') for s in SEEDS)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
src = lambda c: ''.join(c['source'])
cell = lambda t: {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': t.splitlines(keepends=True)}


def sub1(text, old, new):
    assert text.count(old) == 1, ('anchor count != 1', old[:90], text.count(old))
    return text.replace(old, new)


def ours_source(r384):
    t = sub1(r384, '%%writefile /tmp/raptor_run.py\n', '%%writefile /tmp/raptor_ours_run.py\n')
    i0 = t.index("_weight_name = 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt'")
    i1 = t.index('\n\n# %% [code]')
    pre = ('# [rs] OUR Raptor seeds (train_raptor.py = the 746792 recipe, RSNA rows only): every file must be present with its\n'
           '# exact sha256; no substitute model.\n'
           '_SEEDS = ' + json.dumps({s['file']: s['sha256'] for s in SEEDS}, indent=1) + '\n'
           "_ROOTS = [Path('/kaggle/input/rsna-knee-raptor-seeds'), Path('/kaggle/input/datasets/eeshsaxena/rsna-knee-raptor-seeds')]\n"
           'for _n, _s in _SEEDS.items():\n'
           '    _p = next((r / _n for r in _ROOTS if (r / _n).is_file()), None)\n'
           '    if _p is None:\n'
           '        raise FileNotFoundError("Attach eeshsaxena/rsna-knee-raptor-seeds: missing " + _n)\n'
           '    with _p.open("rb") as _f:\n'
           '        assert hashlib.file_digest(_f, "sha256").hexdigest() == _s, "Checkpoint hash differs: " + _n\n')
    t = t[:i0] + pre + t[i1:]
    old_arms = t[t.index('ARMS = ['):t.index(']\n', t.index('ARMS = [')) + 2]
    t = sub1(t, old_arms, 'ARMS = [   # [rs] one arm per seed, equal weights\n' + ''.join(
        '    {"file": %r, "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0},\n' % s['file'] for s in SEEDS) + ']\n')
    t = sub1(t, "    assert observed_sha == '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', \"Bound checkpoint bytes differ\"\n",
             '    assert observed_sha == _SEEDS[os.path.basename(pt_path)], "Bound checkpoint bytes differ"   # [rs] per-seed sha\n')
    t = sub1(t, '    _raw = pd.DataFrame(arm_probs[0].astype(np.float32), columns=LAB)',
             "    _mean32 = np.mean(np.stack(arm_probs), axis=0).astype(np.float32)   # [rs] mean of the seeds' probabilities (the dumped values)\n"
             '    _raw = pd.DataFrame(_mean32, columns=LAB)')
    # [rs] v2: the rank file must be the ranks OF THE DUMPED MEAN. v1 kept the parent's per-arm rank-mean here, a different
    # aggregation than the dumped prob-mean, so the stage-3 consistency check (Spearman raw vs sub >= .99) fired on the commit.
    t = sub1(t, '''    ranks = np.tensordot(_w, np.stack([rankpct(np.clip(p, 0, 1)) for p in arm_probs]),
                         axes=(0, 0))''',
             '''    ranks = rankpct(np.clip(_mean32.astype(np.float64), 0, 1))   # [rs] ranks of the dumped mean: consistent with raptor_ours_raw.csv by construction''')
    t = sub1(t, 'print(f"[blend] weighted rank-mean w=', 'print(f"[rs] seeds prob-mean, ranks of the mean, w=')
    t = sub1(t, '_raw.to_csv("/kaggle/working/raptor_raw.csv", index=False)', '_raw.to_csv("/kaggle/working/raptor_ours_raw.csv", index=False)')
    t = sub1(t, 'out = "/kaggle/working/raptor_sub.csv"', 'out = "/kaggle/working/raptor_ours_sub.csv"')
    assert '7e5315dad125b99f' not in t and 'r384_swa.pt' not in t.split('# %% [code]')[0]
    return t


RUNNER = '''# [rs] Stage 1c: OUR Raptor seeds (RSNA-only retrains of the 746792 recipe, dataset eeshsaxena/rsna-knee-raptor-seeds),
# scored with the r384 source (written to /tmp/raptor_ours_run.py above) as its own process, before the R7 / fleet stages.
# A crash, a non-zero exit, missing files or a run past RS_TIMEOUT_S -> RAPTOR_SEEDS_FALLBACK: stage 3 then uses the parent
# Raptor_mix (r384 + r224), i.e. exactly the parent submission.  This cell never stops the notebook.
import signal
RS_TIMEOUT_S = @TMIN@ * 60
RAPTOR_SEEDS_OK, RS_REASON = False, ''
_rs_t0 = time.time()
for _f in ('raptor_ours_sub.csv', 'raptor_ours_raw.csv'):
    if os.path.isfile('/kaggle/working/' + _f):
        os.remove('/kaggle/working/' + _f)
_rs_p = None
try:
    _rs_p = subprocess.Popen([sys.executable, '-u', '/tmp/raptor_ours_run.py'], start_new_session=True)
    try:
        _rs_rc = _rs_p.wait(timeout=RS_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        _rs_rc = None
    if _rs_rc is None:
        RS_REASON = 'late: killed after %d s' % RS_TIMEOUT_S
    elif _rs_rc != 0:
        RS_REASON = 'exit code %d' % _rs_rc
    else:
        _miss = [_f for _f in ('raptor_ours_sub.csv', 'raptor_ours_raw.csv') if not os.path.isfile('/kaggle/working/' + _f)]
        RS_REASON = ('missing ' + ', '.join(_miss)) if _miss else ''
        RAPTOR_SEEDS_OK = not _miss
except Exception as _e:
    RS_REASON = '%s: %s' % (type(_e).__name__, _e)
finally:
    if _rs_p is not None:
        try:
            os.killpg(_rs_p.pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            _rs_p.wait(timeout=60)
        except Exception:
            pass
RS_SECONDS = time.time() - _rs_t0
if RAPTOR_SEEDS_OK:
    print('RS_STAGE_DONE seconds %.0f' % RS_SECONDS, flush=True)
else:
    print('RAPTOR_SEEDS_FALLBACK stage 1c: %s (%.0f s); stage 3 uses the parent Raptor_mix' % (RS_REASON, RS_SECONDS), flush=True)
subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader'], check=False)
'''

STAGE3_ANCHOR = '''if rr_info['used']:
    print('RR_MIX_OK r384 %.2f / r224 %.2f, r224_failed=%d of %d' % (W_R384, W_R224, rr_info['r224_failed'], n), flush=True)
else:
    print('RR_FALLBACK stage 3 uses Raptor r384 alone (%s)' % RR_REASON, flush=True)
'''
STAGE3_ADD = '''W3 = @W3@                         # [rs] fixed in PREREG_RAPTOR_KERNELS.md before any public read
rs_info = {'used': False, 'weights': W3, 'stage_ok': bool(RAPTOR_SEEDS_OK), 'reason': RS_REASON, 'seconds': round(RS_SECONDS, 1)}
if RAPTOR_SEEDS_OK:
    try:
        raw3 = pd.read_csv(WORK / 'raptor_ours_raw.csv', dtype={'StudyInstanceUID': str})
        rsub3 = pd.read_csv(WORK / 'raptor_ours_sub.csv', dtype={'StudyInstanceUID': str})
        for df, nm in ((raw3, 'raptor_ours_raw'), (rsub3, 'raptor_ours_sub')):
            assert sorted(df['StudyInstanceUID']) == sorted(test_ids), nm + ': rows differ from test.csv'
            assert set(labels) <= set(df.columns), nm + ': columns differ'
        raw3 = raw3.set_index('StudyInstanceUID').loc[test_ids, labels]
        rsub3 = rsub3.set_index('StudyInstanceUID').loc[test_ids, labels]
        assert np.isfinite(raw3.values).all(), 'raptor_ours_raw has non-finite values'
        r3_fail = (raw3.values == 0.5).all(axis=1)
        assert r3_fail.sum() <= 0.03 * n, 'our seeds failed on %d of %d studies' % (r3_fail.sum(), n)
        if n >= 3:
            rho3 = {c: round(float(pd.Series(raw3[c].values).corr(pd.Series(rsub3[c].values), method='spearman')), 4) for c in labels}
            assert min(rho3.values()) >= 0.99, 'our raw dump does not match its own submission ranks: %s' % rho3
            rs_info['spearman_r384_vs_ours'] = {c: round(float(pd.Series(raw[c].values).corr(pd.Series(raw3[c].values), method='spearman')), 4) for c in labels}
        use224 = bool(rr_info['used'])
        if use224:
            raw2_ = raw2; r2f = r2_fail
        mix = np.full((n, len(labels)), np.nan)
        for j, c in enumerate(labels):
            p1 = pct_on(raw[c].values, ~rap_fail)
            p3 = pct_on(raw3[c].values, ~r3_fail)
            num = W3['r384'] * p1; den = np.full(n, W3['r384'])
            ok3 = ~r3_fail
            num[ok3] += W3['ours'] * p3[ok3]; den[ok3] += W3['ours']
            if use224:
                p2 = pct_on(raw2_[c].values, ~r2f); ok2 = ~r2f
                num[ok2] += W3['r224'] * p2[ok2]; den[ok2] += W3['r224']
            mix[:, j] = num / den                       # NaN where r384 failed (p1 NaN): those rows take ours' rank, as in the parent
        rap_src = pd.DataFrame(mix, index=raw.index, columns=labels)
        rs_info.update(used=True, ours_failed=int(r3_fail.sum()), with_r224=use224)
    except Exception as e:
        RS_REASON = 'stage 3 check: %s: %s' % (type(e).__name__, e)
        rs_info['reason'] = RS_REASON
if rs_info['used']:
    print('RAPTOR_SEEDS_MIX_OK %s, ours_failed=%d of %d, with_r224=%s' % (W3, rs_info['ours_failed'], n, rs_info['with_r224']), flush=True)
else:
    print('RAPTOR_SEEDS_FALLBACK stage 3 keeps the parent Raptor_mix (%s)' % RS_REASON, flush=True)
'''


def build(target):
    nb_path, nb_sha, meta_path = BASES[target]
    assert sha(nb_path) == nb_sha, target + ': base notebook changed'
    nb = json.load(open(nb_path, encoding='utf-8')); C = nb['cells']
    i384 = [i for i, c in enumerate(C) if src(c).startswith('%%writefile /tmp/raptor_run.py\n')]
    i1b = [i for i, c in enumerate(C) if src(c).startswith('# [rr] Stage 1b (HR7-RR)')]
    i3 = [i for i, c in enumerate(C) if src(c).startswith('# Stage 3: per-finding rank blend')]
    assert len(i384) == len(i1b) == len(i3) == 1, (i384, i1b, i3)
    s3 = src(C[i3[0]])
    s3 = sub1(s3, STAGE3_ANCHOR, STAGE3_ANCHOR + STAGE3_ADD.replace('@W3@', repr(W3)))
    s3 = sub1(s3, "'rank_basis': 'each model over its own non-failed rows', 'raptor_mix': rr_info}   # [rr]",
              "'rank_basis': 'each model over its own non-failed rows', 'raptor_mix': rr_info, 'raptor_seeds': rs_info}   # [rr] [rs]")
    C[i3[0]] = cell(s3)
    new = [cell(ours_source(src(C[i384[0]]))), cell(RUNNER.replace('@TMIN@', str(A.rs_timeout_min)))]
    C[i1b[0] + 1:i1b[0] + 1] = new                      # 2c (writefile) + 3c (runner) right after the stage-1b runner
    for c in new + [C[i3[0] + 2]]:                      # syntax check of the python cells we wrote / edited
        t = src(c); t = t.split('\n', 1)[1] if t.startswith('%%writefile') else t
        compile(t, target, 'exec')
    d = os.path.join(A.out, 'kernel_' + target); os.makedirs(d, exist_ok=True)
    code = 'rsna_sub_%s.ipynb' % target
    json.dump(nb, open(os.path.join(d, code), 'w', encoding='utf-8'), indent=1)
    meta = json.load(open(meta_path))
    meta.update(id='eeshsaxena/rsna-sub-' + target, title='rsna sub ' + target, code_file=code)
    meta['dataset_sources'] = meta['dataset_sources'] + [DS]
    json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=1)
    print(target, 'cells', len(C), '| sha256', sha(os.path.join(d, code))[:16], '| datasets', meta['dataset_sources'])


for t in (['hr7rs', 'hfr7rs'] if A.target == 'all' else [A.target]):
    build(t)
