#!/usr/bin/env python
# build_hr7rr.py (2026-10-13, Claude): dual-Raptor variants of HR7 and HFR7-R20.
#   HR7-RR  = HR7 (heliosli 0-957 notebook verbatim, our rsna-sub-hr7 v1) + a second public Raptor stage (nartaa .945 r224
#             c08 SWA checkpoint from the SAME attached dataset) + the stage-3 blend with Raptor_mix in place of Raptor.
#   HFR7-RR = HFR7-R20 (rsna-sub-hfr7r20: HR7 + our fleet stage + HF cell at w_f .2) + the same two changes; fleet cells and
#             the HF cell byte-identical.
# Raptor_mix = per-label average-tie percentile ranks over each model's own non-failed rows, r384 .65 / r224 .35 (coordinator
# 10-13, after varshithakolupuri's public recipe; v1 of HR7-RR used .5 / .5), then ranked by the unchanged stage-3 rule.
#   new cells: 2b = %%writefile /tmp/raptor224_run.py (nartaa 0945 notebook cells 3+4, the scored r224 source, with the same
#              2 [blend] edits heliosli made to the r384 source: raw dump at full float32 precision + own output csv),
#              inserted right after the r384 writefile cell;
#              3b = stage 1b runner (own process, 45 min cap, RR_FALLBACK, never raises), right after stage 1.
#   edited:    stage 3 (Raptor_mix edit). Every other cell byte-identical to its base.
# Stage 1b runs BEFORE the R7 assets cell on purpose: that cell pip-installs timm 1.0.25 / pydicom 3.0.1 over the image's
# 1.0.29 / 3.0.2, so 1b sees the same stock environment as stage 1 (in HFR7-RR it also runs before ENV-SNAP / the fleet).
#   usage: python build_hr7rr.py [--target hr7rr|hfr7rr|all]   (writes kernel_<target>/<code>.ipynb + kernel-metadata.json)
import os, json, copy, hashlib, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
HR7_NB = os.path.join(HERE, '..', 'hr7', 'kernel', 'rsna_sub_hr7.ipynb')
HR7_SHA = 'cb2a0f57962ca9b6668661e34461aad772ed9ed10d576d32772e7c2898f2e315'
HR7_META = os.path.join(HERE, '..', 'hr7', 'kernel', 'kernel-metadata.json')
HFR7_NB = os.path.join(HERE, '..', 'hfr7', 'build', 'hfr7r20', 'rsna_sub_hfr7r20.ipynb')
HFR7_SHA = 'a984ae024027574ac035a44e76336e98734bd2a87660b81de9eaa03c1e707344'
HFR7_META = os.path.join(HERE, '..', 'hfr7', 'build', 'hfr7r20', 'kernel-metadata.json')
NB945 = os.path.join(HERE, 'src', 'rsna-knee-0945-efficient-224crop.ipynb')   # kaggle kernels pull nartaa/rsna-knee-0945-efficient-224crop (10-13; == the 10-08 pull)
NB945_SHA = 'b80d3b72a1c34f2b39b3ea0d2952f039b8d27c3feb5c4db0a7d0ab62ae75354e'
W_R384, W_R224 = 0.65, 0.35
ap = argparse.ArgumentParser()
ap.add_argument('--target', choices=['hr7rr', 'hfr7rr', 'all'], default='all')
ARGS = ap.parse_args()


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def src(cell):
    return ''.join(cell['source'])


def code_cell(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': text.splitlines(keepends=True)}


def sub1(text, old, new):
    assert text.count(old) == 1, ('anchor count != 1', old[:80], text.count(old))
    return text.replace(old, new)


assert sha(HR7_NB) == HR7_SHA, 'HR7 notebook changed'
assert sha(HFR7_NB) == HFR7_SHA, 'HFR7-R20 notebook changed'
assert sha(NB945) == NB945_SHA, '0945 notebook changed'
hr7 = json.load(open(HR7_NB, encoding='utf-8'))
hfr7 = json.load(open(HFR7_NB, encoding='utf-8'))
nb945 = json.load(open(NB945, encoding='utf-8'))
C = hr7['cells']
assert len(C) == 7 and src(C[2]).startswith('%%writefile /tmp/raptor_run.py\n')
assert src(C[3]).startswith('# Stage 1: public Raptor model') and src(C[6]).startswith('# Stage 3: per-finding rank blend')

# ---- cell 2b: the r224 scored source (nartaa 0945 notebook cells 3 + 4) -> /tmp/raptor224_run.py --------------------
c3, c4 = src(nb945['cells'][3]), src(nb945['cells'][4])
assert "_weight_name = 'raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt'" in c3
assert "'3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6'" in c3
assert '"arch": "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", "res": 224' in c4 and 'CROP_FRAC = 0.8' in c4
# identical decode to stage 1: same embedded _fastread.py, slots, corpus size and reader variant
r384 = src(C[2])
for key in ('_FASTREAD_B64 = ', 'SLOTS = [', 'IMG = 384', 'K_EVAL = 94', "FASTREAD_VARIANT = 'study_lut'", 'CROP_MM = 140.0'):
    l384 = [l for l in r384.splitlines() if l.startswith(key)]
    l224 = [l for l in c4.splitlines() if l.startswith(key)]
    assert len(l384) == 1 and l384 == l224, ('decode config differs', key)
rr = '%%writefile /tmp/raptor224_run.py\n' + c3 + ('' if c3.endswith('\n') else '\n') + '\n' + c4
rr = sub1(rr, '    print(f"[gpu] done: {done_n[0]}/{N} studies on {len(devs)} device(s) | {time.time()-t0:.0f}s", flush=True)\n',
          '    print(f"[gpu] done: {done_n[0]}/{N} studies on {len(devs)} device(s) | {time.time()-t0:.0f}s", flush=True)\n'
          '    _raw = pd.DataFrame(arm_probs[0].astype(np.float32), columns=LAB)   # [rr] raw probabilities; a failed study stays 0.5\n'
          '    _raw.insert(0, "StudyInstanceUID", test_ids)\n'
          '    _raw.to_csv("/kaggle/working/raptor224_raw.csv", index=False)\n')
rr = sub1(rr, '    out = "/kaggle/working/submission.csv"\n',
          '    out = "/kaggle/working/raptor224_sub.csv"   # [rr] stage 1b output; the blend cell writes submission.csv\n')
CELL_2B = rr

# ---- cell 3b: stage 1b runner ----------------------------------------------------------------------------------------
CELL_3B = '''# [rr] Stage 1b (HR7-RR): the second public Raptor checkpoint of the attached nartaa dataset (.945 r224 c08 SWA), scored
# with its own public source (written to /tmp/raptor224_run.py above), run as its own process like stage 1 and before
# stage 2's pinned wheels (same stock environment as stage 1); writes raptor224_raw.csv / raptor224_sub.csv.
# A crash, a non-zero exit, missing files or a run past RR_TIMEOUT_S -> RR_FALLBACK: stage 3 then uses Raptor r384 alone,
# i.e. exactly HR7.  This cell never stops the notebook.
import signal
RR_TIMEOUT_S = 45 * 60
RR_OK, RR_REASON = False, ''
_rr_t0 = time.time()
for _f in ('raptor224_sub.csv', 'raptor224_raw.csv'):
    if os.path.isfile('/kaggle/working/' + _f):
        os.remove('/kaggle/working/' + _f)
_rr_p = None
try:
    _rr_p = subprocess.Popen([sys.executable, '-u', '/tmp/raptor224_run.py'], start_new_session=True)
    try:
        _rr_rc = _rr_p.wait(timeout=RR_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        _rr_rc = None
    if _rr_rc is None:
        RR_REASON = 'late: killed after %d s' % RR_TIMEOUT_S
    elif _rr_rc != 0:
        RR_REASON = 'exit code %d' % _rr_rc
    else:
        _miss = [_f for _f in ('raptor224_sub.csv', 'raptor224_raw.csv') if not os.path.isfile('/kaggle/working/' + _f)]
        RR_REASON = ('missing ' + ', '.join(_miss)) if _miss else ''
        RR_OK = not _miss
except Exception as _e:
    RR_REASON = '%s: %s' % (type(_e).__name__, _e)
finally:
    if _rr_p is not None:
        try:
            os.killpg(_rr_p.pid, signal.SIGKILL)   # the child's own process group only (prep pool leftovers, or the late run)
        except Exception:
            pass
        try:
            _rr_p.wait(timeout=60)
        except Exception:
            pass
RR_SECONDS = time.time() - _rr_t0
if RR_OK:
    print('RR_STAGE_DONE seconds %.0f' % RR_SECONDS, flush=True)
else:
    print('RR_FALLBACK stage 1b: %s (%.0f s); stage 3 uses Raptor r384 alone' % (RR_REASON, RR_SECONDS), flush=True)
subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader'], check=False)
'''

# ---- cell 6: stage 3 with Raptor_mix ---------------------------------------------------------------------------------
c6 = src(C[6])
c6 = sub1(c6, "# A study that one model failed on takes the other model's rank; more than 3 % Raptor failures stops the notebook.\n",
          "# A study that one model failed on takes the other model's rank; more than 3 % Raptor failures stops the notebook.\n"
          "# [rr] HR7-RR: 'Raptor' = per-finding mix of the two public nartaa Raptor checkpoints: each percentile-ranked (average ties)\n"
          "# over its own non-failed studies, averaged r384 0.65 / r224 0.35, then ranked like the old Raptor column (weights fixed\n"
          "# before any run).  A study r224 failed on keeps its r384 rank.  RR_FALLBACK (stage 1b failed / late / invalid, or more\n"
          "# than 3 % r224 failures) -> Raptor r384 alone, i.e. exactly HR7.\n")
c6 = sub1(c6, "out = np.zeros((n, len(labels)))\n", '''W_R384, W_R224 = %r, %r                             # [rr] fixed before any run''' % (W_R384, W_R224) + '''
rap_src = raw                                           # [rr] RR_FALLBACK -> the r384 raw column, exactly as HR7
rr_info = {'used': False, 'weights': {'r384': W_R384, 'r224': W_R224}, 'stage_ok': bool(RR_OK), 'reason': RR_REASON,
           'r224_seconds': round(RR_SECONDS, 1)}
if RR_OK:
    try:
        raw2 = pd.read_csv(WORK / 'raptor224_raw.csv', dtype={'StudyInstanceUID': str})
        rsub2 = pd.read_csv(WORK / 'raptor224_sub.csv', dtype={'StudyInstanceUID': str})
        for df, nm in ((raw2, 'raptor224_raw'), (rsub2, 'raptor224_sub')):
            assert sorted(df['StudyInstanceUID']) == sorted(test_ids), nm + ': rows differ from test.csv'
            assert set(labels) <= set(df.columns), nm + ': columns differ'
        raw2 = raw2.set_index('StudyInstanceUID').loc[test_ids, labels]
        rsub2 = rsub2.set_index('StudyInstanceUID').loc[test_ids, labels]
        assert np.isfinite(raw2.values).all(), 'raptor224_raw has non-finite values'
        r2_fail = (raw2.values == 0.5).all(axis=1)      # the r224 source also leaves a failed study at 0.5 in every column
        assert r2_fail.sum() <= 0.03 * n, 'r224 failed on %d of %d studies' % (r2_fail.sum(), n)
        if n >= 3:
            rho2 = {c: round(float(pd.Series(raw2[c].values).corr(pd.Series(rsub2[c].values), method='spearman')), 4) for c in labels}
            assert min(rho2.values()) >= 0.99, 'r224 raw dump does not match its own submission ranks: %s' % rho2
            rr_info['spearman_r224_raw_vs_sub'] = rho2
            rr_info['spearman_r384_vs_r224'] = {c: round(float(pd.Series(raw[c].values).corr(pd.Series(raw2[c].values), method='spearman')), 4) for c in labels}
        mix = np.full((n, len(labels)), np.nan)
        for j, c in enumerate(labels):
            p1, p2 = pct_on(raw[c].values, ~rap_fail), pct_on(raw2[c].values, ~r2_fail)
            m = W_R384 * p1 + W_R224 * p2
            m[r2_fail] = p1[r2_fail]
            mix[:, j] = m                               # NaN where r384 failed: those rows take ours' rank below, as in HR7
        rap_src = pd.DataFrame(mix, index=raw.index, columns=labels)
        rr_info.update(used=True, r224_failed=int(r2_fail.sum()))
    except Exception as e:
        RR_REASON = 'stage 3 check: %s: %s' % (type(e).__name__, e)
        rr_info['reason'] = RR_REASON
        rap_src = raw
if rr_info['used']:
    print('RR_MIX_OK r384 %.2f / r224 %.2f, r224_failed=%d of %d' % (W_R384, W_R224, rr_info['r224_failed'], n), flush=True)
else:
    print('RR_FALLBACK stage 3 uses Raptor r384 alone (%s)' % RR_REASON, flush=True)
out = np.zeros((n, len(labels)))
''')
c6 = sub1(c6, "    rr, ro = pct_on(raw[c].values, ~rap_fail), pct_on(ours[c].values, ~our_fail)   # [v3] each over its own non-failed rows\n",
          "    rr, ro = pct_on(rap_src[c].values, ~rap_fail), pct_on(ours[c].values, ~our_fail)   # [v3] each over its own non-failed rows; [rr] rap_src = Raptor_mix (r384 raw on RR_FALLBACK)\n")
c6 = sub1(c6, "         'rank_basis': 'each model over its own non-failed rows'}\n",
          "         'rank_basis': 'each model over its own non-failed rows', 'raptor_mix': rr_info}   # [rr]\n")
CELL_6 = c6

# ---- assemble --------------------------------------------------------------------------------------------------------
for t in (CELL_2B.split('\n', 1)[1], CELL_3B, CELL_6):
    compile(t.replace('display(', 'print('), '<cell>', 'exec')


def build(base, writefile_i, stage1_i, stage3_i, slug, code_file, meta_path, out_dir):
    B = base['cells']
    assert src(B[writefile_i]) == src(C[2]) and src(B[stage1_i]) == src(C[3]) and src(B[stage3_i]) == src(C[6])
    cells = []
    for i, c in enumerate(B):
        if i == stage3_i:
            c6cell = copy.deepcopy(c); c6cell['source'] = CELL_6.splitlines(keepends=True); cells.append(c6cell)
        else:
            cells.append(copy.deepcopy(c))
        if i == writefile_i:
            cells.append(code_cell(CELL_2B))
        if i == stage1_i:
            cells.append(code_cell(CELL_3B))
    assert len(cells) == len(B) + 2
    # every base cell except stage 3 byte-identical, in the same order
    kept = [c for c in cells if c['source'] not in (code_cell(CELL_2B)['source'], code_cell(CELL_3B)['source'])]
    assert len(kept) == len(B) and all(kept[i] == B[i] for i in range(len(B)) if i != stage3_i)
    nb = copy.deepcopy(base); nb['cells'] = cells
    os.makedirs(out_dir, exist_ok=True)
    nb_path = os.path.join(out_dir, code_file)
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    meta = json.load(open(meta_path, encoding='utf-8'))
    meta.update(id=slug, title=slug.split('/')[1].replace('-', ' '), code_file=code_file, is_private=True,
                enable_gpu=True, enable_internet=False)
    with open(os.path.join(out_dir, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    print('BUILT', nb_path, 'sha256', sha(nb_path), 'cells', len(cells), 'datasets', meta['dataset_sources'])


if ARGS.target in ('hr7rr', 'all'):
    build(hr7, 2, 3, 6, 'eeshsaxena/rsna-sub-hr7rr', 'rsna_sub_hr7rr.ipynb', HR7_META, os.path.join(HERE, 'kernel_hr7rr'))
if ARGS.target in ('hfr7rr', 'all'):
    FC = hfr7['cells']
    assert len(FC) == 14 and src(FC[13]).startswith('# === HF BLEND') and "W_F = 0.2\n" in src(FC[13])
    build(hfr7, 3, 4, 12, 'eeshsaxena/rsna-sub-hfr7rr', 'rsna_sub_hfr7rr.ipynb', HFR7_META, os.path.join(HERE, 'kernel_hfr7rr'))
with open(os.path.join(HERE, 'cells_new.txt'), 'w', encoding='utf-8') as f:
    f.write('#### CELL 3b (stage 1b)\n' + CELL_3B + '\n#### STAGE 3 (edited)\n' + CELL_6)
