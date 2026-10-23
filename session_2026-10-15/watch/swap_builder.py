#!/usr/bin/env python
# swap_builder.py (2026-10-14, Claude subagent; RESEARCH_960 lever 3): turn a PUBLIC notebook of this competition into a
# private, canary-armed copy of ours, ready to commit-test. NOTHING is pushed or submitted: it writes kernel_<suffix>/
# (notebook + kernel-metadata.json), build_<suffix>.json (gate results, inputs, runtime of the public commit, fallback audit)
# and prints the push / gate / submit commands.
#
# What it does, in order:
#  1. reads the public notebook page model anonymously (score, scored version, current version, pinned input versions of the
#     scored run, its commit runtime and docker digest); the version copied = --version, or /N in the ref, or the SCORED
#     version (bestSubmissionScore), never silently the latest;
#  2. pulls exactly that version = `kaggle kernels pull OWNER/SLUG/N -m` (same API call, kept in memory so no newline is
#     rewritten) into src/<owner>__<slug>__vN/ (untrusted data: parsed as JSON/text, never executed here);
#  3. gates the inputs: competition data = ours only; every dataset public (anonymous 200) and its current version == the
#     version the scored run pinned (a CLI push attaches the LATEST version; drift FAILS unless --allow-drift); kernel
#     sources public; models visible; internet off; docker pin and accelerator reported;
#  4. copies the notebook VERBATIM (every cell, every output, same JSON serialisation, detected by exact round trip) and
#     APPENDS one canary cell; proof: deleting the appended cell gives back their file byte for byte;
#  5. canary: profile v224 (her notebook structure, detected with build_v224.py's asserts) appends build_v224.py's CANARY
#     text exactly (read from that file, not retyped); any other notebook gets the generic canary: --canary-expr triggers
#     (each name must exist in their code; a trigger that is true OR undefined at run time fires) blank --canary-col to 0.5,
#     plus always-on checks (submission.csv exists, columns/ids == sample_submission, finite in [0,1]; else raise), constant
#     columns and per-study fallback counts in *receipt*.json are logged; prints CANARY_CLEAN / CANARY_FIRED and SWAP_DONE;
#  6. metadata = theirs (dataset list = the scored run's pinned list) with build_v224.py's edits: our id / title / code_file,
#     private, internet off, id_no and keywords dropped.
# --verify-v224: also proves that, on varshitha's notebook with --name rsna-sub-v224, the output equals build_v224.py's
#     V224-exact kernel byte for byte (both the file on disk and a fresh build_v224.py --target v224 rebuild).
#   usage: python swap_builder.py OWNER/SLUG[/VERSION] [--name rsna-sub-XXX] [--out DIR] [--profile auto|v224|generic]
#                                 [--canary-col Fracture] [--canary-expr EXPR]... [--allow-drift] [--allow-unscored] [--verify-v224]
# Exit 0 = BUILD PASS (ready to commit-test), 1 = a gate FAILED (kernel dir still written, marked DO_NOT_PUSH).
import os, sys, re, ast, json, copy, hashlib, argparse, subprocess, datetime, difflib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kgweb                        # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
V224_DIR = os.path.join(ROOT, 'session_2026-10-14', 'v224')
BUILD_V224 = os.path.join(V224_DIR, 'build_v224.py')
SAMPLE = os.path.join(ROOT, '_data', 'sample_submission.csv')
SUBMIT_AT = 'session_2026-10-14/submit_at.py'
TRIGGER_DEFAULTS = {'v224': ['r224 is None']}     # documentation only: the v224 profile uses build_v224's own CANARY

ap = argparse.ArgumentParser()
ap.add_argument('ref', help='OWNER/SLUG or OWNER/SLUG/VERSION of a public notebook of this competition')
ap.add_argument('--version', type=int, default=None)
ap.add_argument('--name', default=None, help='our kernel slug (default rsna-sub-sw-<slug head>-v<N>)')
ap.add_argument('--out', default=os.path.join(HERE, 'swaps'))
ap.add_argument('--profile', choices=['auto', 'v224', 'generic'], default='auto')
ap.add_argument('--canary-col', default='Fracture')
ap.add_argument('--canary-expr', action='append', default=[])
ap.add_argument('--allow-drift', action='store_true')
ap.add_argument('--allow-unscored', action='store_true')
ap.add_argument('--verify-v224', action='store_true')
ap.add_argument('--diff-against', default=os.path.join(V224_DIR, 'src', 'varshitha_v224.ipynb'),
                help='a local .ipynb to diff the copied notebook against, cell by cell (default: the V224 source; "none" = skip)')
A = ap.parse_args()
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

G = []          # gate rows: (status, name, info)


def gate(status, name, info=''):
    G.append((status, name, info))
    print('  %-4s %s %s' % (status, name, info if isinstance(info, str) else json.dumps(info, default=str)[:600]), flush=True)


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode('utf-8')).hexdigest()


def src_of(cell):
    s = cell.get('source', '')
    return ''.join(s) if isinstance(s, list) else s


def v224_constants():
    """CANARY / SRC_NB_SHA / CI_SHA exactly as written in build_v224.py (parsed, not executed)."""
    tree = ast.parse(open(BUILD_V224, encoding='utf-8').read())
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id in ('CANARY', 'SRC_NB_SHA', 'CI_SHA'):
            out[n.targets[0].id] = ast.literal_eval(n.value)
    assert set(out) == {'CANARY', 'SRC_NB_SHA', 'CI_SHA'}, out.keys()
    return out


def v224_structure(C):
    """build_v224.py's structural asserts on her 13 cells, as predicates (all True = the V224 notebook layout)."""
    if len(C) != 13:
        return False, 'cells %d != 13' % len(C)
    kinds = [c['cell_type'] for c in C]
    checks = [
        kinds == ['markdown'] * 3 + ['code', 'code', 'markdown', 'code', 'code', 'markdown', 'code', 'code', 'markdown', 'code'],
        src_of(C[3]).startswith('%%writefile /tmp/raptor_run.py\n'),
        src_of(C[4]).startswith('# Stage 1: public Raptor model') and 'NB_T0 = time.time()' in src_of(C[4]),
        src_of(C[6]).startswith("ARM = 'E_lrprior'\n") and "if manifest.get('flip_tta'):\n    # v4c" in src_of(C[6]),
        "    WORKER_ARGS += ['--flip-tta']\n" in src_of(C[6]) and "print('ASSETS_READY'" in src_of(C[6]),
        src_of(C[7]).lstrip().startswith('deadline = T0 + DEADLINE_HOURS * 3600'),
        src_of(C[9]).startswith('%%writefile /tmp/raptor224_run.py\n'),
        'out = "/kaggle/working/raptor224_sub.csv"' in src_of(C[9]),
        src_of(C[10]).startswith('# Stage 2b [s5]') and '_left = 9 * 3600 - (time.time() - NB_T0) - 40 * 60' in src_of(C[10]),
        src_of(C[12]).startswith('# Stage 3: per-finding rank blend') and 'A224 = 0.35' in src_of(C[12]),
        "sub.to_csv(WORK / 'submission.csv', index=False)" in src_of(C[12]) and 'r224 = None' in src_of(C[12]),
    ]
    bad = [i for i, ok in enumerate(checks) if not ok]
    return not bad, ('all %d structure checks hold' % len(checks)) if not bad else ('failed checks %s' % bad)


SERIALISERS = [   # name -> f(nb) ; the first that reproduces their text exactly is used for our copy
    ('compact, ensure_ascii False', lambda nb: json.dumps(nb, ensure_ascii=False, separators=(',', ':'))),
    ('compact, ensure_ascii True', lambda nb: json.dumps(nb, separators=(',', ':'))),
    ('nbformat (indent 1, sorted keys, ascii False, final newline)',
     lambda nb: json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + '\n'),
    ('indent 1, ascii False', lambda nb: json.dumps(nb, indent=1, ensure_ascii=False)),
    ('indent 1, ascii False, final newline', lambda nb: json.dumps(nb, indent=1, ensure_ascii=False) + '\n'),
    ('indent 1, ascii True', lambda nb: json.dumps(nb, indent=1)),
    ('indent 2, ascii False', lambda nb: json.dumps(nb, indent=2, ensure_ascii=False)),
    ('indent 2, ascii True', lambda nb: json.dumps(nb, indent=2)),
]

GENERIC_CANARY = r'''# [eeshsaxena CANARY (generic, swap_builder.py); not in the public notebook __SRC__] Makes this copy's known fallbacks
# visible on the leaderboard: if a trigger below is true (or its variable was never defined), column __COL__ is set to 0.5 for
# every study (one constant column, about -.03 macro AUC on the display) and CANARY_FIRED is printed. Always checks that
# submission.csv exists and matches sample_submission.csv (columns, ids) with finite values in [0, 1] (else raises, so a broken
# file is a visible error, never a silent score). Constant columns and per-study fallback counts in *receipt*.json are logged.
# Clean path: prints CANARY_CLEAN, submission.csv untouched.
import os as _co, json as _cj
from pathlib import Path as _CP
import numpy as _cnp, pandas as _cpd
_CW = _CP('/kaggle/working')
_CSRC, _CCOL = __SRC_REPR__, __COL_REPR__
_ctrig = {}
__TRIGGERS__
_cs = _cpd.read_csv(_CW / 'submission.csv', dtype={'StudyInstanceUID': str})
_cc = {'profile': 'generic', 'source': _CSRC, 'canary_col': _CCOL, 'triggers': _ctrig, 'fired': [], 'checks': {}}
_cpath = [p + '/sample_submission.csv' for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',
                                                 '/kaggle/input/rsna-knee-abnormality-detection')]
_cpath = [p for p in _cpath if _co.path.isfile(p)]
if _cpath:
    _css = _cpd.read_csv(_cpath[0], dtype={'StudyInstanceUID': str})
    assert list(_cs.columns) == list(_css.columns), ('CANARY columns', list(_cs.columns))
    assert len(_cs) == len(_css) and _cs['StudyInstanceUID'].is_unique and \
        set(_cs['StudyInstanceUID']) == set(_css['StudyInstanceUID']), 'CANARY ids differ from sample_submission'
    _cc['checks']['sample'] = _cpath[0]
else:
    print('CANARY_NO_SAMPLE sample_submission.csv not found under the two standard mounts; schema check skipped', flush=True)
    _cc['checks']['sample'] = None
_clab = [c for c in _cs.columns if c != 'StudyInstanceUID']
_cv = _cs[_clab].values.astype(_cnp.float64)
assert _cnp.isfinite(_cv).all() and (_cv >= 0).all() and (_cv <= 1).all(), 'CANARY values not finite in [0, 1]'
assert _CCOL in _clab, ('CANARY column missing', _CCOL)
_cc['checks']['constant_columns'] = [c for c in _clab if len(_cs) > 1 and _cs[c].nunique() <= 1]
if _cc['checks']['constant_columns']:
    print('CANARY_CONSTANT_COLUMNS', _cc['checks']['constant_columns'], flush=True)
_crf = {}
for _p in sorted(_CW.glob('*receipt*.json')):
    try:
        _d = _cj.loads(_p.read_text())
    except Exception:
        continue
    if isinstance(_d, dict):
        for _k in ('fallback_studies', 'raptor_failed', 'ours_failed', 'both_failed'):
            if isinstance(_d.get(_k), (int, float)) and _d[_k] > 0:
                _crf[_p.name + ':' + _k] = _d[_k]
_cc['checks']['receipt_fallbacks'] = _crf
if _crf:
    print('CANARY_RECEIPT_FALLBACKS (logged, per-study, as in the public runs)', _crf, flush=True)
_cc['fired'] = [k for k, v in _ctrig.items() if v is not False]
if _cc['fired']:
    _cs[_CCOL] = 0.5
    _cs.to_csv(_CW / 'submission.csv', index=False)
    print('CANARY_FIRED %s: %s set to 0.5 for all %d rows' % (_cc['fired'], _CCOL, len(_cs)), flush=True)
else:
    print('CANARY_CLEAN no trigger fired, no column blanked', flush=True)
(_CW / 'canary_receipt.json').write_text(_cj.dumps(_cc, indent=1, default=str))
print('SWAP_DONE', _cj.dumps({k: _cc[k] for k in ('source', 'fired', 'triggers')}, default=str), flush=True)
'''


def trigger_code(exprs):
    if not exprs:
        return '# (no triggers: this copy has no known silent fallback; see the build report fallback audit)'
    lines = []
    for e in exprs:
        lines += ['try:', '    _ctrig[%r] = bool(%s)' % (e, e), 'except NameError:', '    _ctrig[%r] = %r' % (e, 'undefined')]
    return '\n'.join(lines)


AUDIT_RE = re.compile(r'FALLBACK|fallback|SKIP|skipped|TIMEOUT|[Tt]imeout|UNUSABLE|deadline|DEADLINE|except\b|= None\b|fillna\(|'
                      r'sys\.exit|SystemExit|_left\b')


def cell_diff(new_cells, ref_cells):
    """Cell-level diff of the copied notebook vs a reference notebook: identical cells (by sha), changed cells (nearest
    same-type reference cell, unified diff), new cells, and reference cells with no counterpart."""
    rc = [(j, c.get('cell_type'), src_of(c)) for j, c in enumerate(ref_cells)]
    by_sha = {}
    for j, _, s in rc:
        by_sha.setdefault(sha(s), j)
    out, summ, used = [], {'identical': 0, 'changed': 0, 'new': 0}, set()
    for i, c in enumerate(new_cells):
        s = src_of(c)
        if sha(s) in by_sha:
            summ['identical'] += 1
            used.add(by_sha[sha(s)])
            out.append('cell %d == ref cell %d (%s, %d chars)' % (i, by_sha[sha(s)], c.get('cell_type'), len(s)))
            continue
        cands = [(difflib.SequenceMatcher(None, s, r_, autojunk=False).quick_ratio(), j, r_) for j, t, r_ in rc
                 if t == c.get('cell_type') and j not in used]
        best = max(cands, key=lambda x: x[0]) if cands else None
        if best and best[0] >= 0.5:
            summ['changed'] += 1
            used.add(best[1])
            d = list(difflib.unified_diff(best[2].splitlines(), s.splitlines(), 'ref cell %d' % best[1], 'new cell %d' % i,
                                          lineterm='', n=1))
            out.append('cell %d ~ ref cell %d (%s, quick ratio %.2f): %d diff lines' % (i, best[1], c.get('cell_type'), best[0], len(d)))
            out += ['    ' + ln[:220] for ln in d[:400]]
        else:
            summ['new'] += 1
            out.append('cell %d NEW (%s, %d chars): %s' % (i, c.get('cell_type'), len(s), s.strip().splitlines()[0][:120] if s.strip() else ''))
    summ['ref_cells_without_counterpart'] = [j for j, _, _ in rc if j not in used]
    return summ, out


def fallback_audit(code_cells):
    rows = []
    for ci, text in code_cells:
        for li, line in enumerate(text.splitlines()):
            if AUDIT_RE.search(line):
                rows.append('cell %d line %d: %s' % (ci, li + 1, line.strip()[:150]))
    return rows


def main():
    owner_slug = A.ref.strip('/').split('/')
    assert len(owner_slug) in (2, 3), 'ref must be OWNER/SLUG or OWNER/SLUG/VERSION'
    owner, slug = owner_slug[0], owner_slug[1]
    ref = owner + '/' + slug
    print('SWAP_BUILD %s (%s)' % (ref, datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')), flush=True)
    web, api = kgweb.Web(), kgweb.kaggle_api()

    # 1. public page model
    s, j = web.kvm(ref)
    if s != 200 or not j:
        gate('FAIL', 'public notebook page readable anonymously', 'http %s' % s)
        return finish(None)
    kv = kgweb.summarize_kvm(j)
    bv, cv = kv['best_version'], kv['cur_version']
    version = A.version or (int(owner_slug[2]) if len(owner_slug) == 3 else None) or bv
    gate('OK', 'public notebook page readable anonymously', 'score %s (scored v%s), current v%s, %s versions' % (
        kv['best_public'], bv, cv, kv['n_versions']))
    if version is None:
        if A.allow_unscored:
            version = cv
            gate('WARN', 'notebook has no public score: copying the current version', 'v%s' % cv)
        else:
            gate('FAIL', 'notebook has a public score (or --allow-unscored)', 'no bestSubmissionScore')
            return finish(None)
    elif bv is not None and version != bv:
        gate('WARN', 'copying a version other than the scored one', 'v%s (scored v%s)' % (version, bv))
    else:
        gate('OK', 'copying the scored version', 'v%s' % version)
    sv, vj = web.versions(kv['kernel_id'])
    vt = kgweb.version_table(vj) if sv == 200 else {}
    vrec = vt.get(int(version))
    if vrec is None and int(version) == int(cv or -1):
        vrec = {'run_id': kv['run']['id'], 'date': kv['run']['date'], 'run_time_seconds': kv['run']['run_time_seconds'],
                'docker_digest': kv['run']['docker_digest'], 'accelerator': kv['run']['accelerator'], 'sources': kv['sources'],
                'status': kv['run']['status']}
    if vrec is None:
        gate('FAIL', 'run record of the copied version found', 'v%s not in %s' % (version, sorted(vt)))
        return finish(None)
    gate('OK' if (vrec.get('status') in (None, 'COMPLETE')) else 'FAIL', 'copied version ran to COMPLETE on Kaggle',
         {k: vrec.get(k) for k in ('run_id', 'status', 'date', 'run_time_seconds', 'accelerator')})

    # 2. pull exactly that version (= kaggle kernels pull OWNER/SLUG/N -m)
    text, meta_src, fname = kgweb.get_kernel(api, '%s/%s' % (ref, version))
    meta_src = kgweb.plain(meta_src)
    sdir = os.path.join(A.out, 'src', '%s__%s__v%s' % (owner, slug, version))
    os.makedirs(sdir, exist_ok=True)
    with open(os.path.join(sdir, fname), 'w', encoding='utf-8', newline='') as f:
        f.write(text)
    with open(os.path.join(sdir, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:   # as the CLI writes it
        json.dump(meta_src, f, indent=2)
    raw = text.encode('utf-8')
    gate('OK', 'pulled', '%s (%d bytes, sha256 %s) -> %s' % (fname, len(raw), sha(raw)[:16], os.path.relpath(sdir, ROOT)))
    ok_lang = meta_src.get('language') == 'python' and meta_src.get('kernel_type') in ('notebook', 'script')
    gate('OK' if ok_lang else 'FAIL', 'python notebook or script', '%s / %s' % (meta_src.get('language'), meta_src.get('kernel_type')))
    if not ok_lang:
        return finish(None)

    # 3. inputs of the copied version's run (pinned) vs what a push of our copy would attach
    pinned = []
    for x in vrec['sources']:
        if x['ref'].startswith('?/'):
            pool = meta_src.get('dataset_sources') if x['kind'] == 'dataset' else meta_src.get('kernel_sources')
            hit = [p for p in pool or [] if p and p.split('/')[-1] == x['ref'][2:]]
            x = dict(x, ref=hit[0]) if len(hit) == 1 else x
        pinned.append(x)
    comps = [x['ref'] for x in pinned if x['kind'] == 'competition']
    gate('OK' if comps == [kgweb.COMP] and meta_src.get('competition_sources') == [kgweb.COMP] else 'FAIL',
         'competition data = this competition only', {'run': comps, 'meta': meta_src.get('competition_sources')})
    pin_ds = [x for x in pinned if x['kind'] == 'dataset']
    pin_k = [x for x in pinned if x['kind'] == 'kernel']
    pin_m = [x for x in pinned if x['kind'] == 'model']
    other = [x for x in pinned if x['kind'] == 'other' or x['ref'].startswith('?/')]
    gate('OK' if not other else 'FAIL', 'every input resolved to OWNER/SLUG', other)
    inputs = {}
    for x in pin_ds:
        s2, dj = web.dataset(x['ref'])
        pub = s2 == 200 and dj is not None
        cur = (dj or {}).get('datasetVersionId')
        inputs['dataset:' + x['ref']] = {'public': pub, 'pinned_version_id': x['version_id'], 'current_version_id': cur,
                                         'current_version': (dj or {}).get('datasetVersionNumber'),
                                         'bytes': ((dj or {}).get('data') or {}).get('totalSize'), 'http': s2}
        gate('OK' if pub else 'FAIL', 'dataset public', '%s (http %s)' % (x['ref'], s2))
        if pub:
            same = cur is not None and int(cur) == int(x['version_id'])
            gate('OK' if same else ('WARN' if A.allow_drift else 'FAIL'),
                 'dataset version = the version the copied run pinned (a push attaches the latest)',
                 '%s pinned %s, current %s (v%s)' % (x['ref'], x['version_id'], cur, (dj or {}).get('datasetVersionNumber')))
    for x in pin_k:
        s2, _ = web.kvm(x['ref'])
        inputs['kernel:' + x['ref']] = {'public': s2 == 200, 'pinned_version_id': x['version_id'], 'http': s2}
        gate('OK' if s2 == 200 else 'FAIL', 'kernel source public', '%s (http %s; a push attaches its latest version)' % (x['ref'], s2))
    for x in pin_m:
        try:
            api.model_get('/'.join(x['ref'].split('/')[:2]))
            inputs['model:' + x['ref']] = {'visible': True}
            gate('OK', 'model source visible', x['ref'])
        except Exception as e:
            inputs['model:' + x['ref']] = {'visible': False, 'err': str(e)[:120]}
            gate('FAIL', 'model source visible', '%s: %s' % (x['ref'], str(e)[:120]))
    want_ds = [x['ref'] for x in pin_ds]
    meta_ds = [d for d in meta_src.get('dataset_sources') or [] if d]
    if set(meta_ds) == set(want_ds):
        ds_list = list(meta_src.get('dataset_sources'))           # their order, as pulled
        gate('OK', 'pulled metadata datasets == the copied run\'s pinned datasets', ds_list)
    else:
        ds_list = want_ds
        gate('WARN', 'pulled metadata datasets != the copied run\'s pinned datasets: using the pinned list',
             {'meta': meta_ds, 'pinned': want_ds})
    k_list = [x['ref'] for x in pin_k]
    if set(k_list) != set(d for d in meta_src.get('kernel_sources') or [] if d):
        gate('WARN', 'kernel sources: using the pinned list', {'meta': meta_src.get('kernel_sources'), 'pinned': k_list})
    gate('OK' if not meta_src.get('enable_internet') else 'WARN', 'their internet off (ours is forced off)',
         meta_src.get('enable_internet'))
    dock = meta_src.get('docker_image') or ''
    gate('OK' if dock else 'WARN', 'docker image pinned in their metadata (kept)',
         dock or 'NOT pinned: our copy runs on the image of push time; their run used digest %s' % vrec.get('docker_digest'))
    if dock and vrec.get('docker_digest') and vrec['docker_digest'] not in dock:
        gate('WARN', 'their pinned docker != the digest their copied run used', {'meta': dock, 'run': vrec.get('docker_digest')})
    gate('OK' if meta_src.get('machine_shape') == 'NvidiaTeslaT4' and meta_src.get('enable_gpu') else 'WARN',
         'machine 2 x T4 (as the hidden rerun)', '%s gpu %s; run accelerator %s' % (
             meta_src.get('machine_shape'), meta_src.get('enable_gpu'), vrec.get('accelerator')))

    # 4. verbatim copy + one appended canary
    import pandas as pd
    labels = list(pd.read_csv(SAMPLE, nrows=1).columns[1:])
    v224 = v224_constants()
    is_nb = fname.endswith('.ipynb')
    profile, info = 'generic', ''
    if is_nb:
        nb = json.loads(text)
        ser = next(((n, f) for n, f in SERIALISERS if f(nb) == text), None)
        gate('OK' if ser else 'WARN', 'their JSON serialisation reproduced exactly', ser[0] if ser else
             'none of %d known serialisations; proof falls back to JSON equality' % len(SERIALISERS))
        cells = nb['cells']
        code_cells = [(i, src_of(c)) for i, c in enumerate(cells) if c.get('cell_type') == 'code']
        is_v224, info = v224_structure(cells)
    else:
        code_cells = [(0, text)]
        is_v224, info = False, 'script kernel'
    if A.profile == 'v224' or (A.profile == 'auto' and is_v224):
        profile = 'v224'
        gate('OK' if is_v224 else 'FAIL', 'profile v224: build_v224.py structure checks', info)
        canary = v224['CANARY']
        if A.canary_expr or A.canary_col != 'Fracture':
            gate('WARN', 'profile v224 ignores --canary-expr / --canary-col (uses build_v224.py CANARY verbatim)')
        gate('OK' if sha(raw) == v224['SRC_NB_SHA'] else 'WARN', 'source == the notebook build_v224.py was built from',
             'sha %s (build_v224 %s)' % (sha(raw)[:16], v224['SRC_NB_SHA'][:16]))
    else:
        gate('OK', 'profile generic', info)
        gate('OK' if A.canary_col in labels else 'FAIL', 'canary column is a label', A.canary_col)
        code_all = '\n'.join(t for _, t in code_cells)
        for e in A.canary_expr:
            try:
                names = {n.id for n in ast.walk(ast.parse(e, mode='eval')) if isinstance(n, ast.Name)}
                missing = [n for n in names if not re.search(r'\b%s\b' % re.escape(n), code_all)]
                gate('OK' if not missing else 'FAIL', 'canary trigger names exist in their code', '%r missing %s' % (e, missing))
            except SyntaxError as ex:
                gate('FAIL', 'canary trigger parses', '%r: %s' % (e, ex))
        if not A.canary_expr:
            gate('WARN', 'no canary trigger given: only the always-on checks run; read the fallback audit and add '
                         '--canary-expr for every silent fallback before submitting')
        canary = (GENERIC_CANARY.replace('__SRC__', '%s v%s' % (ref, version))
                  .replace('__SRC_REPR__', repr('%s/%s' % (ref, version))).replace('__COL_REPR__', repr(A.canary_col))
                  .replace('__COL__', A.canary_col).replace('__TRIGGERS__', trigger_code(A.canary_expr)))
    compile(canary, '<canary>', 'exec')
    gate('OK', 'canary compiles', '%d chars, profile %s' % (len(canary), profile))
    audit = fallback_audit(code_cells)
    diff_summary = None
    if is_nb and A.diff_against and A.diff_against.lower() != 'none' and os.path.isfile(A.diff_against):
        ref_nb = json.loads(open(A.diff_against, 'rb').read().decode('utf-8'))
        diff_summary, diff_lines = cell_diff(cells, ref_nb['cells'])
        diff_summary['against'] = os.path.relpath(A.diff_against, ROOT)
        os.makedirs(A.out, exist_ok=True)
        diff_path = os.path.join(A.out, 'diff_%s__v%s.txt' % (slug, version))
        with open(diff_path, 'w', encoding='utf-8') as f:
            f.write('cell diff of %s v%s vs %s\n%s\n\n' % (ref, version, diff_summary['against'], json.dumps(diff_summary)))
            f.write('\n'.join(diff_lines) + '\n')
        diff_summary['file'] = os.path.relpath(diff_path, ROOT)
        gate('INFO', 'cell diff vs reference notebook', diff_summary)

    name = A.name or ('rsna-sub-sw-%s-v%s' % (re.sub('[^a-z0-9]+', '-', slug.lower())[:24].strip('-'), version))
    assert re.fullmatch(r'[a-z0-9][a-z0-9-]{4,49}', name), 'bad kernel slug %r' % name
    try:                               # read-only: does eeshsaxena/<name> exist already (a push would add a version to it)?
        kgweb.get_kernel(api, 'eeshsaxena/' + name)
        slug_exists = True
        gate('WARN', 'our kernel slug already exists: a push makes a NEW VERSION of eeshsaxena/%s (pick --name for a new kernel)' % name)
    except Exception:
        slug_exists = False
        gate('OK', 'our kernel slug is new', 'eeshsaxena/' + name)
    suffix = name[len('rsna-sub-'):] if name.startswith('rsna-sub-') else name
    code_file = name.replace('-', '_') + ('.ipynb' if is_nb else '.py')
    kdir = os.path.join(A.out, 'kernel_' + suffix)
    os.makedirs(kdir, exist_ok=True)
    if is_nb:
        has_ids = any('id' in c for c in cells)
        list_src = any(isinstance(c.get('source'), list) for c in cells if c.get('cell_type') == 'code')
        cell = {'cell_type': 'code', 'metadata': {}, 'source': canary.splitlines(True) if list_src else canary,
                'outputs': [], 'execution_count': None}
        if has_ids:
            cell = dict({'id': 'eeshsaxena-canary'}, **cell)
        out_nb = copy.deepcopy(nb)
        out_nb['cells'] = copy.deepcopy(cells) + [cell]
        dump = ser[1] if ser else (lambda o: json.dumps(o, ensure_ascii=False, separators=(',', ':')))
        out_text = dump(out_nb)
        back = copy.deepcopy(out_nb)
        back['cells'] = back['cells'][:-1]
        proof = (dump(back) == text) if ser else (back == nb)
        gate('OK' if proof else 'FAIL', 'verbatim: dropping the appended cell gives back their notebook',
             'byte for byte' if ser else 'JSON-equal (no byte-exact serialisation found)')
        gate('OK' if len(out_nb['cells']) == len(cells) + 1 else 'FAIL', 'exactly one cell appended, at the end',
             '%d -> %d cells' % (len(cells), len(out_nb['cells'])))
    else:
        out_text = text + ('' if text.endswith('\n') else '\n') + '\n' + canary
        gate('OK' if out_text.startswith(text) else 'FAIL', 'verbatim: their script is an exact prefix of ours')
    nb_path = os.path.join(kdir, code_file)
    with open(nb_path, 'w', encoding='utf-8', newline='') as f:
        f.write(out_text)
    meta = {k: copy.deepcopy(v) for k, v in meta_src.items() if k not in ('id_no', 'keywords')}
    meta.update(id='eeshsaxena/' + name, title=name.replace('-', ' '), code_file=code_file, is_private=True,
                enable_gpu=bool(meta_src.get('enable_gpu')), enable_tpu=False, enable_internet=False)
    meta['dataset_sources'] = ds_list
    if k_list or meta.get('kernel_sources'):
        meta['kernel_sources'] = k_list
    with open(os.path.join(kdir, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:   # as build_v224.py writes it
        json.dump(meta, f, indent=1)
    out_sha = sha(open(nb_path, 'rb').read())
    gate('OK', 'written', '%s sha256 %s' % (os.path.relpath(nb_path, ROOT), out_sha[:16]))

    rep = {'built_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source': ref, 'version': version,
           'scored_version': bv, 'current_version': cv, 'public_score': kv['best_public'], 'profile': profile,
           'source_file': os.path.relpath(os.path.join(sdir, fname), ROOT), 'source_sha256': sha(raw),
           'kernel': 'eeshsaxena/' + name, 'kernel_slug_exists': slug_exists, 'kernel_dir': os.path.relpath(kdir, ROOT), 'notebook_sha256': out_sha,
           'metadata': meta, 'inputs': inputs, 'copied_run': {k: vrec.get(k) for k in ('run_id', 'date', 'run_time_seconds',
                                                                                        'docker_digest', 'accelerator')},
           'canary': {'col': A.canary_col if profile == 'generic' else 'Fracture', 'triggers':
                      A.canary_expr if profile == 'generic' else TRIGGER_DEFAULTS['v224']},
           'fallback_audit': audit[:120], 'cell_diff': diff_summary, 'gates': [{'status': s_, 'name': n_, 'info': i_} for s_, n_, i_ in G]}
    return finish(rep, name, suffix, kdir, nb_path, ref, version, cv)


def finish(rep, name=None, suffix=None, kdir=None, nb_path=None, ref=None, version=None, cv=None):
    fails = [g for g in G if g[0] == 'FAIL']
    warns = [g for g in G if g[0] == 'WARN']
    verdict = 'PASS' if not fails and rep is not None else 'FAIL'
    if rep is not None:
        rep['verdict'] = verdict
        if verdict != 'PASS':
            open(os.path.join(kdir, 'DO_NOT_PUSH.txt'), 'w', encoding='utf-8').write(
                'swap_builder gate FAILED:\n' + '\n'.join('%s %s' % (n, i) for _, n, i in fails) + '\n')
        if A.verify_v224:
            rep['verify_v224'] = verify_v224(nb_path, kdir)
            open(os.path.join(kdir, 'DO_NOT_PUSH.txt'), 'w', encoding='utf-8').write(
                'Reproduction proof only (--verify-v224): this folder is byte-identical to session_2026-10-14/v224/kernel_v224, '
                'already pushed and submitted. Pushing it would only add a version to eeshsaxena/%s.\n' % name)
            if rep['verify_v224']['verdict'] != 'PASS':
                verdict = rep['verdict'] = 'FAIL'
        with open(os.path.join(A.out, 'build_%s.json' % suffix), 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=1, ensure_ascii=False, default=str)
        rt = fnum(rep['copied_run'].get('run_time_seconds'))
        out_rel = os.path.relpath(A.out, ROOT).replace('\\', '/')
        k_rel = os.path.relpath(kdir, ROOT).replace('\\', '/')
        print('\nFALLBACK AUDIT (%d lines matching fallback / skip / timeout / except / None patterns; first 25):' % len(rep['fallback_audit']))
        for r in rep['fallback_audit'][:25]:
            print('   ', r)
        print('\nNEXT (nothing below was run; run from %s):' % ROOT)
        print('  # 0. never more than 2 of our commits at once:  kaggle kernels list --mine --sort-by dateRun --page-size 10')
        print('  # 1. commit test (~%s min of GPU quota; their commit ran %s s):  kaggle kernels push -p %s' % (
            '%.0f' % (rt / 60 + 4) if rt else '11', rep['copied_run'].get('run_time_seconds'), k_rel))
        print('  #    then: kaggle kernels status eeshsaxena/%s ; kaggle kernels output eeshsaxena/%s -p %s/out_%s_v1' % (
            name, name, out_rel, suffix))
        print('  # 2. their own commit output (read-only; it is the CURRENT version v%s%s): kaggle kernels output %s -p %s/her_%s' % (
            cv, '' if str(cv) == str(version) else ' != copied v%s: compare with care' % version, ref, out_rel, suffix))
        print('  # 3. gate: python %s/swap_check.py %s/out_%s_v1 --her %s/her_%s%s' % (
            os.path.relpath(HERE, ROOT).replace('\\', '/'), out_rel, suffix, out_rel, suffix,
            '' if rep['profile'] == 'generic' else '   (and session_2026-10-14/v224/v224_check.py OUT --her HER)'))
        print('  # 4. submit only after the gate PASSES and a PREREG amendment:  python %s eeshsaxena/%s %s "<msg>"' % (
            SUBMIT_AT, name, '<N = the version the push created>' if rep.get('kernel_slug_exists') else '1'))
    print('\nSWAP_BUILD %s: %d OK, %d WARN, %d FAIL%s' % (verdict, sum(1 for g in G if g[0] == 'OK'), len(warns), len(fails),
                                                     ('  (kernel %s)' % kdir) if kdir else ''), flush=True)
    return 0 if verdict == 'PASS' else 1


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def verify_v224(nb_path, kdir):
    """Byte-for-byte: our output vs (a) build_v224.py's kernel_v224 on disk, (b) a fresh build_v224.py --target v224 rebuild."""
    print('\nVERIFY_V224 (swap_builder output vs build_v224.py V224-exact):', flush=True)
    res = {'checks': []}
    disk = os.path.join(V224_DIR, 'kernel_v224')
    tmp = os.path.join(A.out, 'verify_v224_rebuild')
    os.makedirs(tmp, exist_ok=True)
    r = subprocess.run([sys.executable, BUILD_V224, '--target', 'v224', '--out', tmp], capture_output=True, text=True)
    res['rebuild_rc'] = r.returncode
    res['rebuild_stdout'] = r.stdout.strip()[-400:]
    for label, other in (('build_v224 kernel on disk', disk), ('fresh build_v224 rebuild', os.path.join(tmp, 'kernel_v224'))):
        for fn in ('rsna_sub_v224.ipynb', 'kernel-metadata.json'):
            a = open(os.path.join(kdir, fn), 'rb').read() if os.path.isfile(os.path.join(kdir, fn)) else b''
            b = open(os.path.join(other, fn), 'rb').read() if os.path.isfile(os.path.join(other, fn)) else None
            ok = b is not None and a == b
            res['checks'].append({'vs': label, 'file': fn, 'equal': ok, 'ours_sha': sha(a)[:16], 'theirs_sha': sha(b)[:16] if b else None,
                                  'bytes': [len(a), len(b) if b else None]})
            print('  %-4s %s: %s ours %s / %s (%s bytes)' % ('OK' if ok else 'FAIL', label, fn, sha(a)[:16],
                                                          sha(b)[:16] if b else 'missing', len(a)), flush=True)
    res['verdict'] = 'PASS' if r.returncode == 0 and all(c['equal'] for c in res['checks']) else 'FAIL'
    print('SWAP_V224_REPRO %s %d/%d' % (res['verdict'], sum(c['equal'] for c in res['checks']), len(res['checks'])), flush=True)
    return res


if __name__ == '__main__':
    sys.exit(main())
