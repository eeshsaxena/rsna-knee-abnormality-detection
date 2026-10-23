#!/usr/bin/env python
# stack_kgate.py (10-20 ~23:17 IST, Claude, STACK BUILD round). LOCAL offline gate for stack/kernels/ (pattern push/e1/gate_e1_offline.py:
# EXECUTES each notebook's own cell 14 (stage 3), cell 15 (V224 canary), emulates cell 16's freeze, then cell 18's _s6f_blend, on the
# gold58 serve predictions). Cell 14 is unchanged in every stack (asserted), so the stage-3 replay must give pw30's stage 3 byte for byte.
# Gates (all must hold -> STACK_KGATE PASS):
#   H  harness: pw30 replay = kernel-path gold58 macro .94336 (a21 / stack_gate logs); stage 3 == numpy members->c4 construction; clean path.
#   K  kernels: nb sha == build receipt, base sha == scored pw30, only cell 18 differs from pw30 (whole-cell JSON), cell 14 identical.
#   F  factorial, per stack: every on/off combination of its components (dose .25 -> .3, A_LOAMM -> 0.0, A_MCLMM -> 0.0, by exact
#      single-literal substitution in ITS OWN cell 18) is replayed:
#        F1 all components off -> submission.csv byte-identical to scored pw30's replay; receipt a_oa .3 pw, no a_loamm / a_mclmm;
#        F2 changed columns vs pw30 == the union of the ON components' columns; every other column byte-identical (CSV text);
#        F3 all 12 columns == independent numpy recompute (route A: from this run's stage 3 + S6fall B; route B: from the member serve
#           files, no notebook code) at maxdiff < 1e-6;
#        F4 receipt keys / values and stdout markers (EOA_APPLIED value, LOAMM_APPLIED, MCLMM_APPLIED, in that order, before S6F_BLEND_OK).
#   C  CTRL_off kernel (both blocks at 0) byte-identical to pw30; cross-stack: MCL column identical in all 4 stacks; S_wide == S_alt and
#      S_mid == S_safe except LOA; S_alt == val pw25 except MCL; S_mid LOA == night lomm05 LOA; S_safe == pw30 except MCL.
#   INFO gold58 macro / per-label deltas vs pw30 (n 58, never a selection ruler).
# No OAI data. Writes only stack/kernels/stack_kgate.log (temp dirs are removed).
import json, os, sys, io, time, hashlib, tempfile, shutil, itertools, contextlib, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
R = 'C:/Users/eeshs/Downloads/rsna-knee'; G = R + '/session_2026-10-18/next/g58'; S = R + '/session_2026-10-20'
HERE = os.path.dirname(os.path.abspath(__file__)).replace(chr(92), '/')
LOG = open(HERE + '/stack_kgate.log', 'w', encoding='utf-8')
def P(s=''):
    print(s, flush=True); LOG.write(s + '\n')
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OA = ['Medial OA', 'Lateral OA', 'PF OA']
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    p = y.sum(); q = len(y) - p; r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
shab = lambda b: hashlib.sha256(b).hexdigest()
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); g = g[g.source == 'original_gold58'].reset_index(drop=True)
ids = g.StudyInstanceUID.tolist(); Y = g[L].values.astype(int); n = len(ids)
assert pd.read_csv(G + '/gold58_r224_ids.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist() == ids
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
frame = lambda A: pd.DataFrame(A, columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L]
BASE_NB_SHA = '5a903187415505e9e3566ebd749cae1328b20df6b60bd6593e6443b17fc0c9fd'
nbp = lambda d, t: d + '/kernel_' + t + '/rsna_sub_' + t + '.ipynb'
REF = {'pw30': nbp(S + '/a21', 'v224r6c4pw30j'), 'pw25': nbp(S + '/val/kernels', 'v224r6c4pw25j'),
       'lomm05': S + '/night/c12_kernel_v224r6c4pw30lomm05j/rsna_sub_v224r6c4pw30lomm05j.ipynb'}
IDX = json.load(open(HERE + '/build_index.json', encoding='utf-8'))
ok = True
def chk(name, cond):
    global ok
    ok = ok and bool(cond); P(('OK   ' if cond else 'FAIL ') + name)


def replay(nb, label, c18_sub=None):
    """cells 14 + 15 + (16 freeze) + 18 of notebook dict nb on gold58; c18_sub = list of (old, new) exact single-occurrence substitutions."""
    c14, c15, c18 = (src(nb['cells'][k]) for k in (14, 15, 18))
    for old, new in (c18_sub or []):
        assert c18.count(old) == 1, (label, old, c18.count(old)); c18 = c18.replace(old, new)
    T = tempfile.mkdtemp().replace(chr(92), '/'); D, W, Pp = Path(T + '/data'), Path(T + '/work'), Path(T + '/pred')
    for d in (D, W, Pp / 'worker0', W / 's6f' / 'B'): d.mkdir(parents=True)
    pd.DataFrame({'StudyInstanceUID': ids}).to_csv(D / 'test.csv', index=False)
    frame(np.full((n, 12), .5)).iloc[:1].to_csv(D / 'sample_submission.csv', index=False)
    frame(R6).to_csv(W / 'ours_sub.csv', index=False); frame(R6).to_csv(Pp / 'worker0' / 'predictions.csv', index=False)
    frame(R384).to_csv(W / 'raptor_raw.csv', index=False); frame(R384).to_csv(W / 'raptor_sub.csv', index=False)
    frame(R224).to_csv(W / 'raptor224_sub.csv', index=False)
    ns = dict(DATA=D, WORK=W, pred=Pp, R224_OK=True, RAPTOR_SECONDS=0.0, NB_T0=time.time(), time=time, json=json,
              display=lambda *a, **k: None, FLIP_TTA_FORCED_ON=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(c14, label + '#cell14', 'exec'), ns)
        exec(compile(c15, label + '#cell15', 'exec'), ns)
        shutil.copyfile(W / 'submission.csv', W / 'submission_v224flip.csv')
        setup = {'base_sha256': sha(W / 'submission_v224flip.csv'), 'v224_canary': dict(ns['_canary'])}
        frame(S6).to_csv(W / 's6f' / 'B' / 'B_probs.csv', index=False)
        run = {'status': 'RAN', 'killed': False, 'driver': {'ok': True, 'B': {'probs_sha256': sha(W / 's6f' / 'B' / 'B_probs.csv')}}}
        fn = c18.split('\n\ntry:\n    _S6F_BLEND = ')[0]; assert fn.count("W = '/kaggle/working'") == 1
        ns18 = {}; exec(compile(fn.replace("W = '/kaggle/working'", 'W = %r' % str(W).replace(chr(92), '/')), label + '#cell18', 'exec'), ns18)
        rec = ns18['_s6f_blend'](setup, run)
    so = buf.getvalue()
    r = dict(rec=rec, canary=dict(ns['_canary']), st3_bytes=(W / 'submission_v224flip.csv').read_bytes(), bytes=(W / 'submission.csv').read_bytes(),
             st3=pd.read_csv(W / 'submission_v224flip.csv', dtype={'StudyInstanceUID': str}),
             out=pd.read_csv(W / 'submission.csv', dtype={'StudyInstanceUID': str}),
             txt=pd.read_csv(W / 'submission.csv', dtype=str), stdout=so)
    shutil.rmtree(T, ignore_errors=True); return r


# ---- independent numpy recompute (no notebook code) -------------------------------------------------------------------------
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
PW = {('Medial OA', 'Lateral OA'): .876, ('Medial OA', 'PF OA'): .708, ('Lateral OA', 'PF OA'): 1.416}
def v_members():
    V = np.zeros((n, 12))
    for j, c in enumerate(L):
        w = C4W.get(c, .5); V[:, j] = w * (.75 * pct(R384[:, j]) + .25 * pct(R224[:, j])) + (1 - w) * pct(R6[:, j])
    return V.astype(np.float32).astype(np.float64)
def final_from(V, aoa, al, am):
    F = np.stack([.8 * pct(V[:, j]) + .2 * pct(S6[:, j]) for j in range(12)], 1).astype(np.float32).astype(np.float64)
    U = {c: pct(F[:, L.index(c)]) for c in OA}; F2 = F.copy()
    for c in OA:
        o = [k for k in OA if k != c]; F2[:, L.index(c)] = pct(U[c] - aoa / 2.0 * sum(PW[tuple(sorted((c, k), key=OA.index))] * U[k] for k in o))
    F = F2.astype(np.float32).astype(np.float64)
    jl, jm, jc = L.index('Lateral OA'), L.index('Medial Meniscus'), L.index('MCL')
    if al:
        F[:, jl] = pct(pct(F[:, jl]) - al * pct(F[:, jm])); F = F.astype(np.float32).astype(np.float64)
    if am:
        F[:, jc] = pct(pct(F[:, jc]) - am * pct(F[:, jm])); F = F.astype(np.float32).astype(np.float64)
    return F


P('== H harness ==')
NBJ = {k: json.load(open(p, encoding='utf-8')) for k, p in REF.items()}
assert sha(REF['pw30']) == BASE_NB_SHA
ref = {k: replay(v, k) for k, v in NBJ.items()}
mac = lambda o: np.mean([auc(o[c].values, Y[:, j]) for j, c in enumerate(L)])
T0 = tempfile.mkdtemp(); frame(v_members().astype(np.float32)).to_csv(T0 + '/v.csv', index=False); vb = open(T0 + '/v.csv', 'rb').read(); shutil.rmtree(T0)
chk('H1 pw30 cell-14 replay stage 3 byte-identical to the numpy members->c4 construction (a20 / a21 / stack_gate V)', ref['pw30']['st3_bytes'] == vb)
chk('H2 pw30 replay gold58 macro %.5f == .94336 (a21/gate_a21_offline.log, stack/stack_gate.log); pw25 %.5f == .94368; lomm05 %.5f == .94384' % (
    mac(ref['pw30']['out']), mac(ref['pw25']['out']), mac(ref['lomm05']['out'])),
    abs(mac(ref['pw30']['out']) - .94336) < 5e-6 and abs(mac(ref['pw25']['out']) - .94368) < 5e-6 and abs(mac(ref['lomm05']['out']) - .94384) < 5e-6)
for k, r in ref.items():
    chk('H3 %-6s clean path: canary none fired, r224 used, S6F mode S6F, base = frozen file, fired []' % k,
        r['canary']['fired'] == [] and r['canary']['r224_used'] and r['rec'].get('mode') == 'S6F' and r['rec'].get('base_source') == 'submission_v224flip.csv'
        and r['rec'].get('fired') == [])
PW30B, PW30T = ref['pw30']['bytes'], ref['pw30']['txt']

# ---- the stacks ---------------------------------------------------------------------------------------------------------------
COMP_COLS = {'dose': set(OA), 'loamm': {'Lateral OA'}, 'mclmm': {'MCL'}}
OFF = {'dose': ('        A_OA = 0.25   #', '        A_OA = 0.3   #'), 'loamm': ('        A_LOAMM = 0.05   #', '        A_LOAMM = 0.0   #'),
       'mclmm': ('        A_MCLMM = 0.1   #', '        A_MCLMM = 0.0   #')}
out_by = {}
for name, e in IDX.items():
    tag = e['tag']; p = nbp(HERE, tag); rc = json.load(open(HERE + '/build_receipt_%s.json' % tag, encoding='utf-8'))
    nb = json.load(open(p, encoding='utf-8')); b0 = NBJ['pw30']
    P('== %s %s ==' % (name, tag))
    chk('K1 %-9s nb sha %s == build receipt == build_index; base sha == scored pw30 5a903187' % (name, sha(p)[:16]),
        sha(p) == rc['out_nb_sha256'] == e['nb_sha256'] and rc['base_nb_sha256'] == BASE_NB_SHA and sha(nbp(HERE, tag).rsplit('/', 1)[0] + '/kernel-metadata.json') == rc['meta_sha256'])
    chk('K2 %-9s only cell 18 differs from scored pw30 (whole-cell JSON); cell 14 identical -> no stage-3 change' % name,
        [j for j, (x, y) in enumerate(zip(b0['cells'], nb['cells'])) if x != y] == [18] and len(b0['cells']) == len(nb['cells']) and nb['cells'][14] == b0['cells'][14])
    comps = [c for c, on in (('dose', rc['a_oa'] != 0.3), ('loamm', bool(rc['a_loamm'])), ('mclmm', bool(rc['a_mclmm']))) if on]
    if rc['control']:
        r = replay(nb, name)
        chk('C1 %-9s (A_LOAMM 0, A_MCLMM 0 in the code) submission.csv byte-identical to scored pw30; stage 3 identical; no a_loamm / a_mclmm; no LOAMM / MCLMM marker' % name,
            r['bytes'] == PW30B and r['st3_bytes'] == ref['pw30']['st3_bytes'] and 'a_loamm' not in r['rec'] and 'a_mclmm' not in r['rec']
            and 'LOAMM_APPLIED' not in r['stdout'] and 'MCLMM_APPLIED' not in r['stdout'])
        continue
    for on in itertools.chain.from_iterable(itertools.combinations(comps, k) for k in range(len(comps) + 1)):
        on = set(on); subs = [OFF[c] for c in comps if c not in on]
        lab = '%s[%s]' % (name, '+'.join(c for c in comps if c in on) or 'all off')
        r = replay(nb, lab, subs)
        aoa = .25 if 'dose' in on else .3; al = .05 if 'loamm' in on else 0.; am = .1 if 'mclmm' in on else 0.
        want = set().union(*[COMP_COLS[c] for c in on]) if on else set()
        chg = {c for c in L if not (r['txt'][c] == PW30T[c]).all()}
        same_ids = (r['txt']['StudyInstanceUID'] == PW30T['StudyInstanceUID']).all()
        if not on:
            chk('F1 %-30s byte-identical to scored pw30 (sha %s); stage 3 identical; receipt a_oa .3 pw, no a_loamm / a_mclmm' % (lab, shab(r['bytes'])[:12]),
                r['bytes'] == PW30B and r['st3_bytes'] == ref['pw30']['st3_bytes'] and r['rec'].get('a_oa') == .3 and r['rec'].get('oa_mode') == 'pw'
                and 'a_loamm' not in r['rec'] and 'a_mclmm' not in r['rec'])
        chk('F2 %-30s changed columns vs pw30 %s == expected %s; all other columns byte-identical (CSV text); ids identical' % (lab, sorted(chg), sorted(want)),
            chg == want and same_ids and r['st3_bytes'] == ref['pw30']['st3_bytes'])
        FA = final_from(r['st3'][L].values.astype(np.float64), aoa, al, am); FB = final_from(v_members(), aoa, al, am)
        ma = float(np.abs(r['out'][L].values - FA).max()); mb = float(np.abs(r['out'][L].values - FB).max())
        chk('F3 %-30s == numpy recompute, 12 columns: route A (stage 3 + B) maxdiff %.1e, route B (members) maxdiff %.1e' % (lab, ma, mb), ma < 1e-6 and mb < 1e-6)
        so = r['stdout']; mk = [(m.start(), m.group(0)) for m in re.finditer(r'(EOA_APPLIED a_oa [0-9.]+|LOAMM_APPLIED a_loamm [0-9.]+|MCLMM_APPLIED a_mclmm [0-9.]+|S6F_BLEND_OK)', so)]
        seq = [x for _, x in mk]
        wseq = ['EOA_APPLIED a_oa %g' % aoa] + (['LOAMM_APPLIED a_loamm %g' % al] if al else []) + (['MCLMM_APPLIED a_mclmm %g' % am] if am else []) + ['S6F_BLEND_OK']
        rr = r['rec']
        rok = (rr.get('a_oa') == aoa and rr.get('oa_mode') == 'pw' and rr.get('mode') == 'S6F' and rr.get('fired') == []
               and (rr.get('a_loamm') == al if al else 'a_loamm' not in rr) and (rr.get('a_mclmm') == am if am else 'a_mclmm' not in rr))
        chk('F4 %-30s receipt (a_oa %s, a_loamm %s, a_mclmm %s) and stdout markers %s in order' % (lab, rr.get('a_oa'), rr.get('a_loamm'), rr.get('a_mclmm'), seq), rok and seq == wseq)
        if on == set(comps):
            out_by[name] = r
            if name == 'S_safe':
                chk('F5 S_safe as built == stack/stack_gate.log G58 macro .94412', abs(mac(r['out']) - .94412) < 5e-6)

P('== C cross-stack ==')
T = {k: v['txt'] for k, v in out_by.items()}; T.update({k: ref[k]['txt'] for k in ref})
eq = lambda a, b, cols: all((T[a][c] == T[b][c]).all() for c in cols)
others = lambda ex: [c for c in L if c not in ex]
chk('C2 MCL column byte-identical in all 4 stacks (MCL does not depend on A_OA or LOAMM)', all(eq(k, 'S_safe', ['MCL']) for k in ('S_alt', 'S_wide', 'S_mid')))
chk('C3 S_wide == S_alt except Lateral OA; S_mid == S_safe except Lateral OA', eq('S_wide', 'S_alt', others({'Lateral OA'})) and eq('S_mid', 'S_safe', others({'Lateral OA'})))
chk('C4 S_alt == val pw25 (64ed7eba) except MCL; S_safe == pw30 except MCL; S_mid == night lomm05 (c0dd7ec8) except MCL',
    eq('S_alt', 'pw25', others({'MCL'})) and eq('S_safe', 'pw30', others({'MCL'})) and eq('S_mid', 'lomm05', others({'MCL'})))
chk('C5 the MCL change is live: S_safe MCL differs from pw30 in rank order on gold58', (rankdata(T['S_safe']['MCL'].astype(float)) != rankdata(T['pw30']['MCL'].astype(float))).any())

P('== INFO gold58 (n 58; code check only, never a selection ruler): per-label delta vs pw30, macro ==')
b0 = ref['pw30']['out']
for k in ('pw25', 'lomm05') + tuple(out_by):
    o = (ref[k] if k in ref else out_by[k])['out']; d = [auc(o[c].values, Y[:, j]) - auc(b0[c].values, Y[:, j]) for j, c in enumerate(L)]
    P('  %-7s macro %.5f (%+.5f) | %s' % (k, mac(o), np.mean(d), ' '.join('%s %+.4f' % (L[j][:4], d[j]) for j in range(12) if abs(d[j]) > 0)))
P('replay output sha (local CRLF bytes): ' + ' | '.join('%s %s' % (k, shab(v['bytes'])[:12]) for k, v in list(ref.items()) + list(out_by.items())))
P('STACK_KGATE %s' % ('PASS' if ok else 'FAIL'))
LOG.close()
sys.exit(0 if ok else 1)
