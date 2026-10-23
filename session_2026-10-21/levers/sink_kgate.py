#!/usr/bin/env python
# sink_kgate.py (10-21 ~11:05 IST, Claude, SINK track). LOCAL offline gate for levers/kernels/ (pattern stack/kernels/stack_kgate.py:
# EXECUTES each notebook's own cell 14 (stage 3), cell 15 (V224 canary), emulates cell 16's freeze, then cell 18's _s6f_blend, on the
# gold58 serve predictions). No OAI data. Writes only levers/kernels/sink_kgate.log (temp dirs removed).
# Gates (all must hold -> SINK_KGATE PASS):
#   H  harness: each base notebook replays to its known gold58 macro (pw30 .94336, lomm30 .94384, pw25 .94368, spl .94417, swide .94492,
#      salt .94444; stack/kernels/stack_kgate.log, stack/skeptic/kernels/build_spl.log); pw30 stage 3 == numpy members->c4; clean path.
#   K  kernels: nb sha == receipt == build_index; base nb sha == registry; only cell 18 differs from ITS base; cell 14 == scored pw30's.
#   F  factorial per build over its components {dose (A_OA .25 -> .3), loamm (A_LOAMM .05 -> 0), mclmm (A_MCLMM .1 -> 0), cvx (P_OA 2 -> 1)}
#      by exact single-literal substitution in its own cell 18:
#        F1 all off -> submission.csv byte-identical to scored pw30's replay; receipt a_oa .3 pw; no a_loamm / a_mclmm / p_oa; no markers;
#        F1b cvx off, base components on -> submission.csv AND blend receipt byte-identical to the BASE notebook's own replay (the A22 entry);
#        F2 changed columns vs pw30 == union of the ON components' columns; every other column byte-identical as CSV text;
#        F3 all 12 columns == independent numpy recompute (route A from this run's stage 3 + S6fall B; route B from member serve files);
#        F4 receipt keys / values and stdout markers EOA_APPLIED (value) < CVX_APPLIED p_oa 2 < LOAMM_APPLIED < MCLMM_APPLIED < S6F_BLEND_OK.
#   C  control (P_OA 1.0 in the code) byte-identical to pw30 with the pw30 receipt; cvx_pw25 output byte-identical to the independent
#      c4-lineage convex twin levers/kernel_v224r6c4pw25p2j (session_2026-10-20, 9750dfbd); cross-build column identities.
#   INFO gold58 macro / per-label deltas (n 58, never a selection ruler).
import json, os, sys, io, time, hashlib, tempfile, shutil, itertools, contextlib, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
R = 'C:/Users/eeshs/Downloads/rsna-knee'; G = R + '/session_2026-10-18/next/g58'; S20 = R + '/session_2026-10-20'
HERE = os.path.dirname(os.path.abspath(__file__)).replace(chr(92), '/'); KD = HERE + '/kernels'
LOG = open(KD + '/sink_kgate.log', 'w', encoding='utf-8')
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
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
frame = lambda A: pd.DataFrame(A, columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L]
REG = json.load(open(S20 + '/val/slate_a22.json', encoding='utf-8'))['entries']
IDX = json.load(open(KD + '/build_index.json', encoding='utf-8'))
nbf = lambda d: [d + '/' + f for f in os.listdir(d) if f.endswith('.ipynb')][0]
BASEDIR = {'pw30': S20 + '/a21/kernel_v224r6c4pw30j', 'lomm30': S20 + '/night/c12_kernel_v224r6c4pw30lomm05j',
           'pw25': R + '/' + REG['pw25']['kernel_dir'], 'spl': R + '/' + REG['spl']['kernel_dir'], 'swide': R + '/' + REG['swide']['kernel_dir'],
           'salt': R + '/' + REG['salt']['kernel_dir']}
MAC = {'pw30': .94336, 'lomm30': .94384, 'pw25': .94368, 'spl': .94417, 'swide': .94492, 'salt': .94444}
BASECOMP = {'pw30': (.3, 0., 0.), 'lomm30': (.3, .05, 0.), 'pw25': (.25, 0., 0.), 'spl': (.25, .05, 0.), 'swide': (.25, .05, .1), 'salt': (.25, 0., .1)}
TWIN = S20 + '/levers/kernel_v224r6c4pw25p2j/rsna_sub_v224r6c4pw25p2j.ipynb'
ok = True
def chk(name, cond):
    global ok
    ok = ok and bool(cond); P(('OK   ' if cond else 'FAIL ') + name)


def replay(nb, label, c18_sub=None):
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
             out=pd.read_csv(W / 'submission.csv', dtype={'StudyInstanceUID': str}), txt=pd.read_csv(W / 'submission.csv', dtype=str), stdout=so)
    shutil.rmtree(T, ignore_errors=True); return r


C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
PW = {('Medial OA', 'Lateral OA'): .876, ('Medial OA', 'PF OA'): .708, ('Lateral OA', 'PF OA'): 1.416}
def v_members():
    V = np.zeros((n, 12))
    for j, c in enumerate(L):
        w = C4W.get(c, .5); V[:, j] = w * (.75 * pct(R384[:, j]) + .25 * pct(R224[:, j])) + (1 - w) * pct(R6[:, j])
    return V.astype(np.float32).astype(np.float64)
def final_from(V, aoa, al, am, p):
    F = np.stack([.8 * pct(V[:, j]) + .2 * pct(S6[:, j]) for j in range(12)], 1).astype(np.float32).astype(np.float64)
    U = {c: pct(F[:, L.index(c)]) for c in OA}; Sx = {c: U[c] ** p for c in OA}; F2 = F.copy()
    for c in OA:
        o = [k for k in OA if k != c]; F2[:, L.index(c)] = pct(U[c] - aoa / 2.0 * sum(PW[tuple(sorted((c, k), key=OA.index))] * Sx[k] for k in o))
    F = F2.astype(np.float32).astype(np.float64)
    jl, jm, jc = L.index('Lateral OA'), L.index('Medial Meniscus'), L.index('MCL')
    if al:
        F[:, jl] = pct(pct(F[:, jl]) - al * pct(F[:, jm])); F = F.astype(np.float32).astype(np.float64)
    if am:
        F[:, jc] = pct(pct(F[:, jc]) - am * pct(F[:, jm])); F = F.astype(np.float32).astype(np.float64)
    return F


mac = lambda o: np.mean([auc(o[c].values, Y[:, j]) for j, c in enumerate(L)])
P('== H harness: base notebooks ==')
NBJ = {k: json.load(open(nbf(d), encoding='utf-8')) for k, d in BASEDIR.items()}
ref = {k: replay(v, k) for k, v in NBJ.items()}
T0 = tempfile.mkdtemp(); frame(v_members().astype(np.float32)).to_csv(T0 + '/v.csv', index=False); vb = open(T0 + '/v.csv', 'rb').read(); shutil.rmtree(T0)
chk('H1 pw30 cell-14 replay stage 3 byte-identical to the numpy members->c4 construction', ref['pw30']['st3_bytes'] == vb)
chk('H2 base gold58 macros: ' + ' '.join('%s %.5f (%.5f)' % (k, mac(ref[k]['out']), MAC[k]) for k in ref), all(abs(mac(ref[k]['out']) - MAC[k]) < 5e-6 for k in ref))
for k, r in ref.items():
    chk('H3 %-6s clean path (canary none, r224 used, S6F, frozen base, fired []); stage 3 == pw30 stage 3' % k,
        r['canary']['fired'] == [] and r['canary']['r224_used'] and r['rec'].get('mode') == 'S6F' and r['rec'].get('base_source') == 'submission_v224flip.csv'
        and r['rec'].get('fired') == [] and r['st3_bytes'] == ref['pw30']['st3_bytes'])
PW30B, PW30T = ref['pw30']['bytes'], ref['pw30']['txt']
COLS = {'dose': set(OA), 'loamm': {'Lateral OA'}, 'mclmm': {'MCL'}, 'cvx': set(OA)}
OFF = {'dose': ('        A_OA = 0.25   #', '        A_OA = 0.3   #'), 'loamm': ('        A_LOAMM = 0.05   #', '        A_LOAMM = 0.0   #'),
       'mclmm': ('        A_MCLMM = 0.1   #', '        A_MCLMM = 0.0   #'), 'cvx': ('        P_OA = 2.0   #', '        P_OA = 1.0   #')}
out_by = {}
for name, e in IDX.items():
    tag = e['tag']; p = KD + '/kernel_%s/rsna_sub_%s.ipynb' % (tag, tag); rc = json.load(open(KD + '/build_receipt_%s.json' % tag, encoding='utf-8'))
    nb = json.load(open(p, encoding='utf-8')); b0 = NBJ[e['base']]
    P('== %s %s (base %s, role %s) ==' % (name, tag, e['base'], e['role']))
    chk('K1 %-11s nb sha %s == receipt == build_index; base nb sha == %s; meta sha == receipt' % (name, sha(p)[:16], e['base_nb_sha256'][:16]),
        sha(p) == rc['out_nb_sha256'] == e['nb_sha256'] and sha(nbf(BASEDIR[e['base']])) == e['base_nb_sha256'] == rc['base_nb_sha256']
        and sha(KD + '/kernel_%s/kernel-metadata.json' % tag) == rc['meta_sha256'] and (e['base'] not in REG or REG[e['base']]['nb_sha256'] == e['base_nb_sha256']))
    chk('K2 %-11s only cell 18 differs from its base (whole-cell JSON); cell 14 == scored pw30 cell 14' % name,
        [j for j, (x, y) in enumerate(zip(b0['cells'], nb['cells'])) if x != y] == [18] and len(b0['cells']) == len(nb['cells'])
        and nb['cells'][14] == NBJ['pw30']['cells'][14])
    aoa0, al0, am0 = BASECOMP[e['base']]
    comps = [c for c, on in (('dose', aoa0 != .3), ('loamm', bool(al0)), ('mclmm', bool(am0)), ('cvx', e['p_oa'] != 1.0)) if on]
    if e['p_oa'] == 1.0:
        r = replay(nb, name)
        chk('C1 %-11s (P_OA 1.0 in the code) submission.csv byte-identical to scored pw30; blend receipt == pw30 receipt; no CVX marker' % name,
            r['bytes'] == PW30B and r['rec'] == ref['pw30']['rec'] and 'CVX_APPLIED' not in r['stdout'])
        continue
    for on in itertools.chain.from_iterable(itertools.combinations(comps, k) for k in range(len(comps) + 1)):
        on = set(on); subs = [OFF[c] for c in comps if c not in on]
        lab = '%s[%s]' % (name, '+'.join(c for c in comps if c in on) or 'all off')
        r = replay(nb, lab, subs)
        aoa = .25 if 'dose' in on else .3; al = .05 if 'loamm' in on else 0.; am = .1 if 'mclmm' in on else 0.; pp = 2. if 'cvx' in on else 1.
        want = set().union(*[COLS[c] for c in on]) if on else set()
        chg = {c for c in L if not (r['txt'][c] == PW30T[c]).all()}
        if not on:
            chk('F1  %-36s byte-identical to scored pw30 (%s); receipt a_oa .3 pw, no a_loamm / a_mclmm / p_oa' % (lab, shab(r['bytes'])[:12]),
                r['bytes'] == PW30B and r['rec'].get('a_oa') == .3 and r['rec'].get('oa_mode') == 'pw' and not ({'a_loamm', 'a_mclmm', 'p_oa'} & set(r['rec'])))
        if on == set(comps) - {'cvx'}:
            chk('F1b %-36s == BASE %s replay: submission.csv bytes and blend receipt identical (the build is the base + CVX only)' % (lab, e['base']),
                r['bytes'] == ref[e['base']]['bytes'] and r['rec'] == ref[e['base']]['rec'])
        chk('F2  %-36s changed columns vs pw30 %s == expected %s; others byte-identical; stage 3 identical' % (lab, sorted(chg), sorted(want)),
            chg == want and (r['txt']['StudyInstanceUID'] == PW30T['StudyInstanceUID']).all() and r['st3_bytes'] == ref['pw30']['st3_bytes'])
        FA = final_from(r['st3'][L].values.astype(np.float64), aoa, al, am, pp); FB = final_from(v_members(), aoa, al, am, pp)
        ma = float(np.abs(r['out'][L].values - FA).max()); mb = float(np.abs(r['out'][L].values - FB).max())
        chk('F3  %-36s == numpy recompute: route A maxdiff %.1e, route B maxdiff %.1e' % (lab, ma, mb), ma < 1e-6 and mb < 1e-6)
        so = r['stdout']
        seq = [m.group(0) for m in re.finditer(r'(EOA_APPLIED a_oa [0-9.]+|CVX_APPLIED p_oa [0-9.]+|LOAMM_APPLIED a_loamm [0-9.]+|MCLMM_APPLIED a_mclmm [0-9.]+|S6F_BLEND_OK)', so)]
        wseq = (['EOA_APPLIED a_oa %g' % aoa] + (['CVX_APPLIED p_oa %g' % pp] if pp != 1 else []) + (['LOAMM_APPLIED a_loamm %g' % al] if al else [])
                + (['MCLMM_APPLIED a_mclmm %g' % am] if am else []) + ['S6F_BLEND_OK'])
        rr = r['rec']
        rok = (rr.get('a_oa') == aoa and rr.get('oa_mode') == 'pw' and rr.get('mode') == 'S6F' and rr.get('fired') == []
               and (rr.get('a_loamm') == al if al else 'a_loamm' not in rr) and (rr.get('a_mclmm') == am if am else 'a_mclmm' not in rr)
               and (rr.get('p_oa') == pp if pp != 1 else 'p_oa' not in rr))
        chk('F4  %-36s receipt (a_oa %s, p_oa %s, a_loamm %s, a_mclmm %s) + markers %s' % (lab, rr.get('a_oa'), rr.get('p_oa'), rr.get('a_loamm'), rr.get('a_mclmm'), seq),
            rok and seq == wseq)
        if on == set(comps):
            out_by[name] = r

P('== C cross-build ==')
tw = replay(json.load(open(TWIN, encoding='utf-8')), 'levers_pw25p2j')
chk('C2 cvx_pw25 submission.csv byte-identical to the independent c4-lineage convex twin levers/kernel_v224r6c4pw25p2j (sha %s)' % sha(TWIN)[:8],
    out_by['cvx_pw25']['bytes'] == tw['bytes'])
T = {k: v['txt'] for k, v in out_by.items()}; T.update({'b_' + k: ref[k]['txt'] for k in ref})
eq = lambda a, b, cols: all((T[a][c] == T[b][c]).all() for c in cols)
others = lambda ex: [c for c in L if c not in ex]
chk('C3 cvx_spl == cvx_pw25 except LOA; cvx_swide == cvx_salt except LOA; cvx_swide == cvx_spl except MCL; cvx_lomm30 == cvx_pw30 except LOA',
    eq('cvx_spl', 'cvx_pw25', others({'Lateral OA'})) and eq('cvx_swide', 'cvx_salt', others({'Lateral OA'})) and eq('cvx_swide', 'cvx_spl', others({'MCL'}))
    and eq('cvx_lomm30', 'cvx_pw30', others({'Lateral OA'})))
chk('C4 every build == its base except the OA trio (MCL / non-OA columns untouched by CVX)',
    all(eq(k, 'b_' + IDX[k]['base'], others(set(OA))) for k in out_by))
chk('C5 the CVX change is live: every build differs from its base in OA rank order on gold58',
    all(any((rankdata(T[k][c].astype(float)) != rankdata(T['b_' + IDX[k]['base']][c].astype(float))).any() for c in OA) for k in out_by))
P('== INFO gold58 (n 58; code check only, never a selection ruler): macro, delta vs its base, delta vs pw30 ==')
for k, r in out_by.items():
    b = ref[IDX[k]['base']]['out']; o = r['out']
    d = [auc(o[c].values, Y[:, j]) - auc(b[c].values, Y[:, j]) for j, c in enumerate(L)]
    P('  %-11s macro %.5f | vs base %-6s %+.5f | vs pw30 %+.5f | %s' % (k, mac(o), IDX[k]['base'], np.mean(d), mac(o) - MAC['pw30'],
      ' '.join('%s %+.4f' % (L[j][:4], d[j]) for j in range(12) if abs(d[j]) > 0)))
P('replay output sha (local CRLF): ' + ' | '.join('%s %s' % (k, shab(v['bytes'])[:12]) for k, v in out_by.items()))
P('SINK_KGATE %s' % ('PASS' if ok else 'FAIL'))
LOG.close()
sys.exit(0 if ok else 1)
