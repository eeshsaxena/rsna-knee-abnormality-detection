#!/usr/bin/env python
# build_spl.py (10-21 ~00:20 IST, Claude, STACK SKEPTIC round). LOCAL BUILD ONLY, UNARMED: nothing pushed, committed or registered.
# S_pl = pw25 + lomm05 WITHOUT MCLmm = the S_wide notebook (stack/kernels, nb c9801a80) with ONE literal changed in cell 18:
#   '        A_MCLMM = 0.1   #'  ->  '        A_MCLMM = 0.0   #'     (the exact substitution stack_kgate.py used for S_wide[dose+loamm])
# so the MCLMM block is present but a no-op (no receipt key, no marker), like the builder's CTRL_off kernel.
# Steps (all asserted; any failure exits 1 and prints SPL_BUILD FAIL):
#   B  build: bytes-level edit of the S_wide notebook (1 occurrence), metadata = S_wide's with the tag swapped (3 occurrences);
#      whole-cell JSON: only cell 18 differs from S_wide, only cell 18 differs from scored pw30, cell 14 identical.
#   O  offline: EXECUTES cells 14 + 15 + (16 freeze) + 18 of the new notebook on gold58 with stack_kgate.py's own replay() (its
#      definitions exec'd with the log redirected here): output bytes == S_wide replayed with A_MCLMM -> 0.0 == numpy recompute
#      final_from(V, .25, .05, 0) (12 columns); == val pw25 except Lateral OA; LOA == pw25 LOA + C12' line; changed columns vs pw30 ==
#      OA trio; receipt a_oa .25 pw, a_loamm .05, no a_mclmm; markers EOA (.25) < LOAMM (.05) < S6F_BLEND_OK, no MCLMM marker.
#   S  commit-gate sim: cell 18 replayed on the REAL scored pw30 committed inputs (a21/out_v224r6c4pw30j_v1, read-only, copied), the
#      same way stack_ksim.py does (LF line endings, replayed stdout spliced into the real log), then stack/kernels/stack_commit_gate.py:
#      POS own flags (.25 / .05 / 0) -> PASS 26/26; NEG (.25 / .05 / .1), (.25 / 0 / 0), (.3 / .05 / 0), builder S_wide sim with S_pl
#      flags, REAL pw30 dir with S_pl flags -> FAIL.
# Writes only under stack/skeptic/kernels/. No OAI data.
import json, os, sys, io, shutil, tempfile, subprocess, contextlib, hashlib, time
import numpy as np, pandas as pd
os.linesep = '\n'
R = 'C:/Users/eeshs/Downloads/rsna-knee'; S = R + '/session_2026-10-20'; K = S + '/stack/kernels'; HERE = S + '/stack/skeptic/kernels'
LOGF = open(HERE + '/build_spl.log', 'w', encoding='utf-8')
def P(s=''):
    print(s, flush=True); LOGF.write(s + '\n'); LOGF.flush()
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest(); shab = lambda b: hashlib.sha256(b).hexdigest()
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
OK = True
def chk(name, cond):
    global OK
    OK = OK and bool(cond); P(('OK   ' if cond else 'FAIL ') + name)

TAG_W, TAG = 'v224r6c4pw25lomm05mcl10j', 'v224r6c4pw25lomm05mcl00j'
NB_W = K + '/kernel_%s/rsna_sub_%s.ipynb' % (TAG_W, TAG_W); MT_W = K + '/kernel_%s/kernel-metadata.json' % TAG_W
SHA_W, MSHA_W = 'c9801a80bbc64ecbd1aba15e9470657fca4631ee7ac091f68b7742d1e17d7478', '6a2745e7f68841c50f200bdc40e1259e95452237163885c67ed73473ec7f8d1e'
BASE_NB = S + '/a21/kernel_v224r6c4pw30j/rsna_sub_v224r6c4pw30j.ipynb'; BASE_SHA = '5a903187415505e9e3566ebd749cae1328b20df6b60bd6593e6443b17fc0c9fd'
OLD, NEW = '        A_MCLMM = 0.1   #', '        A_MCLMM = 0.0   #'

# ---------------- B build ----------------
P('== B build ==')
chk('B0 sources: S_wide nb sha %s / meta sha %s (stack/kernels build) ; scored pw30 nb sha %s' % (SHA_W[:16], MSHA_W[:16], BASE_SHA[:16]),
    sha(NB_W) == SHA_W and sha(MT_W) == MSHA_W and sha(BASE_NB) == BASE_SHA)
rawW = open(NB_W, 'rb').read(); metW = open(MT_W, 'rb').read()
chk('B1 the literal %r occurs exactly once in the S_wide notebook bytes; the tag occurs 0x in the nb and 3x in the metadata' % OLD.strip(),
    rawW.count(OLD.encode()) == 1 and rawW.count(TAG_W.encode()) == 0 and metW.count(TAG_W.encode()) == 3)
rawN = rawW.replace(OLD.encode(), NEW.encode()); metN = metW.replace(TAG_W.encode(), TAG.encode())
D = HERE + '/kernel_' + TAG; os.makedirs(D, exist_ok=True)
NB_N, MT_N = D + '/rsna_sub_%s.ipynb' % TAG, D + '/kernel-metadata.json'
open(NB_N, 'wb').write(rawN); open(MT_N, 'wb').write(metN)
open(D + '/NOT_ARMED.txt', 'w').write('Skeptic-round build of S_pl (pw25 + lomm05, MCLmm block present at dose 0). Needs Eesh GO + PREREG amendment + '
                                      'stack/kernels/stack_commit_gate.py PASS 26/26 with --aoa 0.25 --aloamm 0.05 --amclmm 0.0 before any push.\n')
nbW, nbN, nb0 = (json.load(open(p, encoding='utf-8')) for p in (NB_W, NB_N, BASE_NB))
diffW = [i for i in range(len(nbW['cells'])) if nbW['cells'][i] != nbN['cells'][i]]
diff0 = [i for i in range(len(nb0['cells'])) if nb0['cells'][i] != nbN['cells'][i]]
c18 = src(nbN['cells'][18]); c18W = src(nbW['cells'][18])
chk('B2 whole-cell JSON: vs S_wide only cell %s differs; vs scored pw30 only cell %s differs; cell 14 identical to pw30; same cell count' % (diffW, diff0),
    diffW == [18] and diff0 == [18] and nbN['cells'][14] == nb0['cells'][14] and len(nbN['cells']) == len(nb0['cells']) == len(nbW['cells']))
chk('B3 cell 18 == S_wide cell 18 with exactly the one substitution; cell-18 outputs / metadata identical; notebook metadata identical',
    c18 == c18W.replace(OLD, NEW) and c18W.count(OLD) == 1 and {k: v for k, v in nbN['cells'][18].items() if k != 'source'} == {k: v for k, v in nbW['cells'][18].items() if k != 'source'}
    and nbN['metadata'] == nbW['metadata'])
mN, mW = json.loads(metN), json.loads(metW)
chk('B4 metadata == S_wide metadata except id / title / code_file (-> %s)' % TAG,
    {k: v for k, v in mN.items() if k not in ('id', 'title', 'code_file')} == {k: v for k, v in mW.items() if k not in ('id', 'title', 'code_file')}
    and mN['id'] == 'eeshsaxena/rsna-sub-' + TAG and mN['code_file'] == 'rsna_sub_%s.ipynb' % TAG)
NB_SHA, META_SHA = sha(NB_N), sha(MT_N)
P('     built %s: nb %s | meta %s' % (TAG, NB_SHA, META_SHA))

# ---------------- O offline gate on gold58 (stack_kgate.py's own replay / recompute, definitions only) ----------------
P('== O offline (gold58, notebook cells executed) ==')
kg = open(K + '/stack_kgate.py', encoding='utf-8').read().split("P('== H harness ==')")[0]
kg = kg.replace("LOG = open(HERE + '/stack_kgate.log', 'w', encoding='utf-8')", "LOG = open(%r, 'w', encoding='utf-8')" % (HERE + '/build_spl_kgate_defs.log'))
kg = kg.replace("HERE = os.path.dirname(os.path.abspath(__file__)).replace(chr(92), '/')", "HERE = %r" % K)
G = {}; exec(compile(kg, 'stack_kgate_defs', 'exec'), G)
replay, final_from, v_members, L, OA, auc, Y = G['replay'], G['final_from'], G['v_members'], G['L'], G['OA'], G['auc'], G['Y']
NB_PW25 = S + '/val/kernels/kernel_v224r6c4pw25j/rsna_sub_v224r6c4pw25j.ipynb'
r30 = replay(nb0, 'pw30'); rN = replay(nbN, 'S_pl'); rWoff = replay(nbW, 'S_wide[mclmm off]', [(OLD, NEW)]); r25 = replay(json.load(open(NB_PW25, encoding='utf-8')), 'pw25')
mac = lambda o: np.mean([auc(o[c].values, Y[:, j]) for j, c in enumerate(L)])
chk('O1 pw30 replay gold58 macro %.5f == .94336 (harness, as stack_kgate H2)' % mac(r30['out']), abs(mac(r30['out']) - .94336) < 5e-6)
chk('O2 S_pl stage 3 == pw30 stage 3 bytes; S_pl submission.csv bytes == S_wide replayed with A_MCLMM -> 0.0 (stack_kgate F S_wide[dose+loamm])',
    rN['st3_bytes'] == r30['st3_bytes'] and rN['bytes'] == rWoff['bytes'])
F = final_from(v_members(), .25, .05, 0.)
md = max(float(np.abs(rN['out'][c].values - F[:, j]).max()) for j, c in enumerate(L))
chk('O3 all 12 columns == numpy recompute final_from(V, .25, .05, 0) (no notebook code), maxdiff %.1e < 1e-6' % md, md < 1e-6)
chg = sorted(c for c in L if not rN['txt'][c].equals(r30['txt'][c])); chg25 = sorted(c for c in L if not rN['txt'][c].equals(r25['txt'][c]))
chk('O4 changed columns vs pw30 %s == OA trio; vs val pw25 %s == [Lateral OA]; ids identical' % (chg, chg25),
    chg == sorted(OA) and chg25 == ['Lateral OA'] and rN['txt']['StudyInstanceUID'].equals(r30['txt']['StudyInstanceUID']))
rec = rN['rec']; so = [l for l in rN['stdout'].splitlines() if any(m in l for m in ('EOA_APPLIED', 'LOAMM_APPLIED', 'MCLMM_APPLIED', 'S6F_BLEND_OK'))]
chk('O5 receipt a_oa %s mode %s a_loamm %s, no a_mclmm key; markers %s' % (rec.get('a_oa'), rec.get('oa_mode'), rec.get('a_loamm'), [s[:28] for s in so]),
    rec.get('a_oa') == .25 and rec.get('oa_mode') == 'pw' and rec.get('a_loamm') == .05 and 'a_mclmm' not in rec
    and len(so) == 3 and so[0].startswith('EOA_APPLIED a_oa 0.25') and so[1].startswith('LOAMM_APPLIED a_loamm 0.05') and so[2].startswith('S6F_BLEND_OK'))
P('INFO gold58 macro (n 58, never a ruler): pw30 %.5f | pw25 %.5f | S_pl %.5f (%+.5f vs pw30)' % (mac(r30['out']), mac(r25['out']), mac(rN['out']), mac(rN['out']) - mac(r30['out'])))

# ---------------- S commit-gate sim on the REAL scored pw30 committed inputs (logic of stack/kernels/stack_ksim.py) ----------------
P('== S commit-gate sim ==')
PW30 = S + '/a21/out_v224r6c4pw30j_v1'
REAL_LOG = [p for p in os.listdir(PW30) if p.endswith('.log') and 'cufile' not in p]; assert len(REAL_LOG) == 1; REAL_LOG = REAL_LOG[0]
L0 = json.load(open(PW30 + '/' + REAL_LOG, encoding='utf-8')); RC0 = json.load(open(PW30 + '/s6f_receipt.json', encoding='utf-8'))
def sreplay(nbd):
    s_ = src(nbd['cells'][18]); fn = s_.split('\n\ntry:\n    _S6F_BLEND = ')[0]; assert fn.count("W = '/kaggle/working'") == 1
    W = tempfile.mkdtemp().replace(chr(92), '/'); fn = fn.replace("W = '/kaggle/working'", "W = %r" % W)
    shutil.copyfile(PW30 + '/submission_v224flip.csv', W + '/submission_v224flip.csv'); os.makedirs(W + '/s6f/B')
    shutil.copyfile(PW30 + '/s6f/B/B_probs.csv', W + '/s6f/B/B_probs.csv')
    buf = io.StringIO(); ns = {}
    with contextlib.redirect_stdout(buf):
        exec(fn, ns); r_ = ns['_s6f_blend'](json.loads(json.dumps(RC0['setup'])), json.loads(json.dumps(RC0['run'])))
    b = open(W + '/submission.csv', 'rb').read(); shutil.rmtree(W, ignore_errors=True)
    return r_, b, [l + '\n' for l in buf.getvalue().splitlines()]
i_e = [i for i, e in enumerate(L0) if 'EOA_APPLIED' in e.get('data', '')]; i_b = [i for i, e in enumerate(L0) if 'S6F_BLEND_OK' in e.get('data', '')]
assert len(i_e) == 1 and len(i_b) == 1 and i_b[0] == i_e[0] + 1
real = open(PW30 + '/submission.csv', 'rb').read()
h = sreplay(nb0)
chk('S0 harness: pw30 cell 18 replayed on its REAL committed inputs == REAL submission.csv bytes, receipt sha, log lines, blend receipt',
    h[1] == real and h[0].get('submission_sha256') == RC0['blend']['submission_sha256'] == shab(real) and h[2] == [L0[i_e[0]]['data'], L0[i_b[0]]['data']] and h[0] == RC0['blend'])
recS, outS, soS = sreplay(nbN)
SIMD = HERE + '/sim/out_%s_sim' % TAG; shutil.rmtree(SIMD, ignore_errors=True)
shutil.copytree(PW30, SIMD, ignore=shutil.ignore_patterns('__pycache__'))
open(SIMD + '/submission.csv', 'wb').write(outS)
rc = json.loads(json.dumps(RC0)); rc['blend'] = recS; rc['mode'] = recS.get('mode'); rc['fired'] = list(recS.get('fired') or [])
rc['SIMULATED'] = 'stack/skeptic/kernels/build_spl.py: committed inputs of a21/out_v224r6c4pw30j_v1 + cell 18 of %s replayed locally; NOT a Kaggle run' % TAG
json.dump(rc, open(SIMD + '/s6f_receipt.json', 'w', encoding='utf-8'), indent=1, default=str)
t0 = L0[i_e[0]]['time']
L_ = L0[:i_e[0]] + [{'stream_name': 'stdout', 'time': t0 + 1e-6 * k, 'data': line, 'simulated': True} for k, line in enumerate(soS)] + L0[i_b[0] + 1:]
json.dump(L_, open(SIMD + '/' + REAL_LOG, 'w', encoding='utf-8'))
chk('S1 sim written; 3-row submission.csv byte-identical to the REAL pw30 one (sha %s): the visible rows cannot show S_pl either' % shab(real)[:8], outS == real)
T = '{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker\'s": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}'
Pa = lambda aoa, al, am: ['--a224', '0.25', '--table', T, '--aoa', str(aoa), '--mode', 'pw', '--aloamm', str(al), '--amclmm', str(am)]
gate = ['py', '-3.12', '-I', K + '/stack_commit_gate.py']
ROWS = [('POS S_pl sim, own flags (.25 / .05 / 0)', gate + [SIMD] + Pa(0.25, 0.05, 0.0), 'PASS 26/26'),
        ('NEG S_pl sim checked as S_wide (amclmm .1)', gate + [SIMD] + Pa(0.25, 0.05, 0.1), 'FAIL'),
        ('NEG S_pl sim checked as val pw25 (aloamm 0)', gate + [SIMD] + Pa(0.25, 0.0, 0.0), 'FAIL'),
        ('NEG S_pl sim checked as night lomm05 (aoa .3)', gate + [SIMD] + Pa(0.3, 0.05, 0.0), 'FAIL'),
        ('NEG builder S_wide sim checked with S_pl flags', gate + [K + '/sim/out_%s_sim' % TAG_W] + Pa(0.25, 0.05, 0.0), 'FAIL'),
        ('NEG REAL scored pw30 dir checked with S_pl flags', gate + [PW30] + Pa(0.25, 0.05, 0.0), 'FAIL')]
for name, cmd, want in ROWS:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    last = [l for l in r.stdout.splitlines() if l.startswith('PROBE_CHECK')]; last = last[-1] if last else 'NO PROBE_CHECK LINE ' + r.stderr[-200:]
    fails = [l.strip()[5:45] for l in r.stdout.splitlines() if l.strip().startswith('FAIL')]
    good = (want in last and r.returncode == 0) if want.startswith('PASS') else (last.startswith('PROBE_CHECK FAIL') and r.returncode == 1)
    chk('%-50s want %-10s got %-24s %s' % (name, want, last, ('| ' + ' ; '.join(fails)) if fails else ''), good)

# ---------------- receipt + registry entry ----------------
json.dump(dict(probe='stack-skeptic', stack='S_pl', control=False, tag=TAG, slug='eeshsaxena/rsna-sub-' + TAG, base=K + '/kernel_' + TAG_W,
               base_nb_sha256=SHA_W, base_meta_sha256=MSHA_W, scored_root='a21/kernel_v224r6c4pw30j (pw30 57425970)', scored_root_nb_sha256=BASE_SHA,
               a_oa=.25, oa_mode='pw', a_loamm=.05, a_mclmm=0.0, components=['pw25 (A_OA .3 -> .25)', 'lomm05 (LOA <- MM a .05)', 'MCLmm block present, dose 0 (no-op)'],
               substitution=[OLD, NEW], out_nb_sha256=NB_SHA, meta_sha256=META_SHA, cells_changed_vs_scored_pw30=[18],
               offline_ref='stack/kernels/stack_kgate.log rows F2/F3/F4 S_wide[dose+loamm] + build_spl.log O1-O5', sim_dir=SIMD.replace(R + '/', ''),
               builder_sha256=sha(os.path.abspath(__file__)), built_at_ist=time.strftime('%Y-%m-%d %H:%M:%S'), armed=False,
               note='local build only; needs Eesh GO + PREREG amendment + commit gate PASS 26/26 before any push'),
          open(HERE + '/build_receipt_%s.json' % TAG, 'w', encoding='utf-8'), indent=1)
entry = {'spl': dict(tag=TAG, slug='eeshsaxena/rsna-sub-' + TAG, kernel_dir='session_2026-10-20/stack/skeptic/kernels/kernel_' + TAG, nb_sha256=NB_SHA,
                     meta_sha256=META_SHA, base='c4', base_scored_nb_sha256=BASE_SHA, a_oa=.25, oa_mode='pw', a_loamm=.05, a_mclmm=0.0,
                     gate_script='session_2026-10-20/stack/kernels/stack_commit_gate.py', gate_args=Pa(0.25, 0.05, 0.0), gate_pass='PASS 26/26',
                     sim_dir='session_2026-10-20/stack/skeptic/kernels/sim/out_%s_sim' % TAG,
                     desc='STACK-SKEPTIC S_pl: pw30 with A_OA .25 + Lateral OA <- MM a .05 (lomm05); MCLmm block at dose 0 (cell 18)')}
json.dump({'note': 'Skeptic-round registry entry (10-21 ~00:30 IST). NOT merged anywhere. Key has no underscore (GO-parse typo guard). Merge only on Eesh GO.',
           'entries': entry}, open(HERE + '/registry_entry_spl.json', 'w', encoding='utf-8'), indent=1)
P('SPL_BUILD %s | %s nb %s meta %s' % ('PASS' if OK else 'FAIL', TAG, NB_SHA[:16], META_SHA[:16]))
LOGF.close(); sys.exit(0 if OK else 1)
