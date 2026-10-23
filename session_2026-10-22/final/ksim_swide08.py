#!/usr/bin/env python
# ksim_swide08.py (10-22 ~09:15 IST, Claude, PUBLIC-PUSH track). LOCAL ONLY. Commit-gate self-test for swide08 (kernels_pub/kernel_v224r6c4pw25lomm08mcl10j)
# on the REAL swide committed output (session_2026-10-20/val/a22/out_v224r6c4pw25lomm05mcl10j, read-only, copied), stack_ksim.py pattern:
# each notebook's own cell 18 is replayed on that dir's submission_v224flip.csv / s6f/B/B_probs.csv / setup + run dicts with os.linesep '\n'
# (Kaggle is Linux). Harness first: the swide replay reproduces the REAL submission.csv bytes, the REAL receipt's submission_sha256, the
# REAL log marker lines and the REAL blend receipt. Then the swide08 sim (flagged SIMULATED) + tampers are run through the UNCHANGED gate
# session_2026-10-20/stack/kernels/stack_commit_gate.py (26 checks). Writes only under session_2026-10-22/final/kernels_pub/.
import json, os, sys, io, shutil, tempfile, subprocess, contextlib, hashlib
import pandas as pd
os.linesep = '\n'
R = 'C:/Users/eeshs/Downloads/rsna-knee'
HERE = os.path.dirname(os.path.abspath(__file__)).replace(chr(92), '/'); KD = HERE + '/kernels_pub'
REAL = R + '/session_2026-10-20/val/a22/out_v224r6c4pw25lomm05mcl10j'
GATE = R + '/session_2026-10-20/stack/kernels/stack_commit_gate.py'
NB = {'swide': R + '/session_2026-10-20/stack/kernels/kernel_v224r6c4pw25lomm05mcl10j/rsna_sub_v224r6c4pw25lomm05mcl10j.ipynb',
      'swide08': KD + '/kernel_v224r6c4pw25lomm08mcl10j/rsna_sub_v224r6c4pw25lomm08mcl10j.ipynb'}
SHA = {'swide': 'c9801a80bbc64ecbd1aba15e9470657fca4631ee7ac091f68b7742d1e17d7478'}
LOGF = open(KD + '/ksim_swide08.log', 'w', encoding='utf-8')


def P(s=''):
    print(s, flush=True); LOGF.write(s + '\n')


shab = lambda b: hashlib.sha256(b).hexdigest()
rc08 = json.load(open(KD + '/build_receipt_v224r6c4pw25lomm08mcl10j.json', encoding='utf-8'))
assert shab(open(NB['swide'], 'rb').read()) == SHA['swide'] and shab(open(NB['swide08'], 'rb').read()) == rc08['out_nb_sha256']
REAL_LOG = [p for p in os.listdir(REAL) if p.endswith('.log') and 'cufile' not in p]; assert len(REAL_LOG) == 1; REAL_LOG = REAL_LOG[0]
L0 = json.load(open(REAL + '/' + REAL_LOG, encoding='utf-8'))
RC0 = json.load(open(REAL + '/s6f_receipt.json', encoding='utf-8'))


def replay(p):
    nb = json.load(open(p, encoding='utf-8')); s_ = nb['cells'][18]['source']; s_ = ''.join(s_) if isinstance(s_, list) else s_
    fn = s_.split('\n\ntry:\n    _S6F_BLEND = ')[0]; assert fn.count("W = '/kaggle/working'") == 1
    W = tempfile.mkdtemp().replace(chr(92), '/'); fn = fn.replace("W = '/kaggle/working'", "W = %r" % W)
    shutil.copyfile(REAL + '/submission_v224flip.csv', W + '/submission_v224flip.csv'); os.makedirs(W + '/s6f/B')
    shutil.copyfile(REAL + '/s6f/B/B_probs.csv', W + '/s6f/B/B_probs.csv')
    buf = io.StringIO(); ns = {}
    with contextlib.redirect_stdout(buf):
        exec(fn, ns); rec = ns['_s6f_blend'](json.loads(json.dumps(RC0['setup'])), json.loads(json.dumps(RC0['run'])))
    b = open(W + '/submission.csv', 'rb').read(); shutil.rmtree(W, ignore_errors=True)
    return rec, b, [l + '\n' for l in buf.getvalue().splitlines()]


RP = {k: replay(p) for k, p in NB.items()}
real = open(REAL + '/submission.csv', 'rb').read()
i_e = [i for i, e in enumerate(L0) if 'EOA_APPLIED' in e.get('data', '')]; i_b = [i for i, e in enumerate(L0) if 'S6F_BLEND_OK' in e.get('data', '')]
assert len(i_e) == 1 and len(i_b) == 1 and i_b[0] == i_e[0] + 3, (i_e, i_b)
h1 = RP['swide'][1] == real; h2 = RP['swide'][0].get('submission_sha256') == RC0['blend']['submission_sha256'] == shab(real)
h3 = RP['swide'][2] == [L0[i]['data'] for i in range(i_e[0], i_b[0] + 1)]
h4 = dict(RP['swide'][0]) == RC0['blend']
HARNESS = h1 and h2 and h3 and h4
P('HARNESS swide replay on the REAL committed inputs (%s): submission.csv bytes == REAL %s | kernel-computed submission_sha256 == REAL receipt %s | '
  'stdout lines == REAL log lines %s | blend receipt == REAL blend receipt %s' % (REAL.replace(R + '/', ''), h1, h2, h3, h4))
for k, (rec, b, so) in RP.items():
    P('  replay %-8s out sha %s | 3-row output == swide REAL: %s | receipt a_loamm %s a_mclmm %s | markers %s' % (
        k, shab(b)[:12], b == real, rec.get('a_loamm'), rec.get('a_mclmm'), [l.strip()[:34] for l in so if 'APPLIED' in l]))


def write_sim(tag, rec, outb, so, d):
    shutil.rmtree(d, ignore_errors=True); shutil.copytree(REAL, d, ignore=shutil.ignore_patterns('__pycache__'))
    open(d + '/submission.csv', 'wb').write(outb)
    rc = json.loads(json.dumps(RC0)); rc['blend'] = rec; rc['mode'] = rec.get('mode'); rc['fired'] = list(rec.get('fired') or [])
    rc['SIMULATED'] = 'session_2026-10-22/final/ksim_swide08.py: committed inputs of val/a22/out_v224r6c4pw25lomm05mcl10j + cell 18 of %s replayed locally; NOT a Kaggle run' % tag
    json.dump(rc, open(d + '/s6f_receipt.json', 'w', encoding='utf-8'), indent=1, default=str)
    t0 = L0[i_e[0]]['time']
    new = [{'stream_name': 'stdout', 'time': t0 + 1e-6 * k, 'data': line, 'simulated': True} for k, line in enumerate(so)]
    json.dump(L0[:i_e[0]] + new + L0[i_b[0] + 1:], open(d + '/' + REAL_LOG, 'w', encoding='utf-8'))
    return d


SIMD = KD + '/sim'; os.makedirs(SIMD, exist_ok=True)
SIM = write_sim('v224r6c4pw25lomm08mcl10j', *RP['swide08'], SIMD + '/out_v224r6c4pw25lomm08mcl10j_sim')
P('SIM swide08 %s | receipt a_oa %s a_loamm %s a_mclmm %s' % (SIM.replace(R + '/', ''), RP['swide08'][0].get('a_oa'), RP['swide08'][0].get('a_loamm'), RP['swide08'][0].get('a_mclmm')))


def tamper(name, fn):
    d = SIMD + '/neg_' + name; shutil.rmtree(d, ignore_errors=True); shutil.copytree(SIM, d); fn(d); return d


def _log_edit(d, f):
    L_ = json.load(open(d + '/' + REAL_LOG, encoding='utf-8')); L_ = f(L_); json.dump(L_, open(d + '/' + REAL_LOG, 'w', encoding='utf-8'))


def t_loaswap(d):
    s = pd.read_csv(d + '/submission.csv', dtype=str); v = s['Lateral OA'].values.copy(); v[[0, 2]] = v[[2, 0]]; s['Lateral OA'] = v
    open(d + '/submission.csv', 'wb').write(s.to_csv(index=False).encode())


def t_logvalue(d):          # receipt says .08 but the log says .05
    _log_edit(d, lambda L_: [dict(e, data=e['data'].replace('LOAMM_APPLIED a_loamm 0.08', 'LOAMM_APPLIED a_loamm 0.05')) for e in L_])


def t_dupmarker(d):         # a second LOAMM marker with a conflicting value
    _log_edit(d, lambda L_: L_ + [{'stream_name': 'stdout', 'time': 0, 'data': 'LOAMM_APPLIED a_loamm 0.05\n', 'simulated': True}])


def t_order(d):             # LOAMM marker moved before EOA
    def f(L_):
        k = [i for i, e in enumerate(L_) if 'LOAMM_APPLIED' in e['data']][0]; e = L_.pop(k)
        j = [i for i, e2 in enumerate(L_) if 'EOA_APPLIED' in e2['data']][0]; L_.insert(j, e); return L_
    _log_edit(d, f)


def t_extrakey(d):
    rc = json.load(open(d + '/s6f_receipt.json', encoding='utf-8')); rc['blend']['a_moacf'] = [0.05, 0.05]
    json.dump(rc, open(d + '/s6f_receipt.json', 'w', encoding='utf-8'), indent=1)


def t_dupsame(d):           # benign: every log entry from E-OA to S6F_DONE repeated once
    def f(L_):
        i0 = [i for i, e in enumerate(L_) if 'EOA_APPLIED' in e['data']][0]; i1 = [i for i, e in enumerate(L_) if 'S6F_DONE' in e['data']][0]
        out = L_[:i0]
        for e in L_[i0:i1 + 1]:
            out += [e, dict(e)]
        return out + L_[i1 + 1:]
    _log_edit(d, f)


NEG = {k: tamper(k, f) for k, f in (('loaswap', t_loaswap), ('logvalue', t_logvalue), ('dupmarker', t_dupmarker), ('order', t_order),
                                     ('extrakey', t_extrakey), ('dupsame', t_dupsame))}
T = '{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker\'s": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}'
gate = ['py', '-3.12', '-I', GATE]
Pa = lambda al, am=0.1, aoa=0.25: ['--a224', '0.25', '--table', T, '--aoa', str(aoa), '--mode', 'pw', '--aloamm', str(al), '--amclmm', str(am)]
ROWS = [('POS swide08 sim with its own flags (.25 / .08 / .1)', gate + [SIM] + Pa(0.08), 'PASS 26/26'),
        ('POS REAL swide dir with its flags (.25 / .05 / .1)', gate + [REAL] + Pa(0.05), 'PASS 26/26'),
        ('NEG REAL swide dir checked as swide08 (aloamm .08)', gate + [REAL] + Pa(0.08), 'FAIL'),
        ('NEG swide08 sim checked as swide (aloamm .05)', gate + [SIM] + Pa(0.05), 'FAIL'),
        ('NEG swide08 sim checked with aloamm 0', gate + [SIM] + Pa(0.0), 'FAIL'),
        ('NEG swide08 sim checked with amclmm .15', gate + [SIM] + Pa(0.08, 0.15), 'FAIL'),
        ('NEG swide08 sim checked as aoa .3', gate + [SIM] + Pa(0.08, 0.1, 0.3), 'FAIL'),
        ('NEG tamper: Lateral OA rows 1 / 3 swapped', gate + [NEG['loaswap']] + Pa(0.08), 'FAIL'),
        ('NEG tamper: log says a_loamm .05, receipt .08', gate + [NEG['logvalue']] + Pa(0.08), 'FAIL'),
        ('NEG tamper: 2nd LOAMM marker with a conflicting value .05', gate + [NEG['dupmarker']] + Pa(0.08), 'FAIL'),
        ('NEG tamper: LOAMM marker before EOA', gate + [NEG['order']] + Pa(0.08), 'FAIL'),
        ('NEG tamper: extra transform key a_moacf in receipt', gate + [NEG['extrakey']] + Pa(0.08), 'FAIL'),
        ('POS benign: E-OA..S6F_DONE log lines each repeated', gate + [NEG['dupsame']] + Pa(0.08), 'PASS 26/26')]
ok = HARNESS
for name, cmd, want in ROWS:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    last = [l for l in r.stdout.splitlines() if l.startswith('PROBE_CHECK')]; last = last[-1] if last else 'NO PROBE_CHECK LINE ' + r.stderr[-200:]
    fails = [l.strip()[5:60] for l in r.stdout.splitlines() if l.strip().startswith('FAIL')]
    got_ok = (want in last and r.returncode == 0) if want.startswith('PASS') else (last.startswith('PROBE_CHECK FAIL') and r.returncode == 1)
    ok = ok and got_ok
    P('%-4s %-60s want %-10s got %-22s %s' % ('ok' if got_ok else 'BAD', name, want, last, ('| ' + ' ; '.join(fails)) if fails else ''))
    if name.startswith('POS swide08 sim'):
        for l in r.stdout.splitlines():
            if l.strip().startswith('info: visible'):
                P('       ' + l.strip())
P('SWIDE08_KSIM_GATE_SELFTEST %s (%d rows, harness %s)' % ('PASS' if ok else 'FAIL', len(ROWS), HARNESS))
LOGF.close(); sys.exit(0 if ok else 1)
