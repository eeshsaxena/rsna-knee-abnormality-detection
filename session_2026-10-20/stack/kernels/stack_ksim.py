#!/usr/bin/env python
# stack_ksim.py (10-20 ~23:20 IST, Claude, STACK BUILD round). LOCAL ONLY. Simulated commit outputs for the stack/kernels builds + the
# self-test of stack_commit_gate.py. Made from the REAL committed dir of scored pw30 (a21/out_v224r6c4pw30j_v1, read-only, copied):
# each kernel's own cell 18 is replayed on that dir's submission_v224flip.csv / s6f/B/B_probs.csv / setup + run dicts with os.linesep = '\n'
# (Kaggle is Linux), so the kernel code itself writes the LF submission.csv and its submission_sha256. The replayed stdout lines replace
# the real log's EOA_APPLIED .. S6F_BLEND_OK entries (flagged 'simulated': true), so markers sit where a real run prints them.
# Harness check first: the pw30 replay reproduces the REAL submission.csv bytes, the real receipt's submission_sha256 and the real log lines.
# Writes only under stack/kernels/sim/ and stack/kernels/stack_ksim.log.
import json, os, sys, io, shutil, tempfile, subprocess, contextlib, hashlib
import pandas as pd
os.linesep = '\n'                     # pandas to_csv reads os.linesep at call time -> Linux line endings, as on Kaggle
N = os.path.dirname(os.path.abspath(__file__)).replace(chr(92), '/'); STK = N.rsplit('/', 1)[0]; S = STK.rsplit('/', 1)[0]
PW30 = S + '/a21/out_v224r6c4pw30j_v1'
LOGF = open(N + '/stack_ksim.log', 'w', encoding='utf-8')
def P(s=''):
    print(s, flush=True); LOGF.write(s + '\n')
shab = lambda b: hashlib.sha256(b).hexdigest()
IDX = json.load(open(N + '/build_index.json', encoding='utf-8'))
nbp = lambda t: N + '/kernel_%s/rsna_sub_%s.ipynb' % (t, t)
NB = {'pw30': S + '/a21/kernel_v224r6c4pw30j/rsna_sub_v224r6c4pw30j.ipynb'}; NB.update({k: nbp(v['tag']) for k, v in IDX.items()})
REAL_LOG = [p for p in os.listdir(PW30) if p.endswith('.log') and 'cufile' not in p]; assert len(REAL_LOG) == 1; REAL_LOG = REAL_LOG[0]
L0 = json.load(open(PW30 + '/' + REAL_LOG, encoding='utf-8'))
RC0 = json.load(open(PW30 + '/s6f_receipt.json', encoding='utf-8'))


def replay(p):
    nb = json.load(open(p, encoding='utf-8')); s_ = nb['cells'][18]['source']; s_ = ''.join(s_) if isinstance(s_, list) else s_
    fn = s_.split('\n\ntry:\n    _S6F_BLEND = ')[0]; assert fn.count("W = '/kaggle/working'") == 1
    W = tempfile.mkdtemp().replace(chr(92), '/'); fn = fn.replace("W = '/kaggle/working'", "W = %r" % W)
    shutil.copyfile(PW30 + '/submission_v224flip.csv', W + '/submission_v224flip.csv'); os.makedirs(W + '/s6f/B')
    shutil.copyfile(PW30 + '/s6f/B/B_probs.csv', W + '/s6f/B/B_probs.csv')
    buf = io.StringIO(); ns = {}
    with contextlib.redirect_stdout(buf):
        exec(fn, ns); rec = ns['_s6f_blend'](json.loads(json.dumps(RC0['setup'])), json.loads(json.dumps(RC0['run'])))
    b = open(W + '/submission.csv', 'rb').read(); shutil.rmtree(W, ignore_errors=True)
    return rec, b, [l + '\n' for l in buf.getvalue().splitlines()]


RP = {k: replay(p) for k, p in NB.items()}
real = open(PW30 + '/submission.csv', 'rb').read()
i_e = [i for i, e in enumerate(L0) if 'EOA_APPLIED' in e.get('data', '')]; i_b = [i for i, e in enumerate(L0) if 'S6F_BLEND_OK' in e.get('data', '')]
assert len(i_e) == 1 and len(i_b) == 1 and i_b[0] == i_e[0] + 1
h1 = RP['pw30'][1] == real; h2 = RP['pw30'][0].get('submission_sha256') == RC0['blend']['submission_sha256'] == shab(real)
h3 = RP['pw30'][2] == [L0[i_e[0]]['data'], L0[i_b[0]]['data']]
h4 = {k: v for k, v in RP['pw30'][0].items()} == RC0['blend']
P('HARNESS pw30 replay on the REAL committed inputs: submission.csv bytes == REAL %s | kernel-computed submission_sha256 == REAL receipt %s | '
  'stdout lines == REAL log lines %s | blend receipt == REAL blend receipt %s' % (h1, h2, h3, h4))
HARNESS = h1 and h2 and h3 and h4
for k, (rec, b, so) in RP.items():
    P('  replay %-9s out sha %s | 3-row output == pw30 REAL: %s | markers %s' % (k, shab(b)[:12], b == real, [l.strip()[:40] for l in so if 'APPLIED' in l]))


def write_sim(tag, rec, outb, so, d=None):
    d = d or N + '/sim/out_%s_sim' % tag; shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(PW30, d, ignore=shutil.ignore_patterns('__pycache__'))
    open(d + '/submission.csv', 'wb').write(outb)
    rc = json.loads(json.dumps(RC0)); rc['blend'] = rec; rc['mode'] = rec.get('mode'); rc['fired'] = list(rec.get('fired') or [])
    rc['SIMULATED'] = 'stack/kernels/stack_ksim.py: committed inputs of a21/out_v224r6c4pw30j_v1 + cell 18 of %s replayed locally; NOT a Kaggle run' % tag
    json.dump(rc, open(d + '/s6f_receipt.json', 'w', encoding='utf-8'), indent=1, default=str)
    t0 = L0[i_e[0]]['time']
    new = [{'stream_name': 'stdout', 'time': t0 + 1e-6 * k, 'data': line, 'simulated': True} for k, line in enumerate(so)]
    L_ = L0[:i_e[0]] + new + L0[i_b[0] + 1:]
    json.dump(L_, open(d + '/' + REAL_LOG, 'w', encoding='utf-8'))
    return d


SIM = {k: write_sim(IDX[k]['tag'], *RP[k]) for k in IDX}
for k, d in SIM.items():
    P('SIM %-9s %s | receipt a_oa %s a_loamm %s a_mclmm %s' % (k, d.rsplit('/', 1)[1], RP[k][0].get('a_oa'), RP[k][0].get('a_loamm'), RP[k][0].get('a_mclmm')))

# ---- tampered copies (negatives) -----------------------------------------------------------------------------------------------
def tamper(src, name, fn):
    d = N + '/sim/neg_' + name; shutil.rmtree(d, ignore_errors=True); shutil.copytree(src, d); fn(d); return d
def t_mclswap(d):
    s = pd.read_csv(d + '/submission.csv', dtype=str); v = s['MCL'].values.copy(); v[[0, 2]] = v[[2, 0]]; s['MCL'] = v
    open(d + '/submission.csv', 'wb').write(s.to_csv(index=False).encode())
def t_effswap_stale(d):     # untouched column changed, receipt left stale
    s = pd.read_csv(d + '/submission.csv', dtype=str); v = s['Effusion'].values.copy(); v[[0, 1]] = v[[1, 0]]; s['Effusion'] = v
    open(d + '/submission.csv', 'wb').write(s.to_csv(index=False).encode())
def _log_edit(d, f):
    L_ = json.load(open(d + '/' + REAL_LOG, encoding='utf-8')); L_ = f(L_); json.dump(L_, open(d + '/' + REAL_LOG, 'w', encoding='utf-8'))
def t_dupmarker(d):         # a SECOND MCLMM marker with a conflicting value
    _log_edit(d, lambda L_: L_ + [{'stream_name': 'stdout', 'time': 0, 'data': 'MCLMM_APPLIED a_mclmm 0.2\n', 'simulated': True}])
def t_dupsame(d):           # benign: every log entry from E-OA to S6F_DONE repeated once, the way Kaggle repeats worker lines
    def f(L_):
        i0 = [i for i, e in enumerate(L_) if 'EOA_APPLIED' in e['data']][0]; i1 = [i for i, e in enumerate(L_) if 'S6F_DONE' in e['data']][0]
        out = L_[:i0]
        for e in L_[i0:i1 + 1]:
            out += [e, dict(e)]
        return out + L_[i1 + 1:]
    _log_edit(d, f)
def t_order(d):
    def f(L_):
        k = [i for i, e in enumerate(L_) if 'MCLMM_APPLIED' in e['data']][0]; e = L_.pop(k)
        j = [i for i, e2 in enumerate(L_) if 'EOA_APPLIED' in e2['data']][0]; L_.insert(j, e); return L_
    _log_edit(d, f)
def t_extrakey(d):
    rc = json.load(open(d + '/s6f_receipt.json', encoding='utf-8')); rc['blend']['a_syneff'] = 0.2
    json.dump(rc, open(d + '/s6f_receipt.json', 'w', encoding='utf-8'), indent=1)
def t_logvalue(d):          # receipt says .1 but the log says .2
    _log_edit(d, lambda L_: [dict(e, data=e['data'].replace('MCLMM_APPLIED a_mclmm 0.1', 'MCLMM_APPLIED a_mclmm 0.2')) for e in L_])
NEG = {k: tamper(SIM['S_safe'], k, f) for k, f in (('mclswap', t_mclswap), ('dupmarker', t_dupmarker), ('order', t_order), ('extrakey', t_extrakey),
                                                    ('logvalue', t_logvalue))}
NEG['dupsame'] = tamper(SIM['S_wide'], 'dupsame', t_dupsame)
NEG['effswap_stale'] = tamper(SIM['S_wide'], 'effswap_stale', t_effswap_stale)

# ---- self-test ------------------------------------------------------------------------------------------------------------------
T = '{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker\'s": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}'
PY = ['py', '-3.12', '-I']
gate = PY + [N + '/stack_commit_gate.py']; old = PY + [STK + '/stack_probe_check.py']
Pa = lambda aoa, al, am: ['--a224', '0.25', '--table', T, '--aoa', str(aoa), '--mode', 'pw', '--aloamm', str(al), '--amclmm', str(am)]
SPEC = {k: (0.25 if '25' in v['tag'][8:12] else 0.3, 0.05 if 'lomm05' in v['tag'] else 0.0, 0.1 if 'mcl10' in v['tag'] else 0.0) for k, v in IDX.items()}
ROWS = [('POS %s sim with its own flags %s' % (k, SPEC[k]), gate + [SIM[k]] + Pa(*SPEC[k]), 'PASS 26/26') for k in IDX]
ROWS += [('POS REAL scored pw30 dir (.3 / 0 / 0)', gate + [PW30] + Pa(.3, 0, 0), 'PASS 26/26'),
         ('NEG REAL pw30 dir checked as S_safe (amclmm .1)', gate + [PW30] + Pa(.3, 0, .1), 'FAIL'),
         ('NEG S_safe sim checked with amclmm 0', gate + [SIM['S_safe']] + Pa(.3, 0, 0), 'FAIL'),
         ('NEG S_safe sim checked with wrong dose amclmm .2', gate + [SIM['S_safe']] + Pa(.3, 0, .2), 'FAIL'),
         ('NEG S_mid sim checked with aloamm 0 (= S_safe flags)', gate + [SIM['S_mid']] + Pa(.3, 0, .1), 'FAIL'),
         ('NEG S_alt sim checked as aoa .3 (= S_safe flags)', gate + [SIM['S_alt']] + Pa(.3, 0, .1), 'FAIL'),
         ('NEG S_mid sim checked as aoa .25 (= S_wide flags)', gate + [SIM['S_mid']] + Pa(.25, .05, .1), 'FAIL'),
         ('NEG S_wide sim checked as S_alt (aloamm 0)', gate + [SIM['S_wide']] + Pa(.25, 0, .1), 'FAIL'),
         ('NEG CTRL_off sim checked as S_safe', gate + [SIM['CTRL_off']] + Pa(.3, 0, .1), 'FAIL'),
         ('NEG tamper: MCL rows 1 / 3 swapped (S_safe sim)', gate + [NEG['mclswap']] + Pa(*SPEC['S_safe']), 'FAIL'),
         ('NEG tamper: Effusion swapped, stale receipt (S_wide sim)', gate + [NEG['effswap_stale']] + Pa(*SPEC['S_wide']), 'FAIL'),
         ('NEG tamper: 2nd MCLMM marker with a conflicting value .2 (S_safe sim)', gate + [NEG['dupmarker']] + Pa(*SPEC['S_safe']), 'FAIL'),
         ('POS benign: E-OA..S6F_DONE log lines each repeated (S_wide sim)', gate + [NEG['dupsame']] + Pa(*SPEC['S_wide']), 'PASS 26/26'),
         ('NEG tamper: MCLMM marker before EOA (S_safe sim)', gate + [NEG['order']] + Pa(*SPEC['S_safe']), 'FAIL'),
         ('NEG tamper: extra transform key in receipt (S_safe sim)', gate + [NEG['extrakey']] + Pa(*SPEC['S_safe']), 'FAIL'),
         ('NEG tamper: log says a_mclmm .2, receipt .1 (S_safe sim)', gate + [NEG['logvalue']] + Pa(*SPEC['S_safe']), 'FAIL'),
         ('NEG inventory sim stack/sim pw25mcl10 (2 EOA markers appended)', gate + [STK + '/sim/out_v224r6c4pw25mcl10j_sim'] + Pa(.25, 0, .1), 'FAIL')]
ROWS += [('XCHK inventory gate stack_probe_check.py on %s sim -> PASS 19/19' % k, old + [SIM[k]] + Pa(*SPEC[k]), 'PASS 19/19') for k in IDX if k != 'CTRL_off']
ROWS += [('INFO inventory gate on tamper logvalue (it does not parse the logged dose)', old + [NEG['logvalue']] + Pa(*SPEC['S_safe']), 'INFO'),
         ('INFO inventory gate on tamper extrakey (no key whitelist)', old + [NEG['extrakey']] + Pa(*SPEC['S_safe']), 'INFO')]
ok = HARNESS; n_exp = 0
for name, cmd, want in ROWS:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    last = [l for l in r.stdout.splitlines() if l.startswith('PROBE_CHECK')]; last = last[-1] if last else 'NO PROBE_CHECK LINE ' + r.stderr[-200:]
    fails = [l.strip()[5:50] for l in r.stdout.splitlines() if l.strip().startswith('FAIL')]
    got_ok = (want in last and r.returncode == 0) if want.startswith('PASS') else (last.startswith('PROBE_CHECK FAIL') and r.returncode == 1) if want == 'FAIL' else True
    if want != 'INFO':
        ok = ok and got_ok; n_exp += 1
    P('%-4s %-70s want %-10s got %-22s %s' % ('ok' if got_ok else 'BAD', name, want, last, ('| ' + ' ; '.join(fails)) if fails else ''))
    if want == 'PASS 26/26' and SIM.get(name.split()[1]):
        for l in r.stdout.splitlines():
            if l.strip().startswith('info: visible'):
                P('       ' + l.strip())
P('STACK_KSIM_GATE_SELFTEST %s (%d rows with an expectation%s, harness %s)' % ('PASS' if ok else 'FAIL', n_exp, ', all as expected' if ok else '', HARNESS))
LOGF.close()
sys.exit(0 if ok else 1)
