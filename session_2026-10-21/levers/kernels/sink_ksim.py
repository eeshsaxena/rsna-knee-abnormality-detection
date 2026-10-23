#!/usr/bin/env python
# sink_ksim.py (10-21 ~11:30 IST, Claude, SINK track). LOCAL ONLY. Simulated commit outputs for the levers/kernels sink builds + the
# self-test of sink_commit_gate.py (pattern stack/kernels/stack_ksim.py). REAL committed dirs are read-only and copied, never changed:
#   scored pw30 a21/out_v224r6c4pw30j_v1 and the four A22 commits val/a22/out_<tag> (pw25, spl, swide, salt; committed 10-21 ~09:00 IST).
# Each build's own cell 18 is replayed on the REAL committed inputs of ITS BASE (submission_v224flip.csv, s6f/B/B_probs.csv, setup + run
# dicts) with os.linesep = '\n' (Kaggle is Linux); replayed stdout lines replace the base log's EOA_APPLIED .. S6F_BLEND_OK entries.
# Harness first: each base notebook's cell 18 replayed on its OWN real dir reproduces the real submission.csv bytes, the real receipt's
# submission_sha256, the real log lines and the whole blend receipt (pw30 + 4 A22 entries).
# Writes only under levers/kernels/sim/ and levers/kernels/sink_ksim.log.
import json, os, sys, io, shutil, tempfile, subprocess, contextlib, hashlib
import pandas as pd
os.linesep = '\n'
K = os.path.dirname(os.path.abspath(__file__)).replace(chr(92), '/'); LV = K.rsplit('/', 1)[0]
R = 'C:/Users/eeshs/Downloads/rsna-knee'; S20 = R + '/session_2026-10-20'
LOGF = open(K + '/sink_ksim.log', 'w', encoding='utf-8')
def P(s=''):
    print(s, flush=True); LOGF.write(s + '\n')
shab = lambda b: hashlib.sha256(b).hexdigest()
IDX = json.load(open(K + '/build_index.json', encoding='utf-8'))
REG = json.load(open(S20 + '/val/slate_a22.json', encoding='utf-8'))['entries']
nbf = lambda d: [d + '/' + f for f in os.listdir(d) if f.endswith('.ipynb')][0]
BASE_NB = {'pw30': S20 + '/a21/kernel_v224r6c4pw30j', 'lomm30': S20 + '/night/c12_kernel_v224r6c4pw30lomm05j'}
BASE_NB.update({k: R + '/' + REG[k]['kernel_dir'] for k in ('pw25', 'spl', 'swide', 'salt')})
REALDIR = {'pw30': S20 + '/a21/out_v224r6c4pw30j_v1'}
REALDIR.update({k: S20 + '/val/a22/out_' + REG[k]['tag'] for k in ('pw25', 'spl', 'swide', 'salt')})
INPUT_DIR = dict(REALDIR, lomm30=REALDIR['pw30'])          # lomm30 (never pushed) is replayed on the pw30 commit's inputs
FLAGS = {'pw30': (.3, 0., 0.), 'lomm30': (.3, .05, 0.), 'pw25': (.25, 0., 0.), 'spl': (.25, .05, 0.), 'swide': (.25, .05, .1), 'salt': (.25, 0., .1)}


def reallog(d):
    f = [p for p in os.listdir(d) if p.endswith('.log') and 'cufile' not in p]; assert len(f) == 1, (d, f); return f[0]


def replay(nbpath, d):
    nb = json.load(open(nbpath, encoding='utf-8')); s_ = nb['cells'][18]['source']; s_ = ''.join(s_) if isinstance(s_, list) else s_
    fn = s_.split('\n\ntry:\n    _S6F_BLEND = ')[0]; assert fn.count("W = '/kaggle/working'") == 1
    W = tempfile.mkdtemp().replace(chr(92), '/'); fn = fn.replace("W = '/kaggle/working'", "W = %r" % W)
    shutil.copyfile(d + '/submission_v224flip.csv', W + '/submission_v224flip.csv'); os.makedirs(W + '/s6f/B')
    shutil.copyfile(d + '/s6f/B/B_probs.csv', W + '/s6f/B/B_probs.csv')
    RC = json.load(open(d + '/s6f_receipt.json', encoding='utf-8'))
    buf = io.StringIO(); ns = {}
    with contextlib.redirect_stdout(buf):
        exec(fn, ns); rec = ns['_s6f_blend'](json.loads(json.dumps(RC['setup'])), json.loads(json.dumps(RC['run'])))
    b = open(W + '/submission.csv', 'rb').read(); shutil.rmtree(W, ignore_errors=True)
    return rec, b, [l + '\n' for l in buf.getvalue().splitlines()]


def markers(L0):
    i_e = [i for i, e in enumerate(L0) if 'EOA_APPLIED' in e.get('data', '')]; i_b = [i for i, e in enumerate(L0) if 'S6F_BLEND_OK' in e.get('data', '')]
    assert len(i_e) == 1 and len(i_b) == 1 and i_b[0] > i_e[0]
    return i_e[0], i_b[0]


# ---- harness on the 5 REAL commits ----
HARNESS = True
for k, d in REALDIR.items():
    rec, b, so = replay(nbf(BASE_NB[k]), d); real = open(d + '/submission.csv', 'rb').read()
    RC = json.load(open(d + '/s6f_receipt.json', encoding='utf-8')); L0 = json.load(open(d + '/' + reallog(d), encoding='utf-8')); ie, ib = markers(L0)
    h = (b == real, rec.get('submission_sha256') == RC['blend']['submission_sha256'] == shab(real), so == [e['data'] for e in L0[ie:ib + 1]], rec == RC['blend'])
    HARNESS = HARNESS and all(h)
    P('HARNESS %-6s base nb replay on its REAL commit: bytes %s | sha %s | log lines %s | blend receipt %s | real sha %s' % (k, *h, shab(real)[:12]))


def write_sim(tag, rec, outb, so, d):
    out = K + '/sim/out_%s_sim' % tag; shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(d, out, ignore=shutil.ignore_patterns('__pycache__'))
    open(out + '/submission.csv', 'wb').write(outb)
    RC = json.load(open(d + '/s6f_receipt.json', encoding='utf-8'))
    rc = json.loads(json.dumps(RC)); rc['blend'] = rec; rc['mode'] = rec.get('mode'); rc['fired'] = list(rec.get('fired') or [])
    rc['SIMULATED'] = 'levers/kernels/sink_ksim.py: committed inputs of %s + cell 18 of %s replayed locally; NOT a Kaggle run' % (d.rsplit('/', 1)[1], tag)
    json.dump(rc, open(out + '/s6f_receipt.json', 'w', encoding='utf-8'), indent=1, default=str)
    lf = reallog(d); L0 = json.load(open(d + '/' + lf, encoding='utf-8')); ie, ib = markers(L0); t0 = L0[ie]['time']
    new = [{'stream_name': 'stdout', 'time': t0 + 1e-6 * k, 'data': line, 'simulated': True} for k, line in enumerate(so)]
    json.dump(L0[:ie] + new + L0[ib + 1:], open(out + '/' + lf, 'w', encoding='utf-8'))
    return out


SIM = {}
for k, e in IDX.items():
    rec, b, so = replay(K + '/kernel_%s/rsna_sub_%s.ipynb' % (e['tag'], e['tag']), INPUT_DIR[e['base']])
    SIM[k] = write_sim(e['tag'], rec, b, so, INPUT_DIR[e['base']])
    base_real = open(INPUT_DIR[e['base']] + '/submission.csv', 'rb').read()
    P('SIM %-11s on %-34s | receipt a_oa %s p_oa %s a_loamm %s a_mclmm %s | 3-row output == base REAL: %s | markers %s' % (
        k, INPUT_DIR[e['base']].rsplit('/', 1)[1], rec.get('a_oa'), rec.get('p_oa'), rec.get('a_loamm'), rec.get('a_mclmm'), b == base_real,
        [l.strip()[:28] for l in so if 'APPLIED' in l]))


def tamper(src, name, fn):
    d = K + '/sim/neg_' + name; shutil.rmtree(d, ignore_errors=True); shutil.copytree(src, d); fn(d); return d
def _log_edit(d, f):
    lf = reallog(d); L_ = json.load(open(d + '/' + lf, encoding='utf-8')); json.dump(f(L_), open(d + '/' + lf, 'w', encoding='utf-8'))
def t_oaswap(d):
    s = pd.read_csv(d + '/submission.csv', dtype=str); v = s['Medial OA'].values.copy(); v[[0, 2]] = v[[2, 0]]; s['Medial OA'] = v
    open(d + '/submission.csv', 'wb').write(s.to_csv(index=False).encode())
def t_cvxconflict(d):
    _log_edit(d, lambda L_: L_ + [{'stream_name': 'stdout', 'time': 0, 'data': 'CVX_APPLIED p_oa 3\n', 'simulated': True}])
def t_cvxorder(d):
    def f(L_):
        k = [i for i, e in enumerate(L_) if 'CVX_APPLIED' in e['data']][0]; e = L_.pop(k)
        j = [i for i, e2 in enumerate(L_) if 'EOA_APPLIED' in e2['data']][0]; L_.insert(j, e); return L_
    _log_edit(d, f)
def t_receipt3(d):
    rc = json.load(open(d + '/s6f_receipt.json', encoding='utf-8')); rc['blend']['p_oa'] = 3.0
    json.dump(rc, open(d + '/s6f_receipt.json', 'w', encoding='utf-8'), indent=1)
def t_extrakey(d):
    rc = json.load(open(d + '/s6f_receipt.json', encoding='utf-8')); rc['blend']['a_subh'] = 1.0
    json.dump(rc, open(d + '/s6f_receipt.json', 'w', encoding='utf-8'), indent=1)
def t_dupsame(d):
    def f(L_):
        i0 = [i for i, e in enumerate(L_) if 'EOA_APPLIED' in e['data']][0]; i1 = [i for i, e in enumerate(L_) if 'S6F_DONE' in e['data']][0]
        out = L_[:i0]
        for e in L_[i0:i1 + 1]:
            out += [e, dict(e)]
        return out + L_[i1 + 1:]
    _log_edit(d, f)
NEG = {nm: tamper(SIM['cvx_swide'], nm, f) for nm, f in (('oaswap', t_oaswap), ('cvxconflict', t_cvxconflict), ('cvxorder', t_cvxorder),
                                                        ('receipt3', t_receipt3), ('extrakey', t_extrakey), ('dupsame', t_dupsame))}

T = '{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker\'s": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}'
PY = ['py', '-3.12', '-I']; gate = PY + [K + '/sink_commit_gate.py']; parent = PY + [S20 + '/stack/kernels/stack_commit_gate.py']
Pa = lambda f, p: ['--a224', '0.25', '--table', T, '--aoa', str(f[0]), '--mode', 'pw', '--aloamm', str(f[1]), '--amclmm', str(f[2]), '--poa', str(p)]
Pp = lambda f: ['--a224', '0.25', '--table', T, '--aoa', str(f[0]), '--mode', 'pw', '--aloamm', str(f[1]), '--amclmm', str(f[2])]
FL = lambda k: FLAGS[IDX[k]['base']]
ROWS = [('POS %s sim, own flags %s p %g' % (k, FL(k), IDX[k]['p_oa']), gate + [SIM[k]] + Pa(FL(k), IDX[k]['p_oa']), 'PASS 27/27') for k in IDX]
ROWS += [('POS REAL %s commit, its flags, p 1' % k, gate + [d] + Pa(FLAGS[k], 1.0), 'PASS 27/27') for k, d in REALDIR.items()]
ROWS += [('NEG REAL %s commit checked as its CVX stack (p 2)' % k, gate + [d] + Pa(FLAGS[k], 2.0), 'FAIL') for k, d in REALDIR.items()]
ROWS += [('NEG %s sim checked with p 1 (= its base)' % k, gate + [SIM[k]] + Pa(FL(k), 1.0), 'FAIL') for k in IDX if IDX[k]['p_oa'] != 1.0]
ROWS += [('NEG cvx_spl sim checked as cvx_pw25 (aloamm 0)', gate + [SIM['cvx_spl']] + Pa(FLAGS['pw25'], 2.0), 'FAIL'),
         ('NEG cvx_swide sim checked as cvx_salt (aloamm 0)', gate + [SIM['cvx_swide']] + Pa(FLAGS['salt'], 2.0), 'FAIL'),
         ('NEG cvx_swide sim checked as cvx_spl (amclmm 0)', gate + [SIM['cvx_swide']] + Pa(FLAGS['spl'], 2.0), 'FAIL'),
         ('NEG cvx_lomm30 sim checked as cvx_pw30 (aloamm 0)', gate + [SIM['cvx_lomm30']] + Pa(FLAGS['pw30'], 2.0), 'FAIL'),
         ('NEG cvx_pw25 sim checked at dose .3 (as cvx_pw30)', gate + [SIM['cvx_pw25']] + Pa(FLAGS['pw30'], 2.0), 'FAIL'),
         ('NEG cvx_pw30 sim checked at p 3', gate + [SIM['cvx_pw30']] + Pa(FLAGS['pw30'], 3.0), 'FAIL'),
         ('NEG ctrl_pw30p1 sim checked as cvx_pw30 (p 2)', gate + [SIM['ctrl_pw30p1']] + Pa(FLAGS['pw30'], 2.0), 'FAIL'),
         ('NEG tamper: Medial OA rows 1 / 3 swapped (cvx_swide sim)', gate + [NEG['oaswap']] + Pa(FLAGS['swide'], 2.0), 'FAIL'),
         ('NEG tamper: 2nd CVX marker with a conflicting value 3', gate + [NEG['cvxconflict']] + Pa(FLAGS['swide'], 2.0), 'FAIL'),
         ('NEG tamper: CVX marker before EOA', gate + [NEG['cvxorder']] + Pa(FLAGS['swide'], 2.0), 'FAIL'),
         ('NEG tamper: receipt p_oa 3, log 2', gate + [NEG['receipt3']] + Pa(FLAGS['swide'], 2.0), 'FAIL'),
         ('NEG tamper: extra transform key in receipt', gate + [NEG['extrakey']] + Pa(FLAGS['swide'], 2.0), 'FAIL'),
         ('POS benign: E-OA..S6F_DONE log lines each repeated (cvx_swide sim)', gate + [NEG['dupsame']] + Pa(FLAGS['swide'], 2.0), 'PASS 27/27')]
ROWS += [('XCHK parent stack_commit_gate on %s sim with base flags (must FAIL: p_oa key / CVX marker unknown to it)' % k, parent + [SIM[k]] + Pp(FL(k)), 'FAIL')
         for k in IDX if IDX[k]['p_oa'] != 1.0]
ROWS += [('XCHK parent stack_commit_gate on ctrl_pw30p1 sim (= pw30)', parent + [SIM['ctrl_pw30p1']] + Pp(FLAGS['pw30']), 'PASS 26/26')]
ok = HARNESS; n_exp = 0
for name, cmd, want in ROWS:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    last = [l for l in r.stdout.splitlines() if l.startswith('PROBE_CHECK')]; last = last[-1] if last else 'NO PROBE_CHECK LINE ' + r.stderr[-200:]
    fails = [l.strip()[5:40] for l in r.stdout.splitlines() if l.strip().startswith('FAIL')]
    got_ok = (want in last and r.returncode == 0) if want.startswith('PASS') else (last.startswith('PROBE_CHECK FAIL') and r.returncode == 1)
    ok = ok and got_ok; n_exp += 1
    P('%-4s %-92s want %-10s got %-22s %s' % ('ok' if got_ok else 'BAD', name, want, last, ('| ' + ' ; '.join(fails)) if fails else ''))
    if name.startswith('POS') and 'sim, own flags' in name:
        for l in r.stdout.splitlines():
            if l.strip().startswith('info: visible'):
                P('       ' + l.strip())
P('SINK_KSIM_GATE_SELFTEST %s (%d rows with an expectation%s, harness %s)' % ('PASS' if ok else 'FAIL', n_exp, ', all as expected' if ok else '', HARNESS))
LOGF.close()
sys.exit(0 if ok else 1)
