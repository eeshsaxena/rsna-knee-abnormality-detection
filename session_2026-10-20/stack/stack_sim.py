# stack_sim.py (2026-10-21 ~00:35 IST, Claude, STACK). LOCAL ONLY. Simulated commit outputs for the unarmed stack builds, made the way
# night/c12_sim.py made lomm05's: the REAL committed dir of scored pw30 (a21/out_v224r6c4pw30j_v1, read-only, copied) + cell 18 of each
# stack notebook replayed locally on that dir's own submission_v224flip.csv / B_probs.csv / setup + run dicts. Then the stack commit gate
# (stack/stack_probe_check.py) is self-tested on positives and negatives with stated expectations, including the REAL scored pw30 output.
# Writes only under stack/sim/.
import json, os, shutil, tempfile, subprocess, sys, io
import pandas as pd
N = os.path.dirname(os.path.abspath(__file__)).replace(chr(92), '/')
PW30 = N + '/../a21/out_v224r6c4pw30j_v1'
nbp = lambda t: N + '/kernel_%s/rsna_sub_%s.ipynb' % (t, t)
NB = {'pw30': N + '/../a21/kernel_v224r6c4pw30j/rsna_sub_v224r6c4pw30j.ipynb', 'mcl00': nbp('v224r6c4pw30mcl00j'),
      'pw30mcl10': nbp('v224r6c4pw30mcl10j'), 'pw25mcl10': nbp('v224r6c4pw25mcl10j'),
      'pw30lomm05mcl10': nbp('v224r6c4pw30lomm05mcl10j'), 'pw25lomm05mcl10': nbp('v224r6c4pw25lomm05mcl10j')}
lf = lambda b: b.replace(b'\r\n', b'\n')


def replay(p):
    nb = json.load(open(p, encoding='utf-8')); s_ = nb['cells'][18]['source']; s_ = ''.join(s_) if isinstance(s_, list) else s_
    fn = s_.split('\n\ntry:\n    _S6F_BLEND = ')[0]
    W = tempfile.mkdtemp().replace(chr(92), '/'); fn = fn.replace("W = '/kaggle/working'", "W = %r" % W)
    shutil.copyfile(PW30 + '/submission_v224flip.csv', W + '/submission_v224flip.csv'); os.makedirs(W + '/s6f/B')
    shutil.copyfile(PW30 + '/s6f/B/B_probs.csv', W + '/s6f/B/B_probs.csv')
    rc = json.load(open(PW30 + '/s6f_receipt.json', encoding='utf-8'))
    ns = {}; exec(fn, ns); rec = ns['_s6f_blend'](rc['setup'], rc['run'])
    b = open(W + '/submission.csv', 'rb').read(); shutil.rmtree(W, ignore_errors=True)
    return rec, b


R = {k: replay(p) for k, p in NB.items()}
real = open(PW30 + '/submission.csv', 'rb').read()
print('pw30 replay == its REAL committed submission.csv (LF): %s | mcl00 replay == pw30 replay: %s' % (lf(R['pw30'][1]) == lf(real), R['mcl00'][1] == R['pw30'][1]))
o0 = pd.read_csv(PW30 + '/submission.csv')
for k in ('pw30mcl10', 'pw25mcl10', 'pw30lomm05mcl10', 'pw25lomm05mcl10'):
    o = pd.read_csv(io.BytesIO(R[k][1]))
    print('%-16s 3-row MCL pw30 %s -> %s | LOA %s -> %s | MM %s' % (k, [round(v, 4) for v in o0['MCL']], [round(v, 4) for v in o['MCL']],
          [round(v, 4) for v in o0['Lateral OA']], [round(v, 4) for v in o['Lateral OA']], [round(v, 4) for v in o0['Medial Meniscus']]))


def write_sim(tag, rec, outb):
    d = N + '/sim/out_%s_sim' % tag; shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(PW30, d, ignore=shutil.ignore_patterns('__pycache__'))
    open(d + '/submission.csv', 'wb').write(lf(outb))
    rc = json.load(open(d + '/s6f_receipt.json', encoding='utf-8')); rc['blend'] = rec; rc['mode'] = rec.get('mode'); rc['fired'] = list(rec.get('fired') or [])
    rc['SIMULATED'] = 'stack/stack_sim.py: committed inputs of out_v224r6c4pw30j_v1 + cell 18 of %s replayed locally; NOT a Kaggle run' % tag
    json.dump(rc, open(d + '/s6f_receipt.json', 'w', encoding='utf-8'), indent=1, default=str)
    lg = [p for p in os.listdir(d) if p.endswith('.log') and 'cufile' not in p]; assert len(lg) == 1, lg
    L_ = json.load(open(d + '/' + lg[0], encoding='utf-8'))
    if rec.get('a_oa') != 0.3:
        L_.append({'stream_name': 'stdout', 'time': 0, 'data': "EOA_APPLIED a_oa %g mode pw on ['Medial OA', 'Lateral OA', 'PF OA'] [SIMULATED]\n" % rec['a_oa']})
    if rec.get('a_loamm'):
        L_.append({'stream_name': 'stdout', 'time': 0, 'data': 'LOAMM_APPLIED a_loamm %g [SIMULATED]\n' % rec['a_loamm']})
    if rec.get('a_mclmm'):
        L_.append({'stream_name': 'stdout', 'time': 0, 'data': 'MCLMM_APPLIED a_mclmm %g [SIMULATED]\n' % rec['a_mclmm']})
    json.dump(L_, open(d + '/' + lg[0], 'w', encoding='utf-8'))
    return d


SIM = {k: write_sim('v224r6c4%sj' % k, *R[k]) for k in ('pw30mcl10', 'pw25mcl10', 'pw30lomm05mcl10', 'pw25lomm05mcl10')}
for k, d in SIM.items():
    print('SIM written', d, '| receipt a_oa', R[k][0].get('a_oa'), 'a_loamm', R[k][0].get('a_loamm'), 'a_mclmm', R[k][0].get('a_mclmm'))
T = '{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker\'s": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}'
gate = [sys.executable, '-I', N + '/stack_probe_check.py']; c12 = [sys.executable, '-I', N + '/../night/c12_probe_check.py']
Pa = lambda aoa: ['--a224', '0.25', '--table', T, '--aoa', str(aoa), '--mode', 'pw']
ARGS = {'pw30mcl10': Pa(0.3) + ['--aloamm', '0', '--amclmm', '0.1'], 'pw25mcl10': Pa(0.25) + ['--aloamm', '0', '--amclmm', '0.1'],
        'pw30lomm05mcl10': Pa(0.3) + ['--aloamm', '0.05', '--amclmm', '0.1'], 'pw25lomm05mcl10': Pa(0.25) + ['--aloamm', '0.05', '--amclmm', '0.1']}
ROWS = [('stack gate on %s sim' % k, gate + [SIM[k]] + a, 'PASS') for k, a in ARGS.items()]
ROWS += [('control: stack gate (0 / 0) on the REAL scored pw30 dir', gate + [PW30] + Pa(0.3) + ['--aloamm', '0', '--amclmm', '0'], 'PASS'),
         ('NEG stack gate amclmm .1 on the REAL pw30 dir', gate + [PW30] + Pa(0.3) + ['--aloamm', '0', '--amclmm', '0.1'], 'FAIL'),
         ('NEG amclmm 0 on pw30mcl10 sim', gate + [SIM['pw30mcl10']] + Pa(0.3) + ['--aloamm', '0', '--amclmm', '0'], 'FAIL'),
         ('NEG wrong dose amclmm .2 on pw30mcl10 sim', gate + [SIM['pw30mcl10']] + Pa(0.3) + ['--aloamm', '0', '--amclmm', '0.2'], 'FAIL'),
         ('NEG aloamm 0 on pw30lomm05mcl10 sim', gate + [SIM['pw30lomm05mcl10']] + Pa(0.3) + ['--aloamm', '0', '--amclmm', '0.1'], 'FAIL'),
         ('NEG pw25mcl10 sim checked as aoa .3', gate + [SIM['pw25mcl10']] + Pa(0.3) + ['--aloamm', '0', '--amclmm', '0.1'], 'FAIL'),
         ('NEG pw30lomm05mcl10 sim checked as pw25 (aoa .25)', gate + [SIM['pw30lomm05mcl10']] + Pa(0.25) + ['--aloamm', '0.05', '--amclmm', '0.1'], 'FAIL'),
         ('INFO old C12\' gate on pw30mcl10 sim (3 rows may not show MCLmm)', c12 + [SIM['pw30mcl10']] + Pa(0.3) + ['--aloamm', '0'], 'INFO')]
TD = N + '/sim/out_v224r6c4pw30mcl10j_tamper'; shutil.rmtree(TD, ignore_errors=True); shutil.copytree(SIM['pw30mcl10'], TD)
_s = pd.read_csv(TD + '/submission.csv', dtype={'StudyInstanceUID': str}); _v = _s['MCL'].values.copy(); _v[[0, 1]] = _v[[1, 0]]
_s['MCL'] = _v; _s.to_csv(TD + '/submission.csv', index=False)
ROWS.append(('NEG tampered MCL (2 values swapped) on pw30mcl10 sim', gate + [TD] + ARGS['pw30mcl10'], 'FAIL'))
ok = True
for name, cmd, want in ROWS:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    last = [l for l in r.stdout.splitlines() if l.startswith('PROBE_CHECK')]
    got = 'PASS' if last and last[-1].startswith('PROBE_CHECK PASS') else 'FAIL'
    fails = [l.strip() for l in r.stdout.splitlines() if l.strip().startswith('FAIL')]
    if want != 'INFO':
        ok = ok and got == want
    print('%-62s want %s got %s %s %s' % (name, want, got, last[-1] if last else r.stderr[-200:], ('| ' + ' ; '.join(f[:60] for f in fails)) if fails and want != 'PASS' else ''))
print('STACK_SIM_GATE_SELFTEST %s' % ('PASS' if ok else 'FAIL'))
