# c12_sim.py (2026-10-20 night, Claude). LOCAL ONLY. Simulated commit output for the unarmed C12' build, made the way night/syn_sim.py and
# val/ext_a22.py made theirs: the REAL committed dir of scored pw30 (a21/out_v224r6c4pw30j_v1, read-only, copied) + cell 18 of the lomm05
# notebook replayed locally on that dir's own submission_v224flip.csv / B_probs.csv / setup + run dicts. Then the C12' commit gate
# (night/c12_probe_check.py) is self-tested on positives and negatives with stated expectations, including the REAL scored pw30 output.
# Writes only under night/c12_sim/.
import json, os, shutil, tempfile, hashlib, subprocess, sys, io
import pandas as pd
N = os.path.dirname(os.path.abspath(__file__)).replace(chr(92), '/')
PW30 = N + '/../a21/out_v224r6c4pw30j_v1'
NB = {'lomm05': N + '/c12_kernel_v224r6c4pw30lomm05j/rsna_sub_v224r6c4pw30lomm05j.ipynb',
      'lomm00': N + '/c12_kernel_v224r6c4pw30lomm00j/rsna_sub_v224r6c4pw30lomm00j.ipynb',
      'pw30': N + '/../a21/kernel_v224r6c4pw30j/rsna_sub_v224r6c4pw30j.ipynb'}
lf = lambda b: b.replace(b'\r\n', b'\n')


def replay(nbp):
    nb = json.load(open(nbp, encoding='utf-8')); s_ = nb['cells'][18]['source']; s_ = ''.join(s_) if isinstance(s_, list) else s_
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
print('pw30 replay == its REAL committed submission.csv (LF): %s' % (lf(R['pw30'][1]) == lf(real)))
print('lomm00 replay == pw30 replay: %s | lomm05 replay == pw30 replay (3 public rows; C12\' may not change 3-row ranks): %s' % (
    R['lomm00'][1] == R['pw30'][1], R['lomm05'][1] == R['pw30'][1]))
o0 = pd.read_csv(PW30 + '/submission.csv'); o5 = pd.read_csv(io.BytesIO(R['lomm05'][1]))
print('3-row Lateral OA pw30 %s -> lomm05 %s | Medial Meniscus %s' % ([round(v, 4) for v in o0['Lateral OA']], [round(v, 4) for v in o5['Lateral OA']],
                                                                     [round(v, 4) for v in o0['Medial Meniscus']]))


def write_sim(tag, rec, outb):
    d = N + '/c12_sim/out_%s_sim' % tag; shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(PW30, d, ignore=shutil.ignore_patterns('__pycache__'))
    open(d + '/submission.csv', 'wb').write(lf(outb))
    rc = json.load(open(d + '/s6f_receipt.json', encoding='utf-8')); rc['blend'] = rec; rc['mode'] = rec.get('mode'); rc['fired'] = list(rec.get('fired') or [])
    rc['SIMULATED'] = 'night/c12_sim.py: committed inputs of out_v224r6c4pw30j_v1 + cell 18 of %s replayed locally; NOT a Kaggle run' % tag
    json.dump(rc, open(d + '/s6f_receipt.json', 'w', encoding='utf-8'), indent=1, default=str)
    lg = [p for p in os.listdir(d) if p.endswith('.log') and 'cufile' not in p]; assert len(lg) == 1, lg
    if rec.get('a_loamm'):
        L_ = json.load(open(d + '/' + lg[0], encoding='utf-8'))
        L_.append({'stream_name': 'stdout', 'time': 0, 'data': 'LOAMM_APPLIED a_loamm %g [SIMULATED]\n' % rec['a_loamm']})
        json.dump(L_, open(d + '/' + lg[0], 'w', encoding='utf-8'))
    return d


S5 = write_sim('v224r6c4pw30lomm05j', *R['lomm05'])
print('SIM written', S5, '| receipt a_loamm', R['lomm05'][0].get('a_loamm'), 'a_oa', R['lomm05'][0].get('a_oa'), 'oa_mode', R['lomm05'][0].get('oa_mode'))
T = '{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker\'s": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}'
c12 = [sys.executable, '-I', N + '/c12_probe_check.py']; a21 = [sys.executable, '-I', N + '/../a21/probe_check_eoa21.py']
P = ['--a224', '0.25', '--table', T, '--aoa', '0.3', '--mode', 'pw']
P25 = ['--a224', '0.25', '--table', T, '--aoa', '0.25', '--mode', 'pw']
ROWS = [('C12\' gate (aloamm .05) on lomm05 sim', c12 + [S5] + P + ['--aloamm', '0.05'], 'PASS'),
        ('control: C12\' gate aloamm 0 on the REAL scored pw30 dir', c12 + [PW30] + P + ['--aloamm', '0'], 'PASS'),
        ('reference: a21 gate on the REAL scored pw30 dir', a21 + [PW30] + P, 'PASS'),
        ('NEG C12\' gate (aloamm .05) on the REAL pw30 dir', c12 + [PW30] + P + ['--aloamm', '0.05'], 'FAIL'),
        ('NEG C12\' gate aloamm 0 on lomm05 sim', c12 + [S5] + P + ['--aloamm', '0'], 'FAIL'),
        ('NEG wrong dose (aloamm .1) on lomm05 sim', c12 + [S5] + P + ['--aloamm', '0.1'], 'FAIL'),
        ('NEG wrong E-OA dose (aoa .25) on lomm05 sim', c12 + [S5] + P25 + ['--aloamm', '0.05'], 'FAIL'),
        ('INFO old a21 gate on lomm05 sim (3 rows cannot show C12\'; it passes)', a21 + [S5] + P, 'PASS')]
# tamper test of check (d): same sim, receipt and log say .05, but two Lateral OA values swapped in submission.csv
TD = N + '/c12_sim/out_v224r6c4pw30lomm05j_tamper'; shutil.rmtree(TD, ignore_errors=True); shutil.copytree(S5, TD)
_s = pd.read_csv(TD + '/submission.csv', dtype={'StudyInstanceUID': str}); _v = _s['Lateral OA'].values.copy(); _v[[0, 1]] = _v[[1, 0]]
_s['Lateral OA'] = _v; _s.to_csv(TD + '/submission.csv', index=False)
ROWS.append(('NEG tampered LOA (2 values swapped) on lomm05 sim', c12 + [TD] + P + ['--aloamm', '0.05'], 'FAIL'))
ok = True
for name, cmd, want in ROWS:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    last = [l for l in r.stdout.splitlines() if l.startswith('PROBE_CHECK')]
    got = 'PASS' if last and last[-1].startswith('PROBE_CHECK PASS') else 'FAIL'
    fails = [l.strip() for l in r.stdout.splitlines() if l.strip().startswith('FAIL')]
    ok = ok and got == want
    print('%-66s want %s got %s %s %s' % (name, want, got, last[-1] if last else r.stderr[-200:], ('| ' + ' ; '.join(f[:70] for f in fails)) if fails and want == 'FAIL' else ''))
print('C12_SIM_GATE_SELFTEST %s' % ('PASS' if ok else 'FAIL'))
