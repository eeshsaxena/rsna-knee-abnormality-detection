# syn_sim.py (2026-10-20 night, Claude). LOCAL ONLY. Simulated commit outputs for the unarmed E2s builds, made the way
# val/ext_a22.py made its sims: the REAL committed dir of scored pw30 (a21/out_v224r6c4pw30j_v1, read-only) + cell 18 of the syn
# notebook replayed locally on that dir's own submission_v224flip.csv / B_probs.csv / setup + run dicts. Then the E2s commit gate
# (syn_probe_check.py) is run on positives and negatives with stated expectations. Writes only under night/syn_sim/.
import json, os, shutil, tempfile, hashlib, subprocess, sys
import pandas as pd
N = os.path.dirname(os.path.abspath(__file__)).replace(chr(92), '/')
PW30 = N + '/../a21/out_v224r6c4pw30j_v1'
NB = {'syn20': N + '/syn_kernel_v224r6c4pw30syn20j/rsna_sub_v224r6c4pw30syn20j.ipynb',
      'syn00': N + '/syn_kernel_v224r6c4pw30syn00j/rsna_sub_v224r6c4pw30syn00j.ipynb',
      'pw30': N + '/../a21/kernel_v224r6c4pw30j/rsna_sub_v224r6c4pw30j.ipynb'}
shb = lambda b: hashlib.sha256(b).hexdigest()
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
print('syn00 replay == pw30 replay: %s | syn20 replay == pw30 replay (3 public rows; E2s may not change 3-row ranks): %s' % (
    R['syn00'][1] == R['pw30'][1], R['syn20'][1] == R['pw30'][1]))
o0 = pd.read_csv(PW30 + '/submission.csv'); o2 = pd.read_csv(__import__('io').BytesIO(R['syn20'][1]))
print('3-row Synovitis pw30 %s -> syn20 %s' % ([round(v, 4) for v in o0.Synovitis], [round(v, 4) for v in o2.Synovitis]))


def write_sim(tag, rec, outb):
    d = N + '/syn_sim/out_%s_sim' % tag; shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(PW30, d, ignore=shutil.ignore_patterns('__pycache__'))
    open(d + '/submission.csv', 'wb').write(lf(outb))
    rc = json.load(open(d + '/s6f_receipt.json', encoding='utf-8')); rc['blend'] = rec; rc['mode'] = rec.get('mode'); rc['fired'] = list(rec.get('fired') or [])
    rc['SIMULATED'] = 'night/syn_sim.py: committed inputs of out_v224r6c4pw30j_v1 + cell 18 of %s replayed locally; NOT a Kaggle run' % tag
    json.dump(rc, open(d + '/s6f_receipt.json', 'w', encoding='utf-8'), indent=1, default=str)
    lg = [p for p in os.listdir(d) if p.endswith('.log') and 'cufile' not in p]; assert len(lg) == 1, lg
    if rec.get('a_syn'):
        L_ = json.load(open(d + '/' + lg[0], encoding='utf-8'))
        L_.append({'stream_name': 'stdout', 'time': 0, 'data': 'SYN_E2_APPLIED a_syn %g [SIMULATED]\n' % rec['a_syn']})
        json.dump(L_, open(d + '/' + lg[0], 'w', encoding='utf-8'))
    return d


S20 = write_sim('v224r6c4pw30syn20j', *R['syn20'])
print('SIM written', S20, '| receipt a_syn', R['syn20'][0].get('a_syn'), 'a_oa', R['syn20'][0].get('a_oa'))
T = '{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker\'s": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}'
base = [sys.executable, '-I', N + '/syn_probe_check.py']
P = ['--a224', '0.25', '--table', T, '--aoa', '0.3', '--mode', 'pw']
ROWS = [('syn20 gate (asyn .2) on syn20 sim', base + [S20] + P + ['--asyn', '0.2'], 'PASS'),
        ('NEG syn20 gate (asyn .2) on REAL pw30 dir', base + [PW30] + P + ['--asyn', '0.2'], 'FAIL'),
        ('control: syn gate asyn 0 on REAL pw30 dir (= a21 gate)', base + [PW30] + P + ['--asyn', '0'], 'PASS'),
        ('NEG syn gate asyn 0 on syn20 sim', base + [S20] + P + ['--asyn', '0'], 'FAIL'),
        ('NEG syn20 gate with the wrong dose (asyn .1) on syn20 sim', base + [S20] + P + ['--asyn', '0.1'], 'FAIL')]
ok = True
for name, cmd, want in ROWS:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    last = [l for l in r.stdout.splitlines() if l.startswith('PROBE_CHECK')]
    got = 'PASS' if last and last[-1].startswith('PROBE_CHECK PASS') else 'FAIL'
    fails = [l.strip() for l in r.stdout.splitlines() if l.strip().startswith('FAIL')]
    ok = ok and got == want
    print('%-62s want %s got %s %s %s' % (name, want, got, last[-1] if last else r.stderr[-200:], ('| ' + fails[0][:110]) if fails and want == 'FAIL' else ''))
print('SYN_SIM_GATE_SELFTEST %s' % ('PASS' if ok else 'FAIL'))
