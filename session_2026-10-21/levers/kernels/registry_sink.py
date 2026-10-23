#!/usr/bin/env python
# registry_sink.py (10-21, Claude, SINK track). Writes levers/kernels/registry_entries_sink.json (entries for main to merge into a FUTURE
# slate registry on Eesh's GO + a PREREG amendment; val/ is NOT touched) and runs the val/tick_a22.py --selftest registry checks on them,
# re-implemented as in stack/kernels/registry_kernels.py: kernel_ok (nb + metadata sha + slug), the gate argv on the entry's simulated
# commit output (last line contains gate_pass, rc 0), key [a-z0-9]+, no key / slug collision with val/slate_a22.json.
import json, os, re, sys, hashlib, subprocess
R = 'C:/Users/eeshs/Downloads/rsna-knee'; K = R + '/session_2026-10-21/levers/kernels'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
IDX = json.load(open(K + '/build_index.json', encoding='utf-8'))
C4T = json.dumps({"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65})
FLAGS = {'pw30': (.3, 0., 0.), 'lomm30': (.3, .05, 0.), 'pw25': (.25, 0., 0.), 'spl': (.25, .05, 0.), 'swide': (.25, .05, .1), 'salt': (.25, 0., .1)}
KEY = {'cvx_pw30': 'cvxpw30', 'cvx_lomm30': 'cvxlomm30', 'cvx_pw25': 'cvxpw25', 'cvx_spl': 'cvxspl', 'cvx_salt': 'cvxsalt', 'cvx_swide': 'cvxswide'}
ent = {}
for nm, e in IDX.items():
    if nm not in KEY:
        continue
    f = FLAGS[e['base']]
    ent[KEY[nm]] = dict(tag=e['tag'], slug='eeshsaxena/rsna-sub-' + e['tag'], kernel_dir='session_2026-10-21/levers/kernels/kernel_' + e['tag'],
                        nb_sha256=e['nb_sha256'], meta_sha256=e['meta_sha256'], base=e['base'], base_nb_sha256=e['base_nb_sha256'],
                        a_oa=f[0], oa_mode='pw', a_loamm=f[1], a_mclmm=f[2], p_oa=e['p_oa'], role=e['role'],
                        gate_script='session_2026-10-21/levers/kernels/sink_commit_gate.py',
                        gate_args=['--a224', '0.25', '--table', C4T, '--aoa', str(f[0]), '--mode', 'pw', '--aloamm', str(f[1]), '--amclmm', str(f[2]), '--poa', str(e['p_oa'])],
                        gate_pass='PASS 27/27', sim_dir='session_2026-10-21/levers/kernels/sim/out_%s_sim' % e['tag'],
                        desc='SINK %s: %s (base %s nb %s) + convex E-OA exponent p 2 on the subtracted pct (cell 18, +6 lines / 2 edited)' % (e['role'], nm, e['base'], e['base_nb_sha256'][:8]))
json.dump(dict(note='SINK registry entries (10-21, Claude). NOT merged anywhere; needs Eesh GO + PREREG amendment (A23). Keys have no underscore (GO typo guard). '
               'Gate = levers/kernels/sink_commit_gate.py, PASS 27/27; sims in levers/kernels/sim/.', entries=ent),
          open(K + '/registry_entries_sink.json', 'w', encoding='utf-8'), indent=1)
res = []
def t(name, cond):
    res.append(bool(cond)); print(('OK   ' if cond else 'FAIL ') + name, flush=True)
SL = json.load(open(R + '/session_2026-10-20/val/slate_a22.json', encoding='utf-8'))['entries']
for k, e in ent.items():
    kd = R + '/' + e['kernel_dir']
    ok_k = (sha(kd + '/rsna_sub_%s.ipynb' % e['tag']) == e['nb_sha256'] and sha(kd + '/kernel-metadata.json') == e['meta_sha256']
            and json.load(open(kd + '/kernel-metadata.json'))['id'] == e['slug'])
    t('registry %s: kernel sha %s + metadata match, id %s' % (k, e['nb_sha256'][:8], e['slug']), ok_k)
    p = subprocess.run(['py', '-3.12', '-I', R + '/' + e['gate_script'], R + '/' + e['sim_dir']] + e['gate_args'], cwd=R, capture_output=True, text=True, encoding='utf-8', errors='replace')
    out = (p.stdout or '') + (p.stderr or ''); last = out.strip().splitlines()[-1] if out.strip() else ''
    t('gate argv %s on its simulated commit -> %s (rc %d)' % (k, last, p.returncode), e['gate_pass'] in last and p.returncode == 0)
    t('key %s matches [a-z0-9]+; no key / slug collision with val/slate_a22.json' % k, re.fullmatch(r'[a-z0-9]+', k) is not None and k not in SL
      and e['slug'] not in {v.get('slug') for v in SL.values()})
print('REGISTRY_SINK_CHECK %s %d/%d' % ('PASS' if all(res) else 'FAIL', sum(res), len(res)))
sys.exit(0 if all(res) else 1)
