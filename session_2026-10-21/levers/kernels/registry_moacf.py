#!/usr/bin/env python
# registry_moacf.py (10-21, Claude, LEVERS synthesis; pattern levers/kernels/registry_sink.py). Writes levers/kernels/registry_entries_moacf.json
# (entries for main to merge into a FUTURE slate registry on Eesh's GO + a PREREG amendment; val/ is NOT touched) and checks them:
# kernel_ok (nb + metadata sha + slug), the gate argv on the entry's simulated commit output (last line contains gate_pass, rc 0),
# key [a-z0-9]+, no key / slug collision with val/slate_a22.json or levers/kernels/registry_entries_sink.json.
import json, re, sys, hashlib, subprocess
R = 'C:/Users/eeshs/Downloads/rsna-knee'; K = R + '/session_2026-10-21/levers/kernels'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
IDX = json.load(open(K + '/build_index_moacf_all.json', encoding='utf-8'))
C4T = json.dumps({"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65})
ent = {}
for nm, e in IDX.items():
    f = e['flags']; key = nm.replace('_', '')
    ent[key] = dict(tag=e['tag'], slug='eeshsaxena/rsna-sub-' + e['tag'], kernel_dir='session_2026-10-21/levers/kernels/kernel_' + e['tag'],
                    nb_sha256=e['nb_sha256'], meta_sha256=e['meta_sha256'], base=e['base'], base_nb_sha256=e['base_nb_sha256'],
                    a_oa=f['a_oa'], oa_mode='pw', a_loamm=f['a_loamm'], a_mclmm=f['a_mclmm'], a_moacf=e['a_moacf'], role='MOACF(%s)' % e['base'],
                    gate_script='session_2026-10-21/levers/kernels/moacf_commit_gate.py',
                    gate_args=['--a224', '0.25', '--table', C4T, '--aoa', str(f['a_oa']), '--mode', 'pw', '--aloamm', str(f['a_loamm']),
                               '--amclmm', str(f['a_mclmm']), '--poa', '1.0', '--amoacf', '0.05'],
                    gate_pass='PASS 29/29', sim_dir='session_2026-10-21/levers/kernels/sim_moacf/out_%s_sim' % e['tag'],
                    desc='LEVERS MOACF(%s): %s (nb %s) + MOA <- Contusion .05 then MOA <- Fracture .05 on the final output (cell 18, +8 lines)'
                         % (e['base'], e['base'], e['base_nb_sha256'][:8]))
json.dump(dict(note='MOACF registry entries (10-21, Claude, LEVERS synthesis). NOT merged anywhere; needs Eesh GO + PREREG amendment (A23). '
               'Keys have no underscore (GO typo guard). Gate = levers/kernels/moacf_commit_gate.py, PASS 29/29; sims in levers/kernels/sim_moacf/.',
               entries=ent), open(K + '/registry_entries_moacf.json', 'w', encoding='utf-8'), indent=1)
res = []
def t(name, cond):
    res.append(bool(cond)); print(('OK   ' if cond else 'FAIL ') + name, flush=True)
SL = json.load(open(R + '/session_2026-10-20/val/slate_a22.json', encoding='utf-8'))['entries']
SK = json.load(open(K + '/registry_entries_sink.json', encoding='utf-8'))['entries']
for k, e in ent.items():
    kd = R + '/' + e['kernel_dir']
    ok_k = (sha(kd + '/rsna_sub_%s.ipynb' % e['tag']) == e['nb_sha256'] and sha(kd + '/kernel-metadata.json') == e['meta_sha256']
            and json.load(open(kd + '/kernel-metadata.json'))['id'] == e['slug'])
    t('registry %s: kernel sha %s + metadata match, id %s' % (k, e['nb_sha256'][:8], e['slug']), ok_k)
    p = subprocess.run(['py', '-3.12', '-I', R + '/' + e['gate_script'], R + '/' + e['sim_dir']] + e['gate_args'], cwd=R, capture_output=True, text=True, encoding='utf-8', errors='replace')
    out = (p.stdout or '') + (p.stderr or ''); last = out.strip().splitlines()[-1] if out.strip() else ''
    t('gate argv %s on its simulated commit -> %s (rc %d)' % (k, last, p.returncode), e['gate_pass'] in last and p.returncode == 0)
    slugs = {v.get('slug') for v in SL.values()} | {v.get('slug') for v in SK.values()}
    t('key %s matches [a-z0-9]+; no key / slug collision with val/slate_a22.json or registry_entries_sink.json' % k,
      re.fullmatch(r'[a-z0-9]+', k) is not None and k not in SL and k not in SK and e['slug'] not in slugs)
print('REGISTRY_MOACF_CHECK %s %d/%d' % ('PASS' if all(res) else 'FAIL', sum(res), len(res)))
sys.exit(0 if all(res) else 1)
