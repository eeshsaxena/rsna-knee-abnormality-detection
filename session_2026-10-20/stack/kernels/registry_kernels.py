#!/usr/bin/env python
# registry_kernels.py (10-20 ~23:30 IST, Claude, STACK BUILD round). Writes stack/kernels/registry_entries_kernels.json (entries for main to
# merge into val/slate_a22.json on Eesh's GO; val/ is NOT touched here) and runs the two registry checks of val/tick_a22.py --selftest on them,
# re-implemented verbatim (kernel_ok = nb + metadata sha + slug; gate argv on the entry's simulated commit output, last line of stdout+stderr
# must contain gate_pass and rc 0), using the sims in stack/kernels/sim/. Also checks the GO-parse hazard of underscore keys.
# Keys have no underscore on purpose: tick_a22.parse_go treats a token that is not a slate key and not [a-z0-9]+ as free text (break),
# so with underscore keys a typo like 'GO pw25 s_wid' would silently drop the entry instead of invalidating the GO.
import json, os, re, sys, hashlib, subprocess
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
K = os.path.join(R, 'session_2026-10-20', 'stack', 'kernels')
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
IDX = json.load(open(os.path.join(K, 'build_index.json'), encoding='utf-8'))
C4T = json.dumps({"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65})
DESC = {'S_safe': 'STACK S_safe: pw30 57425970 + MCL <- Medial Meniscus a .1 on the final output (cell 18 after E-OA; MCL column only)',
        'S_alt': 'STACK S_alt: pw30 with A_OA .25 (= pw25) + MCL <- MM a .1 (cell 18)',
        'S_wide': 'STACK S_wide: pw30 with A_OA .25 + Lateral OA <- MM a .05 (lomm05) + MCL <- MM a .1 (cell 18)',
        'S_mid': 'STACK S_mid: pw30 + lomm05 + MCL <- MM a .1 (cell 18)'}
KEY = {'S_safe': 'ssafe', 'S_alt': 'salt', 'S_wide': 'swide', 'S_mid': 'smid'}
ent = {}
for nm, e in IDX.items():
    if nm == 'CTRL_off':
        continue
    rc = json.load(open(os.path.join(K, 'build_receipt_%s.json' % e['tag']), encoding='utf-8'))
    ent[KEY[nm]] = dict(tag=e['tag'], slug='eeshsaxena/rsna-sub-' + e['tag'], kernel_dir='session_2026-10-20/stack/kernels/kernel_' + e['tag'],
                        nb_sha256=e['nb_sha256'], meta_sha256=e['meta_sha256'], base='c4', base_scored_nb_sha256=rc['base_nb_sha256'],
                        a_oa=rc['a_oa'], oa_mode='pw', a_loamm=rc['a_loamm'] or 0.0, a_mclmm=rc['a_mclmm'],
                        gate_script='session_2026-10-20/stack/kernels/stack_commit_gate.py',
                        gate_args=['--a224', '0.25', '--table', C4T, '--aoa', str(rc['a_oa']), '--mode', 'pw', '--aloamm', str(rc['a_loamm'] or 0.0), '--amclmm', str(rc['a_mclmm'])],
                        gate_pass='PASS 26/26', sim_dir='session_2026-10-20/stack/kernels/sim/out_%s_sim' % e['tag'], desc=DESC[nm])
reg = dict(note=('STACK BUILD registry entries (10-20 ~23:30 IST, Claude). NOT merged into val/slate_a22.json (val/ is read-only for this round). '
                 'Main merges on Eesh GO + PREREG Amendment 22. Before tick_a22.py --selftest, copy each sim_dir to val/sim/ (the stack/sim/ inventory '
                 'sims of the same tags append a 2nd EOA marker and FAIL this gate by design). Keys have no underscore (GO-parse typo guard).'), entries=ent)
json.dump(reg, open(os.path.join(K, 'registry_entries_kernels.json'), 'w', encoding='utf-8'), indent=1)
res = []
def t(name, cond):
    res.append(bool(cond)); print(('OK   ' if cond else 'FAIL ') + name, flush=True)
def kernel_ok(e):
    kd = os.path.join(R, e['kernel_dir'])
    try:
        return (sha(os.path.join(kd, 'rsna_sub_%s.ipynb' % e['tag'])) == e['nb_sha256'] and sha(os.path.join(kd, 'kernel-metadata.json')) == e['meta_sha256']
                and json.load(open(os.path.join(kd, 'kernel-metadata.json')))['id'] == e['slug'])
    except Exception:
        return False
def run(cmd, timeout=900):
    p = subprocess.run(cmd, cwd=R, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
    return p.returncode, (p.stdout or '') + (p.stderr or '')
for k, e in ent.items():
    t('registry %s: kernel %s sha %s + metadata match, id %s' % (k, e['kernel_dir'], e['nb_sha256'][:8], e['slug']), kernel_ok(e))
    rc, out = run(['py', '-3.12', '-I', os.path.join(R, e['gate_script']), os.path.join(R, e['sim_dir'])] + e['gate_args'])
    last = out.strip().splitlines()[-1] if out.strip() else ''
    t('gate argv %s on its simulated commit output -> %s (rc %d, stderr-free last line)' % (k, last, rc), e['gate_pass'] in last and rc == 0)
    t('key %s matches [a-z0-9]+ (tick_a22.parse_go typo guard applies to it)' % k, re.fullmatch(r'[a-z0-9]+', k) is not None)
    t('%s: not a key of the current val/slate_a22.json (no collision)' % k, k not in json.load(open(os.path.join(R, 'session_2026-10-20', 'val', 'slate_a22.json'), encoding='utf-8'))['entries'])
print('REGISTRY_KERNELS_CHECK %s %d/%d' % ('PASS' if all(res) else 'FAIL', sum(res), len(res)))
sys.exit(0 if all(res) else 1)
