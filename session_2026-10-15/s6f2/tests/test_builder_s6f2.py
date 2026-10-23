#!/usr/bin/env python
# test_builder_s6f2.py (10-15, Claude subagent): build_v224s6f2.py properties. Offline, laptop CPU, -I. Every build goes to a temp dir.
#   B1 --base flip at the defaults == s6f/kernel_v224flips6f byte for byte: notebook (9c0e2558), kernel-metadata, build receipt,
#      celldiff, driver (the FLIP path of the generalised builder reproduces the pushed S6F kernel)
#   B2 --base flrm / flips20 rebuild byte-identical to the s6f2 project files (deterministic), and their receipts record the s6f
#      template shas == the current s6f files
#   B3 refusals: a submit tag with another G / w / budget, a submit tag on the wrong base, --out inside a pushed-kernel dir, a kill
#      control on a submit tag, the old cut-control flags
#   B4 control builds (never submit): flrm skip control (G 0) and flips20 kill control: receipt policy.control; SETUP / BLEND == the
#      submit cells except the tag; RUN == the submit RUN except G (skip) / the s6f kill-control splices (kill)
#   B5 nothing under s6f/ changed (sha of every file the builder reads, before vs after all builds)
#   usage: py -3.12 -I tests/test_builder_s6f2.py  -> TEST_BUILDER_S6F2 PASS/FAIL n/m
import os, sys, json, glob, shutil, hashlib, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
S6F2 = os.path.dirname(HERE)
SESS = os.path.dirname(S6F2)
S6F = os.path.join(SESS, 's6f')
BUILDER = os.path.join(S6F2, 'build_v224s6f2.py')
T = os.path.join(HERE, '_tmp_builder')
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def build(out, *args):
    os.makedirs(out, exist_ok=True)
    r = subprocess.run([sys.executable, '-I', BUILDER, '--out', out] + list(args), capture_output=True, text=True, timeout=300)
    return r.returncode, ((r.stdout + r.stderr).strip().splitlines() or [''])[-1]


def src(c):
    return ''.join(c['source']) if isinstance(c['source'], list) else c['source']


def cells(p):
    return [src(c) for c in json.load(open(p, encoding='utf-8'))['cells']]


def files(tag):
    return ['kernel_%s/rsna_sub_%s.ipynb' % (tag, tag), 'kernel_%s/kernel-metadata.json' % tag, 'build_receipt_%s.json' % tag,
            'celldiff_%s.txt' % tag, 's6f_driver_built_%s.py' % tag]


shutil.rmtree(T, ignore_errors=True)
S6F_READ = sorted(set([os.path.join(S6F, f) for f in ('cell_setup_tpl.py', 'cell_run_tpl.py', 'cell_blend_tpl.py', 's6f_driver_src.py',
                                                     'cell_run_killctl_tpl.py', 'build_receipt_v224flips6f.json', 'celldiff_v224flips6f.txt',
                                                     's6f_driver_built_v224flips6f.py')]
                      + glob.glob(os.path.join(S6F, 'kernel_v224flips6f', '*')) + glob.glob(os.path.join(S6F, 'pulled_v2', '*'))
                      + glob.glob(os.path.join(S6F, 'ref_output', 'k1', 'k1_receipt.json'))))
before = {p: sha(p) for p in S6F_READ}
# ---- B1 ----------------------------------------------------------------------------------------------------------------------
print('B1 flip default build == the pushed S6F kernel files', flush=True)
rc, last = build(os.path.join(T, 'flip'), '--base', 'flip')
same = {f: (rc == 0 and sha(os.path.join(T, 'flip', f)) == sha(os.path.join(S6F, f))) for f in files('v224flips6f')}
chk('B1: --base flip == s6f/kernel_v224flips6f byte for byte (notebook 9c0e2558, metadata, receipt, celldiff, driver)', all(same.values())
    and sha(os.path.join(T, 'flip', files('v224flips6f')[0])).startswith('9c0e2558'), same)
# ---- B2 ----------------------------------------------------------------------------------------------------------------------
print('B2 flrm / flips20 deterministic rebuilds', flush=True)
for base, tag in (('flrm', 'v224flrms6f'), ('flips20', 'v224flips6fs20')):
    rc, last = build(os.path.join(T, base), '--base', base)
    same = {f: (rc == 0 and sha(os.path.join(T, base, f)) == sha(os.path.join(S6F2, f))) for f in files(tag)}
    rec = json.load(open(os.path.join(S6F2, 'build_receipt_%s.json' % tag), encoding='utf-8'))
    tpl = rec.get('s6f_templates_sha256') or {}
    chk('B2 %s: rebuild == the project files byte for byte; receipt template shas == the current s6f files (%d)' % (base, len(tpl)),
        all(same.values()) and tpl and all(sha(os.path.join(S6F, f)) == h for f, h in tpl.items()), {k: v for k, v in same.items() if not v})
# ---- B3 ----------------------------------------------------------------------------------------------------------------------
print('B3 refusals', flush=True)
for name, args, needle in (
        ('flrm submit tag with G 5.75', ['--base', 'flrm', '--g-start-h', '5.75'], 'runtime-reviewed'),
        ('flips20 submit tag with w .3', ['--base', 'flips20', '--w', '0.3'], 'pre-registered'),
        ('flips20 submit tag with budget 3.0', ['--base', 'flips20', '--budget-h', '3.0'], 'runtime-reviewed'),
        ('submit tag v224flips6f on base flrm', ['--base', 'flrm', '--tag', 'v224flips6f'], 'own base'),
        ('kill control on a submit tag', ['--base', 'flips20', '--kill-control'], 'never a submit tag'),
        ('old cut-control flags', ['--base', 'flrm', '--tag', 'v224flrms6fcut', '--budget-h', '0.03', '--min-run-s', '0'], 'never start the driver')):
    rc, last = build(os.path.join(T, 'refuse'), *args)
    chk('B3: refuses %s' % name, rc != 0 and needle in last, last[:150])
r = subprocess.run([sys.executable, '-I', BUILDER, '--base', 'flrm', '--out', S6F], capture_output=True, text=True, timeout=300)
chk('B3: refuses --out inside s6f/ (a pushed-kernel dir)', r.returncode != 0 and 'pushed-kernel dir' in (r.stdout + r.stderr), (r.stdout + r.stderr).strip()[-120:])
r = subprocess.run([sys.executable, '-I', BUILDER, '--base', 'flips20', '--out', os.path.join(SESS, 'v224f')], capture_output=True, text=True, timeout=300)
chk('B3: refuses --out inside v224f/ (a pushed-kernel dir)', r.returncode != 0 and 'pushed-kernel dir' in (r.stdout + r.stderr))
# ---- B4 ----------------------------------------------------------------------------------------------------------------------
print('B4 control builds', flush=True)
rc, last = build(os.path.join(T, 'skip'), '--base', 'flrm', '--tag', 'v224flrms6fskip', '--g-start-h', '0')
sub = cells(os.path.join(S6F2, 'kernel_v224flrms6f', 'rsna_sub_v224flrms6f.ipynb'))
ctl = cells(os.path.join(T, 'skip', 'kernel_v224flrms6fskip', 'rsna_sub_v224flrms6fskip.ipynb')) if rc == 0 else []
rec = json.load(open(os.path.join(T, 'skip', 'build_receipt_v224flrms6fskip.json'), encoding='utf-8')) if rc == 0 else {}
chk('B4 flrm skip control: policy.control skip; cells 0-16 == submit; SETUP / BLEND == submit but the tag; RUN == submit but G 0.0',
    rc == 0 and (rec.get('policy') or {}).get('control') == 'skip' and ctl[:17] == sub[:17]
    and ctl[17] == sub[17].replace("'v224flrms6f'", "'v224flrms6fskip'") and ctl[19] == sub[19].replace("'v224flrms6f'", "'v224flrms6fskip'")
    and ctl[18] == sub[18].replace('MIN_RUN_S = 6.0, 3.3', 'MIN_RUN_S = 0.0, 3.3'), last[:120])
rc, last = build(os.path.join(T, 'kill'), '--base', 'flips20', '--tag', 'v224flips6fs20kill', '--kill-control', '--budget-h', '0.01', '--min-run-s', '-1000')
sub = cells(os.path.join(S6F2, 'kernel_v224flips6fs20', 'rsna_sub_v224flips6fs20.ipynb'))
ctl = cells(os.path.join(T, 'kill', 'kernel_v224flips6fs20kill', 'rsna_sub_v224flips6fs20kill.ipynb')) if rc == 0 else []
rec = json.load(open(os.path.join(T, 'kill', 'build_receipt_v224flips6fs20kill.json'), encoding='utf-8')) if rc == 0 else {}
s6k = None
rk, lk = build(os.path.join(T, 'kill_flip'), '--base', 'flip', '--tag', 'v224flips6fkill', '--kill-control', '--budget-h', '0.01', '--min-run-s', '-1000')
if rk == 0:
    s6k = cells(os.path.join(T, 'kill_flip', 'kernel_v224flips6fkill', 'rsna_sub_v224flips6fkill.ipynb'))[16]
hdr_len = len(ctl[25].split('\n', 2)[0]) + 1 + len(ctl[25].split('\n', 2)[1]) + 1 if rc == 0 else 0
chk('B4 flips20 kill control: policy.control kill; cells 0-23 == submit; SETUP / BLEND == submit but the tag; RUN == the s6f kill-control RUN '
    '(flip base, same flags) with the s6f2 header and G 6.25',
    rc == 0 and rk == 0 and (rec.get('policy') or {}).get('control') == 'kill' and ctl[:24] == sub[:24]
    and ctl[24] == sub[24].replace("'v224flips6fs20'", "'v224flips6fs20kill'") and ctl[26] == sub[26].replace("'v224flips6fs20'", "'v224flips6fs20kill'")
    and ctl[25][:hdr_len] == sub[25][:hdr_len] and ctl[25][hdr_len:].replace('MIN_RUN_S = 6.25, 0.01', 'MIN_RUN_S = 5.75, 0.01') == s6k, (rc, rk, last[:100]))
# ---- B5 ----------------------------------------------------------------------------------------------------------------------
after = {p: sha(p) for p in S6F_READ}
chk('B5: no file under s6f/ changed (%d files the builder reads, sha before == after)' % len(before), before == after,
    [os.path.relpath(p, SESS) for p in before if before[p] != after.get(p)])
shutil.rmtree(T, ignore_errors=True)
print('TEST_BUILDER_S6F2 %s %d/%d' % ('PASS' if all(R) else 'FAIL', sum(R), len(R)), flush=True)
sys.exit(0 if all(R) else 1)
