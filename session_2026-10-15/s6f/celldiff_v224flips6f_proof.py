#!/usr/bin/env python
# celldiff_v224flips6f_proof.py (10-15, Claude subagent): proof that V224-FLIP-AO + S6fall (kernel_v224flips6f) is EXACTLY the
# submitted V224-FLIP notebook (session_2026-10-14/v224/kernel_v224flip = eeshsaxena/rsna-sub-v224flip v1, ref 57174021) with
# ONLY 3 code cells APPENDED, and that the S6F driver those cells write is nartaa's v2 B path (verbatim statements) + our listed
# parts. Independent of build_v224s6f.py (imports nothing from it; re-derives everything from the files). No em dashes.
#   usage: py -3.12 -I celldiff_v224flips6f_proof.py [--nb NOTEBOOK] [--meta kernel-metadata.json]  -> S6F_CELLDIFF PASS/FAIL n/m
import os, sys, ast, json, base64, hashlib, difflib, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = os.path.dirname(HERE)
ROOT = os.path.dirname(SESS)
ap = argparse.ArgumentParser()
ap.add_argument('--nb', default=os.path.join(HERE, 'kernel_v224flips6f', 'rsna_sub_v224flips6f.ipynb'))
ap.add_argument('--meta', default=None)
A = ap.parse_args()
FLIP_NB = os.path.join(ROOT, 'session_2026-10-14', 'v224', 'kernel_v224flip', 'rsna_sub_v224flip.ipynb')
FLIP_META = os.path.join(ROOT, 'session_2026-10-14', 'v224', 'kernel_v224flip', 'kernel-metadata.json')
FLIP_SHA = '73a9d5620d0a637ed171ecec13bcf243bd0af67992b5b4cb8ad7f35bf72b1b1b'
PULLED = os.path.join(SESS, 'v224rm', 'pulled_v224flip', 'rsna-sub-v224flip.ipynb')
NARTAA_V2 = os.path.join(HERE, 'pulled_v2', 'rsna-knee-s6fall-solo.py')
NARTAA_V2_SHA = '6e87350d5ae092b7a889546ca13517303523bc469f7e611ffe945913434d22fe'
BUILD = os.path.join(HERE, 'build_receipt_v224flips6f.json')
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print(f'  {"OK  " if ok else "FAIL"} {name} {info}', flush=True)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def src(c):
    s = c['source']
    return ''.join(s) if isinstance(s, list) else s


def dumps(nb):
    return json.dumps(nb, ensure_ascii=False, separators=(',', ':'))


def tops(text):
    lines = text.splitlines(keepends=True)
    out = {}
    for node in ast.parse(text).body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
                elif isinstance(t, ast.Tuple):
                    names += [e.id for e in t.elts if isinstance(e, ast.Name)]
        for nm in names:
            out[nm] = ''.join(lines[node.lineno - 1:node.end_lineno])
    return out


flip_raw, new_raw = open(FLIP_NB, 'rb').read(), open(A.nb, 'rb').read()
flip, new = json.loads(flip_raw), json.loads(new_raw)
print('S6F_CELLDIFF base', os.path.relpath(FLIP_NB, ROOT), 'new', os.path.relpath(A.nb, ROOT), flush=True)
chk('base file == the V224-FLIP kernel that was pushed and submitted (sha 73a9d562)', sha(flip_raw) == FLIP_SHA, sha(flip_raw)[:16])
pl = json.load(open(PULLED, encoding='utf-8'))
chk('base cells == kaggle kernels pull of eeshsaxena/rsna-sub-v224flip (type + source of all 15 cells)',
    [(c['cell_type'], src(c)) for c in pl['cells']] == [(c['cell_type'], src(c)) for c in flip['cells']])
FC, NC = flip['cells'], new['cells']
chk('cell counts: base 15, new 18', (len(FC), len(NC)) == (15, 18), (len(FC), len(NC)))
sm = difflib.SequenceMatcher(a=[(c['cell_type'], src(c)) for c in FC], b=[(c['cell_type'], src(c)) for c in NC], autojunk=False)
ops = sm.get_opcodes()
for tag, i1, i2, j1, j2 in ops:
    print(f'    {tag:7s} base[{i1}:{i2}] -> new[{j1}:{j2}]', flush=True)
chk('alignment: base[0:15] EQUAL new[0:15], then one INSERT new[15:18] at the END (append only; no replace / delete / reorder)',
    ops == [('equal', 0, 15, 0, 15), ('insert', 15, 15, 15, 18)], ops)
chk('cells 0-14 equal as whole cell dicts (metadata, outputs, execution_count included)', all(NC[i] == FC[i] for i in range(min(15, len(NC)))))
back = json.loads(new_raw)
back['cells'] = back['cells'][:15]
chk('new notebook minus cells 15-17 re-serialises to the base FILE byte for byte', dumps(back).encode('utf-8') == flip_raw,
    sha(dumps(back).encode('utf-8'))[:16])
chk('notebook-level metadata / nbformat equal', new.get('metadata') == flip.get('metadata') and
    (new.get('nbformat'), new.get('nbformat_minor')) == (flip.get('nbformat'), flip.get('nbformat_minor')))
hdr = ['# [eeshsaxena V224-FLIP-S6F 1/3 SETUP', '# [eeshsaxena V224-FLIP-S6F 2/3 RUN', '# [eeshsaxena V224-FLIP-S6F 3/3 BLEND']
chk('appended cells are code cells with the three S6F headers, no outputs', len(NC) == 18 and all(NC[15 + k]['cell_type'] == 'code' and
                                                                                                src(NC[15 + k]).startswith(hdr[k]) and NC[15 + k].get('outputs') == []
                                                                                                for k in range(3)))
chk('base stage order unchanged: r384 run (4) < R7 assets + pip (6) < FLIP forced (7) < R7 run (8) < r224 run (11) < stage 3 (13) < '
    'V224 canary (14) < S6F (15-17)', "subprocess.run([sys.executable, '-u', '/tmp/raptor_run.py']" in src(NC[4]) and "'pip', 'install'" in src(NC[6])
    and 'FLIP_TTA_FORCED_ON = True' in src(NC[7]) and 'probe_dual.py' in src(NC[8]) and '/tmp/raptor224_run.py' in src(NC[11]) and
    "sub.to_csv(WORK / 'submission.csv'" in src(NC[13]) and src(NC[14]).startswith('# [eeshsaxena CANARY'))
needles = ['7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', '3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6',
           '8401812c', 'FLIP_TTA_FORCED', 'CANARY_CLEAN', 'V224_DONE', 'NB_T0 = time.time()']
chk('every base sha / flip / canary needle sits in the same base cells', all([k for k in range(15) if nd in src(NC[k])] ==
                                                                            [k for k in range(15) if nd in src(FC[k])] for nd in needles))
# ---- the driver cell 15 writes ----------------------------------------------------------------------------------------------
s15 = src(NC[15]) if len(NC) > 15 else ''
try:
    b64 = s15.split("DRIVER_B64 = '", 1)[1].split("'", 1)[0]
    dsha = s15.split("DRIVER_SHA = '", 1)[1].split("'", 1)[0]
    drv = base64.b64decode(b64)
except Exception:
    drv, dsha = b'', None
bld = json.load(open(BUILD, encoding='utf-8')) if os.path.isfile(BUILD) else {}
chk('driver embedded in cell 15: sha == its DRIVER_SHA constant == build receipt', sha(drv) == dsha == bld.get('driver_sha256'), str(dsha)[:16])
v2_raw = open(NARTAA_V2, 'rb').read()
chk('nartaa v2 source == the pulled public file (sha 6e87350d) == the self sha in nartaa\'s commit log',
    sha(v2_raw) == NARTAA_V2_SHA)
D, N = tops(drv.decode('utf-8')) if drv else {}, tops(v2_raw.decode('utf-8'))
shared = sorted(set(D) & set(N))
same = [n for n in shared if D[n] == N[n]]
diff = [n for n in shared if D[n] != N[n]]
ours = sorted(set(D) - set(N))
print('    info verbatim (identical statement text) %d: %s' % (len(same), ', '.join(same)), flush=True)
print('    info same name, our text %d: %s' % (len(diff), ', '.join(diff)), flush=True)
print('    info ours only %d: %s' % (len(ours), ', '.join(ours)), flush=True)
chk('B path verbatim: prepare_inputs, probe_b_stack, LAUNCHER, B_PROBE, collect_part, b_cmd, b_env, split_parts, run_group, '
    'extract_payload, B_MODEL, SOURCE_SHA, CODE_SHA, B_GEOMETRY, B_AMP, LAB identical to nartaa v2',
    {'prepare_inputs', 'probe_b_stack', 'LAUNCHER', 'B_PROBE', 'collect_part', 'b_cmd', 'b_env', 'split_parts', 'run_group',
     'extract_payload', 'B_MODEL', 'SOURCE_SHA', 'CODE_SHA', 'B_GEOMETRY', 'B_AMP', 'LAB'} <= set(same))
chk('our text only where listed: paths (KROOT, OUT, TMP), MODE comment, receipt dict, main', set(diff) <= {'KROOT', 'OUT', 'TMP', 'MODE', '_RECEIPT', 'main'}, diff)
excluded = ['remove_submission', '_watchdog', 'arm_guard', 'guard_left', 'run_a', 'a_cmd', 'a_done', 'A_B64', 'run_b', 'blend_and_write',
            'write_submission', 'blend_with_failures', 'run_parity', 'b_fail_cap', 'GUARD_S', 'GUARD_MAX_S']
chk('never included: remove_submission, hang guard, component A, run_b (hard errors), blend / submission writer, parity mode, fail cap',
    not [n for n in excluded if n in D], [n for n in excluded if n in D])
chk('ours: configure, ENV2, env_isolation, collect_part_csv, run_b_s6f, main', {'configure', 'ENV2', 'env_isolation', 'collect_part_csv',
                                                                               'run_b_s6f'} <= set(ours) and 'main' in D, ours)
fm = json.load(open(FLIP_META, encoding='utf-8'))
nm = json.load(open(A.meta or os.path.join(os.path.dirname(A.nb), 'kernel-metadata.json'), encoding='utf-8'))
diffk = sorted(k for k in set(fm) | set(nm) if fm.get(k) != nm.get(k))
chk('kernel-metadata == V224-FLIP except id / title / code_file / dataset_sources (= the 3 V224 datasets + nartaa/rsna-knee-s6fall-weights '
    '+ glorysing/rsna-knee-offline-codecs-r09); docker pin, T4 x2, private, internet off', diffk == ['code_file', 'dataset_sources', 'id', 'title']
    and nm['dataset_sources'] == fm['dataset_sources'] + ['nartaa/rsna-knee-s6fall-weights', 'glorysing/rsna-knee-offline-codecs-r09']
    and nm['is_private'] is True and nm['enable_internet'] is False and nm['enable_gpu'] is True, {k: (fm.get(k), nm.get(k)) for k in diffk})
print('    info new notebook sha256', sha(new_raw), flush=True)
print(f'S6F_CELLDIFF {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}', flush=True)
sys.exit(0 if all(R) else 1)
