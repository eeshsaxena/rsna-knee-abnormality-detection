#!/usr/bin/env python
# celldiff_v224s6fkc_proof.py (10-15, Claude subagent, kcstage): proof that an S6F+KC kernel (build_v224s6fkc.py) is EXACTLY its base
# (the pushed S6F kernel eeshsaxena/rsna-sub-v224flips6f v1, or its J twin eeshsaxena/rsna-sub-v224flips6fj v1) with ONLY 3 code cells
# APPENDED, and that the files those cells write are the reviewed ones. Independent of the builder: imports nothing from it,
# re-derives everything from the files on disk. No em dashes.
#   usage: py -3.12 -I celldiff_v224s6fkc_proof.py KERNEL_DIR [--build build_receipt_TAG.json] [--kc-dir DS_DIR] -> KC_CELLDIFF PASS/FAIL n/m
import os, re, io, sys, json, base64, hashlib, difflib, argparse, tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = os.path.dirname(HERE)
BASES = {'9c0e2558d51e4adb0cd7474fa6c53aa86e4a5f146f02af1acc8e98ec9323ac0e': (os.path.join(SESS, 's6f', 'kernel_v224flips6f'), 18, 's6f',
                                                                              'V224S6F_CHECK PASS 31/31', os.path.join(SESS, 's6f', 'S6F_READY.txt')),
         '1868634ea68ef131035a01a347f4b1c4471f0a3d7a58b8a22f9efec3a309b02a': (os.path.join(SESS, 'jpeg', 'overlay', 'r2', 'kernel_v224flips6fj'), 19,
                                                                              's6fj', 'J_CHECK PASS 16/16',
                                                                              os.path.join(SESS, 'jpeg', 'overlay', 'r2', 'J_READY_v224flips6fj.txt'))}
ap = argparse.ArgumentParser()
ap.add_argument('kernel_dir')
ap.add_argument('--build', default=None)
ap.add_argument('--kc-dir', default=None)
A = ap.parse_args()
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


def code_only(t):
    return ' '.join(tok.string for tok in tokenize.generate_tokens(io.StringIO(t).readline) if tok.type != tokenize.COMMENT)


def env_path_changes(t):
    """AST: any assignment / deletion / mutating call on os.environ or sys.path (or os.putenv) in a cell."""
    import ast
    def dotted(n):
        parts = []
        while isinstance(n, ast.Attribute):
            parts.append(n.attr)
            n = n.value
        if isinstance(n, ast.Name):
            parts.append(n.id)
        return '.'.join(reversed(parts))
    out = []
    for node in ast.walk(ast.parse(t)):
        tg = []
        if isinstance(node, (ast.Assign, ast.Delete)):
            tg = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            tg = [node.target]
        for x in tg:
            base = x.value if isinstance(x, ast.Subscript) else x
            if dotted(base) in ('os.environ', 'sys.path'):
                out.append(('assign', dotted(base), node.lineno))
        if isinstance(node, ast.Call):
            f = dotted(node.func)
            if f in ('os.putenv', 'os.unsetenv') or (f.startswith(('os.environ.', 'sys.path.')) and f.rsplit('.', 1)[1] in
                                                     ('update', 'pop', 'setdefault', 'clear', 'popitem', 'insert', 'append', 'extend',
                                                      'remove', '__setitem__', '__delitem__')):
                out.append(('call', f, node.lineno))
    return out


meta_new = json.load(open(os.path.join(A.kernel_dir, 'kernel-metadata.json'), encoding='utf-8'))
nb_path = os.path.join(A.kernel_dir, meta_new['code_file'])
NEW_RAW = open(nb_path, 'rb').read()
NEW = json.loads(NEW_RAW)
TAG = meta_new['id'].split('rsna-sub-')[-1]
B = json.load(open(A.build or os.path.join(HERE, 'build_receipt_%s.json' % TAG), encoding='utf-8'))
print('KC_CELLDIFF new %s (tag %s, sha %s) build receipt base %s' % (nb_path, TAG, sha(NEW_RAW)[:16], B.get('base')), flush=True)
base_sha = B.get('base_sha256')
chk('base sha is one of the two pushed + gated bases (S6F v1 9c0e2558 / S6F-J v1 1868634e)', base_sha in BASES, base_sha)
bdir, nbase, bname, verdict, ready = BASES[base_sha]
bmeta = json.load(open(os.path.join(bdir, 'kernel-metadata.json'), encoding='utf-8'))
BASE_RAW = open(os.path.join(bdir, bmeta['code_file']), 'rb').read()
chk('base file on disk == the pinned sha; its READY file records the pushed slug + the gate verdict', sha(BASE_RAW) == base_sha and
    verdict in open(ready, encoding='utf-8').read() and bmeta['id'] in open(ready, encoding='utf-8').read(), (sha(BASE_RAW)[:16], verdict))
BASE = json.loads(BASE_RAW)
C0, C1 = BASE['cells'], NEW['cells']
chk('cell counts: base %d, new %d' % (len(C0), len(C1)), len(C0) == nbase and len(C1) == nbase + 3, (len(C0), len(C1)))
ops = difflib.SequenceMatcher(None, [(c['cell_type'], src(c)) for c in C0], [(c['cell_type'], src(c)) for c in C1], autojunk=False).get_opcodes()
for o in ops:
    print('    %-7s base[%d:%d] -> new[%d:%d]' % o, flush=True)
chk('alignment: base[0:%d] EQUAL new[0:%d], then one INSERT new[%d:%d] at the END (append only)' % (nbase, nbase, nbase, nbase + 3),
    ops == [('equal', 0, nbase, 0, nbase), ('insert', nbase, nbase, nbase, nbase + 3)], ops)
chk('base cells equal as whole cell dicts (metadata, outputs, execution_count included)', all(C1[i] == C0[i] for i in range(nbase)))
back = dict(NEW)
back['cells'] = C1[:nbase]
chk('new notebook minus the appended cells re-serialises to the base FILE byte for byte', dumps(back).encode('utf-8') == BASE_RAW)
chk('notebook-level metadata / nbformat equal', NEW.get('metadata') == BASE.get('metadata') and NEW.get('nbformat') == BASE.get('nbformat')
    and NEW.get('nbformat_minor') == BASE.get('nbformat_minor'))
if len(C1) != nbase + 3 or len(C0) != nbase:
    print('KC_CELLDIFF FAIL %d/%d tag %s (cell counts: nothing else is checkable)' % (sum(R), len(R) + 1, TAG), flush=True)
    sys.exit(1)
K = [src(c) for c in C1[nbase:]]
heads = ['# [eeshsaxena %s KC %d/3 ' % (TAG, k) for k in (1, 2, 3)]
chk('appended cells: code cells, the three KC headers in order, no outputs', all(C1[nbase + k]['cell_type'] == 'code' and K[k].startswith(heads[k])
                                                                               and not C1[nbase + k].get('outputs') for k in range(3)))
S0 = [src(c) for c in C0]
s6 = [next((i for i, s in enumerate(S0) if s.startswith('# [eeshsaxena V224-FLIP-S6F %d/3 ' % k)), -1) for k in (1, 2, 3)]
v224 = next((i for i, s in enumerate(S0) if s.startswith('# [eeshsaxena CANARY')), -1)
chk('base stage order unchanged and the KC cells come after the S6F BLEND (the last base cell)', 0 <= v224 < s6[0] < s6[1] < s6[2] == nbase - 1,
    (v224, s6))
ACL_S6F = "'ACL'" in S0[s6[2]] and "W_S6F, COV_MIN, CANARY_COL = 0.2, 0.98, 'ACL'" in S0[s6[2]]
chk("canaries separable: KC canary Baker's (BLEND constant), S6F = ACL, V224 = Fracture, none shared", ACL_S6F and
    "\"Baker's\"" in K[2] and "W_KC, COV_MIN, CANARY_COL, ON_FAIL = 0.2, 0.98, \"Baker's\", 'canary'" in K[2] and
    "_canary['fired'].append('r224->Fracture')" in S0[v224])
# embedded driver / serve wrapper
emb = {}
for nm, var in (('kc_driver.py', 'DRIVER'), ('kc_serve.py', 'SERVE')):
    m64 = re.search(r"%s_B64 = '([A-Za-z0-9+/=]+)'" % var, K[0])
    msh = re.search(r"DRIVER_SHA, SERVE_SHA = '([0-9a-f]{64})', '([0-9a-f]{64})'", K[0])
    b = base64.b64decode(m64.group(1)) if m64 else b''
    emb[nm] = b
    want = msh.group(1 if var == 'DRIVER' else 2) if msh else None
    disk = open(os.path.join(HERE, 'kc_driver_src.py' if var == 'DRIVER' else 'kc_serve.py'), 'rb').read()
    chk('%s embedded in the SETUP cell: sha == its constant == build receipt == the reviewed file on disk' % nm, sha(b) == want ==
        B.get('driver_sha256' if var == 'DRIVER' else 'serve_sha256') == sha(disk), sha(b)[:16])
srv = emb['kc_serve.py'].decode('utf-8', 'replace')
drv = emb['kc_driver.py'].decode('utf-8', 'replace')
chk('kc_serve: kc_worker.py\'s own __main__ block run in its module namespace with ONE substitution (study_inputs <- side table); pins '
    'kc_worker 4af60f6f / kc_common 45255ae0 / fleet models / d96 30aa72fd', srv.count('kw.study_inputs = make_study_inputs(table, kc)') == 1 and
    "exec(compile(main_src, os.path.join(code, 'kc_worker.py'), 'exec'), kw.__dict__)" in srv and
    "'kc_worker.py': '4af60f6ff7b94b3acab9dceb52162a18'" in srv and "'kc_common.py': '45255ae0e361ef787b290d7510554a9d'" in srv and
    '30aa72fd972e29cf42c752aa5394a6c88a175f17b98740b1f6b0e0f55553ac40' in srv and 'kw.serve_probs = serve_probs' in srv)
chk('kc_driver: the side kernel\'s own study() (sha 2979c191) via importlib, one kc_serve process per GPU in its own session, never names '
    'submission.csv in code', "spec_from_file_location('kc_side_kernel'" in drv and '_KS.study(job)' in drv and
    '2979c1915e0cecf2c4792526a5581f93a723fac368eacff18849661dc0bdcb62' in drv and drv.count('start_new_session=True') == 1 and
    "'--device', dev" in drv and not re.search(r'(?<![A-Za-z_])submission(_s6f)?\.csv', code_only(drv)))
# dataset manifest
man_sha = re.search(r"MANIFEST_SHA = '([0-9a-f]{64})'", K[0])
KD = B.get('kc_dataset') or {}
mdir = A.kc_dir or KD.get('local_dir')
mraw = open(os.path.join(mdir, 'KC_MANIFEST.json'), 'rb').read() if mdir and os.path.isfile(os.path.join(mdir, 'KC_MANIFEST.json')) else b''
chk('KC dataset: manifest sha in the SETUP cell == build receipt == the local dataset manifest; slug in the cell == metadata', man_sha and
    man_sha.group(1) == KD.get('manifest_sha256') == sha(mraw) and ("KC_SLUG = '%s'" % KD.get('slug')) in K[0] and
    KD.get('slug') in meta_new.get('dataset_sources', []), (man_sha.group(1)[:16] if man_sha else None, KD.get('slug')))
if mraw:
    man = json.loads(mraw)
    chk('manifest: every member listed, members %s; kind %s; tag %s "standin"' % ([m['file'] for m in man['members']], man.get('kind'),
                                                                                 'contains' if man.get('standin') else 'does not contain'),
        [m['file'] for m in man['members']] == KD.get('members') and bool(man.get('standin')) == ('standin' in TAG))
# the appended cells change nothing the base uses
bad = []
for k, t in enumerate(K):
    ct = code_only(t)
    bad += [(k,) + x for x in env_path_changes(t)]
    for f in ('raptor_raw.csv', 'raptor_sub.csv', 'raptor224_sub.csv', 'ours_sub.csv', 'blend_receipt.json', 'canary_receipt.json',
              'submission_receipt.json', 'submission_v224flip.csv', 's6f_receipt.json', 's6f_setup_receipt.json', 'dec_receipt.json',
              'dec_envwatch.json'):
        if f in ct:
            bad.append((k, f))
chk('appended cells: no os.environ / sys.path change, no base output file named in code', not bad, bad)
chk('RUN cell: process-group kill only of /proc processes whose cmdline names /tmp/kc/, never the notebook group; start guard + hard deadline',
    "ROOT = '/tmp/kc'" in K[1].replace("W, ROOT = '/kaggle/working', '/tmp/kc'", "ROOT = '/tmp/kc'") and "if (root + '/').encode() in cmd:" in K[1]
    and 'groups.discard(me)' in K[1] and 'if age > G_KC_H:' in K[1] and 'hard = min(t_start + BUDGET_H * 3600.0, nb_t0 + DEADLINE_H * 3600.0)' in K[1])
chk('BLEND cell: submission.csv written only via tmp + os.replace (blend or canary), never raises', K[2].count("os.replace(tmp, W + '/submission.csv')") == 2
    and "out.to_csv(W + '/submission.csv'" not in K[2] and K[2].count('except Exception') >= 3)
pol = B.get('policy') or {}
run_line = 'G_KC_H, BUDGET_H, DEADLINE_H, STOP_MARGIN_S, MIN_RUN_S = %r, %r, %r, %r, %r' % (float(pol.get('g_kc_h')), float(pol.get('budget_h')),
                                                                                         float(pol.get('deadline_h')), float(pol.get('stop_margin_s')),
                                                                                         float(pol.get('min_run_s')))
chk('RUN policy constants == build receipt (%s); deadline <= 8.2 h' % run_line.split('= ')[1], run_line in K[1] and float(pol.get('deadline_h')) <= 8.2)
diff = {k: (bmeta.get(k), meta_new.get(k)) for k in set(bmeta) | set(meta_new) if bmeta.get(k) != meta_new.get(k)}
chk('kernel-metadata == base except id / title / code_file / dataset_sources (= base + the KC dataset); docker pin, T4 x2, private, internet off',
    set(diff) <= {'id', 'title', 'code_file', 'dataset_sources'} and meta_new['dataset_sources'] == bmeta['dataset_sources'] + [KD.get('slug')]
    and meta_new['is_private'] is True and meta_new['enable_internet'] is False and meta_new['machine_shape'] == 'NvidiaTeslaT4', sorted(diff))
chk('notebook sha == build receipt', sha(NEW_RAW) == B.get('notebook_sha256'), sha(NEW_RAW)[:16])
print('KC_CELLDIFF %s %d/%d tag %s' % ('PASS' if all(R) else 'FAIL', sum(R), len(R), TAG), flush=True)
sys.exit(0 if all(R) else 1)
