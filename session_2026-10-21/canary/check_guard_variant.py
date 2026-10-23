# check_guard_variant.py (10-21 canary track, Claude). OFFLINE, read-only on both kernels: proves the guard variant differs from
# the A22 r6pw20 kernel ONLY in the three S6F guard constants of cell 17 (+ kernel-metadata id / title / code_file), then runs
# cell 17's own _s6f_run on paths that never launch anything (SETUP_FAILED, SKIPPED_GUARD) to read the deadlines it computes.
import ast, copy, difflib, hashlib, io, json, contextlib, time, sys
R = 'C:/Users/eeshs/Downloads/rsna-knee'
A = R + '/session_2026-10-20/val/kernels/kernel_v224r6flips6fpw20j'
B = R + '/session_2026-10-21/canary/kernels/kernel_v224r6flips6fpw20g85j'
NA, NB = A + '/rsna_sub_v224r6flips6fpw20j.ipynb', B + '/rsna_sub_v224r6flips6fpw20g85j.ipynb'
res = []
def chk(name, cond, info=''):
    res.append(bool(cond)); print('  %s  %s %s' % ('OK  ' if cond else 'FAIL', name, info))
ra, rb = open(NA, 'rb').read(), open(NB, 'rb').read()
chk('source = A22 r6pw20 build (sha e4d92c0a)', hashlib.sha256(ra).hexdigest().startswith('e4d92c0a2c581a83'))
print('  variant sha256', hashlib.sha256(rb).hexdigest())
la, lb = ra.decode('utf-8').splitlines(), rb.decode('utf-8').splitlines()
dl = [l for l in difflib.unified_diff(la, lb, n=0, lineterm='') if l[:1] in '+-' and not l.startswith(('+++', '---'))]
chk('raw file: exactly one line removed + one added', len(dl) == 2 and dl[0][0] == '-' and dl[1][0] == '+', str(len(dl)))
ja, jb = json.loads(ra), json.loads(rb)
chk('notebook metadata / nbformat identical', {k: v for k, v in ja.items() if k != 'cells'} == {k: v for k, v in jb.items() if k != 'cells'})
chk('same cell count (19)', len(ja['cells']) == len(jb['cells']) == 19)
diff_cells = [i for i, (x, y) in enumerate(zip(ja['cells'], jb['cells'])) if x != y]
chk('only cell 17 differs', diff_cells == [17], str(diff_cells))
ca, cb = copy.deepcopy(ja['cells'][17]), copy.deepcopy(jb['cells'][17])
sa, sb = ''.join(ca.pop('source')), ''.join(cb.pop('source'))
chk('cell 17 type / metadata / outputs identical', ca == cb)
ta, tb = ast.parse(sa), ast.parse(sb)
na, nb_ = list(ast.walk(ta)), list(ast.walk(tb))
chk('cell 17 AST has the same node sequence', [type(n).__name__ for n in na] == [type(n).__name__ for n in nb_])
cd = [(x.value, y.value) for x, y in zip(na, nb_) if isinstance(x, ast.Constant) and x.value != y.value]
chk('cell 17: exactly 3 constants changed = (5.75->6.25, 3.3->4.5, 8.2->8.5)', cd == [(5.75, 6.25), (3.3, 4.5), (8.2, 8.5)], str(cd))
strip = lambda t: ast.dump(ast.parse(ast.unparse(t)).body[0] if False else t, include_attributes=False)
class Z(ast.NodeTransformer):
    def visit_Constant(self, n):
        return ast.Constant(value=('#' if isinstance(n.value, float) and n.value in (5.75, 3.3, 8.2, 6.25, 4.5, 8.5) else n.value))
chk('cell 17 AST equal once those 3 floats are masked', ast.dump(Z().visit(copy.deepcopy(ta))) == ast.dump(Z().visit(copy.deepcopy(tb))))
la17 = [l for l in sa.splitlines() if 'G_START_H, BUDGET_H, DEADLINE_H' in l and '=' in l and 'rec' not in l]
lb17 = [l for l in sb.splitlines() if 'G_START_H, BUDGET_H, DEADLINE_H' in l and '=' in l and 'rec' not in l]
print('  old:', la17[0].strip()); print('  new:', lb17[0].strip())
chk('cell 16 (setup + embedded S6F driver) byte-identical', ja['cells'][16] == jb['cells'][16])
chk('cell 18 (blend + S6F canary + E-OA a .2) byte-identical', ja['cells'][18] == jb['cells'][18])
ma, mb = json.load(open(A + '/kernel-metadata.json')), json.load(open(B + '/kernel-metadata.json'))
dk = sorted(k for k in set(ma) | set(mb) if ma.get(k) != mb.get(k))
chk('kernel-metadata: only id / title / code_file differ', dk == ['code_file', 'id', 'title'], str(dk))
chk('kernel-metadata: datasets, docker image, GPU, private, no internet unchanged',
    all(ma[k] == mb[k] for k in ('dataset_sources', 'docker_image', 'machine_shape', 'enable_gpu', 'is_private', 'enable_internet', 'competition_sources')))
chk('variant id is new', mb['id'] == 'eeshsaxena/rsna-sub-v224r6flips6fpw20g85j' and mb['code_file'] == 'rsna_sub_v224r6flips6fpw20g85j.ipynb')

# behaviour: exec cell 17 itself on non-launching paths (setup failed / guard skip) at fixed kernel ages
def run17(src, age_h, setup_ok):
    ns = {'__name__': 'cell17', 'NB_T0': float(time.time() - age_h * 3600.0),
          '_S6F_SETUP': ({'ok': True} if setup_ok else {'ok': False, 'why': 'offline check'}), '_S6F_ENV': ({} if setup_ok else None)}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, 'cell17', 'exec'), ns)
    return ns['_S6F_RUN']
print('  age_h | A22 r6pw20: status, hard_age, stop_age | variant: status, hard_age, stop_age   (setup failed = no launch)')
for age in (4.0, 4.5, 5.0, 5.5, 6.0, 6.2):
    a, b = run17(sa, age, False), run17(sb, age, False)
    print('  %4.1f  | %-13s %6.3f %6.3f | %-13s %6.3f %6.3f' % (age, a['status'], a['hard_deadline_age_h'], a['stop_at_age_h'],
                                                            b['status'], b['hard_deadline_age_h'], b['stop_at_age_h']))
    if age <= 5.0:
        chk('  A22 hard = start + 3.3 h at age %.1f' % age, abs(a['hard_deadline_age_h'] - min(age + 3.3, 8.2)) < 2e-3)
    chk('  variant hard = min(start + 4.5, 8.5) at age %.1f' % age, abs(b['hard_deadline_age_h'] - min(age + 4.5, 8.5)) < 2e-3)
a, b = run17(sa, 5.8, True), run17(sb, 6.3, True)
chk('A22 r6pw20 skips S6F at age 5.8 (G 5.75)', a['status'] == 'SKIPPED_GUARD')
chk('variant skips S6F at age 6.3 (G 6.25)', b['status'] == 'SKIPPED_GUARD')
chk('variant max hard deadline 8.5 h (kill at NB_T0 + 8.5 h; R6 8.3 h and r224 9 h - 40 min untouched)',
    max(run17(sb, x, False)['hard_deadline_age_h'] for x in (3.0, 4.0, 5.0, 6.0, 6.2)) <= 8.5 + 1e-3)
n = sum(res); print('GUARD_VARIANT_CHECK %s %d/%d' % ('PASS' if n == len(res) else 'FAIL', n, len(res)))
sys.exit(0 if n == len(res) else 1)
