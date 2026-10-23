#!/usr/bin/env python
# make_j.py (2026-10-15, Claude subagent): J overlay = a base kernel + ONE inserted code cell (the DEC cell, dec_cell_j_template.py,
# generalised from ../f20j/dec_cell_template.py) that makes the JPEG Lossless decoders (the three PyPI-identical cp313 wheels of the
# public dataset glorysing/rsna-knee-offline-codecs-r09) visible to every later DICOM-decoding process. Nothing else changes: every
# base cell stays byte-identical and in order (deleting the inserted cell re-serialises to the base notebook byte for byte), the
# kernel metadata changes only id / title / code_file and gains the codec dataset (if the base does not have it already).
#   usage: python make_j.py IN_KERNEL_DIR OUT_KERNEL_DIR --slug OWNER/rsna-sub-TAG [--expect-base-sha SHA] [--receipt-dir DIR]
#
# Where the cell goes: the FIRST code cell, i.e. right after the notebook-start clock cell when the base has a separate one (a code
# cell that only sets the clock), otherwise before the first code cell (and before the "## Stage" heading right above it). In the
# V224 family the clock (NB_T0 = time.time()) is set INSIDE the Raptor r384 launch cell, so "after the clock" and "before any
# decoding stage" cannot both hold without editing that cell: the cell goes before it (byte identity wins; the DEC seconds are
# logged and bounded by the cell's own timeouts; the fleet / S6F walls count from /proc start or kernel age anyway).
#
# The builder REFUSES (exit 1, nothing written) unless an AST audit of the base proves that every process that can decode DICOMs
# inherits os.environ['PYTHONPATH']:
#   - every process launch (subprocess.*, os.system / popen / exec* / spawn*, multiprocessing) in every notebook cell, in every
#     %%writefile script and in every embedded source (string literals that parse as Python, *_B64 constants that decode to
#     Python: R7 SOURCES, the Raptor fastread module, the S6F driver) either passes no env= (inherits os.environ) or an env built
#     from os.environ whose PYTHONPATH is untouched or extended with the inherited value; the few forms a static scan cannot
#     prove are allowed only by an exact-text ALLOW entry with its reason and the anchors it needs (listed in the receipt);
#   - no notebook cell sets / removes os.environ['PYTHONPATH'] except the DEC cell; a generic environment restore
#     (HF ENV-RESTORE) is allowed only when its snapshot is taken after the DEC cell;
#   - no code cell before the insertion point launches a process or touches pydicom; in-process pydicom use after it is listed
#     (it is covered by the DEC cell's sys.path insert, made only while pydicom is not imported).
import os, re, ast, sys, json, copy, base64, hashlib, argparse, binascii

HERE = os.path.dirname(os.path.abspath(__file__))
JPEG = os.path.normpath(os.path.join(HERE, '..'))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..', '..'))
TEMPLATE = os.path.join(HERE, 'dec_cell_j_template.py')
FIXTURES = os.path.join(JPEG, 'f20j', 'selftest_fixtures.json')
FIXTURES_SHA = '725bd7d8f528f022a201bb0e89031388f1a58ba0b7994747c92cf66fecfd6b34'
JC_DRIVER = os.path.join(JPEG, 'kernel_src', 'driver_template.py')         # PyPI sha as checked by rsna-jpeg-check v2
JC_RESULT = os.path.join(JPEG, 'out_jc_v2', 'jpeg_check_result.json')       # MEASURED pip --target listing (v2)
CODEC_DS = 'glorysing/rsna-knee-offline-codecs-r09'
DEC_HEADER = '# [eeshsaxena J DECODERS (jpeg/overlay/make_j.py)'
PYPI_SHA = {
    'pylibjpeg-2.1.0-py3-none-any.whl': '25df9496a69e64e98c887fddee12a1271e275b5f74ba804f9bf98a08bb80993e',
    'pylibjpeg_libjpeg-2.4.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl':
        'd2c8da12c79d1f46596b89df94091be01f932d739613e8af93d1d557ac3bab74',
    'pylibjpeg_openjpeg-2.6.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl':
        '3e8ee9e5bd3b62766ca98fc143cd340633e60bb14450162426f32688908ad9e8'}
OWN_NAMES = ('_dec_main', '_DEC_RECEIPT', '_dec_e')
LAUNCH_FUNCS = {'subprocess.run', 'subprocess.Popen', 'subprocess.call', 'subprocess.check_call', 'subprocess.check_output',
                'subprocess.getoutput', 'subprocess.getstatusoutput', 'os.system', 'os.popen', 'os.execv', 'os.execve', 'os.execvp',
                'os.execvpe', 'os.execl', 'os.execle', 'os.execlp', 'os.execlpe', 'os.spawnv', 'os.spawnve', 'os.spawnl', 'os.spawnle',
                'os.posix_spawn', 'os.posix_spawnp', 'multiprocessing.Process', 'multiprocessing.Pool', 'mp.Process', 'mp.Pool',
                'Popen', 'ProcessPoolExecutor', 'concurrent.futures.ProcessPoolExecutor'}
LAUNCH_TAILS = ('.Pool', '.Process', '.get_context')
PP_SOURCES = ("os.environ.get('PYTHONPATH'", 'os.environ.get("PYTHONPATH"', "os.environ['PYTHONPATH']", 'os.environ["PYTHONPATH"]')
# Forms a static scan cannot prove; each is allowed only with its exact text present AND every anchor in 'needs' present in the
# same unit. Reasons are copied into the build receipt.
ALLOW = [
    {'id': 'S6F_POP_NAMED', 'text': "for k in ('PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONEXECUTABLE'):\n            env.pop(k, None)",
     'needs': [], 'why': 'S6F SETUP pops three named variables (not PYTHONPATH) from its subprocess env'},
    {'id': 'S6F_POP_EMPTY', 'text': "        if stock_pp:\n            env['PYTHONPATH'] = os.pathsep.join(stock_pp)\n        else:\n"
                                    "            env.pop('PYTHONPATH', None)",
     'needs': ["stock_pp = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]"],
     'why': 'S6F SETUP drops PYTHONPATH only when the inherited one is empty; after DEC_OK it always holds /tmp/_dec, so in J '
            'that branch cannot run (with J_UNAVAILABLE it runs exactly as in the base)'},
    {'id': 'S6F_RUN_ENV', 'text': "subprocess.Popen([sys.executable, '-u', ROOT + '/s6f_driver.py', '--stop-at', '%.3f' % stop_at, '--nb-t0',"
                                  " '%.3f' % nb_t0],\n                             env=env,",
     'needs': ["_S6F_RUN = _s6f_run(globals().get('_S6F_SETUP'), globals().get('_S6F_ENV'))"],
     'needs_nb': ["    _S6F_SETUP, _S6F_ENV = _s6f_setup()", "        env = dict(os.environ)\n",
                  "        stock_pp = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]",
                  "            env['PYTHONPATH'] = after                                # the S6F subprocess environment only",
                  "    return rec, env\n"],
     'why': 'S6F RUN launches the driver with the env the S6F SETUP cell returned (_S6F_ENV): dict(os.environ) with PYTHONPATH = '
            '/tmp/s6f_dec + the inherited PYTHONPATH entries (the SETUP launches that use the same env are audited above)'},
    {'id': 'S6F_DRIVER_SPECS', 'text': 'subprocess.Popen(s["cmd"], env=s["env"]',
     'needs': ['def b_env(gi, px_dir=None):\n    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gi),', '"env": b_env(gi, None)',
               '"env": b_env(0, None)'],
     'why': 'S6F driver run_group: every spec env is b_env(...) = dict(os.environ, CUDA_VISIBLE_DEVICES=..., threads...), '
            'PYTHONPATH inherited (b_env only adds / pops K1_PX_DUMP)'},
]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def src(c):
    s = c['source']
    return ''.join(s) if isinstance(s, list) else s


class Refuse(Exception):
    pass


# ---- serialisation ------------------------------------------------------------------------------------------------------------
NB_STYLES = {
    'compact': lambda nb: json.dumps(nb, ensure_ascii=False, separators=(',', ':')),
    'indent1': lambda nb: json.dumps(nb, ensure_ascii=False, indent=1),
    'indent1nl': lambda nb: json.dumps(nb, ensure_ascii=False, indent=1) + '\n',
    'indent1sort': lambda nb: json.dumps(nb, ensure_ascii=False, indent=1, sort_keys=True) + '\n',
    'indent2': lambda nb: json.dumps(nb, ensure_ascii=False, indent=2),
}


def nb_style(raw, nb):
    for k, f in NB_STYLES.items():
        if f(nb).encode('utf-8') == raw:
            return k
    raise Refuse('no known serialisation reproduces the base notebook bytes (a pulled or hand-edited notebook?)')


def meta_dumps(meta, style):
    ind, crlf, nl = style
    s = json.dumps(meta, indent=ind) + ('\n' if nl else '')
    return (s.replace('\n', '\r\n') if crlf else s).encode('utf-8')


def meta_style(raw, meta):
    for ind in (1, 2, 4):
        for crlf in (False, True):
            for nl in (False, True):
                if meta_dumps(meta, (ind, crlf, nl)) == raw:
                    return (ind, crlf, nl)
    raise Refuse('no known serialisation reproduces kernel-metadata.json')


def load_kernel(d):
    mp = os.path.join(d, 'kernel-metadata.json')
    if not os.path.isfile(mp):
        raise Refuse('no kernel-metadata.json in %s' % d)
    mraw = open(mp, 'rb').read()
    meta = json.loads(mraw.decode('utf-8'))
    nbp = os.path.join(d, meta['code_file'])
    raw = open(nbp, 'rb').read()
    nb = json.loads(raw.decode('utf-8'))
    return {'dir': d, 'meta': meta, 'meta_raw': mraw, 'meta_style': meta_style(mraw, meta), 'nb_path': nbp, 'raw': raw, 'nb': nb,
            'nb_style': nb_style(raw, nb)}


# ---- code units -----------------------------------------------------------------------------------------------------------------
def strip_magics(text):
    """IPython magics / shell escapes at the start of a logical line -> blank lines (only if the plain text does not parse)."""
    try:
        ast.parse(text)
        return text, []
    except SyntaxError:
        pass
    out, mags = [], []
    for ln in text.split('\n'):
        if ln.lstrip().startswith(('%', '!')) and _logical_start('\n'.join(out) + '\n'):
            mags.append(ln.strip()[:160])
            out.append(re.match(r'\s*', ln).group(0) + 'pass')
        else:
            out.append(ln)
    t = '\n'.join(out)
    ast.parse(t)
    return t, mags


def _logical_start(prefix):
    """True if the text so far ends outside any bracket / string / continuation (the next line starts a statement)."""
    import io, tokenize
    try:
        for _ in tokenize.generate_tokens(io.StringIO(prefix).readline):
            pass
        return not prefix.rstrip('\n').endswith('\\')
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return False


def embedded_sources(text, prefix, depth=0):
    """String literals that are Python sources (len > 200 and parse, with at least one def / import), and *_B64 constants that
    decode to such sources. Returned with names; scanned recursively."""
    out = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names[id(node.value)] = t.id
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant) and isinstance(v.value, str):
                    names[id(v)] = str(k.value)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) > 200):
            continue
        s, label = node.value, names.get(id(node), 'L%d' % getattr(node, 'lineno', 0))
        cand = None
        if label.endswith('_B64') or re.fullmatch(r'[A-Za-z0-9+/=\s]+', s[:4000] or ' '):
            try:
                dec = base64.b64decode(s, validate=False).decode('utf-8')
                if re.search(r'^\s*(import |from |def )', dec, re.M):
                    cand = dec
            except (binascii.Error, UnicodeDecodeError, ValueError):
                pass
        if cand is None and re.search(r'^\s*(import |from |def )', s, re.M):
            cand = s
        if cand is None:
            continue
        try:
            ast.parse(cand)
        except SyntaxError:
            continue
        nm = '%s>%s' % (prefix, label)
        out.append((nm, cand))
        if depth < 3:
            out += embedded_sources(cand, nm, depth + 1)
    return out


def units_of(cells, first=0):
    """[(unit name, kind, cell index, text)] kind: 'cell' (runs in the notebook process), 'script' (%%writefile), 'embedded'."""
    U = []
    for i, c in enumerate(cells):
        if c['cell_type'] != 'code' or i < first:
            continue
        s = src(c)
        if s.startswith('%%writefile'):
            path = s.split('\n', 1)[0].split()[-1]
            body = s.split('\n', 1)[1] if '\n' in s else ''
            U.append(('cell%d:%s' % (i, path), 'script', i, body))
            U += [(n, 'embedded', i, t) for n, t in embedded_sources(body, 'cell%d:%s' % (i, path))]
            continue
        if s.startswith('%%'):
            U.append(('cell%d' % i, 'cellmagic', i, s))
            continue
        t, mags = strip_magics(s)
        U.append(('cell%d' % i, 'cell', i, t))
        if mags:
            U.append(('cell%d:magics' % i, 'magics', i, '\n'.join(mags)))
        U += [(n, 'embedded', i, tt) for n, tt in embedded_sources(t, 'cell%d' % i)]
    return U


# ---- env analysis ---------------------------------------------------------------------------------------------------------------
def _txt(n):
    return ast.unparse(n)


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.Module)


def _parents(tree):
    par = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            par[ch] = node
    return par


def _scope(node, par):
    n = par.get(node)
    while n is not None and not isinstance(n, _SCOPES):
        n = par.get(n)
    return n


def _scope_nodes(tree, par, scope):
    """nodes whose innermost scope is `scope` or the module (globals)."""
    return [n for n in ast.walk(tree) if _scope(n, par) in (scope, tree)]


def _fixpoint(nodes, seed):
    """names (Name ids or unparsed Subscript targets) whose assigned value mentions a seed pattern or an already-marked name."""
    marked = set()
    assigns = []
    for node in nodes:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                assigns.append((_txt(t), _txt(node.value)))
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)) and node.value is not None:
            assigns.append((_txt(node.target), _txt(node.value)))
    changed = True
    while changed:
        changed = False
        for t, v in assigns:
            if t in marked:
                continue
            if any(p in v for p in seed) or any(re.search(r'(?<![\w.])%s(?![\w])' % re.escape(m), v) for m in marked):
                marked.add(t)
                changed = True
    return marked


def env_value_class(v, derived, tainted):
    """class of one env value expression: ('derived', pp) with pp in {'keep', 'extend', 'BAD'} or ('unknown', why)."""
    t = _txt(v)
    if isinstance(v, ast.Constant) and v.value is None:
        return ('inherit', 'keep')
    base = None
    pp_val = None
    if isinstance(v, ast.Call) and _txt(v.func) == 'dict' and v.args:
        base = _txt(v.args[0])
        for k in v.keywords:
            if k.arg == 'PYTHONPATH':
                pp_val = k.value
            if k.arg is None:
                return ('unknown', '**kwargs in dict(): %s' % t[:120])
    elif isinstance(v, ast.Call) and _txt(v.func) in ('os.environ.copy', 'dict') and not v.args:
        return ('unknown', 'empty dict / copy form: %s' % t[:120]) if _txt(v.func) == 'dict' else ('derived', 'keep')
    elif isinstance(v, ast.Dict) and v.keys and v.keys[0] is None:
        base = _txt(v.values[0])
        for k, val in zip(v.keys[1:], v.values[1:]):
            if k is None:
                return ('unknown', 'second ** in env dict: %s' % t[:120])
            if isinstance(k, ast.Constant) and k.value == 'PYTHONPATH':
                pp_val = val
            elif not isinstance(k, ast.Constant):
                return ('unknown', 'non-constant key in env dict: %s' % t[:120])
    else:
        return ('unknown', 'env not built from os.environ: %s' % t[:160])
    if not (base in ('os.environ', 'dict(os.environ)') or base.endswith('.environ') or base in derived):
        return ('unknown', 'env base %s is not os.environ' % base)
    if pp_val is None:
        return ('derived', 'keep')
    pv = _txt(pp_val)
    ok = any(p in pv for p in PP_SOURCES) or any(re.search(r'(?<![\w.])%s(?![\w])' % re.escape(m), pv) for m in tainted)
    return ('derived', 'extend' if ok else 'BAD:%s' % pv[:120])


def classify_env(env_node, tree, text):
    if env_node is None:
        return {'class': 'inherit', 'detail': 'no env= (os.environ inherited)'}
    par = _parents(tree)
    scope = _scope(env_node, par)
    nodes = _scope_nodes(tree, par, scope)                      # the launch's own function + module level
    tainted = _fixpoint(nodes, PP_SOURCES)
    derived = _fixpoint(nodes, ('os.environ',))
    e = _txt(env_node)
    if not isinstance(env_node, ast.Name):
        c, d = env_value_class(env_node, derived, tainted)
        return {'class': 'derived-' + d if c == 'derived' else c, 'detail': e[:200]}
    nm = env_node.id
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) and nm in [a.arg for a in scope.args.args + scope.args.kwonlyargs]:
        return {'class': 'unknown', 'detail': 'env %s is a parameter of %s()' % (nm, scope.name)}
    defs, pp_sets, pops = [], [], []
    seg = lambda n: ast.get_source_segment(text, n) or _txt(n)
    for node in nodes:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == nm:
                    defs.append(node.value)
                if isinstance(t, ast.Subscript) and _txt(t.value) == nm:
                    key = t.slice
                    if isinstance(key, ast.Constant) and key.value == 'PYTHONPATH':
                        pp_sets.append(node.value)
                    elif not isinstance(key, ast.Constant):
                        pops.append(seg(node))
        if isinstance(node, ast.Delete):
            for t in node.targets:
                if isinstance(t, ast.Subscript) and _txt(t.value) == nm:
                    key = t.slice
                    if not (isinstance(key, ast.Constant) and key.value != 'PYTHONPATH'):
                        pops.append(seg(node))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and _txt(node.func.value) == nm:
            if node.func.attr in ('pop', 'popitem', 'clear', 'update', 'setdefault', '__delitem__', '__setitem__'):
                a0 = node.args[0] if node.args else None
                if node.func.attr == 'setdefault':                 # never replaces an existing PYTHONPATH (DEC_OK: always present)
                    continue
                if node.func.attr in ('pop', '__delitem__') and isinstance(a0, ast.Constant) and a0.value != 'PYTHONPATH':
                    continue
                pops.append(seg(node))
    if not defs:
        return {'class': 'unknown', 'detail': 'env name %s has no assignment in this unit' % nm}
    classes = []
    for v in defs:
        c, d = env_value_class(v, derived, tainted)
        if c == 'unknown':
            return {'class': 'unknown', 'detail': d}
        classes.append(d)
    for v in pp_sets:
        pv = _txt(v)
        ok = any(p in pv for p in PP_SOURCES) or any(re.search(r'(?<![\w.])%s(?![\w])' % re.escape(m), pv) for m in tainted)
        classes.append('extend' if ok else 'BAD:%s' % pv[:120])
    bad = [c for c in classes if c.startswith('BAD')]
    if bad:
        return {'class': 'unknown', 'detail': 'PYTHONPATH set without the inherited value: %s' % bad}
    if pops:
        return {'class': 'needs-allow', 'detail': 'env %s is also changed by: %s' % (nm, pops), 'pops': pops}
    return {'class': 'derived-' + ('extend' if 'extend' in classes else 'keep'), 'detail': '; '.join(_txt(v)[:160] for v in defs)}


def stage_of(unit, cmd, text_line):
    s = (cmd or '') + ' ' + (text_line or '')
    for k, v in (('rm_verify.py', 'RM verify'), ('raptor224_run.py', 'Raptor r224'), ('raptor_run.py', 'Raptor r384'),
                 ('probe_dual.py', 'R7 controller'), ('fleet_worker.py', 'fleet workers'), ('s6f_driver.py', 'S6fall driver'),
                 ("'pip'", 'pip (no decoding)'), ('nvidia-smi', 'tool'), ("['df'", 'tool'), ('TEST', 'self-test'),
                 ('ENV2', 'S6F env probe'), ('B_PROBE', 'S6F B stack probe (CPU)'), ("'-c', PROBE]", 'S6F env probe')):
        if k in s:
            return v
    if unit.endswith('probe_dual.py') and 'Popen(cmd' in s:
        return 'R7 workers'
    if 's["cmd"]' in s:
        return 'S6fall B workers (b_env)'
    if 'get_context' in s or '.Pool' in s or 'Process' in s:
        return 'process pool (fork / spawn: inherits env and sys.path)'
    return 'other'


def audit(cells, pos):
    """-> (report dict, problems list). pos = index the DEC cell will take (cells before it must not decode / launch)."""
    problems, launches, inproc, envmut, syspath, allow_used = [], [], [], [], [], []
    units = units_of(cells)
    # cells before the insertion point
    for i in range(pos):
        c = cells[i]
        if c['cell_type'] == 'code':
            s = src(c)
            if re.search(r'subprocess|pydicom|dcmread|os\.system|Popen|multiprocessing|!\s*python|%%writefile|exec\(', s):
                problems.append('cell %d (before the insertion point) launches a process / touches pydicom' % i)
    snaps = {}
    for name, kind, ci, text in units:
        if kind in ('magics', 'cellmagic'):
            for ln in text.split('\n'):
                if re.match(r'\s*(!|%%?(run|system|sx|bash|sh|script))', ln):
                    launches.append({'unit': name, 'line': 0, 'func': 'magic', 'cmd': ln[:120], 'env': None, 'class': 'inherit',
                                     'detail': 'IPython magic / shell escape (os.environ inherited)', 'stage': stage_of(name, ln, ln)})
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            problems.append('%s does not parse: %s' % (name, e))
            continue
        lines = text.split('\n')
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = _txt(node.func)
                if fn in LAUNCH_FUNCS or fn.endswith(LAUNCH_TAILS) or fn.endswith(('subprocess.run', 'subprocess.Popen')):
                    if fn.endswith('.get_context') and not node.args:
                        continue
                    envk = [k for k in node.keywords if k.arg == 'env']
                    cls = classify_env(envk[0].value if envk else None, tree, text)
                    ln = lines[node.lineno - 1] if 0 < node.lineno <= len(lines) else ''
                    seg = ast.get_source_segment(text, node) or ''
                    rec = {'unit': name, 'kind': kind, 'line': node.lineno, 'func': fn,
                           'cmd': (_txt(node.args[0]) if node.args else '')[:160], 'env': _txt(envk[0].value)[:120] if envk else None,
                           'class': cls['class'], 'detail': cls['detail'], 'stage': stage_of(name, _txt(node.args[0]) if node.args else '', seg)}
                    usable = [a for a in ALLOW if a['text'] in text and all(n in text for n in a['needs'])
                              and all(any(n in src(c) for c in cells) for n in a.get('needs_nb', []))]
                    if cls['class'] == 'needs-allow':
                        # every extra change of the env dict must sit inside an ALLOW text whose anchors are present
                        cover = [next((a['id'] for a in usable if p in a['text']), None) for p in cls['pops']]
                        if all(cover):
                            rec['class'] = 'allowed (' + cls['detail'][:0] + 'derived-extend)'
                            rec['allow'] = sorted(set(cover))
                            allow_used += rec['allow']
                    elif cls['class'] == 'unknown':
                        cover = [a['id'] for a in usable if a['text'] in seg]
                        if cover:
                            rec['class'] = 'allowed (derived-keep)'
                            rec['allow'] = cover
                            allow_used += cover
                    launches.append(rec)
                    if rec['class'] in ('unknown', 'needs-allow'):
                        problems.append('%s line %d: %s(env=%s) cannot be shown to inherit PYTHONPATH: %s' % (
                            name, node.lineno, fn, rec['env'], cls['detail']))
            # process-wide environment changes
            tgt = None
            if isinstance(node, (ast.Assign, ast.Delete)):
                for t in (node.targets if isinstance(node, (ast.Assign, ast.Delete)) else []):
                    if isinstance(t, ast.Subscript) and _txt(t.value).endswith('environ'):
                        tgt = (t, node)
                        key = t.slice
                        kt = _txt(key)
                        if isinstance(key, ast.Constant):
                            if key.value == 'PYTHONPATH':
                                envmut.append({'unit': name, 'line': node.lineno, 'what': _txt(node)[:160]})
                                if kind == 'cell':
                                    problems.append('%s line %d sets / deletes os.environ PYTHONPATH' % (name, node.lineno))
                                elif isinstance(node, ast.Delete) or not (
                                        any(p in _txt(node.value) for p in PP_SOURCES)):
                                    problems.append('%s line %d (a subprocess) replaces PYTHONPATH for its children' % (name, node.lineno))
                        else:
                            envmut.append({'unit': name, 'line': node.lineno, 'what': _txt(node)[:160], 'generic_key': kt})
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                o = _txt(node.func.value)
                if o.endswith('environ') and node.func.attr in ('pop', 'clear', 'update', 'setdefault', 'popitem'):
                    a0 = node.args[0] if node.args else None
                    if not (isinstance(a0, ast.Constant) and a0.value != 'PYTHONPATH'):
                        envmut.append({'unit': name, 'line': node.lineno, 'what': _txt(node)[:160]})
                        # setdefault never replaces an existing PYTHONPATH (after DEC_OK it always exists): recorded, allowed
                        if node.func.attr != 'setdefault':
                            problems.append('%s line %d changes os.environ (%s) with a key that may be PYTHONPATH' % (
                                name, node.lineno, node.func.attr))
                if _txt(node.func) in ('os.putenv', 'os.unsetenv') and node.args and _txt(node.args[0]) in ("'PYTHONPATH'", '"PYTHONPATH"'):
                    problems.append('%s line %d %s PYTHONPATH' % (name, node.lineno, _txt(node.func)))
            if isinstance(node, ast.Assign) and kind == 'cell':
                v = _txt(node.value)
                for t in node.targets:
                    if isinstance(t, ast.Name) and re.fullmatch(r'dict\([\w.]*environ\)', v):
                        snaps[t.id] = ci
            # sys.path changes / in-process pydicom
            if kind == 'cell':
                if isinstance(node, ast.Assign) and any(_txt(t) in ('sys.path', 'sys.path[:]') for t in node.targets):
                    syspath.append({'unit': name, 'line': node.lineno, 'what': _txt(node)[:120]})
                if isinstance(node, ast.Call) and _txt(node.func) in ('sys.path.remove', 'sys.path.pop', 'sys.path.clear'):
                    syspath.append({'unit': name, 'line': node.lineno, 'what': _txt(node)[:120]})
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    mods = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                    if any(m and (m == 'pydicom' or m.startswith('pydicom.')) for m in mods):
                        inproc.append({'unit': name, 'line': node.lineno, 'what': _txt(node)[:120]})
                if isinstance(node, ast.Attribute) and node.attr in ('dcmread', 'pixel_array', 'read_file'):
                    inproc.append({'unit': name, 'line': node.lineno, 'what': _txt(node)[:120]})
                if isinstance(node, ast.Call) and _txt(node.func) in ('exec', 'eval', 'runpy.run_path', 'runpy.run_module',
                                                                       'importlib.import_module', '__import__'):
                    inproc.append({'unit': name, 'line': node.lineno, 'what': 'in-process exec / import: ' + _txt(node)[:100]})
    # generic-key environment writes (ENV-RESTORE) must restore a snapshot taken after the DEC cell
    for m in envmut:
        if 'generic_key' in m:
            ci = int(re.match(r'cell(\d+)', m['unit']).group(1))
            ok = bool(snaps) and all(pos <= s_ci <= ci for s_ci in snaps.values())
            m['snapshot_cells'] = snaps
            if not ok and m['unit'].count(':') == 0 and m['unit'].count('>') == 0:
                problems.append('%s line %d writes os.environ[<key>] but the snapshot it restores (%s) is not after the DEC cell' % (
                    m['unit'], m['line'], snaps))
    for name, kind, ci, text in units:
        if kind == 'cell':
            for nm in OWN_NAMES:
                if re.search(r'(?<![\w])%s(?![\w])' % nm, text):
                    problems.append('name collision: %s in %s' % (nm, name))
    rep = {'launches': launches, 'env_mutations': envmut, 'inprocess_dicom': inproc, 'sys_path_changes': syspath,
           'allow_used': sorted(set(allow_used)), 'allow_table': {a['id']: a['why'] for a in ALLOW if a['id'] in set(allow_used)},
           'units': [(n, k, ci, len(t)) for n, k, ci, t in units]}
    return rep, problems


def insertion_point(cells):
    """-> (pos, rule)."""
    first = next((i for i, c in enumerate(cells) if c['cell_type'] == 'code'), None)
    if first is None:
        raise Refuse('no code cell')
    s = src(cells[first])
    clock_only = (re.search(r'(?m)^\s*\w*T0\w*\s*=\s*time\.time\(\)', s) and len(s) < 1500 and not re.search(
        r'subprocess|pydicom|dcmread|Popen|os\.system|multiprocessing|%%writefile|exec\(|^\s*!', s, re.M))
    if clock_only:
        return first + 1, 'after the notebook-start clock cell %d' % first
    pos = first
    while pos > 0 and cells[pos - 1]['cell_type'] == 'markdown' and src(cells[pos - 1]).lstrip().startswith('## Stage'):
        pos -= 1
    clock = next((i for i, c in enumerate(cells) if c['cell_type'] == 'code' and re.search(r'(?m)^NB_T0\s*=\s*time\.time\(\)', src(c))), None)
    return pos, ('before the first code cell %d (and its "## Stage" heading); no separate clock cell: NB_T0 = time.time() is set in '
                 'cell %s together with the first decoding stage launch' % (first, clock))


def render_dec(tag):
    tpl = open(TEMPLATE, encoding='utf-8').read()
    fx_raw = open(FIXTURES, 'rb').read()
    if sha(fx_raw) != FIXTURES_SHA:
        raise Refuse('self-test fixtures changed')
    FIX = json.loads(fx_raw)
    assert sorted(FIX) == ['JLL57_12U', 'JLL70_16S', 'U_16S']
    drv = open(JC_DRIVER, encoding='utf-8').read()
    for n, h in PYPI_SHA.items():
        if ("'%s'" % h) not in drv or n not in drv:
            raise Refuse('PyPI sha differs from the rsna-jpeg-check driver: %s' % n)
    jc = json.load(open(JC_RESULT, encoding='utf-8'))
    TOP = jc['decoders']['cp313']['files']
    if not (jc['decoders']['cp313']['check'] == '3.0.2 2.1.0 2.4.0 2.6.0' and all(jc['decoders']['cp313_sha_match_pypi'].values())):
        raise Refuse('rsna-jpeg-check v2 result does not show the cp313 wheels working')
    assert TOP == sorted(TOP) and len(TOP) == 8 and not any(t.startswith(('numpy', 'pydicom', 'PIL', 'torch')) for t in TOP), TOP
    if not re.fullmatch(r'[a-z0-9]+', tag):
        raise Refuse('tag %r' % tag)
    dec = tpl.replace('__WHEELS__', repr(PYPI_SHA)).replace('__TOP__', repr(TOP)).replace('__FIX__', repr(FIX)).replace('__TAG__', tag)
    for ph in ('__WHEELS__', '__TOP__', '__FIX__', '__TAG__'):
        assert ph not in dec, ph
    compile(dec, '<dec cell>', 'exec')
    assert '\u2014' not in dec and '\u2013' not in dec, 'dash'
    assert dec.startswith(DEC_HEADER)
    assert dec.count("os.environ['PYTHONPATH'] = after") == 1 and dec.count('sys.path.insert(0, TARGET)') == 1
    assert dec.rstrip().endswith('del _dec_main')
    return dec, {'template_sha256': sha(tpl.encode('utf-8')), 'fixtures_sha256': FIXTURES_SHA, 'top': TOP, 'fix': FIX}


def tag_of(slug):
    m = re.fullmatch(r'([a-z0-9-]+)/rsna-sub-([a-z0-9]+)', slug)
    if not m:
        raise Refuse('slug must look like OWNER/rsna-sub-TAG (lower case): %r' % slug)
    return m.group(2)


def build(in_dir, out_dir, slug, expect_sha=None, receipt_dir=None, write=True):
    B = load_kernel(in_dir)
    if 'id_no' in B['meta']:
        raise Refuse('base metadata has id_no (a pulled kernel dir): a push would target the BASE kernel; use the builder output dir')
    if expect_sha and sha(B['raw']) != expect_sha:
        raise Refuse('base notebook sha %s != expected %s' % (sha(B['raw']), expect_sha))
    tag = tag_of(slug)
    if B['meta'].get('id') == slug:
        raise Refuse('slug == the base slug')
    ain, aout = os.path.abspath(in_dir), os.path.abspath(out_dir)
    if aout == ain or aout.startswith(ain + os.sep):
        raise Refuse('OUT is the base dir or inside it')
    if write and os.path.exists(aout) and os.listdir(aout):
        raise Refuse('OUT exists and is not empty (new dirs only): %s' % aout)
    C = B['nb']['cells']
    pos, rule = insertion_point(C)
    rep, problems = audit(C, pos)
    for c in C:
        if src(c).startswith(DEC_HEADER):
            problems.append('the base already has a J DEC cell')
    if problems:
        raise Refuse('audit: ' + ' | '.join(problems))
    dec, dinfo = render_dec(tag)
    nb = copy.deepcopy(B['nb'])
    nb['cells'] = [copy.deepcopy(c) for c in C[:pos]] + [
        {'cell_type': 'code', 'metadata': {}, 'source': dec, 'outputs': [], 'execution_count': None}] + [copy.deepcopy(c) for c in C[pos:]]
    dump = NB_STYLES[B['nb_style']]
    back = copy.deepcopy(nb)
    del back['cells'][pos]
    assert dump(back).encode('utf-8') == B['raw'], 'removing the DEC cell does not give the base notebook'
    nb_raw = dump(nb).encode('utf-8')
    # metadata
    meta = copy.deepcopy(B['meta'])
    code_file = 'rsna_sub_%s.ipynb' % tag
    meta.update(id=slug, title='rsna sub %s' % tag, code_file=code_file)
    had_codec = CODEC_DS in meta.get('dataset_sources', [])
    if not had_codec:
        meta['dataset_sources'] = list(meta.get('dataset_sources', [])) + [CODEC_DS]
    keep = ('is_private', 'enable_gpu', 'enable_tpu', 'enable_internet', 'docker_image', 'machine_shape', 'language', 'kernel_type',
            'competition_sources', 'kernel_sources', 'model_sources')
    assert {k: meta.get(k) for k in keep} == {k: B['meta'].get(k) for k in keep}
    assert set(meta) == set(B['meta']), 'metadata keys changed'
    if B['meta'].get('enable_internet') is not False or B['meta'].get('is_private') is not True:
        raise Refuse('base metadata is not private / internet-off')
    meta_raw = meta_dumps(meta, B['meta_style'])
    stage_launch = [l for l in rep['launches'] if l['stage'] not in ('tool', 'pip (no decoding)', 'self-test', 'S6F env probe')]
    receipt = {'tag': tag, 'slug': slug, 'base_dir': os.path.relpath(ain, ROOT).replace('\\', '/'), 'base_slug': B['meta'].get('id'),
               'base_notebook_sha256': sha(B['raw']), 'base_cells': len(C), 'base_meta_sha256': sha(B['meta_raw']),
               'notebook_style': B['nb_style'], 'meta_style': list(B['meta_style']), 'notebook_sha256': sha(nb_raw),
               'meta_sha256': sha(meta_raw), 'cells': len(nb['cells']), 'dec_cell_index': pos, 'insertion_rule': rule,
               'dec_cell_sha256': sha(dec.encode('utf-8')), 'dec_cell_chars': len(dec), 'template_sha256': dinfo['template_sha256'],
               'fixtures_sha256': FIXTURES_SHA, 'pypi_sha256': PYPI_SHA, 'target_top': dinfo['top'], 'code_file': code_file,
               'dataset_sources': meta['dataset_sources'], 'codec_dataset_already_in_base': had_codec,
               'docker_image': meta.get('docker_image'), 'machine_shape': meta.get('machine_shape'),
               'audit': {k: v for k, v in rep.items() if k != 'units'}, 'units_scanned': rep['units'],
               'decoding_stage_launches': [{k: l.get(k) for k in ('stage', 'unit', 'line', 'func', 'class', 'env', 'allow')}
                                           for l in stage_launch],
               'proof': 'deleting cell %d re-serialises (%s) to the base notebook byte for byte' % (pos, B['nb_style'])}
    if write:
        os.makedirs(aout, exist_ok=True)
        with open(os.path.join(aout, code_file), 'wb') as f:
            f.write(nb_raw)
        with open(os.path.join(aout, 'kernel-metadata.json'), 'wb') as f:
            f.write(meta_raw)
        rdir = receipt_dir or os.path.dirname(aout)
        with open(os.path.join(rdir, 'j_build_receipt_%s.json' % tag), 'w', encoding='utf-8', newline='\n') as f:
            json.dump(receipt, f, indent=1)
        with open(os.path.join(rdir, 'celldiff_%s.txt' % tag), 'w', encoding='utf-8', newline='\n') as f:
            f.write('# %s = %s (sha %s...) + ONE inserted code cell at index %d (%s); every other cell byte-identical and in order\n'
                    '# (deleting the inserted cell gives the base file byte for byte). Metadata: id / title / code_file%s.\n'
                    '#### inserted cell %d (DEC), full text (the self-test fixture dict abbreviated):\n' % (
                        tag, receipt['base_dir'], receipt['base_notebook_sha256'][:16], pos, rule,
                        '' if had_codec else ' + dataset ' + CODEC_DS, pos))
            f.write(dec.replace(repr(dinfo['fix']), "{...3 self-test DICOMs, selftest_fixtures.json sha %s...}" % FIXTURES_SHA[:16]))
    return receipt, nb, meta, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('in_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--slug', required=True)
    ap.add_argument('--expect-base-sha', default=None)
    ap.add_argument('--receipt-dir', default=None, help='default: the parent of OUT_KERNEL_DIR')
    ap.add_argument('--dry-run', action='store_true', help='audit + build in memory, write nothing')
    A = ap.parse_args()
    try:
        r, nb, meta, dec = build(A.in_dir, A.out_dir, A.slug, A.expect_base_sha, A.receipt_dir, write=not A.dry_run)
    except Refuse as e:
        print('MAKE_J REFUSED: %s' % e, flush=True)
        return 1
    print('MAKE_J BUILT %s %s sha256 %s cells %d (base %d) dec cell at %d (%s) dec sha %s datasets %s' % (
        r['tag'], A.out_dir if not A.dry_run else '(dry run)', r['notebook_sha256'], r['cells'], r['base_cells'], r['dec_cell_index'],
        r['insertion_rule'], r['dec_cell_sha256'][:16], r['dataset_sources']), flush=True)
    for l in r['decoding_stage_launches']:
        print('  launch %-34s %-28s %-16s %s' % (l['stage'][:34], '%s:%s' % (l['unit'][:22], l['line']), l['class'],
                                                  (l.get('allow') or l.get('env') or '')), flush=True)
    print('  in-process pydicom use after the DEC cell: %s' % (r['audit']['inprocess_dicom'] or 'none'), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
