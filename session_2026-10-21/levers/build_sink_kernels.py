#!/usr/bin/env python
# build_sink_kernels.py (10-21 ~10:50 IST, Claude, SINK track). LOCAL BUILD ONLY, UNARMED: nothing is pushed, committed or registered.
# Kitchen-sink / conservative stacks for each plausible A22 winner = the winner's OWN notebook (the scored pw30, or the A22 notebook
# exactly as registered in val/slate_a22.json) + ONE edit family in cell 18: the convex E-OA exponent (CVX, levers/build_convex.py spec):
#   [P_OA]  after "_OA = ['Medial OA', 'Lateral OA', 'PF OA']":   P_OA = 2.0 line
#   [_S]    after the "_U = {...}" line:                           _S = _U if P_OA == 1 else {c: _U[c] ** P_OA for c in _OA}
#   [SUB]   in the two subtraction lines, the SUBTRACTED other-compartment pct _U[_o[k]] -> _S[_o[k]] (the own term _U[c] is kept)
#   [MARK]  after the EOA_APPLIED print: if P_OA != 1: rec['p_oa'] = P_OA; print('CVX_APPLIED p_oa %g' % P_OA)
# With P_OA = 1 the arithmetic is the base's (same dict object), the receipt and stdout are the base's: all-off == base byte for byte.
# Bases (sha-asserted against the registry / receipts):
#   pw30   a21/kernel_v224r6c4pw30j (5a903187, scored 57425970)        lomm30 night/c12_kernel_v224r6c4pw30lomm05j (c0dd7ec8, built, never pushed)
#   pw25   val/kernels/kernel_v224r6c4pw25j (64ed7eba, A22 57463742)   spl    stack/skeptic/kernels/kernel_v224r6c4pw25lomm05mcl00j (e641e88f, A22 57464296)
#   swide  stack/kernels/kernel_v224r6c4pw25lomm05mcl10j (c9801a80, A22 57464326)   salt stack/kernels/kernel_v224r6c4pw25mcl10j (051232e4, A22)
# Writes ONLY under session_2026-10-21/levers/kernels/: kernel_<tag>/ + build_receipt_<tag>.json + build_index.json + build_sink_kernels.log.
import json, hashlib, os, difflib, datetime
HERE = os.path.dirname(os.path.abspath(__file__)); KD = os.path.join(HERE, 'kernels'); os.makedirs(KD, exist_ok=True)
S20 = os.path.abspath(os.path.join(HERE, '..', '..', 'session_2026-10-20'))
REG = json.load(open(os.path.join(S20, 'val', 'slate_a22.json'), encoding='utf-8'))['entries']
BASES = {
    'pw30': (os.path.join(S20, 'a21', 'kernel_v224r6c4pw30j'), '5a903187415505e9e3566ebd749cae1328b20df6b60bd6593e6443b17fc0c9fd'),
    'lomm30': (os.path.join(S20, 'night', 'c12_kernel_v224r6c4pw30lomm05j'), None),
    'pw25': (os.path.join(S20, REG['pw25']['kernel_dir'].split('session_2026-10-20/')[1]), REG['pw25']['nb_sha256']),
    'spl': (os.path.join(S20, REG['spl']['kernel_dir'].split('session_2026-10-20/')[1]), REG['spl']['nb_sha256']),
    'swide': (os.path.join(S20, REG['swide']['kernel_dir'].split('session_2026-10-20/')[1]), REG['swide']['nb_sha256']),
    'salt': (os.path.join(S20, REG['salt']['kernel_dir'].split('session_2026-10-20/')[1]), REG['salt']['nb_sha256'])}
LOMM30_SHA8 = 'c0dd7ec8'
# role table: which winner each build serves (KS = kitchen sink, CONS = conservative)
BUILDS = {'cvx_pw30': ('pw30', 2.0, 'CONS(pw30)'), 'cvx_lomm30': ('lomm30', 2.0, 'KS(pw30)'), 'cvx_pw25': ('pw25', 2.0, 'CONS(pw25)'),
          'cvx_spl': ('spl', 2.0, 'KS(pw25) = KS(spl) = CONS(spl)'), 'cvx_salt': ('salt', 2.0, 'CONS(salt)'),
          'cvx_swide': ('swide', 2.0, 'KS(salt) = KS(swide) = CONS(swide)'), 'ctrl_pw30p1': ('pw30', 1.0, 'CONTROL (CVX block at P_OA 1): never push')}
CELL = 18
sha = lambda b: hashlib.sha256(b).hexdigest()
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
dump = lambda nb: json.dumps(nb, indent=1, ensure_ascii=False).replace('\n', '\r\n').encode('utf-8')
LOG = open(os.path.join(KD, 'build_sink_kernels.log'), 'w', encoding='utf-8')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n')
builder_sha = sha(open(os.path.abspath(__file__), 'rb').read())
L_OA = "        _OA = ['Medial OA', 'Lateral OA', 'PF OA']"
L_U = "            _U = {c: pct(out[c].values.astype(np.float64)) for c in _OA}"
L_SYM = "                    out[c] = pct(_U[c] - A_OA * (_U[_o[0]] + _U[_o[1]]) / 2.0).astype(np.float32)"
L_PW = "                    out[c] = pct(_U[c] - A_OA / 2.0 * (_w[0] * _U[_o[0]] + _w[1] * _U[_o[1]])).astype(np.float32)"
N_SYM = "                    out[c] = pct(_U[c] - A_OA * (_S[_o[0]] + _S[_o[1]]) / 2.0).astype(np.float32)"
N_PW = "                    out[c] = pct(_U[c] - A_OA / 2.0 * (_w[0] * _S[_o[0]] + _w[1] * _S[_o[1]])).astype(np.float32)"
L_EOA = "            print('EOA_APPLIED a_oa %g mode %s on %s' % (A_OA, rec['oa_mode'], _OA), flush=True)"
N_P = lambda p: "        P_OA = %r   # [sink CVX] convex exponent on the SUBTRACTED other-compartment pct (levers/build_convex.py spec); 1 = linear E-OA" % p
N_S = "            _S = _U if P_OA == 1 else {c: _U[c] ** P_OA for c in _OA}"
N_MARK = ["            if P_OA != 1:", "                rec['p_oa'] = P_OA", "                print('CVX_APPLIED p_oa %g' % P_OA, flush=True)"]


def one(lines, old):
    k = [j for j, l in enumerate(lines) if l == old]; assert len(k) == 1, (old, k); return k[0]


def build(name, base, p, role):
    bdir, bsha = BASES[base]
    meta0b = open(os.path.join(bdir, 'kernel-metadata.json'), 'rb').read(); meta0 = json.loads(meta0b)
    raw = open(os.path.join(bdir, meta0['code_file']), 'rb').read()
    if bsha:
        assert sha(raw) == bsha, (base, sha(raw))
    else:
        assert sha(raw).startswith(LOMM30_SHA8), (base, sha(raw))
    if base in REG:
        assert sha(meta0b) == REG[base]['meta_sha256'], (base, 'meta sha')
    assert dump(json.loads(raw)) == raw, (base, 'json round-trip not byte-exact')
    nb = json.loads(raw); lines = src(nb['cells'][CELL]).split('\n')
    i = one(lines, L_OA); lines.insert(i + 1, N_P(p))
    i = one(lines, L_U); lines.insert(i + 1, N_S)
    i = one(lines, L_SYM); lines[i] = N_SYM
    i = one(lines, L_PW); lines[i] = N_PW
    i = one(lines, L_EOA); lines[i + 1:i + 1] = N_MARK
    nb['cells'][CELL]['source'] = '\n'.join(lines)
    btag = meta0['id'].split('rsna-sub-')[1]; assert btag.endswith('j')
    tag = btag[:-1] + ('cvx%gj' % p).replace('.', '')
    out = os.path.join(KD, 'kernel_' + tag); os.makedirs(out, exist_ok=True); cf = 'rsna_sub_%s.ipynb' % tag
    newb = dump(nb); open(os.path.join(out, cf), 'wb').write(newb)
    meta = dict(meta0); meta.update(id='eeshsaxena/rsna-sub-' + tag, title='rsna sub ' + tag, code_file=cf)
    json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
    metab = open(os.path.join(out, 'kernel-metadata.json'), 'rb').read()
    if p == 1.0:
        open(os.path.join(out, 'DO_NOT_PUSH.txt'), 'w').write('CONTROL (CVX block at P_OA 1.0 = linear E-OA). Never push, never register. levers/build_sink_kernels.py\n')
    else:
        open(os.path.join(out, 'NOT_ARMED.txt'), 'w').write('Local build only. Needs Eesh GO + PREREG amendment + commit gate levers/sink_commit_gate.py PASS before any push.\n')
    A_, B_ = json.loads(raw), json.loads(newb)
    assert len(A_['cells']) == len(B_['cells']) and A_['metadata'] == B_['metadata'] and A_.get('nbformat') == B_.get('nbformat')
    diff_cells = [j for j, (x, y) in enumerate(zip(A_['cells'], B_['cells'])) if x != y]; assert diff_cells == [CELL], diff_cells
    cx, cy = dict(A_['cells'][CELL]), dict(B_['cells'][CELL]); cx.pop('source'); cy.pop('source'); assert cx == cy
    ud = [l for l in difflib.unified_diff(src(A_['cells'][CELL]).split('\n'), src(B_['cells'][CELL]).split('\n'), lineterm='', n=0)
          if l[:1] in '+-' and not l.startswith(('+++', '---'))]
    exp = ['+' + N_P(p), '+' + N_S, '-' + L_SYM, '+' + N_SYM, '-' + L_PW, '+' + N_PW] + ['+' + l for l in N_MARK]
    assert sorted(ud) == sorted(exp), ('cell-18 diff is not exactly the CVX edits', ud)
    m0, m1 = meta0, json.loads(metab); assert {q for q in set(m0) | set(m1) if m0.get(q) != m1.get(q)} == {'id', 'title', 'code_file'}
    rc = dict(probe='sink', build=name, role=role, control=p == 1.0, tag=tag, slug='eeshsaxena/rsna-sub-' + tag, base=base,
              base_dir=os.path.relpath(bdir, os.path.join(S20, '..')).replace(chr(92), '/'), base_nb_sha256=sha(raw), base_meta_sha256=sha(meta0b),
              base_registry_key=base if base in REG else None, p_oa=p, components_added=['CVX (convex E-OA exponent p %g on the subtracted pct)' % p] if p != 1.0 else [],
              out_nb_sha256=sha(newb), meta_sha256=sha(metab), cell=CELL, cells_changed=diff_cells, cell18_diff=ud, builder_sha256=builder_sha,
              built_at_ist=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), armed=False)
    json.dump(rc, open(os.path.join(KD, 'build_receipt_%s.json' % tag), 'w', encoding='utf-8'), indent=1)
    P('SINK_BUILT %-11s %-34s base %-6s %s -> nb %s | meta %s | %s' % (name, tag, base, sha(raw)[:8], sha(newb)[:16], sha(metab)[:16], role))
    return dict(tag=tag, base=base, p_oa=p, role=role, nb_sha256=sha(newb), meta_sha256=sha(metab), base_nb_sha256=sha(raw))


IDX = {k: build(k, *v) for k, v in BUILDS.items()}
json.dump(IDX, open(os.path.join(KD, 'build_index.json'), 'w'), indent=1)
P('SINK_BUILD DONE %d kernels (6 stacks + 1 control), builder %s' % (len(IDX), builder_sha[:16]))
LOG.close()
