#!/usr/bin/env python3
"""wdiff.py (2026-10-09, Claude; kfn kit, byte copy in k10/ written by make_n70_tools.py): prove that an N-weight
variant FN notebook (build_kfn.py / build_k10fn.py --w_fn F,N --slug_suffix -n70) differs from its TESTED twin ONLY in
the F/N weight config. Every other byte (FN-IDS cell, fleet cell incl. K2_CFG, N cell body, embedded child, every cell's
metadata, notebook metadata) must be identical.

Allowed differences:
  md cell (0)          the kernel slug in the title, 'F .40 / N .60' and '.40 rank(F) + .60 rank(N)' (each exactly once)
  N cell (last)        line 1 only (the K9N_CFG literal), and inside it only the keys W3 and W3_VARIANTS
  kernel-metadata.json the keys id, title, code_file (the slug)
check() raises AssertionError on anything else and returns the unified text diff of the two notebook files.
"""
import ast, difflib, json

ALLOWED_CFG = {'W3', 'W3_VARIANTS'}
ALLOWED_META = {'id', 'title', 'code_file'}


def wtxt(x):
    """0.4 -> '.40' (the md cell's weight format)."""
    return f'{float(x):.2f}'[1:]


def cfg_line(src, key='K9N_CFG = '):
    l1, rest = src.split('\n', 1)
    assert l1.startswith(key), f'first line is not {key!r}: {l1[:60]!r}'
    lit, sep, comment = l1[len(key):].partition('   #')
    return ast.literal_eval(lit), sep + comment, rest


def check(new_nb, ref_nb, new_slug, ref_slug, new_w, ref_w, new_meta=None, ref_meta=None):
    """new_w / ref_w: the main P/F/N triples. Returns (diff_text, summary)."""
    ta, tb = open(new_nb, encoding='utf-8').read(), open(ref_nb, encoding='utf-8').read()
    a, b = json.loads(ta), json.loads(tb)
    assert {k: v for k, v in a.items() if k != 'cells'} == {k: v for k, v in b.items() if k != 'cells'}, 'notebook metadata differs'
    assert len(a['cells']) == len(b['cells']), (len(a['cells']), len(b['cells']))
    last = len(a['cells']) - 1
    md_lines, cfg_diff = set(), {}
    for i, (ca, cb) in enumerate(zip(a['cells'], b['cells'])):
        rest_a = {k: v for k, v in ca.items() if k != 'source'}
        rest_b = {k: v for k, v in cb.items() if k != 'source'}
        sa, sb = ''.join(ca['source']), ''.join(cb['source'])
        if i == 0:
            assert ca['cell_type'] == cb['cell_type'] == 'markdown' and rest_a == rest_b, 'md cell type / metadata differs'
            reps = [(f'F {wtxt(new_w[1])} / N {wtxt(new_w[2])}', f'F {wtxt(ref_w[1])} / N {wtxt(ref_w[2])}'),
                    (f'{wtxt(new_w[1])} rank(F) + {wtxt(new_w[2])} rank(N)', f'{wtxt(ref_w[1])} rank(F) + {wtxt(ref_w[2])} rank(N)')]
            if new_slug != ref_slug:
                reps.insert(0, (f'# {new_slug}:', f'# {ref_slug}:'))
            r = sa
            for x, y in reps:
                assert r.count(x) == 1, f'md: {x!r} found {r.count(x)} times (want exactly 1)'
                r = r.replace(x, y)
            assert r == sb, 'md cell differs beyond the slug + the two weight phrases'
            md_lines = {l for l in sa.splitlines() if l not in sb.splitlines()} | \
                       {l for l in sb.splitlines() if l not in sa.splitlines()}
        elif i == last:
            assert rest_a == rest_b, 'N cell metadata / outputs differ'
            da, ca_c, ra = cfg_line(sa)
            db, cb_c, rb = cfg_line(sb)
            assert ra == rb, 'N cell differs after its K9N_CFG line (child b64 / body)'
            assert ca_c == cb_c, 'N cell K9N_CFG trailing comment differs'
            assert set(da) == set(db), f'K9N_CFG keys differ {set(da) ^ set(db)}'
            cfg_diff = {k: (db[k], da[k]) for k in da if da[k] != db[k]}
            assert set(cfg_diff) <= ALLOWED_CFG, f'K9N_CFG differs outside W3 / W3_VARIANTS: {sorted(set(cfg_diff) - ALLOWED_CFG)}'
            assert da['W3'] == list(new_w) and db['W3'] == list(ref_w), (da['W3'], db['W3'])
        else:
            assert ca == cb, f'cell {i} ({ca.get("metadata")}) differs'
    # the raw text diff: every changed line is a md source line or the K9N_CFG line
    diff = list(difflib.unified_diff(tb.splitlines(), ta.splitlines(), fromfile=f'TESTED {ref_slug}', tofile=f'NEW {new_slug}',
                                     n=0, lineterm=''))
    changed = [l for l in diff if l[:1] in '+-' and not l.startswith(('+++', '---'))]
    for l in changed:
        try:
            body = json.loads(l[1:].strip().rstrip(','))
        except ValueError:
            body = None
        assert isinstance(body, str) and (body.rstrip('\n').startswith('K9N_CFG = ') or body.rstrip('\n') in md_lines), \
            f'unexpected changed line {l[:120]!r}'
    summ = dict(cells=len(a['cells']), identical_cells=[i for i in range(1, last)], changed_text_lines=len(changed),
                md_changed_lines=sorted(md_lines), k9n_cfg_changed=cfg_diff)
    if new_meta is not None:
        ma, mb = json.load(open(new_meta)), json.load(open(ref_meta))
        assert set(ma) == set(mb), 'kernel-metadata keys differ'
        md = {k: (mb[k], ma[k]) for k in ma if ma[k] != mb[k]}
        assert set(md) <= ALLOWED_META, f'kernel-metadata differs outside id / title / code_file: {sorted(set(md) - ALLOWED_META)}'
        assert ma['id'] == f'eeshsaxena/{new_slug}' and ma['title'] == new_slug.replace('-', ' ') \
            and ma['code_file'] == new_slug.replace('-', '_') + '.ipynb', ma
        summ['meta_changed'] = md
    return '\n'.join(diff), summ
