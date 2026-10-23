#!/usr/bin/env python3
"""k11_nbdiff.py (2026-10-09, Claude): prove that an FN11 notebook differs from its TESTED FN10-3 twin ONLY in the fleet
member list / fleet dataset / kernel naming. Every other byte (FN-IDS cell, fleet cell body modulo the dataset slug, every
K2_CFG key except MEMBERS / TTA, the N cell body incl. the embedded child and its sha, every K9N_CFG key except TAG, all
cell metadata, notebook metadata) must be identical.

Allowed differences (anything else raises AssertionError):
  md cell (0)    the slug in the title, the fleet description bullet (member count / sets), the dataset slug
  fleet cell (2) line 1: K2_CFG keys MEMBERS and TTA only; MEMBERS = the twin's MEMBERS (same order) + new members
                 (E2 / D2 keys only, f{F}_s1046t288f75 / f{F}_s1044f75), TTA = the twin's TTA + 'AB' for each new member;
                 the comment after the literal (writer path / set list). Body: == the twin's body with its 3 dataset slug
                 strings replaced (k10w -> k11w), i.e. both reverse to the K5 cell.
  N cell (3)     line 1: K9N_CFG key TAG only (and W3 / W3_VARIANTS when the two are N-weight twins of each other:
                 --weights_may_differ); the comment's writer path. Line 2+ (child b64 + FN cell): byte-identical.
  metadata       id, title, code_file (slug) and dataset_sources (the fleet dataset slug only; N weights unchanged).
  python k11_nbdiff.py NEW_KERNEL_DIR REF_KERNEL_DIR [--out NB_DIFF.txt] [--weights_may_differ]
Last line: K11NBDIFF OK ... (exit 0) or an AssertionError.
"""
import argparse, ast, difflib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
sys.path.insert(0, HERE)
import make_k11_tools as mt                                           # noqa: E402

NEWKEY = re.compile(r'^f[0-4]_s(1046t288f75|1044f75)_raw_ttr(288|224)-320x24_cv\d{4}_best\.pt$')


def cfg_line(src, key):
    l1, rest = src.split('\n', 1)
    assert l1.startswith(key), f'first line is not {key!r}: {l1[:60]!r}'
    lit, sep, comment = l1[len(key):].partition('   #')
    return ast.literal_eval(lit), comment, rest


def load(kdir):
    meta = json.load(open(f'{kdir}/kernel-metadata.json', encoding='utf-8'))
    return meta, json.load(open(f'{kdir}/{meta["code_file"]}', encoding='utf-8')), f'{kdir}/{meta["code_file"]}'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('new'); ap.add_argument('ref')
    ap.add_argument('--out', default=None)
    ap.add_argument('--weights_may_differ', action='store_true')
    a = ap.parse_args(argv)
    mn, A, pa = load(a.new.replace('\\', '/'))
    mr, R, pr = load(a.ref.replace('\\', '/'))
    sn, sr = mn['id'].split('/')[1], mr['id'].split('/')[1]
    dsn = [d for d in mn['dataset_sources'] if d.startswith('eeshsaxena/rsna-knee-cnxt-fleet')]
    dsr = [d for d in mr['dataset_sources'] if d.startswith('eeshsaxena/rsna-knee-cnxt-fleet')]
    assert len(dsn) == len(dsr) == 1, (dsn, dsr)
    dn, dr = dsn[0].split('/')[1], dsr[0].split('/')[1]
    summ = dict(new=sn, ref=sr, ds_new=dn, ds_ref=dr)
    # ---- notebook metadata, cell count, cell 1
    assert {k: v for k, v in A.items() if k != 'cells'} == {k: v for k, v in R.items() if k != 'cells'}, 'notebook metadata differs'
    assert len(A['cells']) == len(R['cells']) == 4
    for i in range(4):
        assert {k: v for k, v in A['cells'][i].items() if k != 'source'} == {k: v for k, v in R['cells'][i].items() if k != 'source'}, \
            f'cell {i} metadata / type / outputs differ'
    assert A['cells'][1]['source'] == R['cells'][1]['source'], 'FN-IDS cell differs'
    # ---- md cell
    ma, mr_ = ''.join(A['cells'][0]['source']).splitlines(), ''.join(R['cells'][0]['source']).splitlines()
    assert len(ma) == len(mr_)
    md_changed = [(x, y) for x, y in zip(ma, mr_) if x != y]
    for x, y in md_changed:
        assert (x.startswith(f'# {sn}:') and y.startswith(f'# {sr}:') and x[len(sn) + 2:] == y[len(sr) + 2:]) or \
               (x.startswith('* **F**: our ') and y.startswith('* **F**: our ') and f'`eeshsaxena/{dn}`' in x and f'`eeshsaxena/{dr}`' in y) or \
               (a.weights_may_differ and x.replace(sn, sr) != x), f'md line not allowed to differ: {x!r} vs {y!r}'
    summ['md_changed_lines'] = len(md_changed)
    # ---- fleet cell
    fa, fr = ''.join(A['cells'][2]['source']), ''.join(R['cells'][2]['source'])
    ca, cma, ba = cfg_line(fa, 'K2_CFG = ')
    cr, cmr, br = cfg_line(fr, 'K2_CFG = ')
    assert set(ca) == set(cr), (set(ca) ^ set(cr))
    other = {k for k in ca if ca[k] != cr[k]}
    assert other <= {'MEMBERS', 'TTA'}, f'K2_CFG differs in {other}'
    assert ca['MEMBERS'][:len(cr['MEMBERS'])] == cr['MEMBERS'], 'the twin members are not the leading members, in order'
    new_m = ca['MEMBERS'][len(cr['MEMBERS']):]
    assert all(NEWKEY.match(m) for m in new_m), f'unexpected new member names {new_m}'
    assert ca['TTA'] == dict(cr['TTA'], **{m: 'AB' for m in new_m}) and list(ca['TTA'])[:len(cr['TTA'])] == list(cr['TTA'])
    assert ba == br.replace(dr, dn) and br.count(dr) == 3 and ba.count(dn) == 3, 'fleet cell body != the twin with the dataset slug swapped'
    assert mt.unpatch_cell(ba, dn) == mt.unpatch_cell(br, dr), 'fleet bodies do not reverse to the same K5 cell'
    summ.update(members_ref=len(cr['MEMBERS']), members_new=len(ca['MEMBERS']), new_members=new_m,
                k2cfg_changed=sorted(other), fleet_comment_changed=cma != cmr)
    # ---- N cell
    na, nr = ''.join(A['cells'][3]['source']), ''.join(R['cells'][3]['source'])
    ka, kma, ra = cfg_line(na, 'K9N_CFG = ')
    kr, kmr, rr = cfg_line(nr, 'K9N_CFG = ')
    assert ra == rr, 'N cell differs after its K9N_CFG line (child b64 / FN cell body)'
    allowed = {'TAG'} | ({'W3', 'W3_VARIANTS'} if a.weights_may_differ else set())
    assert set(ka) == set(kr) and {k for k in ka if ka[k] != kr[k]} <= allowed, f'K9N_CFG differs in {[k for k in ka if ka[k] != kr[k]]}'
    summ.update(k9ncfg_changed=sorted(k for k in ka if ka[k] != kr[k]), tag_new=ka.get('TAG'), tag_ref=kr.get('TAG'),
                child_sha256=ka['CHILD_SHA256'], w3_new=ka['W3'], w3_ref=kr['W3'])
    # ---- kernel metadata
    dk = {k for k in set(mn) | set(mr) if mn.get(k) != mr.get(k)}
    assert dk <= {'id', 'title', 'code_file', 'dataset_sources'}, f'kernel-metadata differs in {dk}'
    assert [d if d != dsr[0] else dsn[0] for d in mr['dataset_sources']] == mn['dataset_sources'], 'dataset_sources differ beyond the fleet slug'
    summ['meta_changed'] = sorted(dk)
    # ---- the text diff (the child b64 line is identical, so it never shows)
    la = open(pa, encoding='utf-8').read().splitlines()
    lr = open(pr, encoding='utf-8').read().splitlines()
    d = list(difflib.unified_diff(lr, la, fromfile=os.path.relpath(pr, os.path.dirname(HERE)), tofile=os.path.relpath(pa, os.path.dirname(HERE)), n=0, lineterm=''))
    d = [x if len(x) < 2000 else x[:2000] + f'... [{len(x)} chars]' for x in d]
    dm = list(difflib.unified_diff(json.dumps(mr, indent=2).splitlines(), json.dumps(mn, indent=2).splitlines(),
                                   fromfile='ref kernel-metadata.json', tofile='new kernel-metadata.json', n=0, lineterm=''))
    summ['changed_text_lines'] = sum(1 for x in d if x.startswith(('+', '-')) and not x.startswith(('+++', '---')))
    txt = ('\n'.join(d) + '\n\n' + '\n'.join(dm) + '\n\nSUMMARY ' + json.dumps(summ) + '\n')
    if a.out:
        open(a.out, 'w', encoding='utf-8').write(txt)
    print(txt[-3000:] if len(txt) > 3000 else txt)
    print(f'K11NBDIFF OK: {sn} == TESTED {sr} except {summ["changed_text_lines"]} text lines: md {summ["md_changed_lines"]}, '
          f'K2_CFG {summ["k2cfg_changed"]} (+{len(new_m)} members), fleet body slug {dr} -> {dn} (3), K9N_CFG '
          f'{summ["k9ncfg_changed"]}, metadata {summ["meta_changed"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
