#!/usr/bin/env python3
"""k12_nbdiff.py (2026-10-09, Claude): prove that a K12 notebook == its TESTED FN11 twin (rsna-sub-k11fn or -n70) except:
  md cell            the title slug and the F bullet (fleet description + the two dataset slugs)
  NEW cell 2         the K12-MERGE cell: 'K12M_CFG = <literal>' + cells/k12_merge_cell.py byte for byte, metadata {'k2': 'k12_merge'}
  fleet cell         K2_CFG keys MEMBERS / TTA only (R: the 5 set-D names replaced in place by the 5 D-OAI names; A: the
                     5 D-OAI names appended), the comment; body = cells/k12_fleet_cell.py, which reverses to the same K5
                     cell as the twin's k11w body
  N cell             K9N_CFG key TAG only (FN11 -> K12) ; line 2+ (child b64 + FN cell) byte-identical
  metadata           id, title, code_file, dataset_sources (k11w kept first, k12w inserted second, N weights unchanged)
Everything else (notebook metadata, FN-IDS cell, cell metadata, every other config key) must be identical.
  python k12_nbdiff.py NEW_KERNEL_DIR REF_KERNEL_DIR --variant R|A [--out FILE]
"""
import argparse, ast, difflib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
sys.path.insert(0, HERE)
import make_k12_tools as mt, make_k10_tools as mt10                    # noqa: E402

DONAME = re.compile(r'^f([0-4])_s1043f75oai_raw_ttr224-320x24_cv\d{4}_best\.pt$')


def cfg_line(src, key):
    l1, rest = src.split('\n', 1)
    assert l1.startswith(key), l1[:60]
    lit, sep, comment = l1[len(key):].partition('   #')
    return ast.literal_eval(lit), comment, rest


def load(kdir):
    meta = json.load(open(f'{kdir}/kernel-metadata.json', encoding='utf-8'))
    p = f'{kdir}/{meta["code_file"]}'
    return meta, json.load(open(p, encoding='utf-8')), p


def check(new, ref, variant):
    mn, A, pa = load(new.replace('\\', '/'))
    mr, R, pr = load(ref.replace('\\', '/'))
    sn, sr = mn['id'].split('/')[1], mr['id'].split('/')[1]
    assert {k: v for k, v in A.items() if k != 'cells'} == {k: v for k, v in R.items() if k != 'cells'}, 'notebook metadata'
    assert len(A['cells']) == 5 and len(R['cells']) == 4
    m = {0: 0, 1: 1, 3: 2, 4: 3}                                          # new index -> twin index
    for i, j in m.items():
        assert {k: v for k, v in A['cells'][i].items() if k != 'source'} == {k: v for k, v in R['cells'][j].items() if k != 'source'}, f'cell {i} meta'
    assert A['cells'][1]['source'] == R['cells'][1]['source'], 'FN-IDS cell differs'
    la, lr = ''.join(A['cells'][0]['source']).split('\n'), ''.join(R['cells'][0]['source']).split('\n')
    assert len(la) == len(lr)
    chg = [(i, x, y) for i, (x, y) in enumerate(zip(la, lr)) if x != y]
    assert [i for i, _, _ in chg] in ([0, 2], [2]) and la[0][len(f'# {sn}'):] == lr[0][len(f'# {sr}'):], f'md lines {chg}'
    assert '`eeshsaxena/rsna-knee-cnxt-fleet-k11w`' in la[2] and '`eeshsaxena/rsna-knee-cnxt-fleet-k12w`' in la[2]
    mc = A['cells'][2]
    assert mc['cell_type'] == 'code' and mc['metadata'] == {'k2': 'k12_merge'} and mc.get('outputs') == []
    k12m, _, mbody = cfg_line(''.join(mc['source']), 'K12M_CFG = ')
    assert mbody == open(f'{HERE}/cells/k12_merge_cell.py', encoding='utf-8').read(), 'merge cell body != cells/k12_merge_cell.py'
    assert k12m['base_ds'] == 'rsna-knee-cnxt-fleet-k11w' and k12m['add_ds'] == 'rsna-knee-cnxt-fleet-k12w'
    ca, _, ba = cfg_line(''.join(A['cells'][3]['source']), 'K2_CFG = ')
    cr, _, br = cfg_line(''.join(R['cells'][2]['source']), 'K2_CFG = ')
    assert set(ca) == set(cr) and {k for k in ca if ca[k] != cr[k]} <= {'MEMBERS', 'TTA'}
    mem, B = ca['MEMBERS'], cr['MEMBERS']
    if variant == 'R':
        assert len(mem) == len(B) and all((x == y) or (y.startswith(f'f{DONAME.match(x).group(1)}_s1043f75_raw_') if DONAME.match(x) else False)
                                          for x, y in zip(mem, B)), 'R: not the twin with D replaced in place by D-OAI'
        assert sum(bool(DONAME.match(x)) for x in mem) == 5
    else:
        assert mem[:len(B)] == B and all(DONAME.match(x) for x in mem[len(B):]) and len(mem) == len(B) + 5
    assert ca['TTA'] == {n: 'AB' for n in mem} and set(cr['TTA'].values()) == {'AB'}
    assert ba == mt.make()['cells/k12_fleet_cell.py'] and mt.unpatch_k12(ba) == mt10.unpatch_cell(br, 'rsna-knee-cnxt-fleet-k11w')
    assert k12m['merged_manifest']['checkpoints'] == k12m['base_manifest']['checkpoints'] + k12m['add_manifest']['checkpoints']
    assert set(mem) <= set(k12m['merged_manifest']['checkpoints'])
    na, nma, ra = cfg_line(''.join(A['cells'][4]['source']), 'K9N_CFG = ')
    nr, nmr, rr = cfg_line(''.join(R['cells'][3]['source']), 'K9N_CFG = ')
    assert ra == rr and nma == nmr, 'N cell body / comment differs'
    assert set(na) == set(nr) and {k for k in na if na[k] != nr[k]} == {'TAG'} and na['TAG'] == 'K12'
    dk = {k for k in set(mn) | set(mr) if mn.get(k) != mr.get(k)}
    assert dk <= {'id', 'title', 'code_file', 'dataset_sources'}, dk
    assert mn['dataset_sources'] == [mr['dataset_sources'][0], 'eeshsaxena/rsna-knee-cnxt-fleet-k12w'] + mr['dataset_sources'][1:]
    tl = open(pa, encoding='utf-8').read().splitlines(); rl = open(pr, encoding='utf-8').read().splitlines()
    d = [x if len(x) < 1500 else x[:1500] + f'... [{len(x)} chars]'
         for x in difflib.unified_diff(rl, tl, fromfile=f'TESTED {sr}', tofile=f'NEW {sn}', n=0, lineterm='')]
    summ = dict(new=sn, ref=sr, variant=variant, members=len(mem), members_ref=len(B), md_lines=len(chg),
                k2cfg_changed=sorted(k for k in ca if ca[k] != cr[k]), k9n_changed=['TAG'], meta_changed=sorted(dk),
                merge_cell_lines=len(''.join(mc['source']).splitlines()),
                changed_text_lines=sum(1 for x in d if x.startswith(('+', '-')) and not x.startswith(('+++', '---'))))
    return '\n'.join(d) + '\n\nSUMMARY ' + json.dumps(summ) + '\n', summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('new'); ap.add_argument('ref'); ap.add_argument('--variant', required=True, choices=('R', 'A'))
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    txt, summ = check(a.new, a.ref, a.variant)
    if a.out:
        open(a.out, 'w', encoding='utf-8').write(txt)
    print(f'K12NBDIFF OK: {summ}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
