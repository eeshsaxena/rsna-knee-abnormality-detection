#!/usr/bin/env python3
"""t_k8_build.py (laptop; TEST ONLY; 2026-10-07, Claude): build_k8.py refusals and the KF cell provenance, fast (no build is
written: every case fails inside load_members() / main() before any copy, or checks a string).
Input: a collected members.json + its ckpt dir (default the simulation collect k8/_sim).
  B1 the sim members.json loads: 10 members, keys f{F}_s1043 (A) and f{F}_s1044t288 (C)
  B2 a set-B member (s1044 TTR224) -> refused ('DROPPED')
  B3 an EMA pick -> refused (K8 ships raw)
  B4 a set-A fold missing (4/5) -> refused
  B5 duplicate (seed, fold, ttr_size) -> refused
  B6 C 4/5 without do_not_submit -> refused (INTERIM needs K8_INTERIM=1); C 5/5 marked do_not_submit -> refused
  B7 a train_labels other than mix50 -> refused
  B8 KF cell: k2_fleet_cell_kf.py == make_kf_cell.make(); strip_kf == the K5 cell; a one-character change to the KF cell
     breaks the equality (make_k2_kernel asserts it before writing any KF notebook); the KF block sha is the tested one
  B9 make_k2_kernel --outer_flat with --main_mode blend / solo / --combined -> AssertionError (nothing written)
  python tests/t_k8_build.py [members.json] [ckpt dir]
"""
import copy, json, os, sys, tempfile

T_ = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
K8 = os.path.dirname(T_)
SP = os.path.dirname(K8)
sys.path.insert(0, K8)
import build_k8, make_kf_cell, make_k2_kernel  # noqa: E402

R = []


def ok(name, cond, extra=''):
    R.append(bool(cond)); print(f'{"PASS" if cond else "FAIL"} {name} {str(extra)[:200]}', flush=True)


def refuses(m, ck, needle, main_args=None):
    d = tempfile.mkdtemp(prefix='k8b_')
    p = f'{d}/members.json'
    json.dump(m, open(p, 'w'))
    try:
        if main_args is not None:
            build_k8.main(['--members_json', p, '--ckpt_dir', ck, '--out', f'{d}/out'] + main_args)
        else:
            build_k8.load_members(p, ck)
    except AssertionError as e:
        return needle in str(e), str(e)[:160]
    return False, 'not refused'


def main():
    mj = sys.argv[1] if len(sys.argv) > 1 else f'{K8}/_sim/members.json'
    ck = sys.argv[2] if len(sys.argv) > 2 else f'{K8}/_sim/ckpt'
    base = json.load(open(mj))
    _, mem = build_k8.load_members(mj, ck)
    keys = [r['key'] for r in mem]
    ok('B1 sim members.json loads: A f{F}_s1043 + C f{F}_s1044t288', keys == [f'f{F}_s1043' for F in range(5)] +
       [f'f{F}_s1044t288' for F in range(5)], keys)
    m = copy.deepcopy(base); r = m['members'][6]
    m['members'][6] = dict(r, ttr_size=224, file=r['file'].replace('_ttr288-', '_ttr224-'))
    ok('B2 set-B member refused', *refuses(m, ck, 'DROPPED'))
    m = copy.deepcopy(base); m['members'][7]['tag'] = 'ema'
    ok('B3 EMA pick refused', *refuses(m, ck, 'RAW'))
    m = copy.deepcopy(base); m['members'] = [x for x in m['members'] if not (x['set'] == 'A' and x['fold'] == 4)]
    ok('B4 set A 4/5 refused', *refuses(m, ck, 'set A folds'))
    m = copy.deepcopy(base); m['members'].append(copy.deepcopy(m['members'][5]))
    ok('B5 duplicate member refused', *refuses(m, ck, 'duplicate'))
    m = copy.deepcopy(base); m['members'] = [x for x in m['members'] if not (x['set'] == 'C' and x['fold'] == 4)]
    good, det = refuses(m, ck, 'k8/ckpt files', None)   # load_members: the on-disk check sees the extra C f4 file first
    m2 = copy.deepcopy(m)
    with tempfile.TemporaryDirectory() as d2:            # same 9 members with a matching ckpt dir view
        for x in m2['members']:
            os.link(f'{ck}/{x["file"]}', f'{d2}/{x["file"]}') if hasattr(os, 'link') else None
        g1, d1 = refuses(m2, d2, 'INTERIM build needs', [])
        m3 = copy.deepcopy(base); m3['do_not_submit'] = True; m3['interim'] = True
        g2, dd2 = refuses(m3, ck, 'marked do_not_submit', [])
    ok('B6 C 4/5 without do_not_submit refused; C 5/5 marked do_not_submit refused', g1 and g2, f'{d1} | {dd2}')
    m = copy.deepcopy(base); m['members'][2]['train_labels'] = 'llm_labels_v4_blend.csv'
    ok('B7 non-mix50 train_labels refused', *refuses(m, ck, 'mix50'))
    kf = open(f'{K8}/cells/k2_fleet_cell_kf.py', encoding='utf-8').read()
    k5 = open(f'{K8}/cells/k2_fleet_cell.py', encoding='utf-8').read()
    bad = kf.replace("rebuild check max abs", "rebuild check max  abs", 1)
    ok('B8 KF cell == make_kf_cell.make(), strip_kf == K5 cell, a 1-char change breaks it, block sha pinned',
       kf == make_kf_cell.make() and make_kf_cell.strip_kf(kf) == k5 and bad != make_kf_cell.make()
       and make_kf_cell.KF_BLOCK_SHA == '0fa113ce1bad17e0642366fed0d8af5b57e2aa29df7417a34bed6e1fc7f1e4e2')
    res = []
    for extra in (['--main_mode', 'blend'], ['--main_mode', 'solo'], ['--main_mode', 'strict', '--combined']):
        d = tempfile.mkdtemp(prefix='k8mk_')
        try:
            make_k2_kernel.main(['--out', f'{d}/x', '--slug', 'rsna-sub-k8-x', '--outer_flat', '0.6'] + extra)
            res.append(False)
        except AssertionError as e:
            res.append('outer_flat' in str(e) and not os.path.exists(f'{d}/x'))
    ok('B9 make_k2_kernel --outer_flat with blend / solo / combined refused, nothing written', all(res), res)
    print(f'TK8BUILD {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}')
    return 0 if all(R) else 1


if __name__ == '__main__':
    sys.exit(main())
