#!/usr/bin/env python3
"""t_k9_build.py (laptop; TEST ONLY; 2026-10-08, Claude; from t_k8_build.py): build_k9.py refusals, K9-KEY and the built
simulation kit, fast (no build is written by the refusal cases: each fails inside load_members() / main() before any copy).
Input: a collected members.json + its ckpt dir (default the simulation collect k9/_sim) and the sim build k9/build_sim.
  B1  the sim members.json loads: 10 members, keys f{F}_s1043f75 (D) then f{F}_s1044t288 (C)
  B2  a set-A member (D fold relabelled mix50) -> refused ('REPLACED')
  B3  a set-B member (s1044 TTR224 mix50) -> refused ('DROPPED')
  B4  an EMA pick -> refused (K9 ships raw)
  B5  duplicate (seed, fold, ttr_size, train_labels) -> refused
  B6  D 4/5 without do_not_submit -> refused (INTERIM needs K9_INTERIM=1); C 5/5 + D 5/5 marked do_not_submit -> refused
  B7  a train_labels that is neither set's (v4) -> refused ('not a K9 set')
  B8  K9-KEY: every key tagged (no plain f{F}_s1043 / f{F}_s1044, so no K9 key equals a K5/K7/K8 set A or set B key);
      staged names f{F}_s1043f75_raw_ttr224-... / f{F}_s1044t288_raw_ttr288-...
  B9  a members.json whose 'set' label disagrees with the member identity -> refused
  B10 KF cell == the K8 KF cell (sha256) == make_kf_cell.make(); strip_kf == the K5 cell
  B11 the sim build: 3 kernels, cell 31 == K5 cell (main, solo) / KF cell (kf) + K2_CFG, MEMBERS == the 10 staged names,
      OUTER_FLAT only on kf; upload manifest == the 10 tagged names; worker keys unique, 5 fold groups of 2 (D + C);
      K5_BUILD.json k8 + k9 + simulated + do_not_submit true; DO_NOT_PUSH_SIMULATION.txt present; the K8-INT guard in
      k2_check fires on this K5_BUILD.json
  python tests/t_k9_build.py [members.json] [ckpt dir] [build dir]
"""
import copy, hashlib, json, os, sys, tempfile

T_ = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
K9 = os.path.dirname(T_)
SP = os.path.dirname(K9)
sys.path.insert(0, K9)
import build_k9, make_kf_cell  # noqa: E402

R = []


def ok(name, cond, extra=''):
    R.append(bool(cond)); print(f'{"PASS" if cond else "FAIL"} {name} {str(extra)[:220]}', flush=True)


def refuses(m, ck, needle, main_args=None):
    d = tempfile.mkdtemp(prefix='k9b_')
    p = f'{d}/members.json'
    json.dump(m, open(p, 'w'))
    try:
        if main_args is not None:
            build_k9.main(['--members_json', p, '--ckpt_dir', ck, '--out', f'{d}/out'] + main_args)
        else:
            build_k9.load_members(p, ck)
    except AssertionError as e:
        return needle in str(e), str(e)[:200]
    return False, 'not refused'


def idx(m, s, F):
    return [i for i, x in enumerate(m['members']) if x['set'] == s and x['fold'] == F][0]


def main():
    mj = sys.argv[1] if len(sys.argv) > 1 else f'{K9}/_sim/members.json'
    ck = sys.argv[2] if len(sys.argv) > 2 else f'{K9}/_sim/ckpt'
    bd = sys.argv[3] if len(sys.argv) > 3 else f'{K9}/build_sim'
    base = json.load(open(mj))
    _, mem = build_k9.load_members(mj, ck)
    keys = [r['key'] for r in mem]
    ok('B1 sim members.json loads: D f{F}_s1043f75 + C f{F}_s1044t288', keys == [f'f{F}_s1043f75' for F in range(5)] +
       [f'f{F}_s1044t288' for F in range(5)], keys)
    m = copy.deepcopy(base); m['members'][idx(m, 'D', 2)]['train_labels'] = 'labels_mix50_logit.csv'
    ok('B2 set-A member (s1043 TTR224 mix50) refused', *refuses(m, ck, 'REPLACED'))
    m = copy.deepcopy(base); i = idx(m, 'C', 1); r = m['members'][i]
    m['members'][i] = dict(r, ttr_size=224, file=r['file'].replace('_ttr288-', '_ttr224-'))
    ok('B3 set-B member (s1044 TTR224) refused', *refuses(m, ck, 'DROPPED'))
    m = copy.deepcopy(base); m['members'][idx(m, 'D', 3)]['tag'] = 'ema'
    ok('B4 EMA pick refused', *refuses(m, ck, 'RAW'))
    m = copy.deepcopy(base); m['members'].append(copy.deepcopy(m['members'][idx(m, 'D', 0)]))
    ok('B5 duplicate member refused', *refuses(m, ck, 'duplicate'))
    m = copy.deepcopy(base); m['members'] = [x for x in m['members'] if not (x['set'] == 'D' and x['fold'] == 4)]
    with tempfile.TemporaryDirectory() as d2:            # the 9 members with a matching ckpt dir view (hard links)
        for x in m['members']:
            os.link(f'{ck}/{x["file"]}', f'{d2}/{x["file"]}')
        g1, d1 = refuses(m, d2, 'INTERIM build needs', [])
    m3 = copy.deepcopy(base); m3['do_not_submit'] = True; m3['interim'] = True
    g2, dd2 = refuses(m3, ck, 'marked do_not_submit', [])
    ok('B6 D 4/5 without do_not_submit refused; C 5/5 + D 5/5 marked do_not_submit refused', g1 and g2, f'{d1} | {dd2}')
    m = copy.deepcopy(base); m['members'][idx(m, 'D', 1)]['train_labels'] = 'llm_labels_v4_blend.csv'
    ok('B7 train_labels of neither set refused', *refuses(m, ck, 'not a K9 set'))
    old_keys = {f'f{F}_s{S}' for F in range(5) for S in (1042, 1043, 1044)}
    staged = [r['staged'] for r in mem]
    ok('B8 K9-KEY: every key tagged, none equals a K5/K7/K8 key; staged names carry the tag',
       not (set(keys) & old_keys) and all(s.startswith(f'f{r["fold"]}_s1043f75_raw_ttr224-') if r['set'] == 'D'
                                          else s.startswith(f'f{r["fold"]}_s1044t288_raw_ttr288-') for s, r in zip(staged, mem)),
       staged)
    m = copy.deepcopy(base); m['members'][idx(m, 'D', 0)]['set'] = 'C'
    ok('B9 members.json set label != member identity refused', *refuses(m, ck, 'its identity is set D'))
    kf = open(f'{K9}/cells/k2_fleet_cell_kf.py', 'rb').read()
    k5 = open(f'{K9}/cells/k2_fleet_cell.py', encoding='utf-8').read()
    ok('B10 KF cell == K8 KF cell == make_kf_cell.make(), strip_kf == K5 cell',
       kf == open(f'{SP}/k8/cells/k2_fleet_cell_kf.py', 'rb').read() and kf.decode('utf-8') == make_kf_cell.make()
       and make_kf_cell.strip_kf(kf.decode('utf-8')) == k5, hashlib.sha256(kf).hexdigest()[:16])
    # B11 the built simulation kit
    bj = json.load(open(f'{bd}/K5_BUILD.json'))
    man = json.load(open(f'{bd}/upload/rsna-knee-cnxt-fleet/fleet_manifest.json'))
    good = man['checkpoints'] == bj['members'] == staged
    wk = ['_'.join(n.split('_')[:2]) for n in man['checkpoints']]
    groups = {}
    for n in man['checkpoints']:
        groups.setdefault(n.split('_')[0], []).append(n)
    good &= len(set(wk)) == len(wk) and sorted(groups) == [f'f{F}' for F in range(5)] and all(len(v) == 2 for v in groups.values())
    cells = {}
    for slug, ksub in (('rsna-sub-k9-fleet', 'fleet'), ('rsna-sub-k9-solo', 'solo'), ('rsna-sub-k9-kf', 'kf')):
        nb = json.load(open(f'{bd}/kernel/{slug}/rsna_sub_k9_{ksub}.ipynb', encoding='utf-8'))
        src = ''.join(nb['cells'][31]['source'])
        hdr, body = src.split('\n', 1)
        cfg = eval(hdr[len('K2_CFG = '):])                                # the literal dict make_k2_kernel wrote
        cells[ksub] = (body == (kf.decode('utf-8') if ksub == 'kf' else k5), cfg.get('MEMBERS') == staged,
                       cfg.get('OUTER_FLAT') == (0.6 if ksub == 'kf' else None), cfg['MAIN_MODE'])
    good &= all(c[0] and c[1] and c[2] for c in cells.values()) and [c[3] for c in cells.values()] == ['strict', 'solo', 'strict']
    good &= bj.get('k8') is True and bj.get('k9') is True and bj.get('simulated') is True and bj.get('do_not_submit') is True
    good &= os.path.isfile(f'{bd}/DO_NOT_PUSH_SIMULATION.txt') and not os.path.exists(
        f'{bd}/upload/rsna-knee-cnxt-fleet/DO_NOT_PUSH_SIMULATION.txt')
    ok('B11 sim build: 3 kernels (K5 / KF cells, MEMBERS, OUTER_FLAT kf only), manifest == staged, 5 fold groups of 2, '
       'K5_BUILD k8/k9/simulated/do_not_submit, marker file', good, {k: v for k, v in cells.items()})
    import subprocess
    r = subprocess.run([sys.executable, f'{K9}/k2_check.py', '--work', f'{bd}/upload/rsna-knee-cnxt-fleet',
                        '--expect_tta', f'{bd}/K5_BUILD.json'], capture_output=True, text=True)
    ok('B12 k2_check --expect_tta <sim K5_BUILD.json> FAILS with the K8-INT line (a sim/interim build is never submitted)',
       r.returncode != 0 and 'an INTERIM build is NEVER submitted' in r.stdout, r.stdout.strip().splitlines()[-1:] if r.stdout else r.stderr[-200:])
    print(f'TK9BUILD {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}')
    return 0 if all(R) else 1


if __name__ == '__main__':
    sys.exit(main())
