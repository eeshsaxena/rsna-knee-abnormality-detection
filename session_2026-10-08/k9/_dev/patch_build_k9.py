"""patch_build_k9.py (2026-10-08, Claude): turn the copied build_k8.py into build_k9.py (sets C + D, K9-KEY). Run once."""
import os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(HERE, 'build_k9.py')
s = open(p, encoding='utf-8').read()
assert 'build_k8.py (2026-10-07' in s, 'already patched?'


def rep(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, (old[:80], c)
    s = s.replace(old, new)


HDR = '''#!/usr/bin/env python3
"""build_k9.py (2026-10-08, Claude; a COPY of build_k8.py with the K9 member sets). Local only: nothing is pushed, versioned
or submitted.

K9 = parent + [5 x TTR288 mix50 s1044 (set C) + 5 x TTR224 frac75 s1043 (set D)] (NOVEL_PLAN_10-07 sec 4 row 10-09):
  set C  f{0..4}_s1044_ttr288mix50   TTR288, seed 1044, labels_mix50_logit.csv,  raw best (box chain), key f{F}_s1044t288
  set D  f{0..4}_s1043_ttr_frac75    TTR224, seed 1043, labels_frac75_logit.csv, raw best (Kaggle swapiy f01 E17,
                                     kmrsuman f23 + f4x E~21, run receipts),       key f{F}_s1043f75
  set A (s1043 TTR224 mix50, the K5 members) is REPLACED by set D and refused here; set B (s1044 TTR224) is refused.
Every member TTR (fleet_model_ttr, model_sha f141aeed136dfc01), every member A+B TTA, equal members (fleet = raw-prob mean,
each 1/10), parent W .50. Three kernels from ONE upload folder (one dataset version of eeshsaxena/rsna-knee-cnxt-fleet):
  K9-main  eeshsaxena/rsna-sub-k9-fleet  strict W .50 (variants .40 / .60), parent cell-28 map    K5 cell (byte for byte)
  K9-solo  eeshsaxena/rsna-sub-k9-solo   solo (submission.csv = the fleet alone)                  K5 cell (byte for byte)
  K9-KF    eeshsaxena/rsna-sub-k9-kf     strict W .50 with the KF flat parent K2_CFG['OUTER_FLAT'] = 0.60
                                         (cells/k2_fleet_cell_kf.py == the K8 KF cell, make_kf_cell.py)
K9-KEY: member identity = (seed, fold, ttr_size, train_labels). The shipped key carries a set tag so no K9 key can equal a
K5/K7/K8 key: set C f{F}_s1044 -> f{F}_s1044t288 (the K7-KEY rule), set D f{F}_s1043 -> f{F}_s1043f75 (labels tag; a frac75
member would otherwise ship as f{F}_s1043_raw_ttr224-..., the same key as the K5 set A member of that fold). File name only:
the bytes are identical (pack_fleet_ds.py P-TAG; the worker's fold grouping uses field 1 'f{F}', its duplicate check fields
1-2).

  python build_k9.py [--members_json k9/members.json] [--out DIR] [--overwrite] [--w_main 0.50] [--outer_flat 0.60]

Default --out: k9/build when C 5/5 and D 5/5 (FINAL); else k9/build_i<N> (INTERIM: members.json must carry do_not_submit
(k9_collect.sh K9_INTERIM=1); K5_BUILD.json then says interim + do_not_submit, the summary line says NEVER SUBMIT, and
k2_check.py --expect_tta <K5_BUILD.json> refuses it). A FINAL build refuses a members.json marked do_not_submit. A build
from the simulation root (~/k9_test/sim) is SIMULATION + do_not_submit. Interim / simulation builds get a marker file
DO_NOT_PUSH_SIMULATION.txt / DO_NOT_SUBMIT_INTERIM.txt in the build dir (never inside upload/).
"""
'''
s = HDR + s[s.index('import argparse'):]
rep("MAIN, SOLO, KF = 'rsna-sub-k8-fleet', 'rsna-sub-k8-solo', 'rsna-sub-k8-kf'",
    "MAIN, SOLO, KF = 'rsna-sub-k9-fleet', 'rsna-sub-k9-solo', 'rsna-sub-k9-kf'")
rep("""SETS = {(1043, 224): 'A', (1044, 288): 'C'}                      # (seed, ttr_size) -> set
SEEDS = (1043, 1044)
TTR_SHA = 'f141aeed136dfc01'
TRAIN_LABELS = 'labels_mix50_logit.csv'                            # settled: mix50 labels only
""", """MIX50, FRAC75 = 'labels_mix50_logit.csv', 'labels_frac75_logit.csv'
SETS = {(1044, 288, MIX50): 'C', (1043, 224, FRAC75): 'D'}         # (seed, ttr_size, train_labels) -> set
KEY_TAG = {'C': 't288', 'D': 'f75'}                                # K9-KEY: shipped key = f{F}_s{S}<tag>
REFUSED = {(1043, 224, MIX50): 'set A (the K5 mix50 s1043 members) is REPLACED by set D in K9',
           (1044, 224, MIX50): 'set B (s1044 TTR224) is DROPPED (NOVEL_PLAN 10-07 sec 2 item 2)'}
SEEDS = (1043, 1044)
TTR_SHA = 'f141aeed136dfc01'
""")
rep('''def member_key(r):
    """K7-KEY: the shipped key. TTR224 members keep f{F}_s{S} (byte-identical names to K5); others get a t{N} suffix."""
    k = f'f{r["fold"]}_s{r["seed"]}'
    return k if int(r['ttr_size']) == 224 else f'{k}t{int(r["ttr_size"])}'
''', '''def set_of(r):
    return SETS.get((int(r['seed']), int(r['ttr_size']), r.get('train_labels')))


def member_key(r):
    """K9-KEY: the shipped key f{F}_s{S}<set tag>: set C t288 (== the K7/K8 rule), set D f75 (labels tag). No K9 key equals
    a plain f{F}_s{S} key, so a frac75 member can never collide with a set A (K5) member of the same fold + seed."""
    return f'f{r["fold"]}_s{r["seed"]}{KEY_TAG[set_of(r)]}'
''')
rep("""    mem = sorted(m['members'], key=lambda r: (int(r['ttr_size']), r['seed'], r['fold']))     # A, then C
    assert mem, 'members.json has no member'
    ids = [(r['seed'], r['fold'], int(r['ttr_size'])) for r in mem]
    assert len(set(ids)) == len(ids), f'duplicate seed/fold/ttr_size members {ids} (one pick per run)'
""", """    mem = sorted(m['members'], key=lambda r: (int(r['ttr_size']), r['seed'], r['fold']))     # D (224), then C (288)
    assert mem, 'members.json has no member'
    ids = [(r['seed'], r['fold'], int(r['ttr_size']), r.get('train_labels')) for r in mem]
    assert len(set(ids)) == len(ids), f'duplicate seed/fold/ttr_size/train_labels members {ids} (one pick per run)'
""")
rep("""        st = (r['seed'], int(r['ttr_size']))
        assert st != (1044, 224), f'{r["file"]}: set B (s1044 TTR224) is DROPPED from K8 (NOVEL_PLAN 10-07 sec 2 item 2)'
        assert st in SETS, f'{r["file"]}: (seed, ttr_size) {st} is not a K8 set (A = (1043, 224), C = (1044, 288))'
        assert r.get('train_labels') == TRAIN_LABELS, f'{r["file"]}: train_labels {r.get("train_labels")} (K8 = mix50 only)'
""", """        st = (int(r['seed']), int(r['ttr_size']), r.get('train_labels'))
        assert st not in REFUSED, f'{r["file"]}: {REFUSED.get(st)}'
        assert st in SETS, (f'{r["file"]}: (seed, ttr_size, train_labels) {st} is not a K9 set '
                            f'(C = (1044, 288, {MIX50}), D = (1043, 224, {FRAC75}))')
        assert r.get('set') in (None, SETS[st]), f'{r["file"]}: members.json says set {r.get("set")}, its identity is set {SETS[st]}'
""")
rep("""(K8 ships RAW best checkpoints; picker --tag raw)""", """(K9 ships RAW best checkpoints; picker --tag raw)""")
rep("""        assert os.path.isfile(p), f'missing {p} (run k8_collect.sh)'""",
    """        assert os.path.isfile(p), f'missing {p} (run k9_collect.sh)'""")
rep("""        r['set'] = SETS[st]
    keys = [r['key'] for r in mem]
    assert len(set(keys)) == len(keys), f'duplicate shipped keys {keys}'
    nA = sorted(r['fold'] for r in mem if r['set'] == 'A')
    assert nA == [0, 1, 2, 3, 4], f'set A folds {nA}: K8 needs the 5 K5 members'
    on_disk = sorted(os.path.basename(p) for sd in SEEDS for p in glob.glob(f'{ckdir}/f*_s{sd}_*_best.pt'))
    assert on_disk == sorted(files), \\
        f'k8/ckpt files {on_disk} != members.json {files} (stale file? re-run k8_collect.sh)'
""", """        r['set'] = SETS[st]
    keys = [r['key'] for r in mem]
    assert len(set(keys)) == len(keys), f'duplicate shipped keys {keys}'
    assert all(k.split('_')[1] not in ('s1043', 's1044') for k in keys), f'K9-KEY: an untagged key in {keys}'
    on_disk = sorted(os.path.basename(p) for sd in SEEDS for p in glob.glob(f'{ckdir}/f*_s{sd}_*_best.pt'))
    assert on_disk == sorted(files), \\
        f'k9/ckpt files {on_disk} != members.json {files} (stale file? re-run k9_collect.sh)'
""")
rep("""help='default k8/build (A 5/5 + C 5/5) or k8/build_i<N> (interim)')""",
    """help='default k9/build (C 5/5 + D 5/5) or k9/build_i<N> (interim)')""")
rep("""    nA = sum(r['set'] == 'A' for r in mem); nC = sum(r['set'] == 'C' for r in mem)
    interim = not (nA == 5 and nC == 5)
    sim = any('/k8_test/sim/' in str(r.get('run', '')) for r in mem)   # collected from the SIMULATION root (test only)
    if interim:
        assert mj.get('do_not_submit') and mj.get('interim'), \\
            f'set C {nC}/5: an INTERIM build needs a members.json marked do_not_submit (k8_collect.sh K8_INTERIM=1)'
    else:
        assert not mj.get('do_not_submit'), 'members.json is marked do_not_submit but holds A 5/5 + C 5/5: re-collect'
""", """    nC = sum(r['set'] == 'C' for r in mem); nD = sum(r['set'] == 'D' for r in mem)
    fC = sorted(r['fold'] for r in mem if r['set'] == 'C'); fD = sorted(r['fold'] for r in mem if r['set'] == 'D')
    interim = not (fC == [0, 1, 2, 3, 4] and fD == [0, 1, 2, 3, 4])
    sim = any('/k9_test/sim/' in str(r.get('run', '')) for r in mem)   # collected from the SIMULATION root (test only)
    if interim:
        assert mj.get('do_not_submit') and mj.get('interim'), \\
            f'set C {nC}/5, set D {nD}/5: an INTERIM build needs a members.json marked do_not_submit (k9_collect.sh K9_INTERIM=1)'
    else:
        assert not mj.get('do_not_submit'), 'members.json is marked do_not_submit but holds C 5/5 + D 5/5: re-collect'
""")
rep("""'k8/code/fleet_worker.py != the K5 worker (K8 must ship the worker that scored .946, byte for byte)'""",
    """'k9/code/fleet_worker.py != the K5 worker (K9 must ship the worker that scored .946, byte for byte)'""")
rep("""'k8 pack tool != K7 pack tool'""", """'k9 pack tool != K7 pack tool'""")
rep("""'k8/code/fleet_worker.py is not the K5 TTA worker'""", """'k9/code/fleet_worker.py is not the K5 TTA worker'""")
rep("""'shipped worker != k8/code'""", """'shipped worker != k9/code'""")
rep("""        assert (int(r['fold']) == x['fold'] and int(r['seed']) == x['seed'] and int(r['ttr_size']) == int(x['ttr_size'])
                and r['epoch_1based'] == x['epoch'] and abs(float(r['cv']) - x['cv']) < 1e-12), (r, x)""",
    """        assert (int(r['fold']) == x['fold'] and int(r['seed']) == x['seed'] and int(r['ttr_size']) == int(x['ttr_size'])
                and r['train_labels'] == x['train_labels'] and r['key'] == x['key']
                and r['epoch_1based'] == x['epoch'] and abs(float(r['cv']) - x['cv']) < 1e-12), (r, x)""")
rep("""'k8 cell minus the K5a block != the K3 cell (only the K5a block may differ)'
    assert sha256(f'{HERE}/cells/k2_fleet_cell.py') == sha256(f'{SP}/k5/cells/k2_fleet_cell.py'), 'k8 cell != the K5 cell'""",
    """'k9 cell minus the K5a block != the K3 cell (only the K5a block may differ)'
    assert sha256(f'{HERE}/cells/k2_fleet_cell.py') == sha256(f'{SP}/k5/cells/k2_fleet_cell.py'), 'k9 cell != the K5 cell'
    assert sha256(f'{HERE}/cells/k2_fleet_cell_kf.py') == sha256(f'{SP}/k8/cells/k2_fleet_cell_kf.py'), 'k9 KF cell != K8 KF cell'""")
rep("""k8=True, interim=interim, simulated=sim,""", """k8=True, k9=True, interim=interim, simulated=sim,""")
rep("""sets=dict(A=nA, C=nC), folds=""", """sets=dict(C=nC, D=nD), folds=""")
rep("""                picks=[dict({k: r[k] for k in ('set', 'seed', 'fold', 'ttr_size', 'tag', 'epoch', 'cv', 'other_tag_cv', 'file',
                                               'staged', 'sha256', 'source')}, suffix=r.get('suffix'), oof=r.get('oof'))
                       for r in mem],""",
    """                picks=[dict({k: r[k] for k in ('set', 'seed', 'fold', 'ttr_size', 'train_labels', 'tag', 'epoch', 'epochs',
                                               'cv', 'other_tag_cv', 'file', 'key', 'staged', 'sha256', 'source')},
                            suffix=r.get('suffix'), oof=r.get('oof'), kaggle_receipt=r.get('kaggle_receipt'))
                       for r in mem],""")
rep("""set_c=mj.get('set_c'))
    json.dump(info, open(f'{a.out}/K5_BUILD.json', 'w'), indent=1)
    keys8 = [f'{r["set"]}:{r["key"]}' for r in mem]
    tag = (f'INTERIM ({len(mem)} members: A {nA}/5, C {nC}/5: {keys8}) -- TEST ONLY, do_not_submit, NEVER SUBMIT'
           if interim else f'FINAL ({len(mem)} members: A 5/5, C 5/5: {keys8})')
    if sim:
        tag = 'SIMULATION (members from ~/k8_test/sim: TEST ONLY, NEVER VERSION / PUSH / SUBMIT) ' + tag
    print(f'\\nK8 BUILD OK {tag}""",
    """set_c=mj.get('set_c'),
                set_d=mj.get('set_d'))
    json.dump(info, open(f'{a.out}/K5_BUILD.json', 'w'), indent=1)
    for fn, cond, txt in (('DO_NOT_PUSH_SIMULATION.txt', sim, 'SIMULATION BUILD (members from the ~/k9_test/sim root).'),
                          ('DO_NOT_SUBMIT_INTERIM.txt', interim, f'INTERIM BUILD (set C {nC}/5, set D {nD}/5).')):
        if cond:                                                         # marker in the build dir, never inside upload/
            open(f'{a.out}/{fn}', 'w').write(txt + ' TEST ONLY. NEVER version the upload folder, NEVER push these '
                                             'kernels, NEVER submit. K5_BUILD.json: do_not_submit true.\\n')
    keys9 = [f'{r["set"]}:{r["key"]}' for r in mem]
    tag = (f'INTERIM ({len(mem)} members: C {nC}/5, D {nD}/5: {keys9}) -- TEST ONLY, do_not_submit, NEVER SUBMIT'
           if interim else f'FINAL ({len(mem)} members: C 5/5, D 5/5: {keys9})')
    if sim:
        tag = 'SIMULATION (members from ~/k9_test/sim: TEST ONLY, NEVER VERSION / PUSH / SUBMIT) ' + tag
    print(f'\\nK9 BUILD OK {tag}""")
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched', p)
