import sys
p = sys.argv[1]; s = open(p, encoding='utf-8').read()
def rep(o, n):
    global s
    assert s.count(o) == 1, o[:60]; s = s.replace(o, n)
rep('"""k5_ids.py (box, CPU, read-only; == k4_ids.py with ~/k6_test paths):', '"""k5_ids.py, K7 COPY (~/k7_test paths; K7: studies are picked PER FOLD, not per member: every member of a fold (K7: A, B and C\nall have a fold-0 member) gets the SAME A0 studies, chosen from the studies present in ALL of that fold\'s OOFs, so each\nfold-0 member gets its own per-member OOF parity on the same 2 studies and the CPU A0 stays 2 studies long).\n(K6 text) k5_ids.py (box, CPU, read-only; == k4_ids.py with ~/k6_test paths):')
s = s.replace('k6_test', 'k7_test')
rep('''    for r in sorted(m['members'], key=lambda r: r['fold']):
        oof = pd.read_csv(r['oof'], dtype={'StudyInstanceUID': str})['StudyInstanceUID'].tolist()
        ok = [s for s in oof if s in trs and os.path.isdir(f'{COMP}/train_series/{s}')]''',
'''    byfold = {}
    for r in m['members']:
        byfold.setdefault(r['fold'], []).append(r)
    for F in sorted(byfold):                                            # K7: one pick per fold, shared by its members
        r = byfold[F][0]
        oofs = [set(pd.read_csv(x['oof'], dtype={'StudyInstanceUID': str})['StudyInstanceUID']) for x in byfold[F]]
        oof = pd.read_csv(r['oof'], dtype={'StudyInstanceUID': str})['StudyInstanceUID'].tolist()
        ok = [s for s in oof if all(s in o for o in oofs) and s in trs and os.path.isdir(f'{COMP}/train_series/{s}')]''')
rep('''        print(f'fold {r["fold"]}: {len(ok)} val studies with DICOMs, picked {[s[-12:] for s in pick]}')''',
    '''        print(f'fold {r["fold"]} ({len(byfold[F])} member(s)): {len(ok)} val studies with DICOMs in every member OOF, '
              f'picked {[s[-12:] for s in pick]}')''')
open(p, 'w', encoding='utf-8', newline='\n').write(s); print('patched', p)
