# parse_check.py (10-20 r5, Claude): CPU check that the mainline native prepare + geometry bank accept the rebuilt DICOMs
# (first 3 gold58a + first 3 OAI ids of ids_r5.json). Prints status counts / plane / side / shape aggregates only.
import sys, os, json, io, contextlib, shutil, collections
sys.dont_write_bytecode = True
H = os.path.expanduser('~'); R = H + '/r5_oai'
sys.path[:0] = [H + '/r7port/code', R + '/r6src']
import probe_infer as PI
import training.geometry_bank as GB
native = PI.install_native(True)
ids = json.load(open(R + '/ids_r5.json'))
pick = [u for u in ids if not u.startswith('oai_')][:3] + [u for u in ids if u.startswith('oai_')][:3]
out = collections.Counter()
for u in pick:
    work = __import__('pathlib').Path(R + '/tmp_check/' + str(abs(hash(u)) % 10 ** 8))
    shutil.rmtree(work, ignore_errors=True)
    PI.OneStudySource.current = u
    with contextlib.redirect_stdout(io.StringIO()):
        m = native.prepare(R + '/data_r5', work, split='r5', study_ids=[u], source_directory=True)
        GB.build(root=R + '/data_r5', manifest=str(work / 'manifest.json'), output=str(work / 'geometry'), split='r5', source_directory=True)
    st = m['studies'][0]
    kind = 'oai' if u.startswith('oai_') else 'g58a'
    out[(kind, 'series', len(st['series']), 'errors', len(st['errors']))] += 1
    for s in st['series']:
        out[(kind, s['plane'], s['slot'], 'side', s['side'], tuple(s['native_shape']))] += 1
    geos = [json.load(open(p)) for p in (work / 'geometry').glob('*.json')]
    for g in geos:
        out[(kind, 'geo', g.get('laterality_status'), g.get('status', g.get('ok')))] += 1
    shutil.rmtree(work, ignore_errors=True)
for k, v in sorted(out.items(), key=str):
    print(v, k)
