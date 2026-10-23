# Lists every member of the 12 local prep tars (same bytes as the Kaggle prep outputs) and hashes 2 members per part.
import tarfile, hashlib, json, os, sys
SRC = sys.argv[1]; OUT = sys.argv[2]
parts = sorted(f for f in os.listdir(SRC) if f.startswith('fleet_store_s') and f.endswith('.tar'))
members, sample = {}, []
for p in parts:
    with tarfile.open(os.path.join(SRC, p), 'r:') as t:
        ms = [m for m in t.getmembers() if m.isfile()]
        members[p] = [m.name for m in ms]
        for m in (ms[0], ms[len(ms) // 2]):
            h = hashlib.sha256(t.extractfile(m).read()).hexdigest()
            sample.append(dict(part=p, name=m.name, size=m.size, sha256=h))
    print(p, len(ms), flush=True)
json.dump(dict(members=members, sample=sample), open(OUT, 'w'), indent=0)
print('total members', sum(len(v) for v in members.values()), 'unique', len({n for v in members.values() for n in v}))
