# warm_cache.py (rental_kc, BOX, CPU nice 19 + ionice idle, cores 30-39): pre-hash ~/kc_store/store/*.npz into the pack.sh
# manifest cache ~/rkc/cache/store_sha256.tsv (key = name, size, mtime_ns; same format as pack.sh), so the real pack only
# hashes the code / tables. Writes nothing else. No em dashes.
import os, sys, hashlib
from multiprocessing import Pool
ST = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/kc_store/store'); C = os.path.expanduser('~/rkc/cache/store_sha256.tsv')
def init():
    os.nice(19)
    try: os.sched_setaffinity(0, set(range(30, 40)))
    except Exception: pass
def h(p):
    m = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 22), b''): m.update(blk)
    return m.hexdigest()
os.makedirs(os.path.dirname(C), exist_ok=True)
cache = {}
if os.path.exists(C):
    for l in open(C):
        n, sz, mt, hx = l.rstrip('\n').split('\t'); cache[(n, sz, mt)] = hx
todo = []
for f in sorted(os.listdir(ST)):
    if f.endswith('.npz'):
        st = os.stat(f'{ST}/{f}'); k = (f, str(st.st_size), str(st.st_mtime_ns))
        if k not in cache: todo.append((k, f'{ST}/{f}'))
with Pool(4, initializer=init) as pool:
    for k, hx in zip([k for k, _ in todo], pool.map(h, [p for _, p in todo], chunksize=16)): cache[k] = hx
with open(C + '.tmp', 'w') as fh:
    for (n, sz, mt), hx in sorted(cache.items()): fh.write('\t'.join((n, sz, mt, hx)) + '\n')
os.replace(C + '.tmp', C)
print(f'warm_cache: {len(todo)} hashed now, {len(cache)} cached')
