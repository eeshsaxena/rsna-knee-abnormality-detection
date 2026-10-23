# upload_store_sharded.py: upload ~/fleet_store (4407 competition npz, nothing else) to the Modal volume rsna-e2
# under /fleet_store with N parallel client processes (one TCP stream each; a single `modal volume put` stream
# reached only ~5 MB/s from the box). Each process uploads a disjoint shard (sorted names [k::N]), so no two
# writers ever touch the same file. Refuses anything that is not a regular *.npz directly in ~/fleet_store.
# Usage (box): ~/modalenv/bin/python upload_store_sharded.py N
import os, sys, time, multiprocessing as mp

STORE = "/home/training/fleet_store"


def work(k, n, q):
    import modal
    vol = modal.Volume.from_name("rsna-e2")
    names = sorted(os.listdir(STORE))[k::n]
    t0, tot = time.time(), 0
    with vol.batch_upload(force=True) as b:
        for nm in names:
            p = os.path.join(STORE, nm)
            tot += os.path.getsize(p)
            b.put_file(p, "/fleet_store/" + nm)
    q.put((k, len(names), tot, time.time() - t0))


if __name__ == "__main__":
    n = int(sys.argv[1])
    rp = os.path.realpath(STORE)
    assert rp == STORE and "oai" not in rp.lower() and "union_store" not in rp, rp
    ents = os.listdir(STORE)
    assert len(ents) == 4407 and all(e.endswith(".npz") for e in ents), "store must be exactly 4407 npz"
    assert all(os.path.isfile(os.path.join(STORE, e)) and not os.path.islink(os.path.join(STORE, e)) for e in ents)
    print("[shard-upload %s] %d npz, %d shards" % (time.strftime("%H:%M:%S"), len(ents), n), flush=True)
    q = mp.Queue()
    ps = [mp.Process(target=work, args=(k, n, q)) for k in range(n)]
    t0 = time.time()
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    rcs = [p.exitcode for p in ps]
    done = [q.get() for _ in range(q.qsize())] if not q.empty() else []
    tot = sum(d[2] for d in done)
    dt = time.time() - t0
    print("[shard-upload %s] exit codes %s | %d shards reported, %d files, %.2f GB in %.0fs (%.1f MB/s)"
          % (time.strftime("%H:%M:%S"), rcs, len(done), sum(d[1] for d in done), tot / 1e9, dt, tot / 1e6 / dt), flush=True)
    sys.exit(0 if all(r == 0 for r in rcs) and len(done) == n else 1)
