p = 'code/pack_fleet_ds.py'
s = open(p, encoding='utf-8', newline='').read().replace('\r\n', '\n')
BS = chr(92)


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:70], s.count(a))
    s = s.replace(a, b)


rep("""#         identical). The manifest records every source path and re-key under 'sources' (the worker reads only
#         'checkpoints').""", """#         identical). The manifest records every member's provenance and re-key under 'sources' (the worker reads
#         only 'checkpoints'): original file name, run folder name, sha256[:16] of the bytes, model_mod, model_sha.
#         FIX-5: no absolute paths (they would ship local user names and directory layout inside the Kaggle dataset).""")
rep("""def main(argv=None):""", """def _sha16(path):                                     # FIX-5 provenance: content hash, no local path
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()[:16]


def main(argv=None):""")
old = ("""               'sources': {r['staged']: dict(src=os.path.abspath(r['src']).replace('""" + BS + BS + """', '/'), original=r['base'],
                                             model_mod=r['mod'], model_sha=r['a'].get('model_sha')) for r in rows}},""")
rep(old, """               'sources': {r['staged']: dict(original=r['base'],                                  # FIX-5
                                             run_dir=os.path.basename(os.path.dirname(os.path.abspath(r['src']))),
                                             sha256=_sha16(r['src']), model_mod=r['mod'],
                                             model_sha=r['a'].get('model_sha')) for r in rows}},""")
open(p, 'w', encoding='utf-8', newline='').write(s.replace('\n', '\r\n'))
print('ok')
