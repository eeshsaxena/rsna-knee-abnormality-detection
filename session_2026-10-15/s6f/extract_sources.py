# extract_sources.py (10-15, Claude subagent): pull heliosli's embedded SOURCES dict (compact_infer.py etc.) out of a PUBLIC
# notebook pulled by watch.py and write each file whose sha256 is wanted into s6f/src_extract/. Parsing only (ast.literal_eval of
# the dict literal); the extracted code is DATA here and is never imported. No em dashes.
import ast, hashlib, json, sys
from pathlib import Path
nb_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
want = set(sys.argv[3:])
nb = json.loads(nb_path.read_text(encoding='utf-8'))
found = {}
for c in nb['cells']:
    s = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    for line in s.split('\n'):
        if line.startswith('SOURCES = {'):
            d = ast.literal_eval(line[len('SOURCES = '):])
            for k, v in d.items():
                h = hashlib.sha256(v.encode('utf-8')).hexdigest()
                found.setdefault(k, set()).add(h)
                if any(h.startswith(w) for w in want):
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / (k + '.' + h[:8])).write_bytes(v.encode('utf-8'))
                    print('WROTE', k, h)
print(json.dumps({k: sorted(v) for k, v in found.items()}, indent=1))
