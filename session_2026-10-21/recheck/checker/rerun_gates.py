# rerun_gates.py (checker, 10-21): re-runs the two read-only commit gates on the committed outputs already on disk, with the
# exact registry argv for swide (slate_a22.json) and the documented R6 usage; writes stdout under checker/ only.
import json, subprocess, os, hashlib
R = 'C:/Users/eeshs/Downloads/rsna-knee'
HERE = os.path.dirname(os.path.abspath(__file__))
j = json.load(open(R + '/session_2026-10-20/val/slate_a22.json'))
hit = []
def find(o):
    if isinstance(o, dict):
        if 'gate_args' in o and 'lomm05mcl10j' in str(o.get('kernel_dir', '')): hit.append(o)
        for v in o.values(): find(v)
    elif isinstance(o, list):
        for v in o: find(v)
find(j)
e = hit[0]
runs = {
    'swide': ['py', '-3.12', '-I', os.path.join(R, e['gate_script']), R + '/session_2026-10-20/val/a22/out_v224r6c4pw25lomm05mcl10j'] + e['gate_args'],
    'r6': ['py', '-3.12', '-I', R + '/session_2026-10-15/r6entry/r6_check.py', R + '/session_2026-10-15/r6entry/out_v224r6flips6fj_v1'],
}
for k, argv in runs.items():
    p = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=HERE)
    open(os.path.join(HERE, 'gate_%s.txt' % k), 'w', encoding='utf-8').write(p.stdout)
    last = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ''
    print(k, 'rc', p.returncode, '| last:', last, '| sha', hashlib.sha256(p.stdout.encode('utf-8')).hexdigest()[:12], '| stderr', p.stderr.strip()[-200:])
