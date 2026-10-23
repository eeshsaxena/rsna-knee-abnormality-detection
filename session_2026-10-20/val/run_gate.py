#!/usr/bin/env python
# run_gate.py (10-20, Claude): run the exact commit gate of an A22 slate entry from val/slate_a22.json (no shell quoting of the table).
# Usage: py -3.12 -I session_2026-10-20/val/run_gate.py <key> <OUT_DIR>      keys: pw35 pw40 pw45 r6pw30 r6e20
# Exit code = the gate's (0 only on PROBE_CHECK PASS). Read-only on OUT_DIR.
import json, os, sys, subprocess
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
S = json.load(open(os.path.join(R, 'session_2026-10-20', 'val', 'slate_a22.json'), encoding='utf-8'))['entries']
key, out = sys.argv[1], sys.argv[2]; e = S[key]
argv = [sys.executable, '-I', os.path.join(R, e['gate_script']), out] + e['gate_args']
print('RUN_GATE', key, e['tag'], '|', ' '.join(argv[2:3] + ['<OUT>'] + e['gate_args']), flush=True)
sys.exit(subprocess.run(argv, cwd=R).returncode)
