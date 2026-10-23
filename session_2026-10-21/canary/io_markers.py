# io_markers.py LOG...: read-only; for each commit log prints the gap (s) before four dataset-read markers
# (weights sha256, pydicom wheel processing, ASSETS_READY, GPU arm load) and total wall time.
import json, sys, re
MK = [('weights_sha', r'\[weights\] sha256'), ('wheel_proc', r'Processing /tmp/probe/wheels/pydicom'), ('assets_ready', r'ASSETS_READY'), ('arm_load', r'loaded 1 arm')]
print('%-34s %7s %11s %10s %12s %8s' % ('log', 'total', 'weights_sha', 'wheel_proc', 'assets_ready', 'arm_load'))
for f in sys.argv[1:]:
    j = json.load(open(f, encoding='utf-8'))
    ev = [(e['time'], e.get('data') or '') for e in j if isinstance(e, dict) and e.get('time') is not None]
    out = []
    for name, rx in MK:
        g = [round(ev[i][0] - ev[i-1][0], 1) for i in range(1, len(ev)) if re.search(rx, ev[i][1])]
        out.append(max(g) if g else None)
    print('%-34s %7.1f %11s %10s %12s %8s' % (re.sub(r'rsna-sub-v224r6|\.log', '', f.split('/')[-1]), ev[-1][0], *out))
