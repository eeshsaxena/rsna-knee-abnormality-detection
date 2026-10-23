# commit_gaps.py LOG [LOG...]: read-only; prints the largest time gaps between consecutive log lines of a Kaggle commit log,
# with the line that ended each gap, plus total wall time. Used to see where 10-21 commit runs spent extra time.
import json, sys
for f in sys.argv[1:]:
    j = json.load(open(f, encoding='utf-8'))
    ev = [(e.get('time'), (e.get('data') or '').strip().replace('\n', ' ')[:110]) for e in j if isinstance(e, dict) and e.get('time') is not None]
    gaps = sorted(((ev[i][0] - ev[i-1][0], ev[i-1][0], ev[i][1]) for i in range(1, len(ev))), reverse=True)[:6]
    print('\n##', f.split('/')[-1], 'total', round(ev[-1][0], 1), 's, lines', len(ev))
    for g, t0, txt in gaps: print('  gap %6.1f s at t=%6.1f -> %s' % (g, t0, txt))
