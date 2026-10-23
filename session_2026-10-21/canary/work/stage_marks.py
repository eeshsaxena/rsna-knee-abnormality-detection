# stage_marks.py LOG...: read-only; prints the time of the first log line matching each stage marker (commit logs).
import json, sys, re
MK = [('dec', r'DEC_|J_DEC|decoder'), ('raptor_start', r'[Rr]aptor'), ('raptor_csv', r'raptor_sub|RAPTOR_DONE|Stage 1 done|raptor.*done'),
      ('r7_start', r'ASSETS_READY'), ('r6_done', r'WORKER_DONE|ours_sub'), ('r224', r'224'), ('stage3', r'BLEND|blend'),
      ('s6f_setup', r'S6F_SETUP'), ('s6f_start', r'S6F_START'), ('s6f_done', r'S6F_STAGE_DONE'), ('end', r'S6F_DONE')]
for f in sys.argv[1:]:
    j = json.load(open(f, encoding='utf-8'))
    ev = [(e['time'], e.get('data') or '') for e in j if isinstance(e, dict) and e.get('time') is not None]
    out = []
    for name, rx in MK:
        t = next((round(t, 0) for t, d in ev if re.search(rx, d)), None)
        out.append('%s=%s' % (name, t))
    print(f.split('/')[-1][-34:], ' '.join(out))
