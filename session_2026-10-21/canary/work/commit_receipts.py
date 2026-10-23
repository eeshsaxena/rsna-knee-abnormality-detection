# commit_receipts.py (10-21 canary track). READ-ONLY: per real commit output (out_*_v1 / a22 out_*), S6F stage timings from
# s6f_receipt.json and the commit log's total wall time + mtime. Excludes sim / tests / mock / SYNTH / tamper / neg_.
import json, glob, os, re, time
ROOT = 'C:/Users/eeshs/Downloads/rsna-knee'
rows = []
for f in glob.glob(ROOT + '/session_2026-10-*/**/s6f_receipt.json', recursive=True):
    f = f.replace(chr(92), '/')
    if re.search(r'/(sim|tests|selftest|syn_sim|c12_sim)/|mock_|SYNTH|tamper|neg_', f):
        continue
    d = os.path.dirname(f)
    try:
        r = json.load(open(f))
    except Exception as e:
        continue
    s, ru = r.get('setup') or {}, r.get('run') or {}
    dr = ru.get('driver') or {}
    st = dr.get('stages') or {}
    B = dr.get('B') or {}
    mb = B.get('microbench') or {}
    logs = [p for p in glob.glob(d + '/*.log') if 'rsna' in os.path.basename(p)]
    tot = None
    if logs:
        try:
            j = json.load(open(logs[0], encoding='utf-8'))
            ts = [e.get('time') for e in j if isinstance(e, dict) and e.get('time') is not None]
            tot = round(ts[-1], 1) if ts else None
        except Exception:
            tot = None
    rows.append((time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(f))), d.split('rsna-knee/')[1], r.get('mode'),
                 s.get('age_h'), (st.get('inputs') or {}).get('seconds'), (st.get('B_probe') or {}).get('seconds'),
                 (st.get('B') or {}).get('seconds'), mb.get('decode_s_per_study_per_worker'), mb.get('gpu_s_per_study'), tot))
rows.sort()
print('%-11s %-52s %-16s %7s %7s %7s %6s %7s %6s %7s' % ('mtime', 'out', 'mode', 'setupH', 'inputs', 'Bprobe', 'B_s', 'dec/st', 'gpu/st', 'log_s'))
for x in rows:
    print('%-11s %-52s %-16s %7s %7s %7s %6s %7s %6s %7s' % x)
