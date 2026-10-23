# submit_t.py (2026-10-18, Claude): PREREG_10-15 Amendment 17. RUN ONLY AFTER EESH'S EXPLICIT GO.
# Waits for the 10-18 UTC day (>= 05:30:30 IST), then submits each A17 probe whose gate ends 'PROBE_CHECK PASS 16/16', in the order
# pf50, lm40, bk50, fr65, c4 (robust singles first, the stacked bet last). Idempotent (description prefix PROBE-A17-<P>), <= 5 per UTC day.
# Stops without submitting if the API best-id is no longer HW 57262115 (then the probes must be rebuilt on the new best first).
import time, datetime, os, sys, ctypes, requests
try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
except Exception: pass
from kaggle.api.kaggle_api_extended import KaggleApi
COMP, TEAM, HW = 'rsna-knee-abnormality-detection', 16799346, '57262115'
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = r'C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-14\submits.log'
ORDER = [x for x in ['pf50', 'lm40', 'bk50', 'fr65', 'c4'] if len(sys.argv) < 2 or x in sys.argv[1].split(',')]   # argv[1] = comma list (default all 5)
T0 = datetime.datetime(2026, 10, 18, 0, 0, 30, tzinfo=datetime.timezone.utc)
STOP = datetime.datetime(2026, 10, 18, 9, 30, 0, tzinfo=datetime.timezone.utc)   # 15:00 IST: give up on any probe not gated by then
def log(m):
    line = f'[{datetime.datetime.now().strftime("%m-%d %H:%M:%S")} IST] {m}'
    print(line, flush=True); open(LOG, 'a', encoding='utf-8').write(line + '\n')
def best_id():
    s = requests.Session(); s.get('https://www.kaggle.com/competitions', timeout=60)
    h = {'x-xsrf-token': s.cookies.get('XSRF-TOKEN', '')}
    r = s.post('https://www.kaggle.com/api/i/competitions.LeaderboardService/GetLeaderboard', json={'competitionId': 154281}, headers=h, timeout=90)
    return str([x for x in r.json()['publicLeaderboard'] if x['teamId'] == TEAM][0]['submissionId'])
def gate_ok(p):
    f = os.path.join(HERE, 'gate_v224r6hw%sj_v1.txt' % p)
    if not os.path.isfile(f): return None
    lines = [l.strip() for l in open(f, encoding='utf-8', errors='replace') if l.strip()]
    return bool(lines) and lines[-1] == 'PROBE_CHECK PASS 16/16'
def version(p):
    t = open(os.path.join(HERE, 'push_v224r6hw%sj.log' % p), encoding='utf-8', errors='replace').read()
    import re; m = re.search(r'Kernel version (\d+) successfully pushed', t); return int(m.group(1)) if m else None
while datetime.datetime.now(datetime.timezone.utc) < T0:
    time.sleep(30)
api = KaggleApi(); api.authenticate()
desc = lambda s: str(getattr(s, 'description', '') or '')
todo = list(ORDER)
while todo and datetime.datetime.now(datetime.timezone.utc) < STOP:
    p = todo[0]; g = gate_ok(p)
    if g is None:
        time.sleep(120); continue                      # not gated yet: wait for it (keeps the submit order)
    todo.pop(0)
    if not g:
        log(f'A17 {p} NOT submitted: gate is not PROBE_CHECK PASS 16/16'); continue
    try:
        b = best_id()
        if b != HW:
            log(f'A17 STOP: API best-id is {b}, not HW {HW}; rebuild the probes on the new best before submitting'); sys.exit(2)
        subs = api.competition_submissions(COMP)
        if any(desc(s).startswith('PROBE-A17-' + p.upper()) for s in subs):
            log(f'A17 {p} already submitted: skip'); continue
        day = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')
        if sum(1 for s in subs if str(getattr(s, 'date', '')).startswith(day)) >= 5:
            log(f'A17 {p} NOT submitted: 5 submissions already today'); continue
        v = version(p)
        r = api.competition_submit_code(file_name='submission.csv', competition=COMP,
            message='PROBE-A17-%s: HW best (57262115) one-line per-finding table probe %s; PROBE_CHECK 16/16; PREREG_10-15 A17' % (p.upper(), p),
            kernel='eeshsaxena/rsna-sub-v224r6hw%sj' % p, kernel_version=v)
        log(f'A17 {p} SUBMITTED (v{v}): {r}')
        time.sleep(20)
    except Exception as e:
        resp = getattr(e, 'response', None)
        log(f'A17 {p} submit error {e} {resp.text[:300] if resp is not None else ""}'); todo.insert(0, p); time.sleep(120)
log('A17 submitter done; not submitted: %s' % (todo or 'none'))
