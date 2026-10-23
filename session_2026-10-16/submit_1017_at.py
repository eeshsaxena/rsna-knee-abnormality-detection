# submit_1017_at.py (2026-10-16, Claude): PREREG_10-15 Amendment 13. Records the A11 (HW) / A12 (S6W) reads by polling (submission
# status + anonymous API best-id) every 10 min, then at 10-17 05:31 IST submits the entry the A13 table selects (or nothing).
import time, datetime, json, os, sys, ctypes, requests
try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
except Exception: pass
from kaggle.api.kaggle_api_extended import KaggleApi
COMP = 'rsna-knee-abnormality-detection'; TEAM = 16799346
R6, HW, S6W = '57248084', '57262115', '57264835'
T = datetime.datetime(2026, 10, 17, 0, 1, 0, tzinfo=datetime.timezone.utc)
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, 'reads_tonight.json')
LOG = r'C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-14\submits.log'
GATES = {'hw3': (os.path.join(HERE, 'hw3', 'gate_v224r6hws6f3j_v1.txt'), 'HW3_CHECK PASS 17/17', 'eeshsaxena/rsna-sub-v224r6hws6f3j',
                 'V224R6HWS6F3-J: heliosli weights + S6fall .3 (A13 slot 1: HW positive, S6W positive/UNREAD); HW3_CHECK 17/17'),
         's6w4': (os.path.join(HERE, 's6w4', 'gate_v224r6s6f4j_v1.txt'), 'S6W4_CHECK PASS 15/15', 'eeshsaxena/rsna-sub-v224r6s6f4j',
                  'V224R6S6F4-J: R6 + S6fall .4 (A13 slot 1: HW negative, S6W positive); S6W4_CHECK 15/15')}
def log(m):
    line = f'[{datetime.datetime.now().strftime("%m-%d %H:%M:%S")} IST] {m}'
    print(line, flush=True); open(LOG, 'a', encoding='utf-8').write(line + '\n')
def best_id():
    s = requests.Session(); s.get('https://www.kaggle.com/competitions', timeout=60)
    h = {'x-xsrf-token': s.cookies.get('XSRF-TOKEN', '')}
    r = s.post('https://www.kaggle.com/api/i/competitions.LeaderboardService/GetLeaderboard', json={'competitionId': 154281}, headers=h, timeout=90)
    me = [x for x in r.json()['publicLeaderboard'] if x['teamId'] == TEAM][0]
    return str(me['submissionId']), me.get('rank'), me.get('displayScore')
st = json.load(open(STATE)) if os.path.isfile(STATE) else {}
api = KaggleApi(); api.authenticate()
while True:
    try:
        subs = {str(getattr(s, 'ref', '')): s for s in api.competition_submissions(COMP)}
        done = {k: 'COMPLETE' in str(getattr(subs.get(k), 'status', '')) for k in (HW, S6W)}
        new = [k for k in (HW, S6W) if done[k] and k not in st]
        if new:
            b, rank, sc = best_id()
            for k in new:
                if k == HW:
                    v = 'positive' if b == HW else ('UNREAD' if (S6W in new and b != HW) else 'negative')
                else:
                    v = 'positive' if b == S6W else ('UNREAD' if b == HW else 'negative')
                st[k] = {'read': v, 'best_id': b, 'rank': rank, 'score': str(getattr(subs[k], 'public_score', '')), 'at': datetime.datetime.now().isoformat()}
                log(f'A13 read {"HW" if k == HW else "S6W"} {k}: {v} (public {st[k]["score"]}, best-id {b}, rank {rank})')
            json.dump(st, open(STATE, 'w'), indent=1)
    except Exception as e:
        print('poll error', e, flush=True)
    if datetime.datetime.now(datetime.timezone.utc) >= T:
        break
    time.sleep(600)
hw, s6 = st.get(HW, {}).get('read'), st.get(S6W, {}).get('read')
pick = 'hw3' if (hw == 'positive' and s6 in ('positive', 'UNREAD')) else ('s6w4' if (hw == 'negative' and s6 == 'positive') else None)
log(f'A13 decision: HW {hw}, S6W {s6} -> {pick or "nothing automatic"}')
if not pick: sys.exit(0)
gate, want, kernel, msg = GATES[pick]
last = [l.strip() for l in open(gate, encoding='utf-8', errors='replace') if l.strip()][-1] if os.path.isfile(gate) else ''
if last != want:
    log(f'A13 {pick} NOT submitted: gate last line {last!r}'); sys.exit(1)
subs = api.competition_submissions(COMP)
if any(str(getattr(s, 'description', '') or '').startswith(msg.split(':')[0]) for s in subs):
    log(f'A13 {pick} already submitted: skip'); sys.exit(0)
for attempt in range(6):
    try:
        r = api.competition_submit_code(file_name='submission.csv', competition=COMP, message=msg + '; PREREG_10-15 A13', kernel=kernel, kernel_version=1)
        log(f'A13 {pick} SUBMITTED (v1): {r}'); sys.exit(0)
    except Exception as e:
        log(f'A13 {pick} submit attempt {attempt} error {e}'); time.sleep(120)
