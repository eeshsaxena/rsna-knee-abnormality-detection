# submit_r7_once.py (2026-10-16, Claude): one-time scheduled task (10-17 05:45 IST or next logon) for PREREG_10-15 A14.
# Submits rsna-sub-v224r6hwr7j v1 ONCE, only if: UTC day >= 10-17, the gate line is R7_CHECK PASS 20/20, no description starts with
# 'V224R6HWR7-J', fewer than 5 submissions on the current UTC day, and the hw3 entry is already in (so slot order is hw3 then R7).
import datetime, os, sys, time
from kaggle.api.kaggle_api_extended import KaggleApi
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = r'C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-14\submits.log'
def log(m):
    line = f'[{datetime.datetime.now().strftime("%m-%d %H:%M:%S")} IST] {m}'
    print(line, flush=True); open(LOG, 'a', encoding='utf-8').write(line + '\n')
now = datetime.datetime.now(datetime.timezone.utc)
if now < datetime.datetime(2026, 10, 17, 0, 1, tzinfo=datetime.timezone.utc):
    log('R7 late: before the 10-17 UTC reset, not submitting'); sys.exit(0)
last = [l.strip() for l in open(os.path.join(HERE, 'r7late', 'gate_v224r6hwr7j_v1.txt'), encoding='utf-8', errors='replace') if l.strip()][-1]
if last != 'R7_CHECK PASS 20/20':
    log(f'R7 late: gate line {last!r}: not submitting'); sys.exit(1)
api = KaggleApi(); api.authenticate()
for attempt in range(4):
    subs = api.competition_submissions('rsna-knee-abnormality-detection')
    desc = lambda s: str(getattr(s, 'description', '') or '')
    if any(desc(s).startswith('V224R6HWR7-J') for s in subs):
        log('R7 late: already submitted: skip'); sys.exit(0)
    if any(desc(s).startswith('V224R6HWS6F3-J') for s in subs):
        break
    log('R7 late: hw3 not in yet, waiting 5 min'); time.sleep(300)
day = now.strftime('%Y-%m-%d')
if sum(1 for s in subs if str(getattr(s, 'date', '')).startswith(day)) >= 5:
    log('R7 late: 5 submissions already today: skip'); sys.exit(0)
r = api.competition_submit_code(file_name='submission.csv', competition='rsna-knee-abnormality-detection',
    message='V224R6HWR7-J: HW best (57262115) + R7 late leg (R7 no-flip averaged into ours after S6F if age <= 6.6 h, else HW unchanged); R7_CHECK 20/20; PREREG_10-15 A14',
    kernel='eeshsaxena/rsna-sub-v224r6hwr7j', kernel_version=1)
log(f'R7 late SUBMITTED (v1): {r}')
