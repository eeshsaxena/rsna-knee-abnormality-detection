# submit_hw3_once.py (2026-10-16, Claude): backup for the A13 waiter (run by a one-time Windows scheduled task, 10-17 05:40 IST or at
# the next boot). Submits rsna-sub-v224r6hws6f3j v1 ONCE, only if: the UTC day is 10-17 or later, the gate line is HW3_CHECK PASS 17/17,
# no description starts with 'V224R6HWS6F3-J' (the waiter's message prefix), and fewer than 5 submissions on the current UTC day.
import datetime, os, sys
from kaggle.api.kaggle_api_extended import KaggleApi
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = r'C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-14\submits.log'
def log(m):
    line = f'[{datetime.datetime.now().strftime("%m-%d %H:%M:%S")} IST] {m}'
    print(line, flush=True); open(LOG, 'a', encoding='utf-8').write(line + '\n')
now = datetime.datetime.now(datetime.timezone.utc)
if now < datetime.datetime(2026, 10, 17, 0, 1, tzinfo=datetime.timezone.utc):
    log('HW3 backup: before the 10-17 UTC reset, not submitting'); sys.exit(0)
gate = os.path.join(HERE, 'hw3', 'gate_v224r6hws6f3j_v1.txt')
last = [l.strip() for l in open(gate, encoding='utf-8', errors='replace') if l.strip()][-1]
if last != 'HW3_CHECK PASS 17/17':
    log(f'HW3 backup: gate line {last!r}: not submitting'); sys.exit(1)
api = KaggleApi(); api.authenticate()
subs = api.competition_submissions('rsna-knee-abnormality-detection')
if any(str(getattr(s, 'description', '') or '').startswith('V224R6HWS6F3-J') for s in subs):
    log('HW3 backup: already submitted (by the waiter): skip'); sys.exit(0)
day = now.strftime('%Y-%m-%d')
if sum(1 for s in subs if str(getattr(s, 'date', '')).startswith(day)) >= 5:
    log('HW3 backup: 5 submissions already today: skip'); sys.exit(0)
r = api.competition_submit_code(file_name='submission.csv', competition='rsna-knee-abnormality-detection',
    message='V224R6HWS6F3-J: heliosli weights + S6fall .3 (A13 slot 1, submitted by the backup scheduled task); HW3_CHECK 17/17; PREREG_10-15 A13',
    kernel='eeshsaxena/rsna-sub-v224r6hws6f3j', kernel_version=1)
log(f'HW3 backup SUBMITTED (v1): {r}')
