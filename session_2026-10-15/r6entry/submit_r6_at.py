# submit_r6_at.py (2026-10-15, Claude): waits until 10-16 05:36 IST (00:06 UTC), then submits the R6 entry ONCE
# (PREREG_10-15 Amendment 9 rule 1). Idempotent: skips if any submission description already contains 'V224R6'.
import time, datetime, sys, ctypes
# keep this laptop from IDLE-sleeping while the waiter runs (app-level request, released when the process exits)
try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
except Exception: pass
from kaggle.api.kaggle_api_extended import KaggleApi
T = datetime.datetime(2026, 10, 16, 0, 6, 0, tzinfo=datetime.timezone.utc)
LOG = r'C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-14\submits.log'
def log(m):
    line = f'[{datetime.datetime.now().strftime("%m-%d %H:%M:%S")} IST] {m}'
    print(line, flush=True); open(LOG, 'a', encoding='utf-8').write(line + '\n')
while datetime.datetime.now(datetime.timezone.utc) < T:
    time.sleep(60)
api = KaggleApi(); api.authenticate()
for attempt in range(6):
    try:
        subs = api.competition_submissions('rsna-knee-abnormality-detection')
        if any('V224R6' in str(getattr(s, 'description', '') or '') for s in subs):
            log('R6 already submitted (description V224R6 found): skip'); sys.exit(0)
        r = api.competition_submit_code(file_name='submission.csv', competition='rsna-knee-abnormality-detection',
            message='V224R6-FLIP-S6F-J: J twin of the best (S6F 57201874) with R7 bundle -> R6 fe055406 (one change); R6_CHECK 19/19; PREREG_10-15 A9 rule 1',
            kernel='eeshsaxena/rsna-sub-v224r6flips6fj', kernel_version=1)
        log(f'R6 SUBMITTED: {r}'); sys.exit(0)
    except Exception as e:
        resp = getattr(e, 'response', None)
        log(f'R6 submit attempt {attempt} error {e} {resp.text[:300] if resp is not None else ""}')
        time.sleep(120)
log('R6 submit FAILED after retries'); sys.exit(1)
