# submit_kc1_at.py (2026-10-15, Claude): waits until 10-16 06:22 IST (00:52 UTC), then submits the KC1 entry ONCE
# (PREREG_10-15 Amendment 6b: J form rsna-sub-v224s6fkcj, slot >= 45 min after the R6 entry).
# Preconditions checked at submit time (any failure = no submit, logged):
#   - kcstage/gate_v224s6fkcj_v1.txt ends with 'V224S6FKC_CHECK PASS n/n mode KC tag v224s6fkcj' (n >= 29; the real J build has 30)
#   - an R6 submission ('V224R6' in its description) exists and is >= 45 min old
#   - no submission description already contains 'V224S6FKC' (idempotent)
# Usage: python submit_kc1_at.py KERNEL_VERSION
import time, datetime, sys, os, ctypes
try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # no idle sleep while waiting
except Exception: pass
from kaggle.api.kaggle_api_extended import KaggleApi
VER = int(sys.argv[1])
T = datetime.datetime(2026, 10, 16, 0, 52, 0, tzinfo=datetime.timezone.utc)
HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(HERE, 'gate_v224s6fkcj_v1.txt')
LOG = r'C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-14\submits.log'
COMP = 'rsna-knee-abnormality-detection'
def log(m):
    line = f'[{datetime.datetime.now().strftime("%m-%d %H:%M:%S")} IST] {m}'
    print(line, flush=True); open(LOG, 'a', encoding='utf-8').write(line + '\n')
while datetime.datetime.now(datetime.timezone.utc) < T:
    time.sleep(60)
last = [l.strip() for l in open(GATE, encoding='utf-8', errors='replace') if l.strip()][-1] if os.path.isfile(GATE) else ''
import re
m = re.match(r'^V224S6FKC_CHECK PASS (\d+)/(\d+) mode KC tag v224s6fkcj$', last)
if not (m and m.group(1) == m.group(2) and int(m.group(1)) >= 29):   # full pass, KC mode (real J build = 30 checks incl. C9)
    log(f'KC1 NOT submitted: gate last line is {last!r}'); sys.exit(1)
api = KaggleApi(); api.authenticate()
for attempt in range(8):
    try:
        subs = api.competition_submissions(COMP)
        if any('V224S6FKC' in str(getattr(s, 'description', '') or '') for s in subs):
            log('KC1 already submitted (description V224S6FKC found): skip'); sys.exit(0)
        r6 = [s for s in subs if 'V224R6' in str(getattr(s, 'description', '') or '')]
        if not r6:
            log(f'KC1 waiting: no R6 submission yet (attempt {attempt})'); time.sleep(300); continue
        d = getattr(r6[0], 'date', None)
        if d is not None:
            d = d if d.tzinfo else d.replace(tzinfo=datetime.timezone.utc)
            age = (datetime.datetime.now(datetime.timezone.utc) - d).total_seconds() / 60
            if age < 45:
                log(f'KC1 waiting: R6 is only {age:.0f} min old'); time.sleep(max(60, (45 - age) * 60 + 30)); continue
        r = api.competition_submit_code(file_name='submission.csv', competition=COMP,
            message='V224S6FKC-J KC1: S6F best (57201874) J form + .2 rank KC1 leg (f0 s2001 + AD s2003 RAW ep12, policy B wall-only); one change; V224S6FKC_CHECK 30/30; PREREG_10-15 A6b',
            kernel='eeshsaxena/rsna-sub-v224s6fkcj', kernel_version=VER)
        log(f'KC1 SUBMITTED (v{VER}): {r}'); sys.exit(0)
    except Exception as e:
        resp = getattr(e, 'response', None)
        log(f'KC1 submit attempt {attempt} error {e} {resp.text[:300] if resp is not None else ""}')
        time.sleep(120)
log('KC1 submit FAILED after retries'); sys.exit(1)
