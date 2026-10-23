# submit_combo_at.py (2026-10-16, Claude): waits until 10-16 07:10 IST (01:40 UTC), then submits the R6 + KC1 combo ONCE
# (PREREG_10-15 Amendment 10; Eesh approved the parallel 2x2 submit). Preconditions at submit time (any failure = no submit, logged):
#   - session_2026-10-16/combo/gate_v224r6s6fkcj_v1.txt ends with 'COMBO_CHECK PASS 20/20 r6kc'
#   - an R6 submission ('V224R6-FLIP' in its description) AND a KC1 submission ('V224S6FKC-J' in its description) exist,
#     and the KC1 one is >= 45 min old
#   - no description already starts with 'COMBO-R6KC1' (idempotent)
# Usage: python submit_combo_at.py KERNEL_VERSION
import time, datetime, sys, os, ctypes
try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # no idle sleep while waiting
except Exception: pass
from kaggle.api.kaggle_api_extended import KaggleApi
VER = int(sys.argv[1])
T = datetime.datetime(2026, 10, 16, 1, 40, 0, tzinfo=datetime.timezone.utc)
HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(HERE, 'gate_v224r6s6fkcj_v1.txt')
LOG = r'C:\Users\eeshs\Downloads\rsna-knee\session_2026-10-14\submits.log'
COMP = 'rsna-knee-abnormality-detection'
def log(m):
    line = f'[{datetime.datetime.now().strftime("%m-%d %H:%M:%S")} IST] {m}'
    print(line, flush=True); open(LOG, 'a', encoding='utf-8').write(line + '\n')
while datetime.datetime.now(datetime.timezone.utc) < T:
    time.sleep(60)
last = [l.strip() for l in open(GATE, encoding='utf-8', errors='replace') if l.strip()][-1] if os.path.isfile(GATE) else ''
if last != 'COMBO_CHECK PASS 20/20 r6kc':
    log(f'COMBO NOT submitted: gate last line is {last!r}'); sys.exit(1)
api = KaggleApi(); api.authenticate()
desc = lambda s: str(getattr(s, 'description', '') or '')
for attempt in range(10):
    try:
        subs = api.competition_submissions(COMP)
        if any(desc(s).startswith('COMBO-R6KC1') for s in subs):
            log('COMBO already submitted: skip'); sys.exit(0)
        r6 = [s for s in subs if desc(s).startswith('V224R6-FLIP')]
        kc = [s for s in subs if desc(s).startswith('V224S6FKC-J')]
        if not (r6 and kc):
            log(f'COMBO waiting: R6 {bool(r6)} KC1 {bool(kc)} (attempt {attempt})'); time.sleep(300); continue
        d = getattr(kc[0], 'date', None)
        if d is not None:
            d = d if d.tzinfo else d.replace(tzinfo=datetime.timezone.utc)
            age = (datetime.datetime.now(datetime.timezone.utc) - d).total_seconds() / 60
            if age < 45:
                log(f'COMBO waiting: KC1 is only {age:.0f} min old'); time.sleep(max(60, (45 - age) * 60 + 30)); continue
        r = api.competition_submit_code(file_name='submission.csv', competition=COMP,
            message='COMBO-R6KC1: S6F-J best + R6 swap + .2 rank KC1 leg (2x2 with R6 and KC1); COMBO_CHECK 20/20; PREREG_10-15 A10',
            kernel='eeshsaxena/rsna-sub-v224r6s6fkcj', kernel_version=VER)
        log(f'COMBO SUBMITTED (v{VER}): {r}'); sys.exit(0)
    except Exception as e:
        resp = getattr(e, 'response', None)
        log(f'COMBO submit attempt {attempt} error {e} {resp.text[:300] if resp is not None else ""}')
        time.sleep(120)
log('COMBO submit FAILED / gave up'); sys.exit(1)
