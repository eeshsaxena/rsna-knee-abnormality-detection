# mech_harness.py (recheck, 10-21 ~23:30 IST). READ-ONLY harness: loads a23/tick_a23.py and final/ops_tick_a23.py as modules,
# BLOCKS every Kaggle write call, forces dry mode, points GO / PREREG / logs at temp files inside recheck/, simulates the clock.
# Kaggle calls made: competition_submissions, kernels_status, competition_get_submission_limits (all read-only).
import os, sys, types, datetime, importlib.util, io, contextlib
R = r'C:\Users\eeshs\Downloads\rsna-knee'
H = os.path.join(R, 'session_2026-10-21', 'recheck'); T = os.path.join(H, 'mech_tmp'); os.makedirs(T, exist_ok=True)
from kaggle.api.kaggle_api_extended import KaggleApi
def _blocked(*a, **k): raise RuntimeError('BLOCKED WRITE CALL in harness')
for name in ('competition_submit_code', 'competition_submit', 'competition_submit_cli', 'kernels_push', 'kernels_push_cli'):
    if hasattr(KaggleApi, name): setattr(KaggleApi, name, _blocked)

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def fake_dt(now):
    class FDT(datetime.datetime):
        @classmethod
        def now(cls, tz=None): return now
    return types.SimpleNamespace(datetime=FDT, timedelta=datetime.timedelta, date=datetime.date, timezone=datetime.timezone)

def cap(fn):
    b = io.StringIO()
    with contextlib.redirect_stdout(b): fn()
    return b.getvalue().strip().replace('\n', ' || ')

D = datetime.datetime
print('==== tick_a23.py (a23) ====')
A = load(os.path.join(R, 'session_2026-10-21', 'a23', 'tick_a23.py'), 'tick_a23_h')
A.DRY = True; A.LOG = os.path.join(T, 'a_log.txt'); A.LOCK = os.path.join(T, 'a.lock'); A.SUBLOG = os.path.join(T, 'a_sublog.txt')
PR_OK = os.path.join(T, 'prereg_ok.md'); open(PR_OK, 'w').write('... AMENDMENT 23 ...\n')
PR_NO = os.path.join(T, 'prereg_no.md'); open(PR_NO, 'w').write('... AMENDMENT 22 ...\n')
GOF = os.path.join(T, 'A23_GO.txt')
def runA(go, now, prereg=PR_OK):
    if go is None:
        if os.path.exists(GOF): os.remove(GOF)
    else: open(GOF, 'w').write(go + '\n')
    A.GO = GOF; A.PREREG = prereg; A.datetime = fake_dt(now)
    return cap(A.main)
cases = [
    ('GO absent, 05:31', None, D(2026,10,22,5,31)),
    ('GO c1 c2, AMENDMENT 23 absent, 05:31', 'GO c1 c2 Eesh 23:00', D(2026,10,22,5,31), PR_NO),
    ('GO c1 c2, 05:30:59', 'GO c1 c2 Eesh 23:00', D(2026,10,22,5,30,59)),
    ('GO c1 c2, 05:31:00', 'GO c1 c2 Eesh 23:00', D(2026,10,22,5,31)),
    ('GO c1 c2, 06:20 (c1 not yet on Kaggle: late wake)', 'GO c1 c2 Eesh 23:00', D(2026,10,22,6,20)),
    ('GO c1 c2 c3, 07:10 (late wake)', 'GO c1 c2 c3 Eesh', D(2026,10,22,7,10)),
    ('GO c1 c2, 09:00:01', 'GO c1 c2 Eesh', D(2026,10,22,9,0,1)),
    ('GO "c1 C2" (upper-case C2), 06:20', 'GO c1 C2 Eesh', D(2026,10,22,6,20)),
    ('GO "c1 cc2" typo', 'GO c1 cc2', D(2026,10,22,6,20)),
    ('GO "c2" alone, 06:20', 'GO c2', D(2026,10,22,6,20)),
    ('GO "go c1" lower-case', 'go c1', D(2026,10,22,6,20)),
    ('GO "c1,c2" comma', 'GO c1,c2', D(2026,10,22,6,20)),
    ('GO "c1 c2 MAX=3 Eesh"', 'GO c1 c2 MAX=3 Eesh', D(2026,10,22,5,31)),
]
for c in cases:
    print('A | %-50s -> %s' % (c[0], runA(*c[1:]) or '(silent: nothing)'))
# cap path on REAL list dates: pretend a copy was due on 10-21 and ask at 10-21 23:00 IST (UTC day 10-21 has 5 subs)
A.COPIES = {'c1': D(2026,10,21,0,0)}
print('A | cap check on real API dates at 10-21 23:00 IST       -> %s' % runA('GO c1', D(2026,10,21,23,0)))
print('A | files written by tick_a23 in dry mode: log %s lock %s sublog %s' % (os.path.exists(A.LOG), os.path.exists(A.LOCK), os.path.exists(A.SUBLOG)))

print('==== ops_tick_a23.py (final) ====')
O = load(os.path.join(R, 'session_2026-10-21', 'final', 'ops_tick_a23.py'), 'ops_tick_a23_h')
print('O | GO path the live tick reads: %s (exists %s)' % (O.GO, os.path.exists(O.GO)))
O.F = T  # selftest writes its log into recheck/mech_tmp, not final/
print('O | selftest -> %s' % cap(O.selftest).split(' || ')[-1])
for lbl, t in [('05:30:59', D(2026,10,22,5,30,59)), ('05:31:00', D(2026,10,22,5,31)), ('06:20', D(2026,10,22,6,20)), ('09:00:01', D(2026,10,22,9,0,1))]:
    print('O | dry %-9s -> %s' % (lbl, cap(lambda: O.main(t))))
for g in ['GO c1 c2 MAX=3 Eesh 10-21 23:00: "ok"', 'GO c1 c2 ok', 'GO c1 c2 2300', 'GO c1 c2 by eesh', 'GO c1 C2 Eesh', 'GO c1 c1 c2', 'GO c1 c2 MAX=0', 'GO c1 c2 MAX=x']:
    print('O | parse_go(%r) -> %s' % (g, O.parse_go(g)))
api = KaggleApi(); api.authenticate()
subs = list(api.competition_submissions(O.COMP, page_size=50))
print('O | api s.date type/sample: %s %r' % (type(subs[0].date).__name__, subs[0].date))
print('O | count_api_day at 10-21 23:00 IST -> %s ; at 10-22 05:31 -> %s' % (O.count_api_day(subs, D(2026,10,21,23,0)), O.count_api_day(subs, D(2026,10,22,5,31))))
print('O | done_keys(real list, real submits.log) -> %s' % O.done_keys([s.description for s in subs], O.rd(O.SUBLOG)))
print('O | existing descs starting A23: %s' % [s.description for s in subs if (s.description or '').startswith('A23')])
print('O | top 6 newest: %s' % [(s.ref, str(s.date), str(s.status), s.public_score, (s.description or '')[:22]) for s in subs[:6]])
lim = api.competition_get_submission_limits(O.COMP)
print('O | limits now: %s' % lim)
print('O | kernels_status: %s' % api.kernels_status(O.SLUG))
