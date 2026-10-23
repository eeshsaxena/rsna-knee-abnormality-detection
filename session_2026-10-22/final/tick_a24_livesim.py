# tick_a24_livesim.py (10-22, Claude, private-entries track). OFFLINE simulation of tick_a24.main() on its LIVE path.
# A fake 'kaggle' module replaces the SDK (no network, nothing submitted), the clock is faked, and GO / PREREG / submits.log /
# the tick log + lock are redirected to a temp dir (submits.log is a COPY of the real one). The real files are never written.
# Usage: C:\Python314\python.exe -I session_2026-10-22\final\tick_a24_livesim.py   -> prints A24_LIVESIM PASS n/n
import os, sys, types, shutil, tempfile, importlib.util, datetime as _dt
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('tick_a24', os.path.join(HERE, 'tick_a24.py'))
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
REAL = dict(SUBLOG=T.SUBLOG, PREREG=T.PREREG, GO=T.GO, LOG=T.LOG)

class FDT(_dt.datetime):
    NOW = None
    @classmethod
    def now(cls, tz=None): return cls.NOW
T.datetime = types.SimpleNamespace(datetime=FDT, timedelta=_dt.timedelta, date=_dt.date, timezone=_dt.timezone)

class Row:
    def __init__(s, ref, ist, desc): s.ref = ref; s.date = (ist - T.IST_UTC).strftime('%Y-%m-%dT%H:%M:%SZ'); s.description = desc
class FakeApi:
    rows, calls, status, allowed, list_err, submit_err, submit_lands = [], [], 'KernelWorkerStatus.COMPLETE', 5, False, False, True
    def authenticate(s): pass
    def kernels_status(s, slug): return types.SimpleNamespace(status=FakeApi.status)
    def competition_submissions(s, comp, page_size=20):
        if FakeApi.list_err: raise RuntimeError('SSLError sim')
        return list(reversed(FakeApi.rows)) + [None]
    def competition_get_submission_limits(s, comp): return types.SimpleNamespace(num_allowed_now=FakeApi.allowed)
    def competition_submit_code(s, file_name, message, competition=None, kernel=None, kernel_version=None, quiet=False):
        FakeApi.calls.append(dict(file_name=file_name, message=message, competition=competition, kernel=kernel, kernel_version=kernel_version))
        ref = 59000000 + len(FakeApi.calls)
        if FakeApi.submit_lands: FakeApi.rows.append(Row(ref, FDT.NOW, message))
        if FakeApi.submit_err: raise RuntimeError('503 sim')
        return '{"message": "", "ref": %d}' % ref
m_ext = types.ModuleType('kaggle.api.kaggle_api_extended'); m_ext.KaggleApi = FakeApi
sys.modules.update({'kaggle': types.ModuleType('kaggle'), 'kaggle.api': types.ModuleType('kaggle.api'), 'kaggle.api.kaggle_api_extended': m_ext})

TMP = tempfile.mkdtemp(prefix='a24sim_')
def reset(go='GO c1 c2 c3 MAX=5 Eesh 10-22 09:30: "sim"', prereg=True):
    for f in os.listdir(TMP):
        p = os.path.join(TMP, f); shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    T.SUBLOG = os.path.join(TMP, 'submits.log'); shutil.copy(REAL['SUBLOG'], T.SUBLOG)
    T.PREREG = os.path.join(TMP, 'PREREG.md'); open(T.PREREG, 'w').write('...\n### AMENDMENT 24 (sim)\n' if prereg else '...\n')
    T.GO = os.path.join(TMP, 'A24_GO.txt'); open(T.GO, 'w').write(go)
    T.D = os.path.join(TMP, 'a24'); T.LOG = os.path.join(T.D, 'tick_a24.log'); T.LOCK = os.path.join(T.D, 'tick.lock')
    T.OTHER_GO = [os.path.join(TMP, 'A23_GO_sim.txt')]
    FakeApi.rows, FakeApi.calls = [], []
    FakeApi.status, FakeApi.allowed, FakeApi.list_err, FakeApi.submit_err, FakeApi.submit_lands = 'KernelWorkerStatus.COMPLETE', 5, False, False, True
def tick(h, m, s=0):
    FDT.NOW = FDT(2026, 10, 22, h, m, s); T.main()
def logtxt(): return open(T.LOG, encoding='utf-8').read() if os.path.exists(T.LOG) else ''
def a24_lines(): return [l for l in open(T.SUBLOG, encoding='utf-8').read().splitlines() if ' A24 ' in l]

res = []
def t(name, cond):
    res.append(bool(cond)); print(('OK   ' if cond else 'FAIL ') + name, flush=True)

reset()
tick(9, 40)
c = FakeApi.calls
t('A 09:40 armed -> c1 submitted with kernel %s v1, submission.csv, prefix A24-R6PW20C1:' % T.SLUG,
  len(c) == 1 and c[0]['kernel'] == T.SLUG and c[0]['kernel_version'] == 1 and c[0]['file_name'] == 'submission.csv'
  and c[0]['competition'] == T.COMP and c[0]['message'].startswith('A24-R6PW20C1:'))
t('A submits.log (temp copy) gained exactly one "A24 c1 SUBMITTED" line', len(a24_lines()) == 1 and 'A24 c1 SUBMITTED' in a24_lines()[0])
t('A lock removed after the tick', not os.path.exists(T.LOCK))
tick(9, 45); t('B 09:45 -> nothing (c2 not before 10:10)', len(FakeApi.calls) == 1)
tick(10, 9, 59); t('B 10:09:59 -> nothing', len(FakeApi.calls) == 1)
tick(10, 10, 0); t('C 10:10:00 -> c2 submitted', len(FakeApi.calls) == 2 and FakeApi.calls[1]['message'].startswith('A24-R6PW20C2:'))
FDT.NOW = FDT(2026, 10, 22, 10, 20); FakeApi.rows += [Row(1, FDT.NOW, 'A24-PUB1: moacf'), Row(2, FDT.NOW, 'A24-PUB2: x')]
tick(10, 40); t('D 2 public entries in between (used 4 < 5) -> c3 at 10:40', len(FakeApi.calls) == 3 and FakeApi.calls[2]['message'].startswith('A24-R6PW20C3:'))
tick(11, 30); t('E 11:30 all three done -> nothing', len(FakeApi.calls) == 3)
t('E submits.log has c1, c2, c3 once each', [l.split(' A24 ')[1][:2] for l in a24_lines()] == ['c1', 'c2', 'c3'])

reset(); FDT.NOW = FDT(2026, 10, 22, 6, 0); FakeApi.rows = [Row(i, FDT(2026, 10, 22, 6, i), 'pub %d' % i) for i in range(5)]
tick(9, 40); t('F 5 entries already today -> CAP logged, nothing submitted', len(FakeApi.calls) == 0 and 'CAP c1' in logtxt())
reset(); FDT.NOW = FDT(2026, 10, 22, 6, 0); FakeApi.rows = [Row(i, FDT(2026, 10, 22, 6, i), 'pub %d' % i) for i in range(2)]
reset(go='GO c1 c2 c3 MAX=3 Eesh'); FakeApi.rows = [Row(i, FDT(2026, 10, 22, 6, i), 'pub %d' % i) for i in range(3)]
tick(9, 40); t('F2 MAX=3 with 3 entries today -> CAP, nothing submitted', len(FakeApi.calls) == 0 and 'CAP c1' in logtxt())
reset(); FakeApi.allowed = 0
tick(9, 40); t('F3 num_allowed_now 0 -> CAP, nothing submitted', len(FakeApi.calls) == 0 and 'CAP c1' in logtxt())
reset(); FakeApi.status = 'KernelWorkerStatus.RUNNING'
tick(9, 40); t('G kernel status RUNNING (someone pushed) -> STATUS_ODD, nothing submitted', len(FakeApi.calls) == 0 and 'STATUS_ODD' in logtxt())
reset(); FakeApi.list_err = True
tick(9, 40); t('H list error -> nothing submitted', len(FakeApi.calls) == 0 and 'list err' in logtxt())
reset(); FakeApi.submit_err = True
tick(9, 40); FakeApi.submit_err = False
tick(9, 45); t('I submit raised but landed -> SUBMIT ERR logged, next tick dedupes from the list (no 2nd c1)',
  len(FakeApi.calls) == 1 and 'SUBMIT ERR' in logtxt() and 'c1' not in [x['message'][9:11] for x in FakeApi.calls[1:]])
tick(10, 14, 59); t('I c2 waits 30 min from the API time of the landed c1 (09:40): nothing at 10:09:59 is covered, 10:14:59 -> c2',
  len(FakeApi.calls) == 2 and FakeApi.calls[1]['message'].startswith('A24-R6PW20C2:'))
reset(); FakeApi.rows = [Row(57999999, FDT(2026, 10, 22, 9, 31, 20), 'A24-R6PW20C1: by hand')]
tick(10, 1, 19); t('J by-hand c1 09:31:20 (API row only, no log line) -> nothing at 10:01:19', len(FakeApi.calls) == 0)
tick(10, 1, 20); t('J ... c2 at 10:01:20', len(FakeApi.calls) == 1 and FakeApi.calls[0]['message'].startswith('A24-R6PW20C2:'))
reset(); FakeApi.rows = [Row(57999998, FDT(2026, 10, 22, 9, 30), 'A23-R6PW20C1: by hand with the old prefix')]
tick(9, 59, 59); t('N A23-prefixed c1 at 09:30 (forum.md by-hand command) counts as c1 -> nothing at 09:59:59', len(FakeApi.calls) == 0)
tick(10, 0, 0); t('N ... A24 c2 at 10:00:00 (no 2nd c1)', len(FakeApi.calls) == 1 and FakeApi.calls[0]['message'].startswith('A24-R6PW20C2:'))
reset(); open(T.OTHER_GO[0], 'w').write('GO c1 c2 c3 MAX=5'); tick(9, 40)
t('O an A23_GO.txt exists (other tick may be armed) -> CONFLICT logged, nothing submitted', len(FakeApi.calls) == 0 and 'CONFLICT' in logtxt())
reset(prereg=False); tick(9, 40); t('K no AMENDMENT 24 -> nothing submitted', len(FakeApi.calls) == 0 and 'no AMENDMENT 24' in logtxt())
reset(go='GO c1 c2 ok'); tick(9, 40); t('K GO with a stray lower-case word -> invalid, nothing submitted', len(FakeApi.calls) == 0)
reset(go='GO c1 MAX=5'); tick(9, 40); tick(10, 30); t('K GO c1 only -> exactly one copy', len(FakeApi.calls) == 1)
reset(); tick(14, 30, 1); t('L 14:30:01 -> past CLOSE, nothing submitted', len(FakeApi.calls) == 0)
reset(); os.makedirs(T.D, exist_ok=True); open(T.LOCK, 'w').write('1'); tick(9, 40)
t('M fresh lock (another tick running) -> nothing submitted', len(FakeApi.calls) == 0)
t('real files untouched: submits.log / PREREG / GO / tick log paths were redirected',
  all(not os.path.samefile(T.SUBLOG, REAL['SUBLOG']) for _ in [0]) and not os.path.exists(REAL['GO']) and not os.path.exists(REAL['LOG']))
shutil.rmtree(TMP, ignore_errors=True)
print('A24_LIVESIM %s %d/%d' % ('PASS' if all(res) else 'FAIL', sum(res), len(res)), flush=True)
