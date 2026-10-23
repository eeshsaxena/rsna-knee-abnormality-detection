# tick_a24.py (10-22 ~09:10 IST, Claude, private-entries track). DRAFT. NOT REGISTERED, NEVER RUN LIVE BY ITS AUTHOR
# (only --selftest offline and --dry with read-only API calls).
# Adapted COPY of session_2026-10-21/final/ops_tick_a23.py (self-test 21/21 on 10-21; never armed, A23 lapsed with no GO).
# ONE idempotent tick that resubmits r6pw20 v1 UNCHANGED (eeshsaxena/rsna-sub-v224r6flips6fpw20j, kernel_version 1) as up to
# 3 copies on the 10-22 UTC day. No push, no commit, no output fetch: the version is the one gated PASS 17/17 on 10-21 09:00:50 IST.
# Changes vs A23 (all on purpose):
#   - GO file session_2026-10-22/final/A24_GO.txt; PREREG guard 'AMENDMENT 24'; prefixes 'A24-R6PW20C<n>:'; log dir final/a24.
#   - Windows: c1 as soon as armed (not-before 05:30:00 IST = UTC day open), c2 / c3 each >= 30 min after the previous copy's
#     submit; CLOSE 10-22 14:30:00 IST (reads by ~23:00 IST at the 8.5 h worst latency).
#   - The previous copy's submit time comes from the Kaggle list (description prefix + date) OR submits.log, whichever is later,
#     so a copy submitted BY HAND (no log line) still enforces the 30 min gap (A23 used submits.log only). Done but no time known ->
#     the next copy waits (TIME_UNKNOWN), it never bursts.
#   - Cap: MAX defaults to 5 = the Kaggle daily cap, counted over ALL team entries of the UTC day (public pushes included), so the
#     tick takes "5 minus submissions already today" at most, and never more than its 3 copies. GO may lower it (e.g. MAX=3 keeps
#     2 slots free for public entries submitted later).
#   - None rows from the SDK list are skipped (the SDK return type allows None).
#   - Arming state (PREREG / GO present) is INFO in --selftest, not PASS/FAIL; --armcheck asserts it after main arms.
#   - Cross-dedupe + interlock with the parallel A23-prefixed path (final track's ops_tick_a23_1022.py, forum.md's by-hand
#     command): an 'A23-R6PW20C<n>:' row / 'A23 c<n> SUBMITTED' line on the 10-22 UTC day counts as copy c<n>; if an A23_GO.txt
#     exists (10-22 or 10-21 final), this tick submits nothing (CONFLICT). Arm ONE tick only.
# Guards (all must hold before a submit):
#   G1 GO file, first non-empty line 'GO c1 [c2 [c3]] [MAX=n] <free text>'; unknown lower-case key -> whole GO invalid.
#   G2 session_2026-10-15/PREREG_10-15.md contains 'AMENDMENT 24'.
#   G3 local kernel dir sha == registry (nb e4d92c0a, meta 3a5b8b0f); val/a22 push log says only 'Kernel version 1'; val/a22 gate
#      file's last line is 'PROBE_CHECK PASS 17/17'.
#   G4 api.kernels_status(slug) is COMPLETE (anything else = someone pushed again -> STATUS_ODD, submit nothing).
#   G5 c1 not-before 05:30:00 IST; each later copy >= 30 min after the previous copy's submit; CLOSE 14:30:00 IST; strict order.
#   G6 dedupe per copy against a FRESH Kaggle list every tick (list call fails -> nothing submitted) and submits.log.
#   G7 cap: num_allowed_now >= 1 (if readable) AND used this UTC day < MAX (max of API dates and submits.log SUBMITTED lines).
#   G8 at most ONE submit per tick.
# Usage (from C:\Users\eeshs\Downloads\rsna-knee, NO -I for live / dry: kaggle lives in the user site-packages):
#   C:\Python314\python.exe session_2026-10-22\final\tick_a24.py                 live tick (needs GO + AMENDMENT 24)
#   C:\Python314\python.exe -I session_2026-10-22\final\tick_a24.py --selftest   offline, writes only final/tick_a24_selftest.log
#   C:\Python314\python.exe session_2026-10-22\final\tick_a24.py --dry [ISO-IST]  read-only API calls, prints the decision
#   C:\Python314\python.exe -I session_2026-10-22\final\tick_a24.py --armcheck   offline: GO valid + AMENDMENT 24 + static ok
import os, sys, json, time, datetime, hashlib, re
F = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(F, '..', '..'))
D = os.path.join(F, 'a24'); LOG = os.path.join(D, 'tick_a24.log'); LOCK = os.path.join(D, 'tick.lock')
GO = os.path.join(F, 'A24_GO.txt')
SUBLOG = os.path.join(R, 'session_2026-10-14', 'submits.log')
PREREG = os.path.join(R, 'session_2026-10-15', 'PREREG_10-15.md')
A22 = os.path.join(R, 'session_2026-10-20', 'val', 'a22')
CMD = os.path.join(F, 'tick_a24.cmd')
COMP = 'rsna-knee-abnormality-detection'
SLUG = 'eeshsaxena/rsna-sub-v224r6flips6fpw20j'; VER = 1; TAG = 'v224r6flips6fpw20j'
KDIR = os.path.join(R, 'session_2026-10-20', 'val', 'kernels', 'kernel_' + TAG)
NB_SHA = 'e4d92c0a2c581a8301f87ae4779da21870664814b4af7068dcca55014bbc7757'
META_SHA = '3a5b8b0f6df5fb91255179b5f0aa0b5425f768e88d7d0eff47eb36fd01eba9f8'
PUSHLOG = os.path.join(A22, 'push_%s.log' % TAG); GATE = os.path.join(A22, 'gate_%s.txt' % TAG); GATE_PASS = 'PROBE_CHECK PASS 17/17'
IST_UTC = datetime.timedelta(hours=5, minutes=30)
DAY_OPEN = datetime.datetime(2026, 10, 22, 5, 30, 0)          # 10-22 00:00:00 UTC
COPIES = {   # key -> Kaggle description; the prefix up to ':' is the dedupe key
    'c1': 'A24-R6PW20C1: resubmit r6pw20 v1 unchanged (57463719 hit the S6F canary .916); B hedge copy 1',
    'c2': 'A24-R6PW20C2: resubmit r6pw20 v1 unchanged (57463719 hit the S6F canary .916); B hedge copy 2',
    'c3': 'A24-R6PW20C3: resubmit r6pw20 v1 unchanged (57463719 hit the S6F canary .916); B hedge copy 3',
}
ORDER = ['c1', 'c2', 'c3']; GAP = datetime.timedelta(minutes=30)
# Cross-dedupe with the parallel A23-prefixed path (final track's ops_tick_a23_1022.py and forum.md's by-hand command use
# 'A23-R6PW20C<n>:' / 'A23 c<n> SUBMITTED'). Such a copy made on the 10-22 UTC day counts as copy c<n> here, with its time.
ALIAS = {k: 'A23-R6PW20C%s:' % k[1] for k in ORDER}
# Interlock: if an A23 GO file exists, another r6pw20 tick may be armed -> this tick submits nothing (CONFLICT). Arm ONE tick only.
OTHER_GO = [os.path.join(F, 'A23_GO.txt'), os.path.join(R, 'session_2026-10-21', 'final', 'A23_GO.txt')]
CLOSE = datetime.datetime(2026, 10, 22, 14, 30, 0)
MAX_DEFAULT = 5
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
rd = lambda p: open(p, encoding='utf-8', errors='replace').read() if os.path.exists(p) else ''
pref = lambda k: COPIES[k].split(':')[0] + ':'

def say(m, log=None):
    line = '[%s] %s' % (datetime.datetime.now().strftime('%m-%d %H:%M:%S'), m)
    print(line, flush=True)
    os.makedirs(os.path.dirname(log or LOG), exist_ok=True); open(log or LOG, 'a', encoding='utf-8').write(line + '\n')

def parse_go(text):
    """-> (keys in ORDER, maxday, err). keys None when the GO is absent / invalid."""
    ls = [l.strip() for l in (text or '').splitlines() if l.strip()]
    if not ls or ls[0].split()[0] != 'GO':
        return None, MAX_DEFAULT, 'no GO line'
    keys, mx = [], MAX_DEFAULT
    for t in ls[0].split()[1:]:
        if t.startswith('MAX='):
            try: mx = max(0, min(5, int(t[4:])))
            except ValueError: return None, MAX_DEFAULT, 'bad %s' % t
        elif t in COPIES:
            if t not in keys: keys.append(t)
        elif re.fullmatch(r'[a-z0-9]+', t):
            return None, MAX_DEFAULT, 'unknown key %s' % t
        else:
            break
    keys = [k for k in ORDER if k in keys]
    if keys and keys != ORDER[:len(keys)]:
        return None, MAX_DEFAULT, 'copies must be a prefix of c1 c2 c3'
    return (keys or None), mx, (None if keys else 'no keys')

def prereg_ok():
    return 'AMENDMENT 24' in rd(PREREG).upper()

def static_ok():
    """G3 -> (ok, why)."""
    try:
        if sha(os.path.join(KDIR, 'rsna_sub_%s.ipynb' % TAG)) != NB_SHA: return False, 'nb sha'
        if sha(os.path.join(KDIR, 'kernel-metadata.json')) != META_SHA: return False, 'meta sha'
        if json.load(open(os.path.join(KDIR, 'kernel-metadata.json')))['id'] != SLUG: return False, 'meta id'
    except Exception as ex:
        return False, 'kernel dir %r' % ex
    m = re.findall(r'Kernel version (\d+) successfully pushed', rd(PUSHLOG))
    if m != [str(VER)]: return False, 'push log versions %s' % m
    g = rd(GATE).strip().splitlines()
    if not g or g[-1].strip() != GATE_PASS: return False, 'gate last line %r' % (g[-1] if g else '')
    return True, 'ok'

def utc_day(ist):
    return (ist - IST_UTC).date()

def sublog_times(text):
    """{key: IST datetime} of 'A24 cN SUBMITTED' lines, plus the count of SUBMITTED lines per UTC day."""
    t, per = {}, {}
    for l in (text or '').splitlines():
        m = re.match(r'\[(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) IST\](.*)', l)
        if not m or 'SUBMITTED' not in l: continue
        when = datetime.datetime(2026, *map(int, m.groups()[:5]))
        per[utc_day(when)] = per.get(utc_day(when), 0) + 1
        k = re.search(r'\bA24 (c[123]) SUBMITTED', l) or (re.search(r'\bA23 (c[123]) SUBMITTED', l) if when >= DAY_OPEN else None)
        if k: t.setdefault(k.group(1), when)
    return t, per

def conflicts():
    return [p for p in OTHER_GO if os.path.exists(p)]

def api_ist(d):
    """Kaggle date (datetime or ISO str, UTC) -> naive IST datetime, or None."""
    if d is None: return None
    try:
        if not isinstance(d, datetime.datetime): d = datetime.datetime.fromisoformat(str(d).replace('Z', '+00:00'))
        if d.tzinfo is not None: d = d.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return d + IST_UTC
    except Exception:
        return None

def count_api_day(subs, now):
    n, seen = 0, False
    for s in subs or []:
        w = api_ist(getattr(s, 'date', None)) if s is not None else None
        if w is None: continue
        seen = True; n += utc_day(w) == utc_day(now)
    return n if seen else None

def api_times(subs):
    """{key: IST datetime} of copies on Kaggle (A24 prefix, or the A23 alias when dated on the 10-22 UTC day), earliest row per key."""
    t = {}
    for s in subs or []:
        if s is None: continue
        d = s.description or ''; w = api_ist(getattr(s, 'date', None))
        for k in ORDER:
            if d.startswith(pref(k)) or (d.startswith(ALIAS[k]) and w is not None and w >= DAY_OPEN):
                if w is not None and (k not in t or w < t[k]): t[k] = w
    return t

def done_keys(descs, sublog):
    """copies already on Kaggle (description prefix) or in submits.log."""
    t, _ = sublog_times(sublog)
    out = set(t)
    for k in ORDER:
        if any((d or '').startswith(pref(k)) for d in descs or []): out.add(k)
    return out

def merge_times(*ts):
    out = {}
    for t in ts:
        for k, w in (t or {}).items():
            if w is not None and (k not in out or w > out[k]): out[k] = w     # later of the two sources = conservative gap
    return out

def next_copy(keys, done, sub_t, now):
    """-> (key to submit now or None, why). Strict order; c1 not-before DAY_OPEN; >= GAP after the previous copy's submit."""
    if now > CLOSE: return None, 'closed'
    if now < DAY_OPEN: return None, 'before day open'
    prev = None
    for k in keys:
        if k in done:
            if k not in sub_t: return None, 'TIME_UNKNOWN for done copy %s (waits; check by hand)' % k
            prev = sub_t[k]; continue
        nb = DAY_OPEN if prev is None else max(DAY_OPEN, prev + GAP)
        return (k, 'due') if now >= nb else (None, '%s not before %s' % (k, nb.strftime('%H:%M:%S')))
    return None, 'all GO copies done'

def cap_ok(used, mx, allowed):
    return used < mx and allowed != 0

def main(dry_at=None):
    """dry_at: an IST datetime -> DRY RUN: pretend GO 'c1 c2 c3' + AMENDMENT 24 exist and that it is dry_at; do every READ-ONLY
    API call (status, list, limits) and print the decision. Never submits, never writes submits.log / the tick log / the lock."""
    dry = dry_at is not None
    now = dry_at or datetime.datetime.now()
    out = print if dry else say
    if not dry: os.makedirs(D, exist_ok=True)
    if now > CLOSE:
        print('past CLOSE %s: nothing to do' % CLOSE, flush=True); return
    keys, mx, err = (['c1', 'c2', 'c3'], MAX_DEFAULT, None) if dry else parse_go(rd(GO))
    if keys is None:
        print('no valid GO (%s): nothing to do' % err, flush=True); return
    if not dry and not prereg_ok():
        say('GO %s present but PREREG has no AMENDMENT 24: nothing submitted' % keys); return
    cf = conflicts()
    if cf:
        out('CONFLICT A23 GO file(s) %s exist (another r6pw20 tick may be armed): nothing submitted; arm ONE tick only' % cf)
        if not dry: return
    ok, why = static_ok()
    if not ok:
        out('STATIC_FAIL %s: nothing submitted' % why); return
    if now < DAY_OPEN:
        print('before %s: nothing to do' % DAY_OPEN, flush=True); return
    if not dry and os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) < 8 * 60:
        print('locked', flush=True); return
    if not dry: open(LOCK, 'w').write(str(os.getpid()))
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        try:
            st = str(api.kernels_status(SLUG).status)
        except Exception as ex:
            out('status err %r -> retry next tick' % ex); return
        if 'COMPLETE' not in st:
            out('STATUS_ODD %s for %s (someone pushed again?) -> nothing submitted' % (st, SLUG)); return
        try:
            subs = [s for s in (api.competition_submissions(COMP, page_size=50) or []) if s is not None]
            descs = [s.description or '' for s in subs]
        except Exception as ex:
            out('list err %r -> retry next tick' % ex); return
        sublog = rd(SUBLOG)
        lt, per = sublog_times(sublog)
        at = api_times(subs)
        sub_t = merge_times(lt, at)
        done = done_keys(descs, sublog) | set(at)
        tnow = now if dry else datetime.datetime.now()
        k, kwhy = next_copy(keys, done, sub_t, tnow)
        used = max(per.get(utc_day(tnow), 0), count_api_day(subs, tnow) or 0)
        allowed = None
        try: allowed = int(api.competition_get_submission_limits(COMP).num_allowed_now)
        except Exception: pass
        if dry:
            print('DRY at %s IST: kernel status %s | copies done %s | times %s | next copy %s (%s) | used on UTC day %s = %d (MAX %d) | '
                  'num_allowed_now (live, real clock) %s | static %s' % (now, st, sorted(done) or '-',
                  {a: b.strftime('%H:%M:%S') for a, b in sorted(sub_t.items())} or '-', k, kwhy, utc_day(tnow), used, mx, allowed, why))
            return
        if k is None:
            if kwhy.startswith('TIME_UNKNOWN'): say(kwhy)
            return
        if not cap_ok(used, mx, allowed):
            say('CAP %s: used %d (MAX %d), num_allowed_now %s -> not submitted' % (k, used, mx, allowed)); return
        desc = COPIES[k]
        try:
            r = api.competition_submit_code(file_name='submission.csv', message=desc, competition=COMP, kernel=SLUG, kernel_version=VER)
            line = '[%s IST] A24 %s SUBMITTED %s v%d: %s' % (datetime.datetime.now().strftime('%m-%d %H:%M:%S'), k, SLUG, VER, r)
            open(SUBLOG, 'a', encoding='utf-8').write(line + '\n'); say(line)
        except Exception as ex:
            resp = getattr(ex, 'response', None); body = resp.text[:300] if resp is not None else ''
            say('%s SUBMIT ERR %s %s %s -> retry next tick (the fresh list dedupes a submit that did land)' % (k, type(ex).__name__, str(ex)[:150], body))
    finally:
        if not dry:
            try: os.remove(LOCK)
            except Exception: pass

def armcheck():
    keys, mx, err = parse_go(rd(GO)); ok, why = static_ok(); p = prereg_ok(); cf = conflicts()
    res = keys is not None and p and ok and not cf
    print('A24_ARMCHECK %s | GO %s keys %s MAX %d (%s) | PREREG AMENDMENT 24 %s | static %s | A23 GO conflicts %s | now %s | CLOSE %s'
          % ('PASS' if res else 'FAIL', GO, keys, mx, err or 'ok', p, why, cf or 'none', datetime.datetime.now().strftime('%m-%d %H:%M:%S'), CLOSE), flush=True)
    return res

def selftest():
    SL = os.path.join(F, 'tick_a24_selftest.log'); open(SL, 'w').close(); res = []
    def t(name, cond):
        res.append(bool(cond)); say(('OK   ' if cond else 'FAIL ') + name, SL)
    ok, why = static_ok()
    t('G3 static: kernel sha e4d92c0a / meta 3a5b8b0f, A22 push log = v1 only, gate last line PASS 17/17 (%s)' % why, ok)
    t('GO "GO c1 c2 c3 MAX=5 Eesh 10-22 09:30" -> [c1, c2, c3], MAX 5', parse_go('GO c1 c2 c3 MAX=5 Eesh 10-22 09:30')[:2] == (['c1', 'c2', 'c3'], 5))
    t('GO "GO c1 c2" -> [c1, c2], MAX 5 (default = Kaggle daily cap)', parse_go('GO c1 c2')[:2] == (['c1', 'c2'], 5))
    t('GO "MAX=9" clamps to 5, "MAX=3" -> 3', parse_go('GO c1 MAX=9')[1] == 5 and parse_go('GO c1 MAX=3')[1] == 3)
    t('GO typo "GO c1 cc2" / "GO c1 ok" -> invalid', parse_go('GO c1 cc2')[0] is None and parse_go('GO c1 ok')[0] is None)
    t('GO "GO c2" (skips c1) -> invalid', parse_go('GO c2')[0] is None)
    t('GO lower-case / empty / bare -> invalid', parse_go('go c1')[0] is None and parse_go('')[0] is None and parse_go('GO')[0] is None)
    a22 = '[10-21 09:00:53 IST] A22 r6pw20 SUBMITTED %s v1: {"ref": 57463719}' % SLUG
    t('dedupe: the A22 line and the A22 description never block c1', 'c1' not in done_keys(['A22-R6PW20: B hedge'], a22))
    t('alias: submits.log "A23 c1 SUBMITTED" at 10-22 09:30 IST counts as c1 (time kept); the same line at 10-22 05:29 IST does not',
      sublog_times('[10-22 09:30:05 IST] A23 c1 SUBMITTED x v1: {}')[0] == {'c1': datetime.datetime(2026, 10, 22, 9, 30, 5)}
      and sublog_times('[10-22 05:29:00 IST] A23 c1 SUBMITTED x v1: {}')[0] == {})
    t('interlock: OTHER_GO = the two A23_GO.txt paths (10-22 and 10-21 final)', [os.path.basename(os.path.dirname(os.path.dirname(p))) for p in OTHER_GO]
      == ['session_2026-10-22', 'session_2026-10-21'] and all(p.endswith('A23_GO.txt') for p in OTHER_GO))
    t('dedupe: Kaggle desc "A24-R6PW20C1: ..." blocks c1 only', done_keys(['A24-R6PW20C1: x'], '') == {'c1'})
    t('dedupe: submits.log "A24 c2 SUBMITTED" blocks c2 only', done_keys([], '[10-22 10:01:10 IST] A24 c2 SUBMITTED x v1: {}') == {'c2'})
    K = ['c1', 'c2', 'c3']; dt = lambda h, m, s=0: datetime.datetime(2026, 10, 22, h, m, s)
    t('window: 05:29:59 -> none, 05:30:00 -> c1, 09:40 (arm time) -> c1',
      next_copy(K, set(), {}, dt(5, 29, 59))[0] is None and next_copy(K, set(), {}, dt(5, 30))[0] == 'c1' and next_copy(K, set(), {}, dt(9, 40))[0] == 'c1')
    s1 = {'c1': dt(9, 31, 20)}
    t('gap: c1 done 09:31:20 -> c2 not at 10:01:19, c2 at 10:01:20',
      next_copy(K, {'c1'}, s1, dt(10, 1, 19))[0] is None and next_copy(K, {'c1'}, s1, dt(10, 1, 20))[0] == 'c2')
    s2 = {'c1': dt(9, 31, 20), 'c2': dt(10, 5)}
    t('gap: c2 done 10:05 -> c3 at 10:35, not 10:34', next_copy(K, {'c1', 'c2'}, s2, dt(10, 34))[0] is None and next_copy(K, {'c1', 'c2'}, s2, dt(10, 35))[0] == 'c3')
    t('strict order: c1 not done -> never c2 even at 12:00', next_copy(K, set(), {}, dt(12, 0))[0] == 'c1')
    t('done copy with no known time -> waits (TIME_UNKNOWN), never bursts', next_copy(K, {'c1'}, {}, dt(12, 0)) [0] is None
      and next_copy(K, {'c1'}, {}, dt(12, 0))[1].startswith('TIME_UNKNOWN'))
    t('CLOSE: 14:30:00 open, 14:30:01 closed', next_copy(K, set(), {}, dt(14, 30, 0))[0] == 'c1' and next_copy(K, set(), {}, dt(14, 30, 1))[0] is None)
    t('GO [c1, c2] never yields c3', next_copy(['c1', 'c2'], {'c1', 'c2'}, {'c1': dt(9, 31), 'c2': dt(10, 1)}, dt(13, 0))[0] is None)
    t('UTC day: 10-22 05:29:59 IST -> 10-21 UTC; 05:30:00 -> 10-22 UTC', utc_day(dt(5, 29, 59)) == datetime.date(2026, 10, 21)
      and utc_day(dt(5, 30)) == datetime.date(2026, 10, 22))
    class _S:
        def __init__(s, d, desc=''): s.date = d; s.description = desc
    by_hand = [_S('2026-10-22T04:01:20Z', 'A24-R6PW20C1: x'), _S(datetime.datetime(2026, 10, 22, 4, 2), 'A22-PUB: moacf'), None]
    at = api_times(by_hand)
    t('API time: by-hand c1 at 04:01:20 UTC -> 09:31:20 IST; others ignored; None row skipped', at == {'c1': dt(9, 31, 20)})
    t('API alias: "A23-R6PW20C2:" row at 04:31 UTC 10-22 -> c2 at 10:01 IST; an A23 row dated 10-21 UTC is ignored',
      api_times([_S('2026-10-22T04:31:00Z', 'A23-R6PW20C2: x'), _S('2026-10-21T23:00:00Z', 'A23-R6PW20C3: old')]) == {'c2': dt(10, 1)})
    m = merge_times({'c1': dt(9, 31)}, at)
    t('merge: later of submits.log / API time wins (gap counted from 09:31:20)', m == {'c1': dt(9, 31, 20)}
      and next_copy(K, {'c1'}, m, dt(10, 1, 19))[0] is None and next_copy(K, {'c1'}, m, dt(10, 1, 20))[0] == 'c2')
    t('API date count: 2 of 3 on 10-22 UTC (+ None row skipped)', count_api_day([_S(datetime.datetime(2026, 10, 22, 0, 1)), _S('2026-10-22T01:00:00Z'),
      _S('2026-10-21T23:59:00Z'), None], dt(9, 0)) == 2)
    t('cap: used 4 / MAX 5 ok; used 5 blocked; num_allowed_now 0 blocked; None (unreadable) ok',
      cap_ok(4, 5, None) and not cap_ok(5, 5, 3) and not cap_ok(2, 5, 0) and cap_ok(2, 5, None) and not cap_ok(3, 3, 2))
    _, per = sublog_times(rd(SUBLOG))
    n21, n22 = per.get(datetime.date(2026, 10, 21), 0), per.get(datetime.date(2026, 10, 22), 0)
    t('real submits.log: 10-21 UTC = %d (5 A22 lines expected); 10-22 UTC = %d (<= 5)' % (n21, n22), n21 == 5 and n22 <= 5)
    t('description prefixes unique, A24 only, each <= 100 chars', len({pref(k) for k in ORDER}) == 3
      and all(COPIES[k].startswith('A24-R6PW20C') for k in ORDER) and all(len(COPIES[k]) <= 100 for k in ORDER))
    off = datetime.datetime.now().astimezone().utcoffset()
    t('machine clock zone is IST (+05:30): %s' % off, off == IST_UTC)
    c = rd(CMD)
    t('wrapper tick_a24.cmd: runs this script, logs to final\\a24, no -I', 'session_2026-10-22\\final\\tick_a24.py' in c
      and 'session_2026-10-22\\final\\a24' in c and ' -I ' not in c)
    t('constants: slug %s v%d, CLOSE %s, GAP %s, MAX default %d' % (SLUG, VER, CLOSE, GAP, MAX_DEFAULT),
      VER == 1 and CLOSE == dt(14, 30) and GAP == datetime.timedelta(minutes=30) and MAX_DEFAULT == 5)
    say('INFO PREREG AMENDMENT 24 present: %s | GO file present: %s (%s) | A23 GO conflicts: %s -> a live tick %s' % (prereg_ok(),
        os.path.exists(GO), GO, conflicts() or 'none', 'WOULD ACT' if (prereg_ok() and parse_go(rd(GO))[0] and not conflicts()) else 'does nothing'), SL)
    say('A24_TICK_SELFTEST %s %d/%d' % ('PASS' if all(res) else 'FAIL', sum(res), len(res)), SL)

if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    elif '--armcheck' in sys.argv:
        sys.exit(0 if armcheck() else 1)
    elif '--dry' in sys.argv:      # e.g. --dry 2026-10-22T09:40:00  (READ-ONLY API calls, never submits)
        i = sys.argv.index('--dry')
        main(datetime.datetime.fromisoformat(sys.argv[i + 1]) if len(sys.argv) > i + 1 else datetime.datetime.now())
    else:
        main()
