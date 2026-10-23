# ops_tick_a23_1022.py (10-22 ~09:05 IST, Claude, final track): COPY of session_2026-10-21/final/ops_tick_a23.py with the 10-22 DAYTIME
# window (c1 09:30, c2 10:00, c3 10:30 IST, GAP 30 min, CLOSE 13:30 IST) and default MAX 5 (the cap counts EVERY sub on the UTC day,
# including the 2 public-push entries). GO file = session_2026-10-22/final/A23_GO.txt. Logic otherwise byte-identical.
# DRAFT. NOT REGISTERED, NEVER RUN LIVE BY ITS AUTHOR (only --selftest, offline).
# ONE idempotent tick that resubmits r6pw20 v1 UNCHANGED (eeshsaxena/rsna-sub-v224r6flips6fpw20j, kernel_version 1) as up to 3 copies
# on the 10-22 UTC day. No push, no commit, no output fetch: the version is the one gated PASS 17/17 on 10-21 09:00:50 IST.
# Pattern = val/tick_a22.py, with the two porting traps removed:
#   (1) tick_a22 dedupes on the SLUG in submits.log and on the 'A22-R6PW20:' description; both already exist, so a naive port skips
#       forever. Here dedupe is per COPY: Kaggle description prefix 'A23-R6PW20C<n>:' or 'A23 c<n> SUBMITTED' in submits.log.
#   (2) tick_a22 pushes when its own dir has no push log; a naive port with a new dir would push v2 (new GPU commit, new gate).
#       Here there is no push stage at all and the version is pinned to 1.
# Guards (all must hold before a submit):
#   G1 GO file (GO below), first non-empty line 'GO c1 [c2 [c3]] [MAX=n] <free text>'; unknown key -> whole GO invalid.
#   G2 session_2026-10-15/PREREG_10-15.md contains 'AMENDMENT 23'.
#   G3 local kernel dir sha == registry (nb e4d92c0a, meta 3a5b8b0f); val/a22 push log says 'Kernel version 1'; val/a22 gate file's
#      last line is 'PROBE_CHECK PASS 17/17'.
#   G4 api.kernels_status(slug) is COMPLETE (anything else = someone pushed again -> log STATUS_ODD once and submit nothing).
#   G5 per-copy not-before (IST, this machine's clock): c1 09:30:00, c2 10:00, c3 10:30, and each copy >= 30 min after the previous
#      copy's logged submit; global CLOSE 10-22 13:30:00 IST. Copies go strictly in order c1 -> c2 -> c3.
#   G6 dedupe as in (1), checked against a FRESH Kaggle list on every tick (if the list call fails, nothing is submitted).
#   G7 cap: Kaggle limits endpoint num_allowed_now >= 1 (if readable) AND used this UTC day < MAX (API dates + submits.log).
#   G8 at most ONE submit per tick.
# Usage: python ops_tick_a23.py            (live; needs the Kaggle API)  |  python ops_tick_a23.py --selftest   (offline: no Kaggle import,
#        no network, writes only final/ops_a23_selftest.log)
import os, sys, json, time, datetime, subprocess, hashlib, re
F = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(F, '..', '..'))
D = os.path.join(F, 'a23'); LOG = os.path.join(D, 'tick_a23.log'); LOCK = os.path.join(D, 'tick.lock')
GO = os.path.join(F, 'A23_GO.txt')                          # main may move this; keep it next to the tick
SUBLOG = os.path.join(R, 'session_2026-10-14', 'submits.log')
PREREG = os.path.join(R, 'session_2026-10-15', 'PREREG_10-15.md')
A22 = os.path.join(R, 'session_2026-10-20', 'val', 'a22')
COMP = 'rsna-knee-abnormality-detection'
SLUG = 'eeshsaxena/rsna-sub-v224r6flips6fpw20j'; VER = 1; TAG = 'v224r6flips6fpw20j'
KDIR = os.path.join(R, 'session_2026-10-20', 'val', 'kernels', 'kernel_' + TAG)
NB_SHA = 'e4d92c0a2c581a8301f87ae4779da21870664814b4af7068dcca55014bbc7757'
META_SHA = '3a5b8b0f6df5fb91255179b5f0aa0b5425f768e88d7d0eff47eb36fd01eba9f8'
PUSHLOG = os.path.join(A22, 'push_%s.log' % TAG); GATE = os.path.join(A22, 'gate_%s.txt' % TAG); GATE_PASS = 'PROBE_CHECK PASS 17/17'
IST_UTC = datetime.timedelta(hours=5, minutes=30)
COPIES = {   # key -> (not-before IST, Kaggle description; prefix up to ':' is the dedupe key)
    'c1': (datetime.datetime(2026, 10, 22, 9, 30, 0), 'A23-R6PW20C1: resubmit r6pw20 v1 unchanged (57463719 hit the S6F canary .916); B hedge copy 1'),
    'c2': (datetime.datetime(2026, 10, 22, 10, 0, 0), 'A23-R6PW20C2: resubmit r6pw20 v1 unchanged (57463719 hit the S6F canary .916); B hedge copy 2'),
    'c3': (datetime.datetime(2026, 10, 22, 10, 30, 0), 'A23-R6PW20C3: resubmit r6pw20 v1 unchanged (57463719 hit the S6F canary .916); B hedge copy 3'),
}
ORDER = ['c1', 'c2', 'c3']; GAP = datetime.timedelta(minutes=30)
CLOSE = datetime.datetime(2026, 10, 22, 13, 30, 0)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
rd = lambda p: open(p, encoding='utf-8', errors='replace').read() if os.path.exists(p) else ''

def say(m, log=None):
    line = '[%s] %s' % (datetime.datetime.now().strftime('%m-%d %H:%M:%S'), m)
    print(line, flush=True)
    os.makedirs(os.path.dirname(log or LOG), exist_ok=True); open(log or LOG, 'a', encoding='utf-8').write(line + '\n')

def parse_go(text):
    """-> (keys in ORDER, maxday, err). keys None when the GO is absent / invalid."""
    ls = [l.strip() for l in (text or '').splitlines() if l.strip()]
    if not ls or ls[0].split()[0] != 'GO':
        return None, 5, 'no GO line'
    keys, mx = [], 5
    for t in ls[0].split()[1:]:
        if t.startswith('MAX='):
            try: mx = max(0, min(5, int(t[4:])))
            except ValueError: return None, 3, 'bad %s' % t
        elif t in COPIES:
            if t not in keys: keys.append(t)
        elif re.fullmatch(r'[a-z0-9]+', t):
            return None, 3, 'unknown key %s' % t
        else:
            break
    keys = [k for k in ORDER if k in keys]
    if keys and keys != ORDER[:len(keys)]:
        return None, 3, 'copies must be a prefix of c1 c2 c3'
    return (keys or None), mx, (None if keys else 'no keys')

def prereg_ok():
    return 'AMENDMENT 23' in rd(PREREG).upper()

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
    """{key: IST datetime} of 'A23 cN SUBMITTED' lines, plus the count of SUBMITTED lines per UTC day."""
    t, per = {}, {}
    for l in (text or '').splitlines():
        m = re.match(r'\[(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) IST\](.*)', l)
        if not m or 'SUBMITTED' not in l: continue
        when = datetime.datetime(2026, *map(int, m.groups()[:5]))
        per[utc_day(when)] = per.get(utc_day(when), 0) + 1
        k = re.search(r'\bA23 (c[123]) SUBMITTED', l)
        if k: t.setdefault(k.group(1), when)
    return t, per

def count_api_day(subs, now):
    n, seen = 0, False
    for s in subs or []:
        d = getattr(s, 'date', None)
        if d is None: continue
        try:
            if not isinstance(d, datetime.datetime): d = datetime.datetime.fromisoformat(str(d).replace('Z', '+00:00'))
            if d.tzinfo is not None: d = d.astimezone(datetime.timezone.utc).replace(tzinfo=None)
            seen = True; n += d.date() == utc_day(now)
        except Exception:
            continue
    return n if seen else None

def done_keys(descs, sublog):
    """copies already on Kaggle (description prefix) or in submits.log."""
    t, _ = sublog_times(sublog)
    out = set(t)
    for k in ORDER:
        pref = COPIES[k][1].split(':')[0] + ':'
        if any((d or '').startswith(pref) for d in descs or []): out.add(k)
    return out

def next_copy(keys, done, sub_t, now):
    """-> key to submit now, or None. Strict order; per-copy not-before; >= GAP after the previous copy's logged submit."""
    if now > CLOSE: return None
    prev = None
    for k in keys:
        if k in done:
            prev = sub_t.get(k, prev); continue
        nb = COPIES[k][0]
        if prev is not None: nb = max(nb, prev + GAP)
        return k if now >= nb else None
    return None

def main(dry_at=None):
    """dry_at: an IST datetime -> DRY RUN: pretend GO 'c1 c2 c3' + AMENDMENT 23 exist and that it is dry_at; do every READ-ONLY
    API call (status, list, limits) and print the decision. Never submits, never writes submits.log / the tick log / the lock."""
    dry = dry_at is not None
    now = dry_at or datetime.datetime.now()
    out = print if dry else say
    if not dry: os.makedirs(D, exist_ok=True)
    if now > CLOSE:
        print('past CLOSE %s: nothing to do' % CLOSE, flush=True); return
    keys, mx, err = (['c1', 'c2', 'c3'], 5, None) if dry else parse_go(rd(GO))
    if keys is None:
        print('no valid GO (%s): nothing to do' % err, flush=True); return
    if not dry and not prereg_ok():
        say('GO %s present but PREREG has no AMENDMENT 23: nothing submitted' % keys); return
    ok, why = static_ok()
    if not ok:
        out('STATIC_FAIL %s: nothing submitted' % why); return
    if now < COPIES[keys[0]][0]:
        print('before %s: nothing to do' % COPIES[keys[0]][0], flush=True); return
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
            subs = list(api.competition_submissions(COMP, page_size=50)); descs = [s.description or '' for s in subs]
        except Exception as ex:
            out('list err %r -> retry next tick' % ex); return
        sublog = rd(SUBLOG)
        sub_t, per = sublog_times(sublog)
        done = done_keys(descs, sublog)
        k = next_copy(keys, done, sub_t, now if dry else datetime.datetime.now())
        used = max(per.get(utc_day(now), 0), count_api_day(subs, now) or 0)
        allowed = None
        try: allowed = int(api.competition_get_submission_limits(COMP).num_allowed_now)
        except Exception: pass
        if dry:
            print('DRY at %s IST: kernel status %s | copies done %s | next copy %s | used on UTC day %s = %d (MAX %d) | '
                  'num_allowed_now (live, real clock) %s | static %s' % (now, st, sorted(done) or '-', k, utc_day(now), used, mx, allowed, why))
            return
        if k is None:
            return
        if used >= mx or allowed == 0:
            say('CAP %s: used %d (MAX %d), num_allowed_now %s -> not submitted' % (k, used, mx, allowed)); return
        desc = COPIES[k][1]
        try:
            r = api.competition_submit_code(file_name='submission.csv', message=desc, competition=COMP, kernel=SLUG, kernel_version=VER)
            line = '[%s IST] A23 %s SUBMITTED %s v%d: %s' % (datetime.datetime.now().strftime('%m-%d %H:%M:%S'), k, SLUG, VER, r)
            open(SUBLOG, 'a', encoding='utf-8').write(line + '\n'); say(line)
        except Exception as ex:
            resp = getattr(ex, 'response', None); body = resp.text[:300] if resp is not None else ''
            say('%s SUBMIT ERR %s %s %s -> retry next tick (the fresh list dedupes a submit that did land)' % (k, type(ex).__name__, str(ex)[:150], body))
    finally:
        if not dry:
            try: os.remove(LOCK)
            except Exception: pass

def selftest():
    SL = os.path.join(F, 'ops_a23_selftest.log'); open(SL, 'w').close(); res = []
    def t(name, cond):
        res.append(bool(cond)); say(('OK   ' if cond else 'FAIL ') + name, SL)
    ok, why = static_ok()
    t('G3 static: kernel sha e4d92c0a / meta 3a5b8b0f, A22 push log = v1 only, gate last line PASS 17/17 (%s)' % why, ok)
    t('GO "GO c1 c2 MAX=3 Eesh 10-21 23:00" -> [c1, c2], MAX 3', parse_go('GO c1 c2 MAX=3 Eesh 10-21 23:00')[:2] == (['c1', 'c2'], 3))
    t('GO "GO c1 c2 c3" -> [c1, c2, c3], MAX 5 (default)', parse_go('GO c1 c2 c3')[:2] == (['c1', 'c2', 'c3'], 5))
    t('GO "GO c1 c2 c3 MAX=5 Eesh 10-22 09:20" -> [c1, c2, c3], MAX 5', parse_go('GO c1 c2 c3 MAX=5 Eesh 10-22 09:20')[:2] == (['c1', 'c2', 'c3'], 5))
    t('GO typo "GO c1 cc2" -> invalid', parse_go('GO c1 cc2')[0] is None)
    t('GO "GO c2" (skips c1) -> invalid', parse_go('GO c2')[0] is None)
    t('GO lower-case / empty -> invalid', parse_go('go c1')[0] is None and parse_go('')[0] is None and parse_go('GO')[0] is None)
    a22 = '[10-21 09:00:53 IST] A22 r6pw20 SUBMITTED %s v1: {"ref": 57463719}' % SLUG
    t('dedupe: the A22 line for the same slug does NOT block c1', 'c1' not in done_keys(['A22-R6PW20: B hedge'], a22))
    t('dedupe: Kaggle desc "A23-R6PW20C1: ..." blocks c1 only', done_keys(['A23-R6PW20C1: x'], '') == {'c1'})
    t('dedupe: submits.log "A23 c2 SUBMITTED" blocks c2 only', done_keys([], '[10-22 06:15:10 IST] A23 c2 SUBMITTED x v1: {}') == {'c2'})
    K = ['c1', 'c2', 'c3']
    t('window: 09:29:59 -> none, 09:30:00 -> c1', next_copy(K, set(), {}, datetime.datetime(2026, 10, 22, 9, 29, 59)) is None
      and next_copy(K, set(), {}, datetime.datetime(2026, 10, 22, 9, 30)) == 'c1')
    s1 = {'c1': datetime.datetime(2026, 10, 22, 9, 30, 20)}
    t('order: c1 done 09:30:20 -> c2 not at 10:00:19, c2 at 10:00:20', next_copy(K, {'c1'}, s1, datetime.datetime(2026, 10, 22, 10, 0, 19)) is None
      and next_copy(K, {'c1'}, s1, datetime.datetime(2026, 10, 22, 10, 0, 20)) == 'c2')
    late = {'c1': datetime.datetime(2026, 10, 22, 11, 40)}
    t('late arm: c1 at 11:40 -> c2 not before 12:10', next_copy(K, {'c1'}, late, datetime.datetime(2026, 10, 22, 12, 9)) is None
      and next_copy(K, {'c1'}, late, datetime.datetime(2026, 10, 22, 12, 10)) == 'c2')
    t('strict order: c1 not done -> never c2 even at 10:45', next_copy(K, set(), {}, datetime.datetime(2026, 10, 22, 10, 45)) == 'c1')
    t('CLOSE: 13:30:00 open, 13:30:01 closed', next_copy(K, set(), {}, datetime.datetime(2026, 10, 22, 13, 30, 0)) == 'c1'
      and next_copy(K, set(), {}, datetime.datetime(2026, 10, 22, 13, 30, 1)) is None)
    t('GO [c1, c2] never yields c3', next_copy(['c1', 'c2'], {'c1', 'c2'}, {'c1': datetime.datetime(2026, 10, 22, 9, 30), 'c2': datetime.datetime(2026, 10, 22, 10, 0)}, datetime.datetime(2026, 10, 22, 12, 0)) is None)
    t('UTC day: 10-22 05:29:59 IST -> 10-21 UTC; 05:30:00 -> 10-22 UTC', utc_day(datetime.datetime(2026, 10, 22, 5, 29, 59)) == datetime.date(2026, 10, 21)
      and utc_day(datetime.datetime(2026, 10, 22, 5, 30)) == datetime.date(2026, 10, 22))
    _, per = sublog_times(rd(SUBLOG))
    t('cap from the real submits.log: 10-21 UTC = %d (5 A22), 10-22 UTC = %d (0)' % (per.get(datetime.date(2026, 10, 21), 0), per.get(datetime.date(2026, 10, 22), 0)),
      per.get(datetime.date(2026, 10, 21), 0) == 5 and per.get(datetime.date(2026, 10, 22), 0) == 0)
    class _S:
        def __init__(s, d): s.date = d
    t('API date count: 2 of 3 on 10-22 UTC', count_api_day([_S(datetime.datetime(2026, 10, 22, 0, 1)), _S('2026-10-22T01:00:00Z'), _S('2026-10-21T23:59:00Z')], datetime.datetime(2026, 10, 22, 9, 0)) == 2)
    t('description prefixes are unique and contain no A22 prefix', len({COPIES[k][1].split(':')[0] for k in ORDER}) == 3 and all(not COPIES[k][1].startswith('A22') for k in ORDER))
    t('PREREG guard currently %s (no AMENDMENT 23 yet -> a live tick would do nothing)' % prereg_ok(), prereg_ok() is False)
    t('GO file currently absent (%s)' % GO, not os.path.exists(GO))
    say('A23_TICK_SELFTEST %s %d/%d' % ('PASS' if all(res) else 'FAIL', sum(res), len(res)), SL)

if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    elif '--dry' in sys.argv:      # e.g. --dry 2026-10-22T05:31:00  (READ-ONLY API calls, never submits)
        i = sys.argv.index('--dry')
        main(datetime.datetime.fromisoformat(sys.argv[i + 1]) if len(sys.argv) > i + 1 else datetime.datetime(2026, 10, 22, 9, 30))
    else:
        main()
