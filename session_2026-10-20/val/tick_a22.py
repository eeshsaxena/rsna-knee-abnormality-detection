# tick_a22.py (10-20 ~19:40 IST, Claude, Track C). WRITTEN, NOT REGISTERED, NEVER RUN LIVE BY ITS AUTHOR (only --selftest, offline).
# ONE idempotent tick of an A22 slate chosen by Eesh. Pattern = a21/tick_a21.py (push -> COMPLETE -> fetch 502-safe -> gate -> submit once).
# Guards (all must hold before anything is pushed or submitted):
#   G1 GO file session_2026-10-20/val/A22_GO.txt, first non-empty line: 'GO <key> [<key> ...] [MAX=n]', keys from val/slate_a22.json
#      (pw35 pw40 pw45 r6pw30 r6e20). Any unknown key -> the whole GO is invalid. Order = push / submit priority.
#   G2 session_2026-10-15/PREREG_10-15.md contains 'AMENDMENT 22' (the amendment is written by main, before any push).
#   G3 the kernel dir's notebook + metadata sha256 == the registry (no edits after the offline gate).
#   G4 push only when < 2 GPU commits of the account run (running.py); a push writes a22/push_<tag>.log and records the version.
#   G5 gate = the registry's exact gate argv on the fetched output; submit only on 'PROBE_CHECK PASS 17/17', at most once per entry
#      (desc prefix 'A22-XXX:' on Kaggle or the slug in submits.log), the version that was pushed.
#   G6 submit window: 10-21 05:31 IST (00:01 UTC) .. 10-22 05:20 IST (end of the 10-21 UTC day); pushes are allowed as soon as G1-G3
#      hold, so commits finish before 05:31. Daily cap: MAX (default 5) minus submissions already made this UTC day (Kaggle dates and
#      submits.log, the larger count).
# Usage: python tick_a22.py              (live; needs the Kaggle API)  |  python tick_a22.py --selftest   (offline, no network, no writes
#        outside val/a22_selftest.log)
import os, sys, json, time, datetime, subprocess, shutil, hashlib, re
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
V = os.path.join(R, 'session_2026-10-20', 'val'); D = os.path.join(V, 'a22')
LOG = os.path.join(D, 'tick_a22.log'); LOCK = os.path.join(D, 'tick.lock')
SUBLOG = os.path.join(R, 'session_2026-10-14', 'submits.log'); GO = os.path.join(V, 'A22_GO.txt')
PREREG = os.path.join(R, 'session_2026-10-15', 'PREREG_10-15.md'); SLATE = os.path.join(V, 'slate_a22.json')
OPEN = datetime.datetime(2026, 10, 21, 5, 31); CLOSE = datetime.datetime(2026, 10, 22, 5, 20)   # IST (this machine's clock)
IST_UTC = datetime.timedelta(hours=5, minutes=30); COMP = 'rsna-knee-abnormality-detection'
PY312 = ['py', '-3.12', '-I']
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()

def say(m, log=None):
    line = '[%s] %s' % (datetime.datetime.now().strftime('%m-%d %H:%M:%S'), m)
    print(line, flush=True)
    os.makedirs(os.path.dirname(log or LOG), exist_ok=True); open(log or LOG, 'a', encoding='utf-8').write(line + '\n')

def load_slate():
    return json.load(open(SLATE, encoding='utf-8'))['entries']

def parse_go(text, slate):
    """-> (keys, maxday, err). keys None when the GO is absent / invalid."""
    ls = [l.strip() for l in (text or '').splitlines() if l.strip()]
    if not ls or ls[0].split()[0] != 'GO':
        return None, 5, 'no GO line'
    keys, mx = [], 5
    for t in ls[0].split()[1:]:
        if t.startswith('MAX='):
            try: mx = max(0, min(5, int(t[4:])))
            except ValueError: return None, 5, 'bad %s' % t
        elif t in slate:
            if t not in keys: keys.append(t)
        elif re.fullmatch(r'[a-z0-9]+', t):
            return None, 5, 'unknown key %s' % t      # a typo must not silently drop or swap an entry
        else:
            break                                      # free text after the keys (e.g. 'Eesh 10-21 00:10 ...')
    return (keys or None), mx, (None if keys else 'no keys')

def prereg_ok():
    try: return 'AMENDMENT 22' in open(PREREG, encoding='utf-8', errors='replace').read().upper()
    except Exception: return False

def kernel_ok(e):
    kd = os.path.join(R, e['kernel_dir'])
    try:
        return (sha(os.path.join(kd, 'rsna_sub_%s.ipynb' % e['tag'])) == e['nb_sha256'] and sha(os.path.join(kd, 'kernel-metadata.json')) == e['meta_sha256']
                and json.load(open(os.path.join(kd, 'kernel-metadata.json')))['id'] == e['slug'])
    except Exception:
        return False

def utc_day(ist):
    return (ist - IST_UTC).date()

def count_sublog_day(text, now):
    """submissions in submits.log on now's UTC day (lines '[MM-DD HH:MM:SS IST] ... SUBMITTED ...')."""
    n = 0
    for l in (text or '').splitlines():
        m = re.match(r'\[(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) IST\].*SUBMITTED', l)
        if m:
            t = datetime.datetime(2026, *map(int, m.groups()))
            n += utc_day(t) == utc_day(now)
    return n

def count_api_day(subs, now):
    """submissions on now's UTC day from the Kaggle API dates (None if no date can be read)."""
    n, seen = 0, False
    for s in subs or []:
        d = getattr(s, 'date', None) or getattr(s, 'date_submitted', None)
        if d is None: continue
        try:
            if not isinstance(d, datetime.datetime): d = datetime.datetime.fromisoformat(str(d).replace('Z', '+00:00'))
            if d.tzinfo is not None: d = d.astimezone(datetime.timezone.utc).replace(tzinfo=None)
            seen = True; n += d.date() == utc_day(now)
        except Exception:
            continue
    return n if seen else None

def gate_argv(e, out):
    return PY312 + [os.path.join(R, e['gate_script']), out] + e['gate_args']

def run(cmd, timeout=900):
    try:
        p = subprocess.run(cmd, cwd=R, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
        return p.returncode, (p.stdout or '') + (p.stderr or '')
    except Exception as ex:
        return 99, repr(ex)

def main():
    now = datetime.datetime.now()
    os.makedirs(D, exist_ok=True)
    if now > CLOSE:
        say('past CLOSE %s, nothing to do' % CLOSE); return
    slate = load_slate()
    keys, mx, err = parse_go(open(GO, encoding='utf-8', errors='replace').read() if os.path.exists(GO) else '', slate)
    if keys is None:
        print('no valid GO (%s): nothing to do' % err, flush=True); return
    if not prereg_ok():
        say('GO %s present but PREREG_10-15.md has no AMENDMENT 22: nothing pushed or submitted' % keys); return
    if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) < 50 * 60:
        print('locked', flush=True); return
    open(LOCK, 'w').write(str(os.getpid()))
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        rc, out = run([sys.executable, os.path.join(R, 'session_2026-10-15', 'v224f', 'running.py'), '6'], 300)
        running = 9
        for l in out.splitlines():
            if l.startswith('RUNNING_COUNT'):
                try: running = int(l.split()[1])
                except Exception: pass
        try:
            subs = list(api.competition_submissions(COMP)); descs = [s.description or '' for s in subs]
        except Exception as ex:
            say('list err %r' % ex); subs, descs = None, None
        sublog = open(SUBLOG, encoding='utf-8', errors='replace').read() if os.path.exists(SUBLOG) else ''
        ca = count_api_day(subs, now) if subs is not None else None
        used = max(count_sublog_day(sublog, now), ca or 0)
        logtxt = open(LOG, encoding='utf-8', errors='replace').read() if os.path.exists(LOG) else ''
        for key in keys:
            e = slate[key]; tag, slug, desc = e['tag'], e['slug'], e['desc']
            pushlog = os.path.join(D, 'push_%s.log' % tag); gate = os.path.join(D, 'gate_%s.txt' % tag); out_dir = os.path.join(D, 'out_%s' % tag)
            pref = desc.split(':')[0] + ':'
            if ('rsna-sub-' + tag + ' v') in sublog or (descs is not None and any(d.startswith(pref) for d in descs)):
                continue
            if not kernel_ok(e):
                if ('SHA_MISMATCH_' + key) not in logtxt: say('SHA_MISMATCH_%s: %s differs from the registry -> skipped' % (key, e['kernel_dir']))
                continue
            # 1. push (once)
            ptxt = open(pushlog, encoding='utf-8', errors='replace').read() if os.path.exists(pushlog) else ''
            m = re.search(r'Kernel version (\d+) successfully pushed', ptxt)
            if not m:
                if running >= 2:
                    continue
                rc, out = run([sys.executable, '-m', 'kaggle', 'kernels', 'push', '-p', os.path.join(R, e['kernel_dir'])], 600)
                open(pushlog, 'w', encoding='utf-8').write(out)
                say('%s push rc %s: %s' % (key, rc, out.strip().splitlines()[-1:] if out.strip() else ''))
                running += 1
                continue
            ver = int(m.group(1))
            # 2. gate once the commit is COMPLETE
            glines = open(gate, encoding='utf-8', errors='replace').read().strip().splitlines() if os.path.exists(gate) else []
            gate_last = glines[-1] if glines else ''
            if 'PROBE_CHECK' not in gate_last:
                try: st = str(api.kernels_status(slug).status)
                except Exception as ex: st = 'ERR %r' % ex
                if 'COMPLETE' not in st:
                    if ('ERROR' in st or 'CANCEL' in st) and ('COMMIT_BAD_' + key) not in logtxt:
                        say('COMMIT_BAD_%s status %s -> no gate, no submit' % (key, st))
                    continue
                tmp = out_dir + '.tmp'; shutil.rmtree(tmp, ignore_errors=True)
                rc, out = run([sys.executable, os.path.join(R, 'session_2026-10-15', 'v224f', 'wait_fetch.py'), slug, tmp, '20'], 1800)
                bad = []
                for root, _, files in os.walk(tmp):
                    for f in files:
                        try:
                            if b'Server Error' in open(os.path.join(root, f), 'rb').read(600): bad.append(f)
                        except Exception: bad.append(f)
                if bad or not os.path.isdir(tmp):
                    say('%s fetch incomplete (%d bad files) -> retry next tick' % (key, len(bad))); continue
                shutil.rmtree(out_dir, ignore_errors=True); os.replace(tmp, out_dir)
                rc, out = run(gate_argv(e, out_dir), 900)
                open(gate, 'w', encoding='utf-8').write(out)
                gate_last = out.strip().splitlines()[-1] if out.strip() else ''
                say('%s gate (v%d): %s' % (key, ver, gate_last[:140]))
            # 3. submit on PASS only, inside the window, under the daily cap
            if e['gate_pass'] not in gate_last:
                if 'PROBE_CHECK' in gate_last and ('GATE_FAIL_' + key) not in logtxt:
                    say('GATE_FAIL_%s -> not submitted' % key); logtxt += 'GATE_FAIL_' + key
                continue
            if descs is None or not (OPEN <= datetime.datetime.now() <= CLOSE):
                continue
            if used >= mx:
                if ('CAP_' + key) not in logtxt: say('CAP_%s: %d submissions already this UTC day (MAX %d) -> not submitted' % (key, used, mx)); logtxt += 'CAP_' + key
                continue
            try:
                r = api.competition_submit_code(file_name='submission.csv', message=desc, competition=COMP, kernel=slug, kernel_version=ver)
                line = '[%s IST] A22 %s SUBMITTED %s v%d: %s' % (datetime.datetime.now().strftime('%m-%d %H:%M:%S'), key, slug, ver, r)
                open(SUBLOG, 'a', encoding='utf-8').write(line + '\n'); say(line); sublog += line; used += 1
            except Exception as ex:
                resp = getattr(ex, 'response', None); body = resp.text[:300] if resp is not None else ''
                say('%s SUBMIT ERR %s %s %s' % (key, type(ex).__name__, str(ex)[:150], body))
    finally:
        try: os.remove(LOCK)
        except Exception: pass

def selftest():
    """Offline: no Kaggle import, no network, no push / submit. Writes only val/a22_selftest.log."""
    SL = os.path.join(V, 'a22_selftest.log'); open(SL, 'w').close(); res = []
    def t(name, cond):
        res.append(bool(cond)); say(('OK   ' if cond else 'FAIL ') + name, SL)
    slate = load_slate()
    for k, e in slate.items():
        t('registry %s: kernel %s sha %s + metadata match, id %s' % (k, e['kernel_dir'], e['nb_sha256'][:8], e['slug']), kernel_ok(e))
    t('GO parse: "GO pw35 r6pw30" -> [pw35, r6pw30], MAX 5', parse_go('GO pw35 r6pw30', slate)[:2] == (['pw35', 'r6pw30'], 5))
    t('GO parse: "GO r6pw30 MAX=2 Eesh 10-21 00:10" -> [r6pw30], MAX 2', parse_go('GO r6pw30 MAX=2 Eesh 10-21 00:10', slate)[:2] == (['r6pw30'], 2))
    t('GO parse: typo "GO pw53" -> invalid', parse_go('GO pw53', slate)[0] is None)
    t('GO parse: "go pw35" (lower case) -> invalid', parse_go('go pw35', slate)[0] is None)
    t('GO parse: empty / missing -> invalid', parse_go('', slate)[0] is None and parse_go('GO', slate)[0] is None)
    t('window: 10-21 05:30:59 IST closed, 05:31:00 open, 10-22 05:20 open, 05:20:01 closed',
      not (OPEN <= datetime.datetime(2026, 10, 21, 5, 30, 59) <= CLOSE) and OPEN <= datetime.datetime(2026, 10, 21, 5, 31) <= CLOSE
      and OPEN <= datetime.datetime(2026, 10, 22, 5, 20) <= CLOSE and not (OPEN <= datetime.datetime(2026, 10, 22, 5, 20, 1) <= CLOSE))
    t('UTC day: 10-21 05:31 IST -> 10-21 UTC; 10-21 05:29 IST -> 10-20 UTC',
      utc_day(datetime.datetime(2026, 10, 21, 5, 31)) == datetime.date(2026, 10, 21) and utc_day(datetime.datetime(2026, 10, 21, 5, 29)) == datetime.date(2026, 10, 20))
    sl = open(SUBLOG, encoding='utf-8', errors='replace').read()
    c20, c21 = count_sublog_day(sl, datetime.datetime(2026, 10, 20, 19, 0)), count_sublog_day(sl, datetime.datetime(2026, 10, 21, 6, 0))
    t('cap count from the real submits.log: 10-20 UTC day = %d (eoa20 eoa10 eoa30 pw20 pw30 = 5), 10-21 UTC day = %d (0)' % (c20, c21), c20 == 5 and c21 == 0)
    class _S:
        def __init__(s, d): s.date = d
    t('API date count: 2 of 3 on 10-21 UTC (naive UTC, ISO Z, and a 10-20 one)',
      count_api_day([_S(datetime.datetime(2026, 10, 21, 0, 1)), _S('2026-10-21T03:00:00Z'), _S('2026-10-20T23:59:00Z')], datetime.datetime(2026, 10, 21, 9, 0)) == 2)
    t('API date count: no readable dates -> None (falls back to submits.log)', count_api_day([_S(None)], datetime.datetime(2026, 10, 21, 9, 0)) is None)
    t('PREREG guard currently %s (AMENDMENT 22 not yet written -> a live tick would do nothing)' % prereg_ok(), prereg_ok() is False)
    t('GO file currently absent (%s)' % GO, not os.path.exists(GO))
    for k, e in slate.items():
        simd = os.path.join(V, 'sim', 'out_%s_sim' % e['tag'])
        rc, out = run(gate_argv(e, simd), 900); last = out.strip().splitlines()[-1] if out.strip() else ''
        t('gate argv %s on its simulated commit output -> %s' % (k, last), e['gate_pass'] in last and rc == 0)
    say('A22_TICK_SELFTEST %s %d/%d' % ('PASS' if all(res) else 'FAIL', sum(res), len(res)), SL)

if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    else:
        main()
