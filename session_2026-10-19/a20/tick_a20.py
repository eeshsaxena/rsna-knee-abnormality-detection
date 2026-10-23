# tick_a20.py (10-19, Claude): ONE idempotent tick of the A20 slate (E-OA eoa20 + eoa10 on c4). Eesh 10-19 ~23:40: validate offline,
# then push the 2 if strong. Push, gate and submit ALL require a20/A20_GO.txt (written only when A20_VALIDATION_PREREG.md says strong).
# Per entry: push if GO and < 2 GPU commits run -> when COMPLETE, fetch (502-safe) and gate with a20/probe_check_eoa.py --aoa ->
# submit on 'PROBE_CHECK PASS 17/17' only, after 05:31 IST 10-20, at most once. Lock file. Stops after 05:20 IST 10-21. Log a20/tick_a20.log
import os, sys, json, time, datetime, subprocess, shutil
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
D = os.path.join(R, 'session_2026-10-19', 'a20'); LOG = os.path.join(D, 'tick_a20.log'); SUBLOG = os.path.join(R, 'session_2026-10-14', 'submits.log')
LOCK = os.path.join(D, 'tick.lock'); STOP = datetime.datetime(2026, 10, 21, 5, 20)
HW = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6, "Synovitis": 0.8, "Baker's": 0.35}
C4 = {**HW, "PF OA": 0.5, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
C4T = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
SLATE = [('eoa20', 'v224r6c4eoa20j', 0.25, C4T, 'A20-EOA20: c4 best 57343435 + OA decorrelation a .2 on the final output (one change)'),
         ('eoa10', 'v224r6c4eoa10j', 0.25, C4T, 'A20-EOA10: c4 best 57343435 + OA decorrelation a .1 on the final output (one change)')]
AOA = {'eoa20': 0.2, 'eoa10': 0.1}
GO = os.path.join(D, 'A20_GO.txt')   # written only after the A20 validation rule says strong (A20_VALIDATION_PREREG.md)
PY312 = ['py', '-3.12', '-I']

def say(m):
    line = '[%s] %s' % (datetime.datetime.now().strftime('%m-%d %H:%M:%S'), m)
    print(line, flush=True); open(LOG, 'a', encoding='utf-8').write(line + '\n')

def run(cmd, timeout=900):
    try:
        p = subprocess.run(cmd, cwd=R, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or '') + (p.stderr or '')
    except Exception as e:
        return 99, repr(e)

def main():
    if datetime.datetime.now() > STOP:
        say('past STOP, nothing to do'); return
    if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) < 50 * 60:
        print('locked', flush=True); return
    open(LOCK, 'w').write(str(os.getpid()))
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        # running GPU commits of this account (same helper as every other chain)
        rc, out = run([sys.executable, os.path.join(R, 'session_2026-10-15', 'v224f', 'running.py'), '6'], 300)
        running = 9
        for l in out.splitlines():
            if l.startswith('RUNNING_COUNT'):
                try: running = int(l.split()[1])
                except Exception: pass
        try: descs = [s.description or '' for s in api.competition_submissions('rsna-knee-abnormality-detection')]
        except Exception as e: say('list err %r' % e); descs = None
        sublog = open(SUBLOG, encoding='utf-8', errors='replace').read() if os.path.exists(SUBLOG) else ''
        for key, tag, a224, table, desc in SLATE:
            slug = 'eeshsaxena/rsna-sub-' + tag
            pushlog = os.path.join(D, 'push_%s.log' % tag); gate = os.path.join(D, 'gate_%s_v1.txt' % tag); out_dir = os.path.join(D, 'out_%s_v1' % tag)
            submitted = ('rsna-sub-' + tag) in sublog or (descs is not None and any(d.startswith(desc.split(':')[0] + ':') for d in descs))
            if submitted:
                continue
            # 1. push
            pushed = os.path.exists(pushlog) and 'successfully pushed' in open(pushlog, encoding='utf-8', errors='replace').read()
            if not pushed:
                if not os.path.exists(GO):
                    continue
                if running >= 2:
                    continue
                rc, out = run([sys.executable, '-m', 'kaggle', 'kernels', 'push', '-p', os.path.join(D, 'kernel_' + tag)], 600)
                open(pushlog, 'w', encoding='utf-8').write(out)
                say('%s push rc %s: %s' % (key, rc, out.strip().splitlines()[-1:] if out.strip() else ''))
                running += 1
                continue
            # 2. gate (once the commit is COMPLETE)
            gate_last = ''
            if os.path.exists(gate):
                ls = open(gate, encoding='utf-8', errors='replace').read().strip().splitlines()
                gate_last = ls[-1] if ls else ''
            if 'PROBE_CHECK' not in gate_last:
                try:
                    st = str(api.kernels_status(slug).status)
                except Exception as e:
                    st = 'ERR %r' % e
                if 'COMPLETE' not in st:
                    if 'ERROR' in st or 'CANCEL' in st:
                        say('%s commit status %s -> no gate, no submit' % (key, st))
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
                rc, out = run(PY312 + [os.path.join(D, 'probe_check_eoa.py'), out_dir,
                                       '--a224', str(a224), '--table', json.dumps(table), '--aoa', str(AOA[key])], 900)
                open(gate, 'w', encoding='utf-8').write(out)
                gate_last = out.strip().splitlines()[-1] if out.strip() else ''
                say('%s gate: %s' % (key, gate_last[:140]))
            # 3. submit on PASS only
            if 'PASS 17/17' not in gate_last:
                if 'PROBE_CHECK' in gate_last and 'FAIL' in gate_last and ('GATE_FAIL_' + key) not in open(LOG, encoding='utf-8', errors='replace').read():
                    say('GATE_FAIL_%s -> not submitted' % key)
                continue
            if descs is None or not os.path.exists(GO):
                continue
            if datetime.datetime.now() < datetime.datetime(2026, 10, 20, 5, 31):
                continue
            try:
                r = api.competition_submit_code(file_name='submission.csv', message=desc, competition='rsna-knee-abnormality-detection',
                                                kernel=slug, kernel_version=1)
                line = '[%s IST] A20 %s SUBMITTED %s v1: %s' % (datetime.datetime.now().strftime('%m-%d %H:%M:%S'), key, slug, r)
                open(SUBLOG, 'a', encoding='utf-8').write(line + '\n'); say(line)
                sublog += line
            except Exception as e:
                resp = getattr(e, 'response', None); body = resp.text[:300] if resp is not None else ''
                say('%s SUBMIT ERR %s %s %s' % (key, type(e).__name__, str(e)[:150], body))
    finally:
        try: os.remove(LOCK)
        except Exception: pass

if __name__ == '__main__':
    main()
