#!/usr/bin/env python
# test_ta_driver.py (2026-10-15, Claude subagent): offline tests of compact_infer_ta.py (the time-aware flip driver) against a
# STUB compact_infer.py that speaks the pinned worker's file protocol (predictions.csv flushed per row, progress json lines,
# FLIP_TTA_READY, --deadline, receipt.json). Runs on Windows: the test copy of the driver replaces only the CHILD_SHA constant
# (the stub's sha) and, where os.killpg does not exist, kill_group (taskkill /T /F); every decision / merge line is the real one.
# The POSIX kill path itself is exercised on Kaggle by the forced-time-pressure control commit.
#   usage: python test_ta_driver.py            -> prints TA_TEST PASS n/m (exit 0) or FAIL
import os, sys, json, csv, time, shutil, hashlib, subprocess, tempfile, ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = HERE / 'compact_infer_ta.py'
HER_NB = HERE.parent.parent / 'session_2026-10-14' / 'v224' / 'src' / 'varshitha_v224.ipynb'
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
          "Baker's", "Contusion", "Fracture"]
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print(f'  {"OK  " if ok else "FAIL"} {name} {info}', flush=True)


STUB = r'''
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
LABELS = %r
def val(uid, lab, flip):
    h = hashlib.md5(("%%s|%%s|%%d" %% (uid, lab, int(flip))).encode()).hexdigest()
    return (int(h[:8], 16) + 1) / (16 ** 8 + 2)
ap = argparse.ArgumentParser()
for o in ("--data", "--split", "--code", "--arm", "--assignment", "--output", "--amp", "--workers", "--models", "--backbone"):
    ap.add_argument(o)
ap.add_argument("--deadline", type=float, default=0.)
ap.add_argument("--flip-tta", action="store_true")
a, _ = ap.parse_known_args()
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
uids = json.loads(Path(a.assignment).read_text())
S = float(os.environ.get("STUB_S", "0.2")) * (2 if a.flip_tta else 1)
crash = int(os.environ.get("STUB_CRASH_AFTER", "0")) if a.flip_tta else 0
torn = int(os.environ.get("STUB_TORN_AT", "0")) if a.flip_tta else 0
if a.flip_tta:
    print("FLIP_TTA_READY", json.dumps({"perm": [0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 10, 11], "mcl_plain_pass_only": True}), flush=True)
t0 = time.time()
fh = open(out / "predictions.csv", "w", encoding="utf-8", newline="")
wr = csv.writer(fh); wr.writerow(["StudyInstanceUID"] + LABELS); fh.flush()
time.sleep(float(os.environ.get("STUB_COMPILE_S", "1.0")))
pos, stats, fallback, timed_out = 0, {"decode_s": 0., "render_s": 0., "gpu_s": 0., "windows": 0, "wait_s": 0., "rows": 0, "anchor_s": 0.}, {}, False
for u in uids:
    time.sleep(S)
    wr.writerow([u] + ["%%.9g" %% val(u, L, a.flip_tta) for L in LABELS]); fh.flush()
    pos += 1
    stats["rows"] += 1; stats["gpu_s"] += S; stats["decode_s"] += 1.0; stats["windows"] += 10
    if torn and pos == torn - 1:
        fh.write(u[:3] + "TORN,0.5"); fh.flush(); time.sleep(30)
    if pos %% 20 == 0 or pos == len(uids) or pos <= 3:
        print(json.dumps({"done": pos, "of": len(uids), "scored": stats["rows"], "fallback": 0, "elapsed_s": round(time.time() - t0, 1),
                          "decode_s": stats["decode_s"], "render_s": 0.0, "gpu_s": round(stats["gpu_s"], 1), "anchor_s": 0.0, "windows": stats["windows"]}), flush=True)
    if crash and pos == crash:
        print("STUB_CRASH", flush=True); sys.exit(3)
    if a.deadline and time.time() > a.deadline:
        timed_out = True; break
for u in uids[pos:]:
    fallback[u] = "not reached before the deadline"
fh.close()
rec = {"status": "PASS", "studies": len(uids), "scored": stats["rows"], "fallback": fallback, "timed_out": timed_out, "amp": "fp16",
       "seconds": round(time.time() - t0, 1), "stats": stats, "per_study": []}
if a.flip_tta:
    rec["flip_tta"] = {"perm": [0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 10, 11], "bin_perm": [], "mcl": "plain pass only"}
(out / "receipt.json").write_text(json.dumps(rec, indent=1))
print("WORKER_DONE", json.dumps({"studies": len(uids), "scored": stats["rows"], "timed_out": timed_out}), flush=True)
''' % (LABELS,)

WIN_KILL = '''def kill_group(p):
    subprocess.run(['taskkill', '/T', '/F', '/PID', str(p.pid)], capture_output=True)
    return p.wait()
'''


def val(uid, lab, flip):
    h = hashlib.md5(('%s|%s|%d' % (uid, lab, int(flip))).encode()).hexdigest()
    return (int(h[:8], 16) + 1) / (16 ** 8 + 2)


def setup(root, n):
    src = root / 'src'
    src.mkdir()
    (src / 'compact_infer.py').write_text(STUB, encoding='utf-8')
    stub_sha = hashlib.sha256((src / 'compact_infer.py').read_bytes()).hexdigest()
    d = DRIVER.read_text(encoding='utf-8')
    real_sha = '8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d'
    assert d.count(real_sha) == 1
    d = d.replace(real_sha, stub_sha)
    if not hasattr(os, 'killpg'):
        i, j = d.index('def kill_group(p):'), d.index('def last_progress(')
        d = d[:i] + WIN_KILL + '\n\n' + d[j:]
    (src / 'compact_infer_ta.py').write_text(d, encoding='utf-8')
    # her probe_dual.py (pinned source, from her cell 6) for the end-to-end controller test
    c6 = json.load(open(HER_NB, encoding='utf-8'))['cells'][6]['source']
    c6 = c6 if isinstance(c6, str) else ''.join(c6)
    line = [l for l in c6.split('\n') if l.startswith('SOURCES = ')][0]
    S = ast.literal_eval(line[len('SOURCES = '):])
    (src / 'probe_dual.py').write_bytes(S['probe_dual.py'].encode('utf-8'))
    assert hashlib.sha256((src / 'probe_dual.py').read_bytes()).hexdigest() == 'a6155eb08701f9bb746f838bc8ef8e2129c85f76a5c58e379692ffe2e76bbef0'
    data = root / 'data'
    uids = ['1.2.%03d' % i for i in range(n)]
    for i, u in enumerate(uids):
        sd = data / 'test_series' / u / 's1'
        sd.mkdir(parents=True)
        for f in range(5 + 3 * (i % 4)):
            (sd / ('%d.dcm' % f)).write_bytes(b'x')
    with open(data / 'test.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['StudyInstanceUID']); [w.writerow([u]) for u in uids]
    with open(data / 'sample_submission.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['StudyInstanceUID'] + LABELS); [w.writerow([u] + [0.5] * 12) for u in uids]
    return src, data, uids


def run_worker(src, data, uids, wdir, ta, env_extra, deadline=0.0, ta_args=None):
    wdir.mkdir(parents=True)
    (wdir / 'ids.json').write_text(json.dumps(uids))
    worker_args = ['--backbone', 'x', '--flip-tta'] + (ta_args if ta_args is not None else
                   ['--ta-budget-end', str(ta['budget_end']), '--ta-nb-t0', str(ta['nb_t0']), '--ta-min-studies', str(ta['k_min']),
                    '--ta-switch-min-s', str(ta['switch_min_s']), '--ta-poll-s', '0.05'])
    cmd = [sys.executable, '-u', str(src / 'compact_infer_ta.py'), '--data', str(data), '--split', 'test', '--code', 'c', '--arm', 'A',
           '--assignment', str(wdir / 'ids.json'), '--output', str(wdir), '--amp', 'auto', '--workers', '2', '--deadline', str(deadline),
           '--models', 'b'] + worker_args
    env = dict(os.environ, **env_extra)
    with open(wdir / 'log.txt', 'w', encoding='utf-8') as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300).returncode
    return rc, (wdir / 'log.txt').read_text(encoding='utf-8', errors='replace')


def read_pred(p):
    with open(p, encoding='utf-8') as fh:
        rows = list(csv.reader(fh))
    return rows[0], {r[0]: [float(x) for x in r[1:]] for r in rows[1:]}, [r[0] for r in rows[1:]]


def expect(name, wdir, uids, flipped, rc, log, want_rc=0, decision='switch', k=None):
    """k = the study count at which the switch was DECIDED; the study in flight at the kill may still complete (it then counts as
    flipped), so the flipped set must be the prefix uids[:k] or uids[:k + 1] of the assignment (the driver reports the final count)."""
    hdr, P, order = read_pred(wdir / 'predictions.csv')
    rec = json.loads((wdir / 'receipt.json').read_text())
    ta = rec.get('time_aware_flip', {})
    if decision == 'switch':
        flipped = set(ta.get('flipped_uids') or [])
        kf = len(flipped)
        chk(f'{name}: decided at study {k}, flipped = prefix of {k} or {k + 1}', ta.get('decided_at_study') == k and kf in (k, k + 1)
            and flipped == set(uids[:kf]), (ta.get('decided_at_study'), kf))
    chk(f'{name}: exit code {want_rc}', rc == want_rc, rc)
    chk(f'{name}: header', hdr == ['StudyInstanceUID'] + LABELS)
    chk(f'{name}: every study exactly once', sorted(order) == sorted(uids) and len(order) == len(set(order)), (len(order), len(uids)))
    bad = [u for u in uids if u in P and P[u] != [float('%.9g' % val(u, L, u in flipped)) for L in LABELS]]
    chk(f'{name}: flipped rows = flip pass, others = plain pass', not bad, bad[:3])
    chk(f'{name}: decision {decision}', ta.get('decision') == decision, {x: ta.get(x) for x in ('decision', 'disabled_at_study', 'flipped', 'plain', 'rule')})
    chk(f'{name}: receipt counts', rec.get('studies') == len(uids) and rec.get('scored') == len(uids) and not rec.get('fallback'),
        (rec.get('studies'), rec.get('scored'), list((rec.get('fallback') or {}).items())[:2]))
    if decision == 'none':
        chk(f'{name}: no FLIP_TTA_DISABLED marker, flip block present', 'FLIP_TTA_DISABLED_AT_STUDY' not in log and isinstance(rec.get('flip_tta'), dict))
    else:
        mk = 'FLIP_TTA_DISABLED_AT_STUDY %d ' % len(flipped)
        chk(f'{name}: marker "{mk.strip()}" in the worker log', mk in log)
        chk(f'{name}: disabled_at_study == flipped == {len(flipped)}', ta.get('disabled_at_study') == ta.get('flipped') == len(flipped)
            and ta.get('plain') == len(uids) - len(flipped))
        chk(f'{name}: flip block iff flipped rows', isinstance(rec.get('flip_tta'), dict) == (len(flipped) > 0), rec.get('flip_tta'))
        if k is not None and decision != 'switch':
            chk(f'{name}: disabled at study {k}', len(flipped) == k, len(flipped))
    return rec, ta


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    root = Path(tempfile.mkdtemp(prefix='ta_test_'))
    try:
        src, data, uids = setup(root, 24)
        base = {'STUB_S': '0.15', 'STUB_COMPILE_S': '1.0'}
        far = lambda: {'budget_end': time.time() + 3600, 'nb_t0': time.time() - 60, 'k_min': 10, 'switch_min_s': 600}

        print('A: budget far away -> flip on every study (the expected hidden case)')
        rc, log = run_worker(src, data, uids, root / 'A', far(), base)
        expect('A', root / 'A', uids, set(uids), rc, log, decision='none')
        chk('A: FLIP_TA_START + FLIP_TA_DONE', 'FLIP_TA_START' in log and 'FLIP_TA_DONE' in log)

        print('B: budget 2 s after start, compile 3 s -> lower-bound switch at study 1 (the control commit pattern)')
        ta = far(); ta['budget_end'] = time.time() + 2.0
        rc, log = run_worker(src, data, uids, root / 'B', ta, dict(base, STUB_COMPILE_S='3.0'))
        rec, t = expect('B', root / 'B', uids, None, rc, log, k=1)
        chk('B: rule lower bound', t.get('rule') == 'lower bound (k < min_studies)', t.get('rule'))

        print('C: measured-rate projection: k_min 4, budget start + compile + 4 s, 24 x 0.3 s flip -> switch at study 4')
        ta = far(); ta.update(budget_end=time.time() + 1.0 + 4.0, k_min=4, switch_min_s=0.5)
        rc, log = run_worker(src, data, uids, root / 'C', ta, base)
        rec, t = expect('C', root / 'C', uids, None, rc, log, k=4)
        chk('C: rule measured rate, projected end > budget', t.get('rule') == 'measured rate' and t.get('projected_end_h') > t.get('budget_end_h'), t)

        print('D: same as C but switch_min_s 1000 s (a restart is not worth it) -> no switch')
        ta = far(); ta.update(budget_end=time.time() + 1.0 + 4.0, k_min=4, switch_min_s=1000)
        rc, log = run_worker(src, data, uids, root / 'D', ta, base)
        expect('D', root / 'D', uids, set(uids), rc, log, decision='none')

        print('E: budget already passed at start -> no flip child at all (prestart, study 0)')
        ta = far(); ta['budget_end'] = time.time() - 1
        rc, log = run_worker(src, data, uids, root / 'E', ta, base)
        expect('E', root / 'E', uids, set(), rc, log, decision='prestart', k=0)
        chk('E: no FLIP_TTA_READY (the flip child never started)', 'FLIP_TTA_READY' not in log)

        print('F: flip child crashes after 5 studies -> logged, the rest scored without flip')
        rc, log = run_worker(src, data, uids, root / 'F', far(), dict(base, STUB_CRASH_AFTER='5'))
        expect('F', root / 'F', uids, set(uids[:5]), rc, log, decision='child_failed', k=5)
        chk('F: FLIP_TTA_CHILD_FAILED logged', 'FLIP_TTA_CHILD_FAILED' in log)

        print('G: torn last line at the kill -> dropped and that study re-scored without flip')
        ta = far(); ta.update(budget_end=time.time() + 1.0 + 4.0, k_min=4, switch_min_s=0.5)
        rc, log = run_worker(src, data, uids, root / 'G', ta, dict(base, STUB_TORN_AT='5'))
        rec, t = expect('G', root / 'G', uids, None, rc, log, k=4)
        chk('G: torn_last_line_dropped true', t.get('torn_last_line_dropped') is True, t.get('torn_last_line_dropped'))

        print('H: driver args missing -> fail-safe = plain V224-FLIP worker (flip on all, logged)')
        rc, log = run_worker(src, data, uids, root / 'H', None, base, ta_args=['--ta-nb-t0', str(time.time())])
        hdr, P, order = read_pred(root / 'H' / 'predictions.csv')
        chk('H: FLIP_TA_DRIVER_ERROR logged, rc 0, all rows flip', 'FLIP_TA_DRIVER_ERROR' in log and rc == 0 and sorted(order) == sorted(uids)
            and all(P[u] == [float('%.9g' % val(u, L, True)) for L in LABELS] for u in uids))

        print('I: original R7 deadline passes during flip (budget later) -> the child closes itself, same as V224-FLIP')
        rc, log = run_worker(src, data, uids, root / 'I', far(), base, deadline=time.time() + 1.0 + 1.0)
        rec = json.loads((root / 'I' / 'receipt.json').read_text())
        hdr, P, order = read_pred(root / 'I' / 'predictions.csv')
        chk('I: decision none, timed_out, rest = deadline fallback', rec['time_aware_flip']['decision'] == 'none' and rec['timed_out'] is True
            and len(order) + len(rec['fallback']) == len(uids) and 0 < len(order) < len(uids), (len(order), len(rec['fallback'])))

        print('J: end to end through her probe_dual.py (pinned) with the driver as --worker, lower-bound switch')
        pred = root / 'J'
        wa = ['--backbone', 'x', '--flip-tta', '--ta-budget-end', str(time.time() + 2.5), '--ta-nb-t0', str(time.time() - 60),
              '--ta-min-studies', '10', '--ta-switch-min-s', '600', '--ta-poll-s', '0.05']
        cmd = [sys.executable, '-u', str(src / 'probe_dual.py'), '--data', str(data), '--split', 'test', '--code', 'c', '--bundle', str(root),
               '--arm', 'A', '--output', str(pred), '--amp', 'auto', '--loader-workers', '2', '--deadline', str(time.time() + 3600),
               '--worker', 'compact_infer_ta.py', '--worker-args', json.dumps(wa), '--max-fallback-fraction', '1.0']
        r = subprocess.run(cmd, env=dict(os.environ, **dict(base, STUB_COMPILE_S='4.0')), capture_output=True, text=True, timeout=600)
        print('   ', r.stdout.strip().splitlines()[-1][:200] if r.stdout.strip() else r.stderr[-500:])
        sr = json.loads((pred / 'submission_receipt.json').read_text())
        chk('J: controller PASS, every study scored, 0 fallback', r.returncode == 0 and sr['status'] == 'PASS' and sr['scored'] == len(uids)
            and sr['fallback_studies'] == 0, {k: sr.get(k) for k in ('status', 'scored', 'fallback_studies', 'worker_exit_codes')})
        hdr, P, order = read_pred(pred / 'submission.csv')
        nw = len(sr['workers'])
        flipped = set()
        for w in sr['workers']:
            flipped |= set(w['time_aware_flip'].get('flipped_uids') or [])
        bad = [u for u in uids if P[u] != [float('%.9g' % val(u, L, u in flipped)) for L in LABELS]]
        chk(f'J: submission rows (workers {nw}) = flip for flipped uids (1 or 2 per worker), plain otherwise', not bad and nw <= len(flipped) <= 2 * nw,
            (len(flipped), bad[:2]))
        chk('J: worker receipts carry time_aware_flip switch + flip_tta block', all(w['time_aware_flip']['decision'] == 'switch'
            and w['flip_tta']['mcl'] == 'plain pass only' for w in sr['workers']))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    n = sum(R)
    print(f'TA_TEST {"PASS" if all(R) else "FAIL"} {n}/{len(R)}', flush=True)
    sys.exit(0 if all(R) else 1)


if __name__ == '__main__':
    main()
