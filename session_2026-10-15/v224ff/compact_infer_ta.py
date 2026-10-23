"""compact_infer_ta.py (eeshsaxena V224-F-FLIP, 2026-10-15): time-aware R7 flip TTA driver.

NOT one of the pinned R7 sources. A stdlib-only driver that probe_dual.py launches in place of compact_infer.py (the FLIP-TA
cell sets WORKER to this file). It runs the PINNED compact_infer.py (sha256 8401812c..., unchanged, re-checked here) as a child:
  phase F: the worker's whole assignment with --flip-tta, i.e. exactly the V224-FLIP worker command. predictions.csv is
           flushed per study, so after every study the driver projects this worker's R7 end WITH flip from the measured rate:
             rate = (t_k - t_1) / files(rows 2..k)   (seconds per DICOM file; row 1 carries the ~4 min torch.compile check)
             projected_end = t_k + safety * rate * files(remaining studies)        (k >= min_studies)
             projected_end = t_k                                                   (k <  min_studies: a lower bound)
           If projected_end > budget_end (notebook start + 5.5 h by default) and the remaining flip work is worth a restart
           (safety * rate * files_remaining > switch_min_s; always true while k < min_studies), flip is DISABLED for the
           remaining studies: the child's process group is killed, its complete rows are kept (a torn last line is dropped
           and that study is re-scored), FLIP_TTA_DISABLED_AT_STUDY k is printed, and
  phase P: the pinned compact_infer.py WITHOUT --flip-tta (= the V224-exact worker command) scores the remaining studies into
           <wdir>/ta_plain; the driver then writes <wdir>/predictions.csv = phase-F rows + phase-P rows and <wdir>/receipt.json
           = the phase-P receipt with the merged counts, the flip_tta block (phase F) and a time_aware_flip block.
  budget already passed when this worker starts: phase P only, written straight into <wdir> (FLIP_TTA_DISABLED_AT_STUDY 0).
  phase F child exits non-zero on its own: FLIP_TTA_CHILD_FAILED is printed and the unscored studies go to phase P.
No switch (the expected case): the child wrote <wdir>/predictions.csv and receipt.json exactly as the V224-FLIP worker; the
driver only adds the time_aware_flip block to receipt.json. Exit code: the last child's exit code.
Driver arguments (stripped before the child command): --ta-budget-end EPOCH --ta-nb-t0 EPOCH [--ta-min-studies 10]
[--ta-switch-min-s 600] [--ta-safety 1.05] [--ta-poll-s 1.0].
"""
import csv, hashlib, io, json, math, os, shutil, signal, subprocess, sys, time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CHILD = HERE / 'compact_infer.py'
CHILD_SHA = '8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d'
UIDK = 'StudyInstanceUID'
TA_OPTS = {'--ta-budget-end': float, '--ta-nb-t0': float, '--ta-min-studies': int, '--ta-switch-min-s': float,
           '--ta-safety': float, '--ta-poll-s': float}


def log(tag, obj):
    print(tag, json.dumps(obj), flush=True)


def split_args(argv):
    """-> (driver options, child argv). Driver options are exactly the TA_OPTS keys, each followed by one value."""
    ta, rest, i = {}, [], 0
    while i < len(argv):
        if argv[i] in TA_OPTS:
            ta[argv[i][2:].replace('-', '_')] = TA_OPTS[argv[i]](argv[i + 1])
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    return ta, rest


def get_opt(argv, name):
    idx = [i for i, a in enumerate(argv) if a == name]
    assert len(idx) == 1 and idx[0] + 1 < len(argv), ('need exactly one', name)
    return argv[idx[0] + 1]


def set_opt(argv, name, value):
    out = list(argv)
    out[out.index(name) + 1] = value
    return out


def files_of(data, split, uid):                       # probe_dual.py's scheduling cost: DICOM files of the study
    n = 0
    try:
        for s in os.scandir(Path(data) / (split + '_series') / uid):
            if s.is_dir():
                n += sum(1 for _ in os.scandir(s.path))
    except OSError:
        pass
    return max(1, n)


class Rows:
    """Incremental reader of a predictions.csv that another process appends to and flushes per row. Only complete lines
    (ending in a newline) count; a row counts when it has the header's 13 fields, a uid of this worker's assignment and 12
    finite probabilities in [0, 1] (the same row filter as probe_dual.py, plus the complete-line rule)."""

    def __init__(self, path, uids):
        self.path, self.uids = Path(path), set(uids)
        self.off, self.buf, self.header, self.rows, self.t, self.bad = 0, b'', None, {}, [], 0
        self.header_raw = None

    def poll(self):
        try:
            with open(self.path, 'rb') as fh:
                fh.seek(self.off)
                chunk = fh.read()
        except OSError:
            return []
        self.off += len(chunk)
        self.buf += chunk
        new = []
        while b'\n' in self.buf:
            line, self.buf = self.buf.split(b'\n', 1)
            text = line.decode('utf-8', errors='replace').rstrip('\r')
            rec = next(csv.reader(io.StringIO(text)), None)
            if self.header is None:
                self.header, self.header_raw = rec, line + b'\n'
                continue
            ok = rec is not None and self.header is not None and len(rec) == len(self.header) and rec[0] in self.uids
            if ok:
                try:
                    v = [float(x) for x in rec[1:]]
                    ok = all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in v)
                except ValueError:
                    ok = False
            if ok and rec[0] not in self.rows:
                self.rows[rec[0]] = line + b'\n'
                self.t.append(time.time())
                new.append(rec[0])
            elif not ok:
                self.bad += 1
        return new

    def torn(self):
        return len(self.buf) > 0


def run_child(argv, flip):
    cmd = [sys.executable, '-u', str(CHILD)] + [a for a in argv if flip or a != '--flip-tta']
    assert flip == ('--flip-tta' in cmd)
    return subprocess.Popen(cmd, start_new_session=True)


def kill_group(p):
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    rc = p.wait()
    t = time.time()
    while time.time() - t < 10:                        # DataLoader workers of the child share its process group
        try:
            os.killpg(p.pid, 0)
        except ProcessLookupError:
            break
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            break
        time.sleep(0.5)
    return rc


def last_progress(log_path):
    """the child's last progress json line in this worker's log (cumulative stats of phase F); {} if none"""
    try:
        lines = [l for l in Path(log_path).read_text(encoding='utf-8', errors='replace').splitlines() if l.startswith('{"done"')]
        return json.loads(lines[-1]) if lines else {}
    except Exception:
        return {}


def ready_line(log_path, marker):
    try:
        for l in Path(log_path).read_text(encoding='utf-8', errors='replace').splitlines():
            if l.startswith(marker + ' '):
                return json.loads(l[len(marker) + 1:])
    except Exception:
        pass
    return None


def add_ta(wdir, TA):
    try:
        rec = json.loads((wdir / 'receipt.json').read_text(encoding='utf-8'))
        rec['time_aware_flip'] = TA
        (wdir / 'receipt.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    except Exception as e:
        log('FLIP_TA_RECEIPT_ERROR', {'error': repr(e)[:200]})


def brief(TA):
    return {k: v for k, v in TA.items() if k != 'flipped_uids'}


def main():
    t_start = time.time()
    assert hashlib.sha256(CHILD.read_bytes()).hexdigest() == CHILD_SHA, 'compact_infer.py is not the pinned v4c source'
    ta, argv = split_args(sys.argv[1:])
    try:
        assert '--flip-tta' in argv, 'the time-aware driver expects --flip-tta in the worker args'
        budget_end, nb_t0 = float(ta['ta_budget_end']), float(ta['ta_nb_t0'])
        k_min = max(2, int(ta.get('ta_min_studies', 10)))
        switch_min_s = float(ta.get('ta_switch_min_s', 600.0))
        safety = float(ta.get('ta_safety', 1.05))
        poll_s = float(ta.get('ta_poll_s', 1.0))
        wdir = Path(get_opt(argv, '--output'))
        wdir.mkdir(parents=True, exist_ok=True)
        uids = json.loads(Path(get_opt(argv, '--assignment')).read_text(encoding='utf-8'))
        data, split = get_opt(argv, '--data'), get_opt(argv, '--split')
        cost = {u: files_of(data, split, u) for u in uids}
        assert len(uids) == len(set(uids))
    except Exception as e:                            # fail-safe: the plain V224-FLIP worker (full flip, no time awareness)
        log('FLIP_TA_DRIVER_ERROR', {'stage': 'setup', 'error': repr(e)[:300], 'fallback': 'V224-FLIP worker command'})
        sys.exit(run_child(argv, flip='--flip-tta' in argv).wait())
    h = lambda t: round((t - nb_t0) / 3600.0, 4)
    TA = {'driver': 'compact_infer_ta.py', 'budget_end_h': h(budget_end), 'min_studies': k_min, 'switch_min_s': switch_min_s,
          'safety': safety, 'studies': len(uids), 'files': sum(cost.values()), 'start_h': h(t_start), 'decision': 'none',
          'disabled_at_study': None, 'flipped': 0, 'plain': 0}
    log('FLIP_TA_START', TA)
    log_path = wdir / 'log.txt'
    F = Rows(wdir / 'predictions.csv', uids)
    rc_f = None

    if time.time() >= budget_end:
        TA.update(decision='prestart', reason='flip budget already passed when the worker started', decided_at_study=0,
                  decided_h=h(time.time()))
    else:
        p = run_child(argv, flip=True)
        decided = None
        try:
            while True:
                alive = p.poll() is None
                new = F.poll()
                if new and alive:
                    k = len(F.rows)
                    remaining = [u for u in uids if u not in F.rows]
                    if remaining:
                        now = F.t[-1]
                        files_rem = sum(cost[u] for u in remaining)
                        rate = None
                        if k >= 2:
                            rate = (F.t[-1] - F.t[0]) / max(1, sum(cost[u] for u in list(F.rows)[1:k]))
                        if k >= k_min:
                            work = safety * rate * files_rem
                            proj = now + work
                            trig = proj > budget_end and work > switch_min_s
                        else:
                            work, proj = None, now
                            trig = proj > budget_end
                        if trig:
                            decided = {'decided_at_study': k, 'decided_h': h(now), 'projected_end_h': h(proj),
                                       'rate_s_per_file': None if rate is None else round(rate, 5),
                                       'remaining_flip_work_s': None if work is None else round(work, 1),
                                       'remaining_studies': len(remaining), 'remaining_files': files_rem,
                                       'rule': 'measured rate' if work is not None else 'lower bound (k < min_studies)'}
                            break
                if not alive:
                    F.poll()
                    break
                time.sleep(poll_s)
        except Exception as e:                        # a driver bug never stops the flip child: it then runs to its end
            log('FLIP_TA_DRIVER_ERROR', {'stage': 'monitor', 'error': repr(e)[:300], 'fallback': 'phase F runs to its end'})
            decided = None
            p.wait()
            F.poll()
        if decided is not None:
            TA.update(decision='switch', reason='projected R7 end with flip > budget', kill_rc=kill_group(p), **decided)
            F.poll()                                  # rows completed between the decision and the kill still count
            shutil.rmtree(wdir / 'tmp', ignore_errors=True)
        else:
            rc_f = p.returncode
            if rc_f != 0:
                TA.update(decision='child_failed', reason='phase F child exit %s' % rc_f)
                log('FLIP_TTA_CHILD_FAILED', {'rc': rc_f, 'rows': len(F.rows), 'elapsed_h': h(time.time())})
                shutil.rmtree(wdir / 'tmp', ignore_errors=True)

    TA['flipped'] = len(F.rows)
    if TA['decision'] == 'none':                      # phase F scored (or deadline-closed) the whole assignment = V224-FLIP
        TA.update(phase_f_seconds=round(time.time() - t_start, 1), end_h=h(time.time()))
        add_ta(wdir, TA)
        log('FLIP_TA_DONE', TA)
        sys.exit(rc_f)

    k_final = len(F.rows)
    remaining = [u for u in uids if u not in F.rows]
    TA.update(disabled_at_study=k_final, torn_last_line_dropped=F.torn(), flipped_uids=list(F.rows))
    log('FLIP_TTA_DISABLED_AT_STUDY %d' % k_final, brief(TA))

    if TA['decision'] == 'prestart':                  # no flip at all: the V224-exact worker command, straight into <wdir>
        rc = run_child(argv, flip=False).wait()
        TA.update(plain=len(uids), plain_rc=rc, end_h=h(time.time()))
        add_ta(wdir, TA)
        log('FLIP_TA_DONE', brief(TA))
        sys.exit(rc)

    prog_f = last_progress(log_path) if k_final else {}
    flip_ready = ready_line(log_path, 'FLIP_TTA_READY') if k_final else None
    pdir = wdir / 'ta_plain'
    pdir.mkdir(exist_ok=True)
    P = Rows(pdir / 'predictions.csv', remaining)
    rc, rec_p = 0, None
    try:
        if remaining:
            (pdir / 'ids.json').write_text(json.dumps(remaining), encoding='utf-8')
            argv_p = set_opt(set_opt(argv, '--assignment', str(pdir / 'ids.json')), '--output', str(pdir))
            t_p = time.time()
            log('FLIP_TA_PLAIN_START', {'studies': len(remaining), 'files': sum(cost[u] for u in remaining), 'start_h': h(t_p)})
            rc = run_child(argv_p, flip=False).wait()
            P.poll()
            try:
                rec_p = json.loads((pdir / 'receipt.json').read_text(encoding='utf-8'))
            except Exception as e:
                log('FLIP_TA_PLAIN_NO_RECEIPT', {'error': repr(e)[:200]})
            TA.update(plain_rc=rc, plain_seconds=round(time.time() - t_p, 1))
        TA['plain'] = len(P.rows)
    except Exception as e:
        rc = 1
        log('FLIP_TA_DRIVER_ERROR', {'stage': 'plain phase', 'error': repr(e)[:300], 'fallback': 'only the phase F rows are kept'})

    # merged predictions.csv: phase-F rows (flip) then phase-P rows (plain), each in the child's own bytes
    H = F if F.header is not None else P
    assert H.header is not None and H.header[0] == UIDK, H.header
    if F.header is not None and P.header is not None:
        assert F.header == P.header, (F.header, P.header)
    assert not (set(F.rows) & set(P.rows))
    tmpf = wdir / 'predictions.csv.tmp'
    with open(tmpf, 'wb') as fh:
        fh.write(H.header_raw)
        for u in uids:
            if u in F.rows:
                fh.write(F.rows[u])
        for u in uids:
            if u in P.rows:
                fh.write(P.rows[u])
    os.replace(tmpf, wdir / 'predictions.csv')

    rec = dict(rec_p) if isinstance(rec_p, dict) else {'status': 'PASS' if rc == 0 else 'PLAIN_PHASE_FAILED'}
    fb = dict((rec_p or {}).get('fallback') or {})
    for u in uids:
        if u not in F.rows and u not in P.rows and u not in fb:
            fb[u] = 'no row after the time-aware flip switch'
    stats = {k_: float(v) for k_, v in ((rec_p or {}).get('stats') or {}).items()}
    for k_ in ('decode_s', 'render_s', 'gpu_s', 'windows', 'anchor_s'):
        stats[k_] = round(stats.get(k_, 0.0) + float(prog_f.get(k_, 0.0) or 0.0), 1)
    stats['rows'] = len(P.rows) + int(prog_f.get('scored', 0) or 0)   # phase F part from its last progress line
    rec.update(studies=len(uids), scored=len(F.rows) + len(P.rows), fallback=fb, stats=stats,
               seconds=round(time.time() - t_start, 1),
               stats_note='phase-F stats from its last progress line (every 20 studies) + the phase-P receipt stats')
    if k_final:
        rec['flip_tta'] = {'perm': (flip_ready or {}).get('perm'), 'mcl': 'plain pass only',
                           'phase': 'flip TTA on the first %d of %d studies of this worker only' % (k_final, len(uids))}
    TA['end_h'] = h(time.time())
    rec['time_aware_flip'] = TA
    (wdir / 'receipt.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    log('FLIP_TA_DONE', brief(TA))
    sys.exit(rc)


if __name__ == '__main__':
    main()


