#!/usr/bin/env python
# j_envwatch.py (2026-10-15, Claude subagent, review fix J2): the gate rule for dec_envwatch.json, the file the DEC cell's daemon
# thread writes in-run (/proc reads of every later Python process: argv, PYTHONPATH, parent chain). Imported by j_check.py (the
# GPU-commit gate) and by j_local_harness.py (which runs the real DEC cell watcher against a fake /proc), so the writer in the DEC
# cell and this reader are tested together.
# Rule: the watcher ran (ok, >= 1 sweep, target /tmp/_dec); every EXPECTED stage script was seen at least once as a descendant
# Python process of the notebook with /tmp/_dec among its PYTHONPATH entries and no -I / -E / -S, and never otherwise; no
# descendant Python process at all (any script, '-c' spawn children, pip, ...) lacks /tmp/_dec or runs with -I / -E / -S, except
# scripts waived by name on the command line (printed); every record agrees with the counters it was summed into.
EXPECT = {
    'v224flipj': ['raptor_run.py', 'raptor224_run.py', 'probe_dual.py', 'compact_infer.py'],
    'v224flips6fj': ['raptor_run.py', 'raptor224_run.py', 'probe_dual.py', 'compact_infer.py', 's6f_driver.py', 'k1_b_infer.py'],
    'v224flips20j': ['raptor_run.py', 'raptor224_run.py', 'probe_dual.py', 'compact_infer.py', 'fleet_worker.py'],
}
TARGET = '/tmp/_dec'


def verdict(doc, expect, waive=(), target=TARGET, sep=':'):
    """-> (ok, problems, info)."""
    if not isinstance(doc, dict):
        return False, ['no dec_envwatch.json (the watcher never wrote it)'], {}
    P = []
    waive = set(waive or ())
    if doc.get('ok') is not True:
        P.append('watcher not ok: %s' % (doc.get('why') or doc.get('last_error')))
    if (doc.get('sweeps') or 0) < 1:
        P.append('no completed sweep')
    if doc.get('target') != target:
        P.append('watcher target %r != %r' % (doc.get('target'), target))
    if not expect:
        P.append('no expected stage scripts given (unknown tag: pass --expect-script)')
    sc = doc.get('scripts') or {}
    for s in expect or ():
        c = sc.get(s) or {}
        if not c.get('scope_ok'):
            P.append('stage script %s never seen as a descendant with %s on PYTHONPATH and no -I / -E / -S' % (s, target))
        if c.get('scope_bad'):
            P.append('stage script %s seen %d time(s) WITHOUT %s on PYTHONPATH or with -I / -E / -S' % (s, c['scope_bad'], target))
    nbad = (doc.get('n_scope_bad') or 0) - sum((sc.get(w) or {}).get('scope_bad', 0) for w in waive)
    bad = [r for r in (doc.get('bad') or []) if r.get('script') not in waive]
    if nbad > 0 or bad:
        P.append('%d descendant Python process(es) without %s or with -I / -E / -S: %s' % (
            max(nbad, len(bad)), target, [(r.get('script'), r.get('flags'), (r.get('pythonpath') or '')[:90]) for r in bad[:4]]))
    # the records must agree with the counters (a writer bug must not turn into a PASS)
    recs = doc.get('records') or []
    for r in recs:
        if r.get('environ') and r.get('pythonpath') is not None and len(r['pythonpath']) < 400:
            if bool(r.get('dec_on_path')) != (target in r['pythonpath'].split(sep)):
                P.append('record inconsistent (dec_on_path vs pythonpath): %s' % r.get('script'))
                break
        rb = bool(r.get('flags')) or (bool(r.get('environ')) and not r.get('dec_on_path'))
        if rb and r.get('scope') is not False and r.get('script') not in waive and not any(
                b.get('pid') == r.get('pid') and b.get('script') == r.get('script') for b in doc.get('bad') or []):
            P.append('bad record missing from the bad list: %s pid %s' % (r.get('script'), r.get('pid')))
            break
    if len(recs) < 400:
        for s, c in sc.items():
            n = sum(1 for r in recs if r.get('script') == s)
            if n != c.get('n'):
                P.append('counter of %s (%s) != records (%d)' % (s, c.get('n'), n))
                break
    info = {'sweeps': doc.get('sweeps'), 'n_python': doc.get('n_python'), 'n_scope_python': doc.get('n_scope_python'),
            'n_scope_bad': doc.get('n_scope_bad'), 'waived': sorted(waive),
            'scripts': {s: '%d/%d ok' % (c.get('scope_ok', 0), c.get('scope', 0)) for s, c in sorted(sc.items())}}
    return not P, P, info
