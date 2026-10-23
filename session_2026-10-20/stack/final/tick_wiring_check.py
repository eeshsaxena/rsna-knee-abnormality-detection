# tick_wiring_check.py (10-21 ~00:35 IST, Claude). Read-only wiring check of the proposed registry merge against val/tick_a22.py's OWN
# functions (parse_go, kernel_ok, gate_argv, run). Imports tick_a22 from its file (run with -B: no bytecode written); merges the
# slate IN MEMORY only (val/slate_a22.json is not written); runs each new entry's gate argv on its sim in place (no copy to val/sim).
import importlib.util, json, os, sys
R = 'C:/Users/eeshs/Downloads/rsna-knee'
spec = importlib.util.spec_from_file_location('tick_a22', R + '/session_2026-10-20/val/tick_a22.py'); t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
LOG = open(R + '/session_2026-10-20/stack/final/tick_wiring_check.log', 'w'); res = []
def ok(name, c):
    res.append(bool(c)); s = ('OK   ' if c else 'FAIL ') + name; print(s, flush=True); LOG.write(s + '\n')
val = t.load_slate()
new = {}
new.update(json.load(open(R + '/session_2026-10-20/stack/skeptic/kernels/registry_entry_spl.json'))['entries'])
k2 = json.load(open(R + '/session_2026-10-20/stack/kernels/registry_entries_kernels.json'))['entries']
new.update({k: k2[k] for k in ('swide', 'salt')})
ok('no key collision with val slate %s' % sorted(val), not (set(new) & set(val)))
ok('no slug collision with val slate', not ({e['slug'] for e in new.values()} & {e['slug'] for e in val.values()}))
ok('desc prefixes distinct from val entries and from each other (tick dedup by desc prefix)',
   len({e['desc'].split(':')[0] for e in list(val.values()) + list(new.values())}) == len(val) + len(new))
merged = dict(val); merged.update(new)
for k, e in new.items():
    ok('kernel_ok %s (%s nb %s meta %s)' % (k, e['tag'], e['nb_sha256'][:8], e['meta_sha256'][:8]), t.kernel_ok(e))
    rc, out = t.run(t.gate_argv(e, os.path.join(R, e['sim_dir'])), 900); last = out.strip().splitlines()[-1] if out.strip() else ''
    ok('gate_argv %s on its sim -> %s (rc %s)' % (k, last, rc), e['gate_pass'] in last and rc == 0)
for go, want in (('GO r6pw20 pw25 spl swide MAX=4 Eesh 10-21 03:00 "ok"', (['r6pw20', 'pw25', 'spl', 'swide'], 4)),
                 ('GO r6pw20 pw25 spl MAX=3 Eesh', (['r6pw20', 'pw25', 'spl'], 3)),
                 ('GO r6pw20 pw25 spl salt swide MAX=5 Eesh', (['r6pw20', 'pw25', 'spl', 'salt', 'swide'], 5))):
    ok('parse_go %r -> %s' % (go, want), t.parse_go(go, merged)[:2] == want)
for bad in ('GO r6pw20 pw25 sp1 swide', 'GO r6pw20 pw25 spl swid', 'GO r6pw20 pw25 ssafe'):
    ok('parse_go %r -> invalid (typo / unregistered key voids the GO)' % bad, t.parse_go(bad, merged)[0] is None)
s = 'TICK_WIRING_CHECK %s %d/%d' % ('PASS' if all(res) else 'FAIL', sum(res), len(res)); print(s); LOG.write(s + '\n'); LOG.close()
