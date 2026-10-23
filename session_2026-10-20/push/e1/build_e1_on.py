#!/usr/bin/env python
# build_e1_on.py (2026-10-20, Claude): LOCAL BUILD ONLY (no push / commit / submit). Contingency twin of build_e1eoa.py for the case
# that the A21 read (eoa30 / pw20 / pw30, read ~17:00-19:00 IST 10-20) moves the API best-id: E1 is then stacked on THAT scored kernel,
# not on eoa20 (MORE_1020 sec 3.1: "build only the variant whose base is the API best-id"). Starts from the SCORED notebook itself
# (sha asserted against its A21 build receipt, whose src must be c4 b47c29c3) and inserts exactly the E1 lines of build_e1eoa.py:
#   cell 14: E1 constant after W_RAPTOR_C, `** E1` on pct_on(raw), pct_on(ours), pct_on(r224), `blend['e1'] = E1`;
#   cell 18: `E1 = ..; rec['e1'] = E1` after rec, `rb, rs = rb ** E1, rs ** E1` after `rs[~ok] = rb[~ok]`.
# Usage: python build_e1_on.py E1 BASE_TAG    BASE_TAG in v224r6c4eoa30j, v224r6c4pw20j, v224r6c4pw30j (a21) or v224r6c4eoa20j (a20)
import json, hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
RK = os.path.normpath(os.path.join(HERE, '..', '..', '..'))
C4_SHA = 'b47c29c3d72b9fab09393535a369440e5c77d41f2f019d091b65fd7d2306c563'
e1 = float(sys.argv[1]); base = sys.argv[2]; assert e1 >= 1.0
BD = {'v224r6c4eoa20j': 'session_2026-10-19/a20', 'v224r6c4eoa30j': 'session_2026-10-20/a21', 'v224r6c4pw20j': 'session_2026-10-20/a21',
      'v224r6c4pw30j': 'session_2026-10-20/a21'}[base]
sha = lambda b: hashlib.sha256(b).hexdigest()
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
brc = json.load(open(os.path.join(RK, BD, 'build_receipt_%s.json' % base), encoding='utf-8'))
assert brc['src_sha256'] == C4_SHA and brc['tag'] == base, brc
kd = os.path.join(RK, BD, 'kernel_' + base); meta = json.load(open(os.path.join(kd, 'kernel-metadata.json'), encoding='utf-8'))
raw = open(os.path.join(kd, meta['code_file']), 'rb').read(); assert sha(raw) == brc['out_sha256'], ('base sha != its build receipt', sha(raw))
c4raw = open(os.path.join(RK, 'session_2026-10-18', 'a18', 'kernel_v224r6a25c4j', 'rsna_sub_v224r6a25c4j.ipynb'), 'rb').read()
assert sha(c4raw) == C4_SHA
nb = json.loads(raw); C4 = json.loads(c4raw)
assert [q for q, (x, y) in enumerate(zip(C4['cells'], nb['cells'])) if src(x) != src(y)] == [18], 'base differs from c4 outside cell 18'
tag = base[:-1] + ('e1j' if e1 == 2.0 else 'e1x%gj' % e1)
# ---- cell 18 -----------------------------------------------------------------------------------------------------------------
L18 = src(nb['cells'][18]).split('\n'); L18_0 = list(L18)
E1_18 = ["    E1 = %r; rec['e1'] = E1   # [E1 soft-OR pow2] rb, rs ** E1 after the failed-row fill, before the .8 / .2 mix; 1.0 = no-op" % e1,
         '            rb, rs = rb ** E1, rs ** E1   # [E1] square both arms after the fill (the rho floor above is rank-invariant)']
j = [k for k, l in enumerate(L18) if l == "    rec = {'w_s6f': W_S6F, 'cov_min': COV_MIN, 'canary_col': CANARY_COL, 'mode': None, 'fired': []}"]
assert len(j) == 1, j; L18.insert(j[0] + 1, E1_18[0])
k = [m for m, l in enumerate(L18) if l == '            rs[~ok] = rb[~ok]']; assert len(k) == 1, k; k = k[0]
assert L18[k + 1] == '            out[c] = ((1.0 - W_S6F) * rb + W_S6F * rs).astype(np.float32)', L18[k + 1]
L18.insert(k + 1, E1_18[1])
assert [i for i, l in enumerate(L18) if l.startswith('        A_OA = ')][0] > k + 1, 'E-OA block must follow the E1 square'
nb['cells'][18]['source'] = '\n'.join(L18)
# ---- cell 14 -----------------------------------------------------------------------------------------------------------------
L14 = src(nb['cells'][14]).split('\n'); L14_0 = list(L14)
OLD_A = '    rr, ro = pct_on(raw[c].values, ~rap_fail), pct_on(ours[c].values, ~our_fail)   # [v3] each over its own non-failed rows'
NEW_A = '    rr, ro = pct_on(raw[c].values, ~rap_fail) ** E1, pct_on(ours[c].values, ~our_fail) ** E1   # [v3] each over its own non-failed rows; [E1] each arm ** E1'
OLD_B = '        rr = (1.0 - A224) * rr + A224 * pct_on(r224[c].values, ~rap_fail)'
NEW_B = '        rr = (1.0 - A224) * rr + A224 * pct_on(r224[c].values, ~rap_fail) ** E1   # [E1] no re-rank of the Raptor mix'
CONST = 'E1 = %r   # [E1 soft-OR pow2, THINK_1019 8.4] each arm\'s percentile ** E1 before the stage-3 mix (r384, r224, ours = R6); 1.0 = no-op' % e1
RCPT = "blend['e1'] = E1   # [E1] receipt"
for old, new in ((OLD_A, NEW_A), (OLD_B, NEW_B)):
    m = [q for q, l in enumerate(L14) if l == old]; assert len(m) == 1, (old, m); L14[m[0]] = new
m = [q for q, l in enumerate(L14) if l.startswith('W_RAPTOR_C = {')]; assert len(m) == 1, m; L14.insert(m[0] + 1, CONST)
m = [q for q, l in enumerate(L14) if l == "         'rank_basis': 'each model over its own non-failed rows'}"]; assert len(m) == 1, m
L14.insert(m[0] + 1, RCPT)
nb['cells'][14]['source'] = '\n'.join(L14)
# ---- write + verify ------------------------------------------------------------------------------------------------------------
out = os.path.join(HERE, 'alt', 'kernel_' + tag); os.makedirs(out, exist_ok=True); cf = 'rsna_sub_%s.ipynb' % tag
json.dump(nb, open(os.path.join(out, cf), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
meta.update(id='eeshsaxena/rsna-sub-' + tag, title='rsna sub ' + tag, code_file=cf)
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
newb = open(os.path.join(out, cf), 'rb').read(); A_, B_ = json.loads(raw), json.loads(newb)
diff = [q for q, (x, y) in enumerate(zip(A_['cells'], B_['cells'])) if src(x) != src(y)]
assert diff == [14, 18] and len(A_['cells']) == len(B_['cells']), diff
b14, b18 = src(B_['cells'][14]).split('\n'), src(B_['cells'][18]).split('\n')
assert [{NEW_A: OLD_A, NEW_B: OLD_B}.get(l, l) for l in b14 if l not in (CONST, RCPT)] == L14_0 == src(C4['cells'][14]).split('\n')
assert [l for l in b18 if l not in E1_18] == L18_0 and len(b18) == len(L18_0) + 2
rc = dict(probe='e1_on', e1=e1, base_tag=base, base_dir=BD, base_sha256=sha(raw), base_receipt=dict(mode=brc.get('mode', 'sym'), a_oa=brc['a_oa']),
          c4_sha256=C4_SHA, tag=tag, kernel_id=meta['id'], out_sha256=sha(newb), cells_changed=diff, pushed=False, committed=False)
json.dump(rc, open(os.path.join(HERE, 'alt', 'build_receipt_%s.json' % tag), 'w'), indent=1)
print('E1_ON_BUILT', tag, 'e1 %g on %s (%s a %g)' % (e1, base, rc['base_receipt']['mode'], brc['a_oa']), sha(newb)[:16], '| base', sha(raw)[:16], '| cells', diff)
