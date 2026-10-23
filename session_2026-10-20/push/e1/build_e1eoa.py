#!/usr/bin/env python
# build_e1eoa.py (2026-10-20, Claude): LOCAL BUILD ONLY (no push, no commit, no submit). Track 4 of the 10-20 push:
# c4 kernel (session_2026-10-18/a18/kernel_v224r6a25c4j, 57343435, sha b47c29c3) + E1 soft-OR pow2 (THINK_1019 sec 8.4 /
# MORE_1020 sec 3.1, kernel-natural) + E-OA sym on the final output (the exact 9-line A20 block of build_a20.py, eoa20 = .959).
#   cell 14 (stage 3): each arm's percentile ** E1 before the mix (pct_on of r384 raw, of r224, and ro = R6); NO re-rank of
#                      the Raptor mix; the failed-row fills are untouched (NaN ** E1 stays NaN); blend receipt gets 'e1'.
#   cell 18 (_s6f_blend): rb, rs ** E1 directly after the failed-row fill `rs[~ok] = rb[~ok]` (the rho floor is computed
#                      before and is rank-invariant); then the A20 E-OA block after the blank fill, byte-for-byte as eoa20.
# Usage: python build_e1eoa.py E1 A_OA      e.g.  2 0.2 (candidate), 1 0 (control: must replay byte-identical to c4),
#        1 0.2 (control: must replay byte-identical to the scored a20 eoa20), 2 0 (E1 alone: parity .94111).
# Structural asserts: base sha == c4; changed cells == [14, 18]; every other cell == c4; cell 14 with the E1 lines reverted
# == c4 cell 14; cell 18 with the E1 lines removed == the A20 builder's cell 18 (and, for A_OA .2 / 0, == the cell 18 of the
# shipped a20 kernel_v224r6c4eoa20j (sha d40f43a4) / kernel_v224r6c4eoa00j).
import json, hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
RK = os.path.normpath(os.path.join(HERE, '..', '..', '..'))
SRC = os.path.join(RK, 'session_2026-10-18', 'a18', 'kernel_v224r6a25c4j')
A20 = os.path.join(RK, 'session_2026-10-19', 'a20')
C4_SHA = 'b47c29c3d72b9fab09393535a369440e5c77d41f2f019d091b65fd7d2306c563'
A20_SHA = {0.2: 'd40f43a4b6404fdb3021519d97bc3d7257f41f53da9a6c6c20fcf5781957a768',
           0.0: 'b9cab541941df897dce5219237fb74f265e03704cf0334c1071686a45dfe455a'}
e1 = float(sys.argv[1]); a = float(sys.argv[2]); assert e1 >= 1.0 and 0.0 <= a <= 0.5
tag = 'v224r6c4eoa%02de1%sj' % (round(a * 100), '' if e1 == 2.0 else 'x%g' % e1)
sha = lambda b: hashlib.sha256(b).hexdigest()
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
raw = open(os.path.join(SRC, meta['code_file']), 'rb').read(); assert sha(raw) == C4_SHA, ('base is not c4', sha(raw))
nb = json.loads(raw); assert len(nb['cells']) == 19

# ---- cell 18, step 1: the A20 E-OA block, verbatim from build_a20.py (same anchor, same text) ----------------------------
L18 = src(nb['cells'][18]).split('\n')
i = [k for k, l in enumerate(L18) if l == '        for c in blank:']; assert len(i) == 1, i; i = i[0]
assert L18[i + 1] == '            out[c] = np.float32(0.5)' and L18[i + 2].startswith('        v = out[labels].values.astype(np.float64)'), L18[i:i + 3]
EOA = [
    '        A_OA = %r   # [A20 E-OA] OA decorrelation on the final output (OAI expert check G1 SUPPORT, SHIP_A .2); 0 = no-op' % a,
    "        _OA = ['Medial OA', 'Lateral OA', 'PF OA']",
    '        if A_OA and set(_OA) <= set(labels) and not set(_OA) & set(blank):',
    '            _U = {c: pct(out[c].values.astype(np.float64)) for c in _OA}',
    '            for c in _OA:',
    '                _o = [k for k in _OA if k != c]',
    '                out[c] = pct(_U[c] - A_OA * (_U[_o[0]] + _U[_o[1]]) / 2.0).astype(np.float32)',
    "            rec['a_oa'] = A_OA",
    "            print('EOA_APPLIED a_oa %g on %s' % (A_OA, _OA), flush=True)",
]
L18[i + 2:i + 2] = EOA
A20_CELL18 = list(L18)                                        # what build_a20.py writes for this a
# ---- cell 18, step 2: E1 (two inserted lines) -------------------------------------------------------------------------------
E1_18 = ["    E1 = %r; rec['e1'] = E1   # [E1 soft-OR pow2] rb, rs ** E1 after the failed-row fill, before the .8 / .2 mix; 1.0 = no-op" % e1,
         '            rb, rs = rb ** E1, rs ** E1   # [E1] square both arms after the fill (the rho floor above is rank-invariant)']
j = [k for k, l in enumerate(L18) if l == "    rec = {'w_s6f': W_S6F, 'cov_min': COV_MIN, 'canary_col': CANARY_COL, 'mode': None, 'fired': []}"]
assert len(j) == 1, j; L18.insert(j[0] + 1, E1_18[0])
k = [m for m, l in enumerate(L18) if l == '            rs[~ok] = rb[~ok]']; assert len(k) == 1, k; k = k[0]
assert L18[k + 1] == '            out[c] = ((1.0 - W_S6F) * rb + W_S6F * rs).astype(np.float32)', L18[k + 1]
L18.insert(k + 1, E1_18[1])
nb['cells'][18]['source'] = '\n'.join(L18)

# ---- cell 14: E1 constant, two edited mix lines, receipt line -------------------------------------------------------------
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
assert L14[m[0] + 1] == "(WORK / 'blend_receipt.json').write_text(json.dumps(blend, indent=1))", L14[m[0] + 1]
L14.insert(m[0] + 1, RCPT)
nb['cells'][14]['source'] = '\n'.join(L14)

# ---- write + structural verification ---------------------------------------------------------------------------------------
out = os.path.join(HERE, 'kernel_' + tag); os.makedirs(out, exist_ok=True); cf = 'rsna_sub_%s.ipynb' % tag
json.dump(nb, open(os.path.join(out, cf), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
meta.update(id='eeshsaxena/rsna-sub-' + tag, title='rsna sub ' + tag, code_file=cf)
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
newb = open(os.path.join(out, cf), 'rb').read(); A_, B_ = json.loads(raw), json.loads(newb)
diff = [q for q, (x, y) in enumerate(zip(A_['cells'], B_['cells'])) if src(x) != src(y)]
assert diff == [14, 18] and len(A_['cells']) == len(B_['cells']), diff
assert all(x['cell_type'] == y['cell_type'] for x, y in zip(A_['cells'], B_['cells']))
b14 = src(B_['cells'][14]).split('\n'); b18 = src(B_['cells'][18]).split('\n')
rev14 = [{NEW_A: OLD_A, NEW_B: OLD_B}.get(l, l) for l in b14 if l not in (CONST, RCPT)]
assert rev14 == src(A_['cells'][14]).split('\n') == L14_0, 'cell 14 minus E1 != c4 cell 14'
assert sum(l in (NEW_A, NEW_B) for l in b14) == 2 and b14.count(CONST) == 1 and b14.count(RCPT) == 1
rm18 = [l for l in b18 if l not in E1_18]; assert len(rm18) == len(b18) - 2
assert rm18 == A20_CELL18, 'cell 18 minus E1 != the A20 builder cell 18'
ref = None
if a in A20_SHA:                                              # compare with the SHIPPED a20 kernel of the same a
    rp = os.path.join(A20, 'kernel_v224r6c4eoa%02dj' % round(a * 100), 'rsna_sub_v224r6c4eoa%02dj.ipynb' % round(a * 100))
    rb_ = open(rp, 'rb').read(); assert sha(rb_) == A20_SHA[a], ('a20 ref sha', sha(rb_)); R_ = json.loads(rb_)
    assert rm18 == src(R_['cells'][18]).split('\n'), 'cell 18 minus E1 != shipped a20 cell 18'
    d2 = [q for q, (x, y) in enumerate(zip(R_['cells'], B_['cells'])) if src(x) != src(y)]; assert d2 == [14, 18], d2   # 18 differs only by the 2 E1 lines (asserted above)
    ref = dict(path=os.path.relpath(rp, RK).replace(os.sep, '/'), sha256=sha(rb_), cells_differing_from_ref=d2)
rc = dict(probe='e1eoa', e1=e1, a_oa=a, oa_mode='sym', tag=tag, kernel_id=meta['id'], src='session_2026-10-18/a18/kernel_v224r6a25c4j',
          src_sha256=sha(raw), out_sha256=sha(newb), cells_changed=diff,
          cell14=dict(inserted=[CONST, RCPT], replaced={OLD_A: NEW_A, OLD_B: NEW_B}),
          cell18=dict(inserted_e1=E1_18, inserted_eoa=len(EOA)), a20_ref=ref, pushed=False, committed=False)
json.dump(rc, open(os.path.join(HERE, 'build_receipt_%s.json' % tag), 'w'), indent=1)
print('E1EOA_BUILT', tag, 'e1 %g a_oa %g' % (e1, a), sha(newb)[:16], '| base c4', sha(raw)[:16],
      '| cells', diff, '| a20 ref', (ref or {}).get('sha256', '-')[:8], (ref or {}).get('cells_differing_from_ref'))
