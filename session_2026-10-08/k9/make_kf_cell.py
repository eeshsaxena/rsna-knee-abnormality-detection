#!/usr/bin/env python3
"""make_kf_cell.py (K8 kit, 2026-10-07, Claude): generate cells/k2_fleet_cell_kf.py = the K5 cell (cells/k2_fleet_cell.py,
byte-identical to SP/k5/cells/k2_fleet_cell.py, the cell that scored .946) + the KF flat outer-map hedge spliced in
exactly as designed and tested in gold_prep (parent_audit.md sec 3.2; kf_patch_block.py == the verifier-fixed version;
splice == gold_prep/verify_parentaudit/kf_integration_test.py splice(), which ran 11 integration cases + the e2e test):
  1. K2 defaults: 'MEMBERS': None}  ->  'MEMBERS': None, 'OUTER_FLAT': None}
  2. the KF block (cells/kf_patch_block.py up to its '# ---- KF call lines' comment) right before def _k2_stage
  3. in _k2_stage, after the parent validation: r_pub_src = parent, replaced by _kf_outer_flat(...) when OUTER_FLAT is set
  4. r_pub = _k2_rank(r_pub_src) instead of _k2_rank(pub[...])
  5. status line: outer_flat=K2.get('OUTER_FLAT') in _k2_stage_fleet.json
Only the K8-KF kernel (rsna-sub-k8-kf, MAIN_MODE strict, K2_CFG OUTER_FLAT 0.60) uses the KF cell; rsna-sub-k8-fleet and
rsna-sub-k8-solo ship the K5 cell byte for byte. strip_kf(KF cell) == K5 cell is asserted here and by build_k8.py.

  python make_kf_cell.py [--check]     (--check: regenerate in memory and assert == the file on disk)
"""
import hashlib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace(chr(92), '/')
SP = os.path.dirname(HERE)
K5_CELL = f'{HERE}/cells/k2_fleet_cell.py'
KF_BLOCK = f'{HERE}/cells/kf_patch_block.py'
OUT = f'{HERE}/cells/k2_fleet_cell_kf.py'
GOLD_BLOCK = f'{SP}/gold_prep/kf_patch_block.py'              # the tested block (== verify_parentaudit/kf_patch_block.fixed.py)
KF_BLOCK_SHA = '0fa113ce1bad17e0642366fed0d8af5b57e2aa29df7417a34bed6e1fc7f1e4e2'
K5_CELL_SHA = 'e37f0604997cd40fb4a583e40a3db1f28a2e8d751f72a7aae15cda000d9922dd'

EDITS = [  # (anchor, replacement): each anchor must occur exactly once in the K5 cell
    ("      'MEMBERS': None}", "      'MEMBERS': None, 'OUTER_FLAT': None}"),
    ("    n = len(pub)\n    if not _K2_FLEET_DIR:",
     "    n = len(pub)\n"
     "    r_pub_src = pub[_K2_TARGETS].to_numpy(np.float64)                                          # KF\n"
     "    if K2.get('OUTER_FLAT') is not None:                                                        # KF\n"
     "        r_pub_src = _kf_outer_flat(pub, K2['OUTER_FLAT'], rec)                                  # KF\n"
     "    if not _K2_FLEET_DIR:"),
    ("    r_pub = _k2_rank(pub[_K2_TARGETS].to_numpy(np.float64)); r_fl = _k2_rank(F)",
     "    r_pub = _k2_rank(r_pub_src); r_fl = _k2_rank(F)                                             # KF"),
    ("members_filter=K2.get('MEMBERS'))",
     "members_filter=K2.get('MEMBERS'),\n                  outer_flat=K2.get('OUTER_FLAT'))   # KF"),
]
STAGE = '\ndef _k2_stage(t_cell, rec, procs):'


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def block():
    raw = open(KF_BLOCK, 'rb').read()
    assert sha(raw) == KF_BLOCK_SHA, f'cells/kf_patch_block.py sha {sha(raw)} != the tested KF block {KF_BLOCK_SHA}'
    if os.path.isfile(GOLD_BLOCK):
        assert open(GOLD_BLOCK, 'rb').read() == raw, 'cells/kf_patch_block.py != gold_prep/kf_patch_block.py'
    s = raw.decode('utf-8')
    return s[:s.index('# ---- KF call lines')]


def make():
    k5 = open(K5_CELL, 'rb').read()
    assert sha(k5) == K5_CELL_SHA, f'cells/k2_fleet_cell.py sha {sha(k5)} != the K5 cell {K5_CELL_SHA}'
    s = k5.decode('utf-8')
    for old, new in EDITS:
        assert s.count(old) == 1, f'anchor not unique / absent: {old[:60]!r} ({s.count(old)})'
        s = s.replace(old, new)
    assert s.count(STAGE) == 1
    s = s.replace(STAGE, '\n' + block() + '\n' + STAGE)
    compile(s, 'k2_fleet_cell_kf', 'exec')
    assert strip_kf(s) == k5.decode('utf-8'), 'strip_kf(KF cell) != the K5 cell'
    return s


def strip_kf(s):
    """Undo the 5 KF edits (the inverse of make()); must give the K5 cell byte for byte."""
    b = s.index('# ---- KF begin'); e = s.index('# ---- KF end\n') + len('# ---- KF end\n')
    assert s[b - 1:b] == '\n' and s[e:e + 3] == '\n\n\n', 'KF block framing'        # block's own blank line + the join
    s = s[:b - 1] + s[e + 2:]
    for old, new in reversed(EDITS):
        assert s.count(new) == 1, f'KF edit not found exactly once: {new[:60]!r}'
        s = s.replace(new, old)
    return s


if __name__ == '__main__':
    s = make()
    if '--check' in sys.argv:
        assert open(OUT, 'rb').read() == s.encode('utf-8'), f'{OUT} != make() (stale KF cell: re-run make_kf_cell.py)'
        print(f'KF CELL CHECK OK {OUT} sha256 {sha(s)[:16]}')
    else:
        open(OUT, 'wb').write(s.encode('utf-8'))
        print(f'wrote {OUT} ({len(s)} chars, sha256 {sha(s)[:16]}); strip_kf == K5 cell {K5_CELL_SHA[:16]}')
