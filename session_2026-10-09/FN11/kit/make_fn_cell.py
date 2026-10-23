#!/usr/bin/env python3
"""make_fn_cell.py (kfn, 2026-10-09, Claude): the FN / N-only N-stage cell = the K9N N-stage cell
(kfn/cells/k9n_n949_cell.py, a byte copy of k9n/cells/k9n_n949_cell.py, sha16 pinned below) + the FN-P patches below,
each an exact string replacement that must match exactly once. MODE 'K9N' (the default when K9N_CFG has no MODE) keeps
the K9N behaviour; FN and NONLY are selected by the K9N_CFG literal that build_kfn.py writes.

  FN-P0 header comment (what this cell is); no code.
  FN-P1 config defaults MODE 'K9N', RHO_REF 'P' (FN: 'FN' / 'F'; N-only: 'NONLY' / 'NONE').
  FN-P2 NONLY reads its ids from the FN-IDS placeholder _fn_ids.csv (no fleet cell, no _k2_parent.csv); _N_IDS path.
  FN-P3 FN / NONLY: every weight triple has P weight 0 (P is the id-only placeholder, never ranked into a blend);
        NONLY triples are exactly [0, 0, 1]; RHO_REF must be F (FN) / NONE (NONLY); the placeholder file must exist with
        the 12 targets all 0.5, and in FN _k2_parent.csv must be byte-identical to it (proves no real parent ran).
  FN-P4 NONLY: no fleet stage is required (and none may exist); FN / K9N: the K9N fleet checks unchanged.
  FN-P5 NONLY: F = zeros (rank-constant, weight 0).
  FN-P6 Spearman gate reference leg: P (K9N), F (FN; P is constant), or N's own verbatim csv mapped to parent order
        (NONLY; bar .90 since fallback rows are median-filled), once n >= RHO_MIN_N, as in K9N.
  FN-P7 RESULT text names the reference leg; FN-P8 status records MODE / RHO_REF; FN-P9 start log names MODE.
  python make_fn_cell.py [--check]   -> cells/fn_n949_cell.py (+ prints its sha256)
"""
import argparse, hashlib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
SRC = f'{HERE}/cells/k9n_n949_cell.py'
OUT = f'{HERE}/cells/fn_n949_cell.py'
K9N_CELL_SHA256 = None            # filled from the k9n kit at build time (asserted == the K9N_BUILD.json n_cell_sha16)

S2_START = '    # ---- S2 the fleet stage must be clean (strict) and its files untouched\n'
S2_END = "        raise _NFail('parent / fleetsolo has non-finite values')\n"

P = [
    ('FN-P0',
     '# === K9N N STAGE: appended after the K9 fleet stage (cell 31) by k9n/build_k9n.py (2026-10-08, Claude) ===\n',
     '# === FN / N-ONLY N STAGE (kfn/make_fn_cell.py, 2026-10-09, Claude) = the K9N N stage (k9n/cells/k9n_n949_cell.py)\n'
     '# + the FN-P patches marked "FN-P" below. MODE \'FN\': after the parent-free strict fleet stage, P = the FN-IDS\n'
     '# placeholder with weight 0, Spearman gate on F~N. MODE \'NONLY\': N alone (no fleet stage in the kernel), ids from the\n'
     '# FN-IDS placeholder. MODE \'K9N\' (default) = the K9N behaviour. The K9N header follows unchanged.\n'
     '# === K9N N STAGE: appended after the K9 fleet stage (cell 31) by k9n/build_k9n.py (2026-10-08, Claude) ===\n'),
    ('FN-P1',
     "K9N.update(globals().get('K9N_CFG') or {})\n",
     "K9N.update(globals().get('K9N_CFG') or {})\n"
     "K9N.setdefault('MODE', 'K9N'); K9N.setdefault('RHO_REF', 'P')   # FN-P1: 'K9N' | 'FN' | 'NONLY'; Spearman reference leg\n"),
    ('FN-P2',
     "_N_PAR = f'{_N_WORK}/_k2_parent.csv'\n",
     "_N_PAR = f'{_N_WORK}/_fn_ids.csv' if K9N['MODE'] == 'NONLY' else f'{_N_WORK}/_k2_parent.csv'   # FN-P2\n"
     "_N_IDS = f'{_N_WORK}/_fn_ids.csv'   # FN-P2: the FN-IDS id-only placeholder (FN / NONLY)\n"),
    ('FN-P3',
     "            raise _NFail(f'weight triple {w} must be 3 non-negative weights summing to 1 with N > 0')\n",
     "            raise _NFail(f'weight triple {w} must be 3 non-negative weights summing to 1 with N > 0')\n"
     "    if K9N['MODE'] not in ('K9N', 'FN', 'NONLY') or K9N['RHO_REF'] not in ('P', 'F', 'NONE'):   # FN-P3\n"
     "        raise _NFail(f'MODE {K9N[\"MODE\"]!r} / RHO_REF {K9N[\"RHO_REF\"]!r} unknown')\n"
     "    if K9N['MODE'] in ('FN', 'NONLY'):                                  # FN-P3: the placeholder P never carries weight\n"
     "        trip = [w3] + [[float(x) for x in v] for v in K9N['W3_VARIANTS']]\n"
     "        if any(w[0] != 0.0 for w in trip):\n"
     "            raise _NFail(f'{K9N[\"MODE\"]}: P is the id-only placeholder, every triple needs P weight 0, got {trip}')\n"
     "        if K9N['MODE'] == 'NONLY' and any(w != [0.0, 0.0, 1.0] for w in trip):\n"
     "            raise _NFail(f'NONLY: every triple must be [0, 0, 1], got {trip}')\n"
     "        if K9N['RHO_REF'] != {'FN': 'F', 'NONLY': 'NONE'}[K9N['MODE']]:\n"
     "            raise _NFail(f'{K9N[\"MODE\"]}: RHO_REF {K9N[\"RHO_REF\"]!r} (FN gates on F~N, NONLY on N~its own csv)')\n"
     "        try:\n"
     "            ph = pd.read_csv(_N_IDS, dtype={'StudyInstanceUID': str})\n"
     "            phb = open(_N_IDS, 'rb').read()\n"
     "        except Exception as e:\n"
     "            raise _NFail(f'FN-IDS placeholder {os.path.basename(_N_IDS)} unreadable ({e!r}): N not started')\n"
     "        if ph.columns.tolist() != ['StudyInstanceUID', *_N_TARGETS] or not len(ph) \\\n"
     "                or not (ph[_N_TARGETS].to_numpy(np.float64) == 0.5).all():\n"
     "            raise _NFail('FN-IDS placeholder schema / values (all 0.5) wrong')\n"
     "        if K9N['MODE'] == 'FN' and (not os.path.isfile(_N_PAR) or open(_N_PAR, 'rb').read() != phb):\n"
     "            raise _NFail('FN: _k2_parent.csv != the FN-IDS placeholder bytes (a real parent ran, or none)')\n"
     "        rec['placeholder_sha256'] = hashlib.sha256(phb).hexdigest()\n"),
    ('FN-P5',
     "    P = pub[_N_TARGETS].to_numpy(np.float64); F = fso[_N_TARGETS].to_numpy(np.float64)\n",
     "    P = pub[_N_TARGETS].to_numpy(np.float64)\n"
     "    F = fso[_N_TARGETS].to_numpy(np.float64) if fso is not None else np.zeros_like(P)   # FN-P5: NONLY: no F leg\n"),
    ('FN-P6',
     "    rho = []\n"
     "    for j in range(12):\n"
     "        c = np.corrcoef(rP[:, j], rN[:, j])[0, 1] if n > 2 else np.nan\n"
     "        rho.append(float(c) if np.isfinite(c) else float('nan'))\n"
     "    rho_mean = float(np.nanmean(rho)) if np.isfinite(rho).any() else float('nan')\n"
     "    rec.update(rho_parent_n={t: round(r, 4) for t, r in zip(_N_TARGETS, rho)}, rho_mean=rho_mean)\n"
     "    if n >= int(K9N['RHO_MIN_N']) and not rho_mean >= float(K9N['MIN_RHO_MEAN']):\n"
     "        raise _NFail(f'parent~N Spearman mean {rho_mean:.3f} < {K9N[\"MIN_RHO_MEAN\"]} (label order / decode bug?)')\n",
     "    # FN-P6: reference leg of the Spearman gate: P (K9N), F (FN: the placeholder P is constant) or, for NONLY, N's own\n"
     "    # verbatim csv mapped to parent order (row-mapping / label-order guard; bar .90: fallback rows are median-filled)\n"
     "    if K9N['RHO_REF'] == 'NONE':\n"
     "        ref_r = _n_rank(cs.set_index('StudyInstanceUID').loc[ids_par, _N_TARGETS].to_numpy(np.float64))\n"
     "        ref_name, bar = 'N-own-csv', 0.90\n"
     "    else:\n"
     "        ref_r = rP if K9N['RHO_REF'] == 'P' else rF\n"
     "        ref_name, bar = {'P': 'parent', 'F': 'fleet'}[K9N['RHO_REF']], float(K9N['MIN_RHO_MEAN'])\n"
     "    rho = []\n"
     "    for j in range(12):\n"
     "        c = np.corrcoef(ref_r[:, j], rN[:, j])[0, 1] if n > 2 else np.nan\n"
     "        rho.append(float(c) if np.isfinite(c) else float('nan'))\n"
     "    rho_mean = float(np.nanmean(rho)) if np.isfinite(rho).any() else float('nan')\n"
     "    rec.update(rho_parent_n={t: round(r, 4) for t, r in zip(_N_TARGETS, rho)}, rho_mean=rho_mean, rho_ref=ref_name)\n"
     "    if n >= int(K9N['RHO_MIN_N']) and not rho_mean >= bar:\n"
     "        raise _NFail(f'{ref_name}~N Spearman mean {rho_mean:.3f} < {bar} (label order / decode bug?)')\n"),
    ('FN-P7',
     "mirror mean|d| {rec.get(\"mirror_meanabs\") or 0:.4f}, rho(P,N) mean '\n",
     "mirror mean|d| {rec.get(\"mirror_meanabs\") or 0:.4f}, rho({rec.get(\"rho_ref\")}~N) mean '   # FN-P7\n"),
    ('FN-P8',
     "st.update(W3=K9N['W3'], W3_VARIANTS=K9N['W3_VARIANTS'], TAG=K9N.get('TAG'), main_mode='strict',\n",
     "st.update(W3=K9N['W3'], W3_VARIANTS=K9N['W3_VARIANTS'], TAG=K9N.get('TAG'), main_mode='strict',\n"
     "                  MODE=K9N.get('MODE'), RHO_REF=K9N.get('RHO_REF'),   # FN-P8\n"),
    ('FN-P9',
     "f'clean); fail-loud files {loud}; tag {K9N.get(\"TAG\")}')\n",
     "f'clean); fail-loud files {loud}; tag {K9N.get(\"TAG\")}; MODE {K9N.get(\"MODE\")}')   # FN-P9\n"),
]


def make():
    src = open(SRC, encoding='utf-8').read()          # universal newlines (the k9n file is CRLF on disk; build_k9n reads it so)
    assert src.count(S2_START) == 1 and src.count(S2_END) == 1, 'FN-P4 anchors'
    for name, old, new in P:
        assert src.count(old) == 1, f'{name}: anchor found {src.count(old)} times'
        src = src.replace(old, new)
    i = src.index(S2_START); j = src.index(S2_END) + len(S2_END)
    src = src[:i] + s2_new(src[i:j]) + src[j:]
    compile(src, 'fn_n949_cell', 'exec')
    return src


def s2_new(block):
    """FN-P4: the K9N S2 block (fleet checks) re-indented under `else:` after a NONLY branch."""
    assert block.startswith(S2_START) and block.endswith(S2_END)
    body = block[len(S2_START):]
    ind = ''.join(('    ' + l) if l.strip() else l for l in body.splitlines(keepends=True))
    return ('    # ---- S2 the fleet stage must be clean (strict) and its files untouched (FN-P4: NONLY has no fleet stage)\n'
           "    if K9N['MODE'] == 'NONLY':                                       # FN-P4\n"
           "        fst, fso = {}, None\n"
           "        if os.path.exists(_N_FSTAT) or os.path.exists(_N_FSOLO):\n"
           "            raise _NFail('NONLY: a fleet stage output exists (_k2_stage_fleet.json / submission_fleetsolo.csv)')\n"
           "        pub = pd.read_csv(_N_PAR, dtype={'StudyInstanceUID': str})\n"
           "        if pub.columns.tolist() != ['StudyInstanceUID', *_N_TARGETS] or not pub['StudyInstanceUID'].is_unique:\n"
           "            raise _NFail(f'NONLY: id file columns / unique ids {pub.columns.tolist()[:3]}...')\n"
           "    else:                                                            # FN-P4: K9N / FN unchanged\n" + ind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    src = make()
    h = hashlib.sha256(src.encode()).hexdigest()
    if a.check:
        disk = open(OUT, encoding='utf-8', newline='').read()
        assert disk == src, 'cells/fn_n949_cell.py != make(): run python make_fn_cell.py'
        print(f'fn_n949_cell.py OK (== make()), sha256 {h}')
        return
    open(OUT, 'w', encoding='utf-8', newline='').write(src)
    print(f'wrote {OUT} sha256 {h} ({len(P) + 1} FN-P patches incl. FN-P4)')


if __name__ == '__main__':
    main()
