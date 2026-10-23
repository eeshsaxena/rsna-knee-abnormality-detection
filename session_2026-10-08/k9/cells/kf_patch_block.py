# ---- KF begin (gold_prep/parent_audit.md; additive: this block + the 2 call lines marked KF in _k2_stage and the 1
# status line marked KF in _k2_entry_inner; every output is byte-identical to K5 when K2['OUTER_FLAT'] is None).
# KF = flat outer-map hedge for final pick 2. The parent's cell 28 routes each label as
#   final_l = rank_pct((1 - w_l) * rank_pct(S3_l) + w_l * side_l)
# with w_l = .60, except ACL .75, Medial Meniscus .80, Lateral Meniscus 1.00, Lateral OA .75, Fracture .75: the
# "probe #22" map, tuned on the public split (742050; maverickss26 config cell: 0.939 flat -> 0.941 mapped).
# With OUTER_FLAT = x, the K-kit rebuilds the parent final from cell 28's own persisted inputs with w_l = x for every
# label and blends THAT with the fleet. Inputs (both survive cell 30's cleanup, which only removes pixel caches):
#   btk_v32_rest_input.npz : cell 28's exact _ke_ours (stage 3 = DINO + A5 + Rad + calibrator), saved in _coat_substitute
#   _raptor.csv            : the Raptor/CoAt side after _coat_substitute (0.6 rank(pub) + 0.4 family, not re-ranked)
# Self-check before anything is used: the same rebuild with the parent's live map (cell 28 global _coatnet_weight,
# which must equal the known map) must reproduce _k2_parent.csv to <= 1e-9, and stage 3 must equal cell 26's saved
# diagnostics/btkd_parent_after_rad.npz to <= 1e-9 on all 12 labels (the rebuild alone is blind to stage 3 on Lateral
# Meniscus, where the map weight is 1.00). Any failure raises _K2Fail, so a strict slot errors instead of scoring
# something unintended.
_KF_PARENT_MAP = {'ACL': 0.75, 'Medial Meniscus': 0.80, 'Lateral Meniscus': 1.00, 'Lateral OA': 0.75, 'Fracture': 0.75}
_KF_DEFAULT_W = 0.60


def _kf_outer_flat(pub, flat, rec):
    """Return the parent final rebuilt with a flat outer CoAt weight (n x 12, parent row order)."""
    flat = float(flat)
    if not 0.0 < flat < 1.0:
        raise _K2Fail(f'OUTER_FLAT {flat} outside (0, 1)')
    ids = pub['StudyInstanceUID'].astype(str).tolist()
    want = {t: float(_KF_PARENT_MAP.get(t, _KF_DEFAULT_W)) for t in _K2_TARGETS}
    live = globals().get('_coatnet_weight')                              # cell 28 global (absent in a box stage test)
    if live is not None:
        got = {t: float(live.get(t, -1.0)) for t in _K2_TARGETS} if isinstance(live, dict) else None
        if got != want:
            raise _K2Fail(f'parent outer map drift: cell 28 has {got}, KF expects {want}')
    p_npz, p_side = f'{_K2_WORK}/btk_v32_rest_input.npz', f'{_K2_WORK}/_raptor.csv'
    for p in (p_npz, p_side):
        if not os.path.isfile(p):
            raise _K2Fail(f'KF input {os.path.basename(p)} missing (cell 28 did not persist it?)')
    z = np.load(p_npz, allow_pickle=False)
    if [str(x) for x in z['labels']] != _K2_TARGETS:
        raise _K2Fail(f'KF stage-3 labels {[str(x) for x in z["labels"]]}')
    zu = [str(u) for u in z['study_uids']]
    if len(set(zu)) != len(zu) or sorted(zu) != sorted(ids):
        raise _K2Fail('KF stage-3 study ids != parent ids')
    pos = {u: i for i, u in enumerate(zu)}
    s3 = np.asarray(z['values'], np.float64)[[pos[u] for u in ids]]
    sd = pd.read_csv(p_side, dtype={'StudyInstanceUID': str})
    if sd.columns.tolist() != ['StudyInstanceUID', *_K2_TARGETS] or not sd['StudyInstanceUID'].is_unique \
            or sorted(sd['StudyInstanceUID'].tolist()) != sorted(ids):
        raise _K2Fail('KF side (_raptor.csv) schema / ids != parent')
    side = sd.set_index('StudyInstanceUID').loc[ids, _K2_TARGETS].to_numpy(np.float64)
    if not (np.isfinite(s3).all() and np.isfinite(side).all()):
        raise _K2Fail('KF inputs not finite')
    # Verifier fix (10-06): the parent-map rebuild below cannot see stage 3 on Lateral Meniscus (map w 1.00 drops it),
    # the label the hedge moves most. Cross-check ALL 12 stage-3 columns against cell 26's in-memory copy, saved by
    # rsna_save_predictions('btkd_parent_after_rad') right before it wrote _pipeline_stage.csv (cell 28 reads that
    # CSV back as _ke_ours). Measured: 0.0 on the 3 K5 commit outputs, 1.1e-16 (box CSV parse) on the gold58 mimic.
    p_rad = f'{_K2_WORK}/diagnostics/btkd_parent_after_rad.npz'
    if not os.path.isfile(p_rad):
        raise _K2Fail('KF cross-check input diagnostics/btkd_parent_after_rad.npz missing')
    zr = np.load(p_rad, allow_pickle=False)
    zru = [str(u) for u in zr['study_uids']]
    if [str(x) for x in zr['labels']] != _K2_TARGETS or len(set(zru)) != len(zru) or sorted(zru) != sorted(ids):
        raise _K2Fail('KF cross-check (btkd_parent_after_rad.npz) labels / ids != parent')
    pr = {u: i for i, u in enumerate(zru)}
    d_s3 = float(np.abs(np.asarray(zr['values'], np.float64)[[pr[u] for u in ids]] - s3).max())
    if not d_s3 <= 1e-9:
        raise _K2Fail(f'KF stage 3 (btk_v32_rest_input.npz) != cell 26 output (max abs {d_s3:.3e})')
    tr = _k2_rank(s3)                                                    # cell 28: _ke_tr = rank_pct(_ke_ours)

    def route(wmap):                                                     # cell 28 outer blend, same float order
        w = np.array([wmap[t] for t in _K2_TARGETS], np.float64)[None, :]
        return _k2_rank((1.0 - w) * tr + w * side)

    par = pub[_K2_TARGETS].to_numpy(np.float64)
    d_chk = float(np.abs(route(want) - par).max())
    info = {'flat': flat, 'parent_map': want, 'rebuild_max_abs': d_chk, 'stage3_vs_cell26_max_abs': d_s3}
    rec['outer_flat'] = info
    if not d_chk <= 1e-9:
        raise _K2Fail(f'KF rebuild with the parent map != _k2_parent.csv (max abs {d_chk:.3e})')
    out = route({t: flat for t in _K2_TARGETS})
    if not (np.isfinite(out).all() and (out > 0).all() and (out <= 1).all()):
        raise _K2Fail('KF flat parent not finite / outside (0, 1]')
    n = len(out)
    if n > 2:
        rho = [float(np.corrcoef(_k2_rank(out[:, [j]])[:, 0], _k2_rank(par[:, [j]])[:, 0])[0, 1]) for j in range(12)]
        info['rho_flat_vs_parent'] = {t: round(r, 4) for t, r in zip(_K2_TARGETS, rho)}
    info['mars_flat_vs_parent'] = float(np.abs(out - par).mean())
    info['changed_labels'] = [t for t in _K2_TARGETS if abs(want[t] - flat) > 1e-12]
    _k2_log(f'KF outer map flattened to {flat}: rebuild check max abs {d_chk:.1e}, mean abs rank shift '
            f'{info["mars_flat_vs_parent"]:.4f}, labels changed {info["changed_labels"]}')
    return out
# ---- KF end

# ---- KF call lines (inside _k2_stage, right after the parent validation `n = len(pub)`):
#     r_pub_src = pub[_K2_TARGETS].to_numpy(np.float64)                                          # KF
#     if K2.get('OUTER_FLAT') is not None:                                                        # KF
#         r_pub_src = _kf_outer_flat(pub, K2['OUTER_FLAT'], rec)                                  # KF
# and later replace
#     r_pub = _k2_rank(pub[_K2_TARGETS].to_numpy(np.float64)); r_fl = _k2_rank(F)
# with
#     r_pub = _k2_rank(r_pub_src); r_fl = _k2_rank(F)                                             # KF
# ---- KF status line (in _k2_entry_inner, the st.update(...) call): add  outer_flat=K2.get('OUTER_FLAT')   # KF
# ---- KF default (K2 dict): 'OUTER_FLAT': None
