# [eeshsaxena CANARY; not in the public notebook] V224's only silent fallback is "blend without the 224 model" (stage 2b
# skipped by the 9 h guard, timed out, failed, csv missing or unusable: r224 is None above). Make it visible on the
# leaderboard: Fracture = 0.5 for every study (one constant column, about -.03 macro AUC). Clean path: nothing changes.
import json as _cj
USE_F, FLIP_TTA_ON = False, bool(globals().get('FLIP_TTA_FORCED_ON', False))   # no fleet cell in this kernel
_canary = {'r224_used': r224 is not None, 'R224_OK': bool(R224_OK), 'use_f': USE_F, 'flip_tta_forced': FLIP_TTA_ON, 'fired': []}
if r224 is None:
    assert 'Fracture' in labels
    sub['Fracture'] = np.float32(0.5)
    assert sub['StudyInstanceUID'].tolist() == test_ids and np.isfinite(sub[labels].values).all()
    sub.to_csv(WORK / 'submission.csv', index=False)
    _canary['fired'].append('r224->Fracture')
    print('CANARY_FIRED r224 not used (R224_OK=%s): Fracture set to 0.5 for all %d rows' % (R224_OK, n), flush=True)
else:
    print('CANARY_CLEAN r224 used, no column blanked', flush=True)
(WORK / 'canary_receipt.json').write_text(_cj.dumps(_canary, indent=1))
print('V224_DONE', _cj.dumps(_canary), flush=True)
