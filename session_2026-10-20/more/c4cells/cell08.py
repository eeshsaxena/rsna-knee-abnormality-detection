# [eeshsaxena V224-FLIP; the ONE change vs V224-exact] R7 anatomical flip TTA forced on. The pinned R7 manifest ships
# flip_tta false, so the cell above built WORKER_ARGS without --flip-tta. compact_infer v4c (sha 8401812c, pinned above) then
# also scores the mirrored windows per fold model, swaps Medial/Lateral Meniscus and Medial/Lateral OA, averages the two passes
# in logit space and keeps MCL from the plain pass; the worker prints FLIP_TTA_READY. Nothing is fitted.
assert SOURCE_SHA['compact_infer.py'] == '8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d'
assert '"--flip-tta"' in SOURCES['compact_infer.py'] and 'FLIP_TTA_READY' in SOURCES['compact_infer.py']
assert not manifest.get('flip_tta'), 'the R7 manifest already has flip_tta on: nothing to force'
assert '--flip-tta' not in WORKER_ARGS, WORKER_ARGS
assert manifest['backbone'] != 'ofl_l', 'flip TTA is for timm backbones only'
manifest['flip_tta'] = True          # in memory only; the dataset file and every sha assert above are unchanged
WORKER_ARGS += ['--flip-tta']        # the same flag her cell above adds when the manifest says flip_tta
FLIP_TTA_FORCED_ON = True
print('FLIP_TTA_FORCED', json.dumps({'worker_args': WORKER_ARGS}), flush=True)
