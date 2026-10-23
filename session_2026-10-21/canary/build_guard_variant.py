# build_guard_variant.py (10-21 canary track, Claude). LOCAL BUILD ONLY: never pushes, submits or commits anything.
# Builds the r6pw20 guard variant: the A22 r6pw20 kernel (val/kernels/kernel_v224r6flips6fpw20j, sha e4d92c0a) with ONLY the
# S6F time-guard constants of cell 17 changed: G_START_H 5.75 -> 6.25, BUDGET_H 3.3 -> 4.5, DEADLINE_H 8.2 -> 8.5
# (STOP_MARGIN_S 240 and MIN_RUN_S 1200 unchanged). Byte-level text substitution of one exact, unique substring, so every
# other byte of the notebook is the source's. kernel-metadata.json: only id / title / code_file change.
import hashlib, json, os, sys
R = 'C:/Users/eeshs/Downloads/rsna-knee'
SRC_TAG, NEW_TAG = 'v224r6flips6fpw20j', 'v224r6flips6fpw20g85j'
SRC_DIR = R + '/session_2026-10-20/val/kernels/kernel_' + SRC_TAG
OUT_DIR = R + '/session_2026-10-21/canary/kernels/kernel_' + NEW_TAG
SRC_NB = SRC_DIR + '/rsna_sub_' + SRC_TAG + '.ipynb'
OUT_NB = OUT_DIR + '/rsna_sub_' + NEW_TAG + '.ipynb'
SRC_SHA = 'e4d92c0a2c581a8301f87ae4779da21870664814b4af7068dcca55014bbc7757'   # build_receipt_v224r6flips6fpw20j.json out_sha256
OLD = 'G_START_H, BUDGET_H, DEADLINE_H, STOP_MARGIN_S, MIN_RUN_S = 5.75, 3.3, 8.2, 240.0, 1200.0'
NEW = 'G_START_H, BUDGET_H, DEADLINE_H, STOP_MARGIN_S, MIN_RUN_S = 6.25, 4.5, 8.5, 240.0, 1200.0'
sha = lambda b: hashlib.sha256(b).hexdigest()

raw = open(SRC_NB, 'rb').read()
if sha(raw) != SRC_SHA:
    sys.exit('source notebook sha %s != the A22 r6pw20 build %s' % (sha(raw)[:12], SRC_SHA[:12]))
if raw.count(OLD.encode()) != 1:
    sys.exit('guard constants line found %d times (want exactly 1)' % raw.count(OLD.encode()))
out = raw.replace(OLD.encode(), NEW.encode())
meta = json.load(open(SRC_DIR + '/kernel-metadata.json', encoding='utf-8'))
meta2 = dict(meta, id='eeshsaxena/rsna-sub-' + NEW_TAG, title='rsna sub ' + NEW_TAG, code_file='rsna_sub_' + NEW_TAG + '.ipynb')
os.makedirs(OUT_DIR, exist_ok=True)
open(OUT_NB, 'wb').write(out)
mtxt = json.dumps(meta2, indent=2) + '\n'
open(OUT_DIR + '/kernel-metadata.json', 'w', encoding='utf-8', newline='\n').write(mtxt)
rec = {'tag': NEW_TAG, 'base': SRC_TAG, 'src': SRC_NB.replace(R + '/', ''), 'src_sha256': SRC_SHA, 'out_sha256': sha(out),
       'meta_sha256': sha(mtxt.encode()), 'change': {'old': OLD, 'new': NEW, 'cell': 17},
       'guards': {'G_START_H': [5.75, 6.25], 'BUDGET_H': [3.3, 4.5], 'DEADLINE_H': [8.2, 8.5], 'STOP_MARGIN_S': [240.0, 240.0],
                  'MIN_RUN_S': [1200.0, 1200.0]},
       'note': 'LOCAL BUILD ONLY (canary track 10-21); not pushed. Push / submit needs Eesh go + PREREG Amendment 23 + an A23 registry '
               'entry (new kernel sha) + gate run on its commit output (the r6pw20 gate key works unchanged: it checks outputs only).'}
open(R + '/session_2026-10-21/canary/kernels/build_receipt_' + NEW_TAG + '.json', 'w').write(json.dumps(rec, indent=1))
print('BUILT', NEW_TAG, 'out_sha256', sha(out))
