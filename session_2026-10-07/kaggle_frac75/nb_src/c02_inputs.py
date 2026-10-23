# [step 1] locate the attached inputs, verify every dataset file against kt_manifest.json, copy the trainer code,
# verify the frac75 training labels (dataset eeshsaxena/rsna-knee-frac75-labels) against the sha256 baked in cell 1.
log(f'kernel t0 {ist(T0)} ({T0_SRC}); notebook start +{(T_NB0 - T0) / 60:.1f} min; root {KT_ROOT or "/"}')
save_receipt()
DS, TARS, MANS = locate_inputs()
EXP = json.load(open(f'{DS}/kt_expect.json'))
BOX = json.load(open(f'{DS}/box_facts.json'))
log(f'dataset {DS} | {len(TARS)} tar parts | manifests {sorted(MANS)}')
for n in sorted(TARS):
    log(f'  {n}  {os.path.getsize(TARS[n]) / 1e9:.3f} GB  {TARS[n]}')

_man = json.load(open(f'{DS}/kt_manifest.json'))
_soft = set(LABEL_FILES) | {EXP['timm_wheel']}        # csvs: checked below (CRLF-tolerant); wheel: optional fallback
_bad = [f for f, h in _man['files'].items()
        if f not in _soft and (not os.path.exists(f'{DS}/{f}') or sha256_file(f'{DS}/{f}') != h)]
if _bad:
    abort(f'dataset files missing or changed vs kt_manifest.json: {_bad}')
_w = EXP['timm_wheel']
WHEEL_OK = os.path.exists(f'{DS}/{_w}') and sha256_file(f'{DS}/{_w}') == _man['files'][_w]
if not WHEEL_OK:
    warn(f'{_w} missing or changed: the timm fallback would use PyPI (only needed if timm < 1.0 here)')
log(f'dataset files verified: {len(_man["files"]) - len(_soft)} sha256 match kt_manifest.json (+ csvs below)')

_want = set(EXP['parts'])
if set(TARS) != _want:
    abort(f'prep tar parts: missing {sorted(_want - set(TARS))}, unexpected {sorted(set(TARS) - _want)} '
          f'(attach kernel outputs eeshsaxena/rsna-fleet-prep-s0 and -s1)')
for s, mp in sorted(MANS.items()):
    m = json.load(open(mp))
    for n, h in m['parts'].items():
        if EXP['parts'].get(n) != h:
            abort(f'{mp}: part {n} sha {h} != expected {EXP["parts"].get(n)}')
if sorted(MANS) != [0, 1]:
    warn(f'prep manifests found for shards {sorted(MANS)} only (tar hashes are still checked against kt_expect.json)')

shutil.rmtree(CODE, ignore_errors=True)
os.makedirs(CODE)
for f in CODE_FILES:
    shutil.copy2(f'{DS}/{f}', f'{CODE}/{f}')
    if sha256_file(f'{CODE}/{f}') != BOX['code_sha256'][f]:
        abort(f'{f} differs from the box copy (~/bigjump_build/code/{f})')
with open(f'{CODE}/kt_modelcheck.py', 'w') as f:
    f.write(MODELCHECK_SRC)
shutil.rmtree(LBL, ignore_errors=True)
os.makedirs(LBL)
for f in LABEL_FILES:                       # the trainer reads these copies; bytes == the box ~/rsna_train files
    b = open(f'{DS}/{f}', 'rb').read()
    if hashlib.sha256(b).hexdigest() != EXP['labels_sha256'][f]:
        b = b.replace(bytes([13, 10]), bytes([10]))          # CRLF -> LF
        if hashlib.sha256(b).hexdigest() != EXP['labels_sha256'][f]:
            abort(f'{f} differs from the box ~/rsna_train/{f}')
        warn(f'{f}: CRLF line endings on Kaggle; LF-normalised copy == the box bytes, using it')
    with open(f'{LBL}/{f}', 'wb') as o:
        o.write(b)
shutil.copy2(f'{DS}/box_facts.json', f'{LBL}/box_facts.json')
# the frac75 TRAINING labels (the one recipe change): second dataset, sha256 baked into this notebook (CFG), and the
# box property "only the Fracture column differs from labels_mix50_logit.csv" re-checked on the exact bytes used
FRAC_SRC = locate_train_labels()
_fb = open(FRAC_SRC, 'rb').read()
_fv = labels_variant(_fb, CFG['LABELS_TRAIN_SHA256'])
if _fv is None:
    abort(f'{CFG["LABELS_TRAIN"]} at {FRAC_SRC}: sha256 {hashlib.sha256(_fb).hexdigest()} != baked '
          f'{CFG["LABELS_TRAIN_SHA256"]} (wrong or changed dataset eeshsaxena/rsna-knee-frac75-labels)')
_fb, _fhow = _fv
if _fhow != 'exact':
    warn(f'{CFG["LABELS_TRAIN"]}: line endings rewritten on Kaggle ({_fhow} restores the box bytes), using the box bytes')
if hashlib.md5(_fb).hexdigest() != CFG['LABELS_TRAIN_MD5']:
    abort(f'{CFG["LABELS_TRAIN"]}: md5 != the PREREG md5 {CFG["LABELS_TRAIN_MD5"]}')
if not only_fracture_differs(_fb, open(f'{LBL}/labels_mix50_logit.csv', 'rb').read()):
    abort(f'{CFG["LABELS_TRAIN"]}: not "labels_mix50_logit.csv with only the Fracture column changed"')
with open(f'{LBL}/{CFG["LABELS_TRAIN"]}', 'wb') as o:
    o.write(_fb)
REC['inputs'] = dict(dataset=DS, labels_dir=LBL, tars=TARS, manifests=MANS, code_sha256=BOX['code_sha256'],
                     labels_sha256=EXP['labels_sha256'], wheel_ok=WHEEL_OK, train_labels=CFG['LABELS_TRAIN'],
                     train_labels_src=FRAC_SRC, train_labels_sha256=CFG['LABELS_TRAIN_SHA256'],
                     train_labels_md5=CFG['LABELS_TRAIN_MD5'], train_labels_line_endings=_fhow)
stage('inputs', ok=True)
log('code == box ~/bigjump_build/code (6 files), labels == box ~/rsna_train (4 files), '
    f'{CFG["LABELS_TRAIN"]} == box ~/novel/frac75 (sha256 {CFG["LABELS_TRAIN_SHA256"][:16]}, only Fracture differs '
    f'from mix50) from {FRAC_SRC}')
