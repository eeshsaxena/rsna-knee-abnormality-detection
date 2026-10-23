#!/usr/bin/env python
# build_stack_kernels.py (10-20 ~23:15 IST, Claude, STACK BUILD round). LOCAL BUILD ONLY, UNARMED: nothing is pushed, committed or registered.
# Builds every stack of stack/inventory.md sec 1 from ONE base: the SCORED pw30 notebook (a21/kernel_v224r6c4pw30j, pushed as v1 and
# submitted as 57425970, nb sha 5a903187...). Each stack = these edits in cell 18 (_s6f_blend, S6F path) and nothing else:
#   [DOSE]   'A_OA = 0.3   # [A21 E-OA pw]' -> 'A_OA = 0.25   # [A21 E-OA pw]'         (pw25 stacks only; exactly one line)
#   [LOAMM]  6 lines after the E-OA print: LOA = pct(pct(LOA) - A_LOAMM pct(MM))      (byte-identical to the night/c12 lomm05 build)
#   [MCLMM]  6 lines after that:            MCL = pct(pct(MCL) - A_MCLMM pct(MM))       (byte-identical to the stack/ inventory build)
# Asserts: base nb + metadata sha (full), base json round-trip is byte-exact, the dose edit alone reproduces val/kernels pw25 (64ed7eba)
# byte for byte, only cell 18 differs from the base (every other cell = whole JSON object), the cell-18 diff is exactly the edits above,
# metadata differs only in id / title / code_file, and (parity) the result equals the inventory build in stack/ when one exists.
# Writes ONLY under stack/kernels/: kernel_<tag>/ + build_receipt_<tag>.json (+ DO_NOT_PUSH.txt for controls) + build_stack_kernels.log.
# Usage: py -3.12 -I build_stack_kernels.py            (builds all 4 stacks + the all-off control)
import json, hashlib, os, sys, difflib, datetime
HERE = os.path.dirname(os.path.abspath(__file__)); STK = os.path.dirname(HERE); S = os.path.dirname(STK)
BASE_DIR = os.path.join(S, 'a21', 'kernel_v224r6c4pw30j')
BASE_NB_SHA = '5a903187415505e9e3566ebd749cae1328b20df6b60bd6593e6443b17fc0c9fd'      # a21/build_receipt_v224r6c4pw30j.json out_sha256
BASE_META_SHA = 'ee1a7c44579b78a4c53ccec6e91c69cd4ae97606ffbb88c7018109886665c49d'
PW25_NB = os.path.join(S, 'val', 'kernels', 'kernel_v224r6c4pw25j', 'rsna_sub_v224r6c4pw25j.ipynb'); PW25_SHA = '64ed7eba5880d6efaa8555042d398b39f547baa23b91f5e2ba4f962938fa8697'
NIGHT_NB = os.path.join(S, 'night', 'c12_kernel_v224r6c4pw30lomm05j', 'rsna_sub_v224r6c4pw30lomm05j.ipynb'); NIGHT_SHA8 = 'c0dd7ec8'
CELL = 18
# name: (A_OA, A_LOAMM or None = block absent, A_MCLMM, control?)
STACKS = {'S_safe': (0.3, None, 0.1, False), 'S_alt': (0.25, None, 0.1, False), 'S_wide': (0.25, 0.05, 0.1, False),
          'S_mid': (0.3, 0.05, 0.1, False), 'CTRL_off': (0.3, 0.0, 0.0, True)}
sha = lambda b: hashlib.sha256(b).hexdigest()
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
# == json.dump(.., indent=1, ensure_ascii=False) to a Windows text-mode utf-8 file (CRLF), the way every earlier builder wrote these notebooks
dump = lambda nb: json.dumps(nb, indent=1, ensure_ascii=False).replace('\n', '\r\n').encode('utf-8')
LOG = open(os.path.join(HERE, 'build_stack_kernels.log'), 'w', encoding='utf-8')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n')

meta0b = open(os.path.join(BASE_DIR, 'kernel-metadata.json'), 'rb').read(); meta0 = json.loads(meta0b)
raw = open(os.path.join(BASE_DIR, meta0['code_file']), 'rb').read()
assert sha(raw) == BASE_NB_SHA, ('base nb sha changed', sha(raw)); assert sha(meta0b) == BASE_META_SHA, ('base meta sha changed', sha(meta0b))
assert dump(json.loads(raw)) == raw, 'base json round-trip is not byte-exact (builder would reformat other cells)'
assert meta0['id'] == 'eeshsaxena/rsna-sub-v224r6c4pw30j'
P('BASE scored pw30 57425970: nb %s | meta %s | json round-trip exact' % (BASE_NB_SHA, BASE_META_SHA))
DOSE_OLD = '        A_OA = 0.3   # [A21 E-OA pw]'
LOAMM = lambda al: [
    "        A_LOAMM = %r   # [night C12'] Lateral OA decorrelated from Medial Meniscus on the final output (NIGHT_1020.md); 0 = no-op" % al,
    "        if A_LOAMM and {'Lateral OA', 'Medial Meniscus'} <= set(labels) and not {'Lateral OA', 'Medial Meniscus'} & set(blank):",
    "            out['Lateral OA'] = pct(pct(out['Lateral OA'].values.astype(np.float64))",
    "                                    - A_LOAMM * pct(out['Medial Meniscus'].values.astype(np.float64))).astype(np.float32)",
    "            rec['a_loamm'] = A_LOAMM",
    "            print('LOAMM_APPLIED a_loamm %g' % A_LOAMM, flush=True)"]
MCLMM = lambda am: [
    "        A_MCLMM = %r   # [stack MCLmm] MCL decorrelated from Medial Meniscus on the final output (stack/inventory.md); 0 = no-op" % am,
    "        if A_MCLMM and {'MCL', 'Medial Meniscus'} <= set(labels) and not {'MCL', 'Medial Meniscus'} & set(blank):",
    "            out['MCL'] = pct(pct(out['MCL'].values.astype(np.float64))",
    "                             - A_MCLMM * pct(out['Medial Meniscus'].values.astype(np.float64))).astype(np.float32)",
    "            rec['a_mclmm'] = A_MCLMM",
    "            print('MCLMM_APPLIED a_mclmm %g' % A_MCLMM, flush=True)"]
night_lines = src(json.load(open(NIGHT_NB, encoding='utf-8'))['cells'][CELL]).split('\n')
assert sha(open(NIGHT_NB, 'rb').read()).startswith(NIGHT_SHA8)
assert all(x in night_lines for x in LOAMM(0.05)), 'LOAMM block differs from the night/c12 lomm05 build'
builder_sha = sha(open(os.path.abspath(__file__), 'rb').read())


def build(name, aoa, al, am, ctrl):
    nb = json.loads(raw); lines = src(nb['cells'][CELL]).split('\n'); edits = []
    # [DOSE]
    k = [j for j, l in enumerate(lines) if l.startswith(DOSE_OLD)]; assert len(k) == 1, k
    if aoa != 0.3:
        new = lines[k[0]].replace('A_OA = 0.3   #', 'A_OA = %r   #' % aoa, 1); assert new != lines[k[0]]; lines[k[0]] = new; edits.append('dose')
        nb_d = json.loads(raw); nb_d['cells'][CELL]['source'] = '\n'.join(lines); b_d = dump(nb_d)
        assert aoa == 0.25 and b_d == open(PW25_NB, 'rb').read() and sha(b_d) == PW25_SHA, 'dose edit alone != val/kernels pw25 build'
    # [LOAMM] + [MCLMM] right after the E-OA print line
    i = [j for j, l in enumerate(lines) if l.startswith("            print('EOA_APPLIED a_oa")]; assert len(i) == 1, i; i = i[0]
    assert lines[i + 1].startswith('        v = out[labels].values.astype(np.float64)'), lines[i:i + 2]
    BLOCK = (LOAMM(al) if al is not None else []) + MCLMM(am)
    lines[i + 1:i + 1] = BLOCK; nb['cells'][CELL]['source'] = '\n'.join(lines)
    tag = 'v224r6c4pw%02d%s%sj' % (round(aoa * 100), ('lomm%02d' % round(al * 100)) if al is not None and (al or ctrl) else '', 'mcl%02d' % round(am * 100))
    out = os.path.join(HERE, 'kernel_' + tag); os.makedirs(out, exist_ok=True); cf = 'rsna_sub_%s.ipynb' % tag
    newb = dump(nb); open(os.path.join(out, cf), 'wb').write(newb)
    meta = dict(meta0); meta.update(id='eeshsaxena/rsna-sub-' + tag, title='rsna sub ' + tag, code_file=cf)
    json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
    metab = open(os.path.join(out, 'kernel-metadata.json'), 'rb').read()
    if ctrl:
        open(os.path.join(out, 'DO_NOT_PUSH.txt'), 'w').write('CONTROL (all stack components off). Never push, never register. stack/kernels/build_stack_kernels.py\n')
    # structural asserts against the base
    A_, B_ = json.loads(raw), json.loads(newb)
    assert len(A_['cells']) == len(B_['cells']) and A_['metadata'] == B_['metadata'] and A_.get('nbformat') == B_.get('nbformat')
    diff_cells = [j for j, (x, y) in enumerate(zip(A_['cells'], B_['cells'])) if x != y]; assert diff_cells == [CELL], diff_cells
    cx, cy = dict(A_['cells'][CELL]), dict(B_['cells'][CELL]); cx.pop('source'); cy.pop('source'); assert cx == cy
    ud = [l for l in difflib.unified_diff(src(A_['cells'][CELL]).split('\n'), src(B_['cells'][CELL]).split('\n'), lineterm='', n=0)
          if l[:1] in '+-' and not l.startswith(('+++', '---'))]
    old_dose = [l for l in src(A_['cells'][CELL]).split('\n') if l.startswith(DOSE_OLD)]; assert len(old_dose) == 1
    exp = (['-' + old_dose[0], '+' + old_dose[0].replace('A_OA = 0.3   #', 'A_OA = %r   #' % aoa, 1)] if aoa != 0.3 else []) + ['+' + l for l in BLOCK]
    assert sorted(ud) == sorted(exp), ('cell-18 diff is not exactly the stack edits', ud)
    m0, m1 = meta0, json.loads(metab); assert {q for q in set(m0) | set(m1) if m0.get(q) != m1.get(q)} == {'id', 'title', 'code_file'}
    # parity with the inventory build in stack/ (independent route for the pw25 stacks: built there on the val pw25 notebook)
    par = None; od = os.path.join(STK, 'kernel_' + tag)
    if os.path.isdir(od):
        onb, ometa = open(os.path.join(od, cf), 'rb').read(), open(os.path.join(od, 'kernel-metadata.json'), 'rb').read()
        par = dict(ref='stack/kernel_' + tag, notebook_identical=onb == newb, metadata_identical=ometa == metab, ref_nb_sha256=sha(onb))
        assert par['notebook_identical'] and par['metadata_identical'], par
    rc = dict(probe='stack', stack=name, control=ctrl, tag=tag, slug='eeshsaxena/rsna-sub-' + tag,
              base='session_2026-10-20/a21/kernel_v224r6c4pw30j', base_scored='pw30 57425970 (.959), pushed v1 10-20 11:42 IST',
              base_nb_sha256=BASE_NB_SHA, base_meta_sha256=BASE_META_SHA,
              a_oa=aoa, oa_mode='pw', a_loamm=al, a_mclmm=am,
              components=[c for c, on in (('pw25 (A_OA .3 -> .25)', aoa != 0.3), ('lomm05 (LOA <- MM a .05)', bool(al)), ('MCLmm (MCL <- MM a .1)', bool(am))) if on],
              out_nb_sha256=sha(newb), meta_sha256=sha(metab), cell=CELL, cells_changed=diff_cells, cell18_diff=ud, inserted_lines=len(BLOCK),
              dose_edit_equals_val_pw25=(aoa != 0.3) or None, loamm_block_equals_night=(al is not None) or None, parity_with_stack_inventory_build=par,
              builder_sha256=builder_sha, built_at_ist=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), armed=False,
              note=('CONTROL: never push. ' if ctrl else '') + 'local build only; needs Eesh GO + PREREG amendment + commit gate (stack/kernels/stack_commit_gate.py) before any push')
    json.dump(rc, open(os.path.join(HERE, 'build_receipt_%s.json' % tag), 'w', encoding='utf-8'), indent=1)
    P('STACK_KBUILT %-9s %-26s nb %s | meta %s | %s | +%d lines%s | parity stack/: %s' % (
        name, tag, sha(newb)[:16], sha(metab)[:16], ', '.join(rc['components']) or 'none (control)', len(BLOCK),
        ' + dose line' if aoa != 0.3 else '', 'n/a' if par is None else ('IDENTICAL nb + meta' if par['notebook_identical'] else 'DIFF')))
    return tag, rc


R = {nm: build(nm, *spec) for nm, spec in STACKS.items()}
json.dump({nm: dict(tag=t, nb_sha256=rc['out_nb_sha256'], meta_sha256=rc['meta_sha256']) for nm, (t, rc) in R.items()},
          open(os.path.join(HERE, 'build_index.json'), 'w'), indent=1)
P('STACK_KBUILD DONE %d kernels (4 stacks + 1 control), base pw30 %s, builder %s' % (len(R), BASE_NB_SHA[:16], builder_sha[:16]))
LOG.close()
