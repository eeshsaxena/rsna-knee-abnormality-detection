#!/usr/bin/env python
# patch_s6_model.py (2026-10-15, Claude): S6 entry builder (PREREG_10-15 Amendment 7 sec 5). Takes ANY built kernel that holds
# the S6F setup cell (S6F, its J twin, R6, KC1 ...) and swaps ONLY the S6fall model the S6F driver loads:
#   - the driver (base64 in the setup cell, sha-checked there) gets a new B_MODEL constant (file / sha256 / run_config / done /
#     expect_*) and reads that model trio from a SECOND mount S6M_SLUG (our private dataset); nartaa's dataset stays attached and
#     still supplies the code manifest + Apache-2.0 code extras byte for byte (I.ds unchanged);
#   - DRIVER_SHA / DRIVER_B64 in the setup cell are re-made; metadata: + our dataset, new id / title / code_file.
# Nothing else in the notebook changes (asserted: every other cell byte-identical, the setup cell differs only in those 2 lines).
# Usage:
#   python patch_s6_model.py --base KERNEL_DIR --tag TAG --ds OWNER/SLUG --model-file F.pt --model-sha SHA256
#          --run-config F.run_config.json --done F.DONE.json [--expect-epochs 16] [--expect-train N|none] [--arm E_prior]
# Writes s6entry/kernel_TAG/, s6entry/build_receipt_TAG.json, s6entry/driver_TAG.py, s6entry/driver_TAG.diff
import os, re, sys, json, base64, hashlib, argparse, difflib
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--ds', required=True)
ap.add_argument('--model-file', required=True)
ap.add_argument('--model-sha', required=True)
ap.add_argument('--run-config', required=True)
ap.add_argument('--done', required=True)
ap.add_argument('--expect-epochs', default='16')
ap.add_argument('--expect-train', default='none')
ap.add_argument('--arm', default='E_prior')
A = ap.parse_args()
sha = lambda b: hashlib.sha256(b).hexdigest()
assert re.fullmatch(r'[0-9a-f]{64}', A.model_sha), 'model sha must be 64 hex'
assert re.fullmatch(r'[a-z0-9_]+', A.tag) and 'standin' not in A.tag
owner, slug = A.ds.split('/')
meta = json.load(open(os.path.join(A.base, 'kernel-metadata.json'), encoding='utf-8'))
nb_path = os.path.join(A.base, meta['code_file'])
nb_raw = open(nb_path, 'rb').read()
nb = json.loads(nb_raw)
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']

# 1. the one setup cell holding the S6F driver
RX_SHA = re.compile(r"^    DRIVER_SHA = '([0-9a-f]{64})'$", re.M)
RX_B64 = re.compile(r"^    DRIVER_B64 = '([A-Za-z0-9+/=]+)'$", re.M)
idx = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and RX_SHA.search(src(c)) and RX_B64.search(src(c))
       and 'def _s6f_setup' in src(c)]
assert len(idx) == 1, ('S6F setup cells found', idx)
ci = idx[0]
cell = src(nb['cells'][ci])
old_sha = RX_SHA.search(cell).group(1)
drv_old = base64.b64decode(RX_B64.search(cell).group(1))
assert sha(drv_old) == old_sha, 'embedded driver differs from its sha'
D = drv_old.decode('utf-8')

# 2. the driver patch (each edit asserted to hit exactly the expected count)
def sub1(s, a, b, n=1):
    k = s.count(a)
    assert k == n, (a[:80], k, n)
    return s.replace(a, b)
m = re.findall(r'^B_MODEL = (\{.*\})$', D, re.M)
assert len(m) == 1, 'B_MODEL line'
bm_old = json.loads(m[0])
assert list(bm_old) == ['file', 'source_file', 'sha256', 'run_config', 'done', 'arm', 'expect_epochs', 'expect_train'], list(bm_old)
num = lambda v: None if str(v).lower() == 'none' else int(v)
bm_new = {'file': A.model_file, 'source_file': A.model_file, 'sha256': A.model_sha, 'run_config': A.run_config, 'done': A.done,
          'arm': A.arm, 'expect_epochs': num(A.expect_epochs), 'expect_train': num(A.expect_train)}
D = sub1(D, 'B_MODEL = ' + m[0] + '\n', 'B_MODEL = ' + json.dumps(bm_new) + '\n')
ds_line = re.findall(r'^DS_SLUG = .*\n', D, re.M)
assert len(ds_line) == 1
D = sub1(D, ds_line[0], ds_line[0] + 'S6M_SLUG = %s                    # [eeshsaxena S6 entry] the S6fall-family model trio (B_MODEL)\n'
         % json.dumps(slug))
D = sub1(D, '    I.ds = find_mount(DS_SLUG, "checkpoint-manifest.json")\n',
         '    I.ds = find_mount(DS_SLUG, "checkpoint-manifest.json")\n    I.dsm = find_mount(S6M_SLUG, B_MODEL["file"])\n')
D = sub1(D, 'I.ds / B_MODEL["file"]', 'I.dsm / B_MODEL["file"]', 2)
D = sub1(D, 'I.ds / B_MODEL["run_config"]', 'I.dsm / B_MODEL["run_config"]')
D = sub1(D, 'I.ds / B_MODEL["done"]', 'I.dsm / B_MODEL["done"]')
D = sub1(D, '"ds": str(I.ds), ', '"ds": str(I.ds), "dsm": str(I.dsm), ')
assert 'I.ds / B_MODEL' not in D
compile(D, 's6f_driver.py', 'exec')
assert '—' not in D and '–' not in D
drv_new = D.encode('utf-8')
new_sha = sha(drv_new)

# 3. the setup cell: only the 2 driver lines change
cell_new = sub1(cell, "    DRIVER_SHA = '%s'" % old_sha, "    DRIVER_SHA = '%s'" % new_sha)
old_b64 = RX_B64.search(cell).group(1)
cell_new = sub1(cell_new, "    DRIVER_B64 = '%s'" % old_b64, "    DRIVER_B64 = '%s'" % base64.b64encode(drv_new).decode('ascii'))
dl = [l for l in difflib.unified_diff(cell.splitlines(), cell_new.splitlines(), lineterm='', n=0) if l[:1] in '+-' and l[:3] not in ('+++', '---')]
assert len(dl) == 4 and all('DRIVER_SHA' in l or 'DRIVER_B64' in l for l in dl), dl
others = sum(src(c).count(old_sha) for i, c in enumerate(nb['cells']) if i != ci)
nb['cells'][ci]['source'] = cell_new

# 4. metadata
ds = list(meta['dataset_sources'])
assert 'nartaa/rsna-knee-s6fall-weights' in ds, 'base has no S6F weights dataset'
assert A.ds not in ds
meta['dataset_sources'] = ds + [A.ds]
code_file = 'rsna_sub_%s.ipynb' % A.tag
meta.update(id='eeshsaxena/rsna-sub-%s' % A.tag, title='rsna sub %s' % A.tag.replace('_', ' '), code_file=code_file)
out = os.path.join(HERE, 'kernel_%s' % A.tag)
os.makedirs(out, exist_ok=True)
p = os.path.join(out, code_file)
json.dump(nb, open(p, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
open(os.path.join(HERE, 'driver_%s.py' % A.tag), 'wb').write(drv_new)
ddiff = list(difflib.unified_diff(drv_old.decode('utf-8').splitlines(), D.splitlines(), 'driver_base', 'driver_' + A.tag, lineterm='', n=0))
open(os.path.join(HERE, 'driver_%s.diff' % A.tag), 'w', encoding='utf-8').write('\n'.join(ddiff) + '\n')

# 5. proof: every other cell byte-identical to the base
nb2 = json.load(open(p, encoding='utf-8'))
nb0 = json.loads(nb_raw)
same = [src(a) == src(b) for a, b in zip(nb0['cells'], nb2['cells'])]
assert len(nb0['cells']) == len(nb2['cells']) and sum(not s for s in same) == 1 and not same[ci]
rec = dict(tag=A.tag, base=os.path.abspath(A.base), base_code_file=meta and os.path.basename(nb_path), base_sha256=sha(nb_raw),
           out=p, out_sha256=sha(open(p, 'rb').read()), cells=len(nb2['cells']), setup_cell=ci,
           driver_old_sha256=old_sha, driver_new_sha256=new_sha, old_sha_elsewhere=others, b_model_old=bm_old, b_model_new=bm_new,
           s6m_slug=slug, dataset_sources=meta['dataset_sources'], driver_diff_lines=sum(1 for l in ddiff if l[:1] in '+-' and l[:3] not in ('+++', '---')))
json.dump(rec, open(os.path.join(HERE, 'build_receipt_%s.json' % A.tag), 'w'), indent=1)
print('S6_PATCH OK', A.tag, 'out', rec['out_sha256'][:16], '| driver', old_sha[:12], '->', new_sha[:12], '| diff lines',
      rec['driver_diff_lines'], '| old sha elsewhere', others, '| model', A.model_file, A.model_sha[:12], '| ds', A.ds)
