# smoke_probe_local.py (Claude, 2026-10-14): local smoke run of kernel_src/probe_jj.py 'probe' mode (Python312, no Kaggle data):
# a tiny fake root (2 studies x 1 series x 2 files from the DEC self-test control image), the plain reader, the synthetic reader,
# and the raptor / fleet readers against MOCK modules with the real call signatures. Catches wiring bugs before a Kaggle commit.
import os, sys, io, json, base64, shutil, tempfile, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
tmp = tempfile.mkdtemp(prefix='smoke_')
fx = json.load(open(os.path.join(HERE, '..', 'selftest_fixtures.json')))
import pydicom
for k, sid in (('a', '1.2.826.0.1.3680043.8.498.71'), ('b', '1.2.826.0.1.3680043.8.498.72')):
    d = os.path.join(tmp, 'root', 'test_series', sid, '1.2.826.0.1.3680043.8.498.81')
    os.makedirs(d)
    for i in range(2):
        open(os.path.join(d, '%d.dcm' % i), 'wb').write(base64.b64decode(fx['U_16S'][0]))
open(os.path.join(tmp, 'root', 'test.csv'), 'w').write('StudyInstanceUID\n1.2.826.0.1.3680043.8.498.71\n1.2.826.0.1.3680043.8.498.72\n')
open(os.path.join(tmp, 'root', 'test_series.csv'), 'w').write(
    'StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,Anatomical_Plane\n'
    '1.2.826.0.1.3680043.8.498.71,1.2.826.0.1.3680043.8.498.81,0,0,Sagittal\n1.2.826.0.1.3680043.8.498.72,1.2.826.0.1.3680043.8.498.81,0,0,Sagittal\n')
os.makedirs(os.path.join(tmp, 'synth', 'corpus', 'dicom'))
for n in ('x.dcm', 'y.dcm'):
    open(os.path.join(tmp, 'synth', 'corpus', 'dicom', n), 'wb').write(base64.b64decode(fx['U_16S'][0]))
open(os.path.join(tmp, 'synth', 'corpus', 'dicom', 'bad.dcm'), 'wb').write(b'not dicom')
open(os.path.join(tmp, 'fastread_mock.py'), 'w').write('''
import numpy as np, pydicom, glob, os
def make_builder(mode, img, slots, hdr_threads=1, px_threads=1):
    assert mode == 'study_lut'
    def build(sid, SER, tsdir):
        assert sid in SER
        fs = sorted(glob.glob(os.path.join(tsdir, sid, '*', '*.dcm')))
        vol = np.stack([pydicom.dcmread(f).pixel_array for f in fs]).astype(np.uint8)
        return vol, np.ones(len(vol), np.uint8)
    return build
''')
os.makedirs(os.path.join(tmp, 'fleet'))
open(os.path.join(tmp, 'fleet', 'fleet_common.py'), 'w').write('''
import numpy as np, pydicom, glob, os
VIEWS = [('Axial', 1), ('Sagittal', 0), ('Coronal', 1), ('Sagittal', 1)]
def build_study(df, img_root, sid, return_bytes=False, stats=None, errs=None):
    out = {v: None for v in VIEWS}
    uid = str(df['SeriesInstanceUID'].iloc[0])
    fs = sorted(glob.glob(os.path.join(img_root, sid, uid, '*.dcm')))
    out[('Sagittal', 0)] = (np.stack([pydicom.dcmread(f).pixel_array for f in fs]).astype(np.uint8), 2, None, uid)
    return out
''')
res = os.path.join(tmp, 'res.json')
r = subprocess.run([PY, os.path.join(HERE, '..', 'kernel_src', 'probe_jj.py'), 'probe', '--phase', 'S', '--readers', 'plain,raptor,fleet',
                    '--result', res, '--fake', os.path.join(tmp, 'root'), '--real', os.path.join(tmp, 'root'), '--fastread',
                    os.path.join(tmp, 'fastread_mock.py'), '--fleet', os.path.join(tmp, 'fleet'), '--synth', os.path.join(tmp, 'synth'),
                    '--mods', tmp, '--tmp', tmp], capture_output=True, text=True)
print(r.stdout[-2500:], r.stderr[-1500:])
out = json.load(open(res))
want = ['plain_fake', 'plain_real', 'synthetic', 'raptor_fake', 'raptor_real', 'fleet_fake', 'fleet_real']
ok = all(k in out and 'step_error' not in out[k] for k in want)
ok = ok and out['plain_real']['1.2.826.0.1.3680043.8.498.71']['ok'] == 2 and out['plain_real']['1.2.826.0.1.3680043.8.498.71']['sha'] == \
    out['plain_real']['1.2.826.0.1.3680043.8.498.72']['sha']
ok = ok and out['synthetic']['bad.dcm'].get('error') and out['synthetic']['x.dcm']['sha'] == out['synthetic']['y.dcm']['sha']
ok = ok and out['raptor_real']['1.2.826.0.1.3680043.8.498.71']['mask_sum'] == 2 and out['fleet_real']['1.2.826.0.1.3680043.8.498.72']['views'] == ['Sag0']
ok = ok and out['fleet_real']['1.2.826.0.1.3680043.8.498.72']['series'] == [None, '81', None, None]
print({k: (list(out[k].values())[0] if isinstance(out.get(k), dict) and out[k] else out.get(k)) for k in want})
shutil.rmtree(tmp, ignore_errors=True)
print('SMOKE_PROBE_LOCAL', 'PASS' if ok else 'FAIL')
