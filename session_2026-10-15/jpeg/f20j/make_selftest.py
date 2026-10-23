#!/usr/bin/env python
# make_selftest.py (2026-10-14, Claude subagent): the tiny DICOM self-test images embedded in the V224-F20-J DEC cell.
# Run with a Python that has pydicom 3.x + numpy (here: Python312). Writes selftest_fixtures.json:
#   name -> [base64 DICOM file, expected pixel_array dtype, expected shape, sha256 of the expected array bytes]
#   JLL70_16S : JPEG Lossless SV1 (1.2.840.10008.1.2.4.70), 16-bit signed (BitsStored 16, PixelRepresentation 1)
#   JLL57_12U : JPEG Lossless P14 (1.2.840.10008.1.2.4.57), 12-bit unsigned (BitsStored 12, PixelRepresentation 0)
#   U_16S     : the JLL70_16S array as uncompressed Explicit VR Little Endian (control)
# The JPEG streams come from ../jll.py (our numpy encoder) and are verified here with jll.decode_sv1_reference; the real
# decoder check (pylibjpeg-libjpeg) happens inside the kernel (the DEC cell enables the decoders only if all 3 are exact).
import os, sys, io, json, base64, hashlib, importlib.util
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import JPEGLosslessSV1, JPEGLossless, ExplicitVRLittleEndian, MRImageStorage, UID

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('jll', os.path.join(HERE, '..', 'jll.py'))
jll = importlib.util.module_from_spec(spec); spec.loader.exec_module(jll)

ROWS, COLS = 40, 56
rng = np.random.default_rng(1015)
yy, xx = np.mgrid[0:ROWS, 0:COLS]
base = np.sin(xx / 7.0) * np.cos(yy / 5.0)                      # smooth "anatomy"
edge = (np.hypot(yy - 20, xx - 28) < 11).astype(np.float64)       # a sharp disc edge


def make(bits, signed):
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    mid, span = (lo + hi) / 2.0, (hi - lo) / 2.0
    a = mid + 0.55 * span * base + 0.30 * span * edge + rng.normal(0, span * 0.02, base.shape)
    a = np.clip(np.round(a), lo, hi).astype(np.int64)
    a[0, 0], a[0, 1], a[-1, -1], a[-1, -2] = lo, hi, hi, lo        # the range ends
    return a


def dicom(arr, bits, signed, ts):
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = MRImageStorage
    sop = UID('1.2.826.0.1.3680043.8.498.1015.%d.%d.%d' % (bits, int(signed), len(str(ts))))
    fm.MediaStorageSOPInstanceUID = sop
    fm.TransferSyntaxUID = ts
    ds = Dataset()
    ds.file_meta = fm
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = sop
    ds.StudyInstanceUID = sop + '.1'
    ds.SeriesInstanceUID = sop + '.2'
    ds.Modality = 'MR'
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = 16
    ds.BitsStored = bits
    ds.HighBit = bits - 1
    ds.PixelRepresentation = int(signed)
    if ts == ExplicitVRLittleEndian:
        ds.PixelData = arr.astype('<i2' if signed else '<u2').tobytes()
        ds['PixelData'].VR = 'OW'
    else:
        raw = arr & ((1 << bits) - 1) if signed else arr                       # two's complement in BitsStored bits
        stream = jll.encode_sv1(raw, bits)
        back = jll.decode_sv1_reference(stream)
        assert np.array_equal(back, raw), 'jll reference round trip failed'
        ds.PixelData = encapsulate([stream])
        ds['PixelData'].VR = 'OB'
        ds['PixelData'].is_undefined_length = True
    bio = io.BytesIO()
    ds.save_as(bio, enforce_file_format=True)
    return bio.getvalue()


out = {}
a16 = make(16, True)
a12 = make(12, False)
for name, arr, bits, signed, ts in (('JLL70_16S', a16, 16, True, JPEGLosslessSV1), ('JLL57_12U', a12, 12, False, JPEGLossless),
                                    ('U_16S', a16, 16, True, ExplicitVRLittleEndian)):
    b = dicom(arr, bits, signed, ts)
    want = np.ascontiguousarray(arr.astype('<i2' if signed else '<u2'))
    rd = pydicom.dcmread(io.BytesIO(b))
    assert str(rd.file_meta.TransferSyntaxUID) == str(ts)
    if ts == ExplicitVRLittleEndian:                                            # local pydicom can decode the control
        assert np.array_equal(rd.pixel_array, want) and str(rd.pixel_array.dtype) == str(want.dtype)
    out[name] = [base64.b64encode(b).decode('ascii'), str(want.dtype), list(want.shape),
                 hashlib.sha256(want.tobytes()).hexdigest()]
    print(name, str(ts), len(b), 'bytes', 'range', int(arr.min()), int(arr.max()), out[name][1:])
p = os.path.join(HERE, 'selftest_fixtures.json')
with open(p, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(out, f, indent=1, sort_keys=True)
print('WROTE', p, hashlib.sha256(open(p, 'rb').read()).hexdigest(), 'pydicom', pydicom.__version__, 'numpy', np.__version__)
