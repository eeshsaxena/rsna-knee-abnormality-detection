#!/usr/bin/env python
# stub_wheels.py (2026-10-15, Claude subagent): LOCAL TEST STAND-INS for the three cp313 decoder wheels, used only by
# j_local_harness.py on this Windows laptop (Python 3.12; the real manylinux cp313 wheels cannot be installed here and nothing is
# downloaded). Same distribution names, versions and import names as the real ones (pylibjpeg 2.1.0, pylibjpeg-libjpeg 2.4.0,
# pylibjpeg-openjpeg 2.6.0) so pip --no-index --no-deps --target, pydicom's pylibjpeg plugin discovery (import + __version__ +
# get_pixel_data_decoders(version=2) from installed entry points) and the import path work exactly as with the real wheels.
# The libjpeg stand-in decodes JPEG Lossless (T.81 process 14, one component, any Huffman table, predictors 1-7, point transform)
# in pure Python; it is NOT the C libjpeg (that decoder was measured on Kaggle: rsna-jpeg-check v2, rsna-v224f20j-check v1).
#   variants: good | broken (first pixel + 1) | slow (sleeps 8 s on import) | shadow (also ships a top-level numpy/ package)
#             | crash / hang / raise (review fix J1: on a 37-row image only, the stand-in for a native decoder fault: os._exit(139)
#               like a SIGSEGV, an endless sleep, or a Python ValueError; every other image decodes as in good)
#   usage: python stub_wheels.py OUT_DIR VARIANT  -> writes the 3 wheels, prints a JSON {wheel name: sha256}
import os, sys, io, json, base64, hashlib, zipfile

DT = (2026, 10, 15, 0, 0, 0)

PYLIBJPEG_INIT = '''__version__ = "2.1.0"
'''
PYLIBJPEG_UTILS = '''# stand-in for pylibjpeg.utils (local tests only)
from importlib.metadata import entry_points
from typing import Callable
Decoder = Callable
DecoderType = Callable


def get_pixel_data_decoders(version=1):
    out = {}
    for ep in entry_points(group="pylibjpeg.pixel_data_decoders"):
        out.setdefault(ep.name, {})[ep.value.split(":")[0]] = ep.load()
    return out


def get_decoders(decoder_type="JPEG"):
    return {}
'''
OPENJPEG_INIT = '''__version__ = "2.6.0"


def decode_pixel_data(src, version=2, **kwargs):
    raise NotImplementedError("stand-in openjpeg: no JPEG 2000 decoding in the local harness")
'''
LIBJPEG_INIT = '''__version__ = "2.4.0"
import os as _os
_BROKEN = __BROKEN__
_TRAP = '__TRAP__'
if __SLOW__:
    import time as _t
    _t.sleep(8)


def _decode(buf):
    b = bytes(buf)
    if b[:2] != b"\\xff\\xd8":
        raise ValueError("no SOI")
    i, huff, P, rows, cols = 2, {}, None, None, None
    while True:
        while b[i] != 0xFF:
            i += 1
        while b[i] == 0xFF:
            i += 1
        m = b[i]
        i += 1
        if m == 0xD8 or 0xD0 <= m <= 0xD7 or m == 0x01:
            continue
        n = (b[i] << 8) | b[i + 1]
        seg = b[i + 2:i + n]
        i += n
        if m == 0xC3:
            P, rows, cols, nf = seg[0], (seg[1] << 8) | seg[2], (seg[3] << 8) | seg[4], seg[5]
            if nf != 1:
                raise NotImplementedError("one component only")
            if rows == 37 and _TRAP == 'crash':
                _os._exit(139)
            if rows == 37 and _TRAP == 'hang':
                import time as _t
                _t.sleep(3600)
            if rows == 37 and _TRAP == 'raise':
                raise ValueError("stand-in decoder: trap image")
        elif m == 0xC4:
            j = 0
            while j < len(seg):
                th, counts = seg[j] & 0x0F, seg[j + 1:j + 17]
                j += 17
                vals = seg[j:j + sum(counts)]
                j += sum(counts)
                table, code, k = {}, 0, 0
                for L in range(1, 17):
                    for _ in range(counts[L - 1]):
                        table[(L, code)] = vals[k]
                        k += 1
                        code += 1
                    code <<= 1
                huff[th] = table
        elif m == 0xDD:
            if (seg[0] << 8) | seg[1]:
                raise NotImplementedError("restart intervals")
        elif m == 0xDA:
            ns = seg[0]
            td = seg[2] >> 4
            ss, pt = seg[1 + 2 * ns], seg[3 + 2 * ns] & 0x0F
            break
        elif m in (0xC0, 0xC1, 0xC2, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            raise NotImplementedError("not a lossless Huffman stream")
    data, j = bytearray(), i
    while j < len(b):
        c = b[j]
        if c == 0xFF:
            if b[j + 1] == 0x00:
                data.append(0xFF)
                j += 2
                continue
            break
        data.append(c)
        j += 1
    table = huff[td]
    pos = [0]
    nbits = len(data) * 8

    def bit():
        p = pos[0]
        pos[0] = p + 1
        return (data[p >> 3] >> (7 - (p & 7))) & 1 if p < nbits else 1

    def take(n):
        v = 0
        for _ in range(n):
            v = (v << 1) | bit()
        return v
    out = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            code, L = 0, 0
            while True:
                code = (code << 1) | bit()
                L += 1
                if (L, code) in table:
                    s = table[(L, code)]
                    break
                if L > 16:
                    raise ValueError("bad Huffman code")
            if s == 0:
                d = 0
            elif s == 16:
                d = 32768
            else:
                e = take(s)
                d = e if e >= (1 << (s - 1)) else e - (1 << s) + 1
            if r == 0 and c == 0:
                px = 1 << (P - pt - 1)
            elif r == 0:
                px = out[0][c - 1]
            elif c == 0:
                px = out[r - 1][0]
            else:
                ra, rb, rc = out[r][c - 1], out[r - 1][c], out[r - 1][c - 1]
                px = {1: ra, 2: rb, 3: rc, 4: ra + rb - rc, 5: ra + ((rb - rc) >> 1), 6: rb + ((ra - rc) >> 1),
                      7: (ra + rb) >> 1}[ss]
            out[r][c] = (px + d) & 0xFFFF
    flat = [v << pt for row in out for v in row]
    if _BROKEN:
        flat[0] = (flat[0] + 1) & 0xFFFF
    import numpy as np
    return np.asarray(flat, dtype="<u2" if P > 8 else "u1").tobytes()


def decode_pixel_data(src, version=2, **kwargs):
    return bytearray(_decode(src))


def decode(src, **kwargs):
    return _decode(src)
'''
LIBJPEG_EP = '''[pylibjpeg.pixel_data_decoders]
1.2.840.10008.1.2.4.57 = libjpeg:decode_pixel_data
1.2.840.10008.1.2.4.70 = libjpeg:decode_pixel_data
'''


def _rec_hash(b):
    return 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(b).digest()).rstrip(b'=').decode()


def wheel(path, dist, ver, files, ep=None):
    di = '%s-%s.dist-info' % (dist.replace('-', '_'), ver)
    files = dict(files)
    files[di + '/METADATA'] = ('Metadata-Version: 2.1\nName: %s\nVersion: %s\nSummary: LOCAL TEST STAND-IN (j_local_harness)\n' % (
        dist, ver)).encode()
    files[di + '/WHEEL'] = b'Wheel-Version: 1.0\nGenerator: stub_wheels.py\nRoot-Is-Purelib: true\nTag: py3-none-any\n'
    if ep:
        files[di + '/entry_points.txt'] = ep.encode()
    rec = ''.join('%s,%s,%d\n' % (n, _rec_hash(b), len(b)) for n, b in sorted(files.items())) + di + '/RECORD,,\n'
    files[di + '/RECORD'] = rec.encode()
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for n in sorted(files):
            zi = zipfile.ZipInfo(n, DT)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, files[n])
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


NAMES = {'pylibjpeg': 'pylibjpeg-2.1.0-py3-none-any.whl', 'libjpeg': 'pylibjpeg_libjpeg-2.4.0-py3-none-any.whl',
         'openjpeg': 'pylibjpeg_openjpeg-2.6.0-py3-none-any.whl'}
TOP_STUB = ['libjpeg', 'openjpeg', 'pylibjpeg', 'pylibjpeg-2.1.0.dist-info', 'pylibjpeg_libjpeg-2.4.0.dist-info',
            'pylibjpeg_openjpeg-2.6.0.dist-info']


def build(out, variant='good'):
    os.makedirs(out, exist_ok=True)
    lj = LIBJPEG_INIT.replace('__BROKEN__', str(variant == 'broken')).replace('__SLOW__', str(variant == 'slow')).replace(
        '__TRAP__', variant if variant in ('crash', 'hang', 'raise') else '')
    ljf = {'libjpeg/__init__.py': lj.encode()}
    if variant == 'shadow':
        ljf['numpy/__init__.py'] = b'raise ImportError("shadow numpy from the decoder target")\n'
    s = {}
    s[NAMES['pylibjpeg']] = wheel(os.path.join(out, NAMES['pylibjpeg']), 'pylibjpeg', '2.1.0',
                                  {'pylibjpeg/__init__.py': PYLIBJPEG_INIT.encode(), 'pylibjpeg/utils.py': PYLIBJPEG_UTILS.encode()})
    s[NAMES['libjpeg']] = wheel(os.path.join(out, NAMES['libjpeg']), 'pylibjpeg-libjpeg', '2.4.0', ljf, LIBJPEG_EP)
    s[NAMES['openjpeg']] = wheel(os.path.join(out, NAMES['openjpeg']), 'pylibjpeg-openjpeg', '2.6.0',
                                 {'openjpeg/__init__.py': OPENJPEG_INIT.encode()})
    return s


if __name__ == '__main__':
    print(json.dumps(build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'good')))
