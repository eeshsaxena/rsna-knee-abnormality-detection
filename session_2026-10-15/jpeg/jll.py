# jll.py: minimal JPEG Lossless (ITU T.81 process 14, Huffman, one component) encoder in numpy.
# Used ONLY to build test fixtures (no JPEG Lossless encoder exists in the pinned Kaggle image or in
# the decoder wheels). Output = one complete JPEG stream (SOI SOF3 DHT SOS data EOI), first-order
# prediction with selection value 1 (Ra, left), point transform 0 = DICOM 1.2.840.10008.1.2.4.70.
# Huffman table: the 17 difference categories SSSS 0..16 all get a 5-bit code (canonical: code = SSSS),
# a valid (not optimal) table. Correctness is checked by decoding the result with independent decoders
# (pylibjpeg-libjpeg in the kernel, gdcm and libjpeg locally) and comparing with the source pixels.
import numpy as np

CODE_LEN = 5


def _marker_seg(marker, payload):
    n = len(payload) + 2
    return bytes([0xFF, marker, (n >> 8) & 0xFF, n & 0xFF]) + bytes(payload)


def encode_sv1(arr, precision):
    """arr: 2-D integer array with values in [0, 2**precision). -> JPEG Lossless SV1 bytes."""
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError('2-D only')
    P = int(precision)
    if not 2 <= P <= 16:
        raise ValueError('precision %d' % P)
    x = a.astype(np.int64)
    if x.min() < 0 or x.max() >= (1 << P):
        raise ValueError('values outside [0, 2**P)')
    rows, cols = x.shape
    pred = np.empty_like(x)
    pred[0, 0] = 1 << (P - 1)
    pred[0, 1:] = x[0, :-1]
    pred[1:, 0] = x[:-1, 0]
    pred[1:, 1:] = x[1:, :-1]
    d = (x - pred) & 0xFFFF                              # modulo 2**16 (T.81 H.1.2.1)
    d = np.where(d >= 32768, d - 65536, d)               # signed; -32768 is coded as SSSS 16 (= +32768)
    mag = np.abs(d)
    ssss = np.zeros_like(d)
    for k in range(1, 17):
        ssss[mag >= (1 << (k - 1))] = k
    xlen = np.where(ssss == 16, 0, ssss)                 # SSSS 16 carries no extra bits
    extra = np.where(d >= 0, d, d + (1 << ssss) - 1) & ((1 << xlen) - 1)
    val = ((ssss << xlen) | extra).astype(np.uint64).ravel()
    ln = (CODE_LEN + xlen).astype(np.int64).ravel()
    sh = (64 - ln).astype(np.uint64)
    w = (val << sh).astype('>u8')
    bits = np.unpackbits(w.view(np.uint8)).reshape(-1, 64)
    keep = np.arange(64)[None, :] < ln[:, None]
    stream = bits[keep]
    pad = (-len(stream)) % 8
    if pad:
        stream = np.concatenate([stream, np.ones(pad, np.uint8)])   # pad with 1-bits (T.81 F.1.2.3)
    data = np.packbits(stream)
    ff = np.flatnonzero(data == 0xFF)
    if len(ff):
        data = np.insert(data, ff + 1, 0)                # byte stuffing
    sof = _marker_seg(0xC3, [P, rows >> 8, rows & 0xFF, cols >> 8, cols & 0xFF, 1, 1, 0x11, 0])
    bits_cnt = [0] * 16
    bits_cnt[CODE_LEN - 1] = 17
    dht = _marker_seg(0xC4, [0x00] + bits_cnt + list(range(17)))
    sos = _marker_seg(0xDA, [1, 1, 0x00, 1, 0, 0])      # Ns 1, Cs 1, Td/Ta 0, Ss 1 (predictor), Se 0, Ah/Al 0
    return b'\xFF\xD8' + sof + dht + sos + data.tobytes() + b'\xFF\xD9'


def decode_sv1_reference(buf):
    """Tiny pure-python decoder for THIS encoder's streams (5-bit table only), for self-tests."""
    b = bytearray(buf)
    assert b[:2] == b'\xFF\xD8'
    i = 2; P = rows = cols = None
    while True:
        assert b[i] == 0xFF
        m = b[i + 1]; n = (b[i + 2] << 8) | b[i + 3]
        seg = b[i + 4:i + 2 + n]
        if m == 0xC3:
            P = seg[0]; rows = (seg[1] << 8) | seg[2]; cols = (seg[3] << 8) | seg[4]
        i += 2 + n
        if m == 0xDA:
            break
    data = bytearray(); j = i
    while j < len(b):
        if b[j] == 0xFF:
            if b[j + 1] == 0x00:
                data.append(0xFF); j += 2; continue
            break
        data.append(b[j]); j += 1
    bitarr = np.unpackbits(np.frombuffer(bytes(data), np.uint8))
    pos = 0
    out = np.zeros((rows, cols), np.int64)

    def take(n):
        nonlocal pos
        v = 0
        for _ in range(n):
            v = (v << 1) | int(bitarr[pos]); pos += 1
        return v
    for r in range(rows):
        for c in range(cols):
            s = take(CODE_LEN)
            if s == 0:
                dd = 0
            elif s == 16:
                dd = 32768
            else:
                e = take(s)
                dd = e if e >= (1 << (s - 1)) else e - (1 << s) + 1
            if r == 0 and c == 0:
                p = 1 << (P - 1)
            elif r == 0:
                p = out[0, c - 1]
            elif c == 0:
                p = out[r - 1, 0]
            else:
                p = out[r, c - 1]
            out[r, c] = (p + dd) & 0xFFFF
    return out
