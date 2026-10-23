# JPEG decode check of the V224-F20 stages (private CPU kernel eeshsaxena/rsna-jpeg-check). 2026-10-14, Claude subagent.

No em dashes. Tags: MEASURED (kernel output), MODEL (inference from measured numbers), GUESS.
Nothing submitted, no dataset created, GPU untouched. CPU commits used: 2 of 4 (v1 18:14 IST, v2 18:36 IST; both COMPLETE).

## Verdict
**DROPS_REAL, but only for JPEG Lossless (1.2.840.10008.1.2.4.70 and .4.57), and hidden exposure looks near zero.**
- JPEG Lossless cannot be decoded anywhere in the pinned image (no pylibjpeg / gdcm). Every V224-F20 stage loses it:
  Raptor r384 and r224 zero-fill the slices with no log line; R7 drops the series silently and a fully JPEG Lossless study
  becomes an R7 fallback (stage 3 then uses the Raptor rank, which was computed on a blank volume); the fleet drops the
  series / view silently and a fully JPEG Lossless study is blank (median fill if <= 2 % of studies, else the fleet stage
  fails -> V224-F20 HF_FALLBACK with the Synovitis canary). MEASURED.
- **JPEG 2000 (lossless .4.90 and lossy .4.91), RLE and implicit VR LE decode correctly in every stage of the stock pinned
  image** (pydicom 3.0.2 / 3.0.1 + Pillow 12.3.0 with openjpeg). Lossless J2K output is byte-identical to the uncompressed
  copy in Raptor, R7 and fleet. The mkhamzanov/rsna-b1-ctrl claim is wrong for JPEG 2000 on this pin. MEASURED.
- **The b1-ctrl fix does not work on our pin.** Our docker pin (sha256:2757e0c7...) runs **Python 3.13.15**; goodpjw2008's
  wheels are cp311 / cp312 only and `pip install --target` fails ("no matching distribution"). b1-ctrl uses a different pin
  (37c64f7). MEASURED.
- A working fix exists: **glorysing/rsna-knee-offline-codecs-r09** (public) holds the same versions as cp313 wheels
  (pylibjpeg 2.1.0, pylibjpeg-libjpeg 2.4.0, pylibjpeg-openjpeg 2.6.0), **sha256 byte-identical to the official PyPI wheels**
  (checked locally against pypi.org and asserted in the kernel). On PYTHONPATH they make every stage decode JPEG Lossless
  exactly, and the uncompressed outputs stay **byte-identical** in all 4 stages (same sha with and without). MEASURED.
- Hidden exposure (MODEL): train 24,371 / 24,371 series and 3,000 random train files, and all 557 public-test files, are
  uncompressed explicit VR LE (MEASURED census). Our fleet passed its 98 % coverage gate on the full hidden test in every
  scored HF / solo entry (stock env, no JPEG Lossless decoder) -> at most 2 % of hidden studies (public + private) can be
  fully JPEG Lossless. b1-ctrl (decoders working on its py3.12 pin) and plain R7 both show .956: a fully undecodable study
  costs about 0.91 x its share of macro AUC, so the public split holds at most ~0.1 % such studies (MODEL).

## Main table (v2, out_jc_v2/jpeg_check_table.md)
One real public-test study (`...10062861783145312629332250977456991776`: 5 series, 135 slices, 16-bit signed) transcoded into
each syntax (fake competition root under /tmp, new UIDs keeping the original series order), then run through each reader.
| stage (env it runs in inside V224-F20) | uncompressed (control) | implicit VR LE | JPEG Lossless SV1 .4.70 | JPEG Lossless .4.57 | JPEG 2000 lossless .4.90 | JPEG 2000 lossy .4.91 | RLE .5 | mixed (series 1/3/5 = .4.70) |
|---|---|---|---|---|---|---|---|---|
| plain pydicom, stock (S) | 135/135 | 135/135 exact | **0/135** (RuntimeError: no plugin) | **0/135** | 135/135 exact | 135/135 (lossy) | 135/135 exact | 57/135 |
| Raptor r384 (S, stock) | OK | OK identical | **SILENT DROP 84/84 slices, zero-filled** | **same** | OK identical | lossy diff (max 24/255) | OK identical | **SILENT DROP 44/84 slices** |
| Raptor r224 (R, after R7 pip: pydicom 3.0.1) | OK | OK identical | **SILENT DROP 84/84** | **same** | OK identical | lossy diff | OK identical | **SILENT DROP 44/84** |
| R7 worker (R + CODE on path) | OK (510 windows, 5 series) | OK identical | **STUDY FALLBACK** (item error 'no series file present', 0/5 series) | **same** | OK identical | lossy diff | OK identical | **SILENT DROP 3 of 5 series** (228 vs 510 windows, no error) |
| fleet in V224-F20 (R) | OK (4 views) | OK identical | **STUDY BLANK** (0/4 views) | **same** | OK identical | lossy diff (max 19) | OK identical | **SILENT DROP 2 of 4 views** |
| fleet in HFR7-RR (S) | OK | OK identical | **STUDY BLANK** | **same** | OK identical | lossy diff | OK identical | **SILENT DROP 2 of 4 views** |
| every row above + cp313 wheels on PYTHONPATH (SW / RW / R7W) | OK, **same sha** | OK | **OK identical** | **OK identical** | OK identical | lossy diff | OK | **OK identical** |

U-control output sha by environment (equal = the wheels do not move uncompressed data): Raptor 37f0bb06 in S / SW / R / RW;
fleet 7089b668 in S / SW / R / RW; R7 7d577755 in R7 / R7W. MEASURED.

Fixture validity (v2, in-kernel, decoded with the cp313 wheels): JPEG Lossless .4.70 / .4.57 135/135 exact (own numpy
encoder `jll.py`, also checked locally against pylibjpeg-libjpeg on 12- / 16-bit and signed data and against its own reference
decoder), J2K lossless 135/135 exact (openjpeg via pydicom.compress), RLE 135/135 exact, implicit 135/135 exact, J2K lossy
cr 20 max |d| 528 (16-bit). The prvsiyan synthetic fixtures dataset contains **no compressed file** (48 uncompressed, its README
says so); used only as an uncompressed control (45/48 decode, the 3 failures are its intended refusals).

## How each stage handles an undecodable slice (code read + MEASURED behaviour)
- **Raptor r384 / r224** (embedded `_fastread` 'study_lut', identical B64 in cells 3 and 9): `fast_raw` skips non-LE syntaxes,
  falls back to pydicom; an exception becomes `(None, None)` and `_fill_*` advances the slot index leaving a zero slice
  (mask 0). No log line. An all-zero study still goes through the CoAtNet (no FALLBACK print), so stage 3 never sees it as
  failed (`rap_fail` only catches all-0.5 rows). Partial: windows are drawn between the first and last valid slice.
- **R7** (mainline `training.native_cache.prepare` + `dicom_geometry.pixels_from_dataset`): a slice that raises goes to that
  series' `bad` list; a series with no usable slice is dropped (study `errors`, not printed). No series at all -> the item
  raises -> the study is in the worker's fallback list (BLEND_FALLBACK count in stage 3) and takes the Raptor slot rank.
- **Fleet** (`fleet_common.build_study`): any slice error makes `load_series` raise, the next candidate series of that view
  is tried, else the view is absent (zeros, a state the model trained on). 0 views -> NaN row (`n_blank`). Gates: coverage
  > 98 % and views/study >= 3.0, else the stage fails (V224-F20: HF_FALLBACK, Synovitis = 0.5 canary).

## Environment facts (MEASURED, kernel v1 + v2)
- Pin 2757e0c7: Python 3.13.15, pydicom 3.0.2, Pillow 12.3.0 (jpg + jpg_2000), numpy 2.1.3, cv2 4.14.0, torch 2.11.0+cu128,
  timm 1.0.29; **no** pylibjpeg, pylibjpeg-libjpeg/-openjpeg/-rle, python-gdcm, imagecodecs, pyjpegls.
- pydicom decoder plugins available, stock: RLE = pydicom; JPEG baseline/extended = pillow; J2K .90/.91 = pillow;
  JPEG Lossless .57/.70 = none; JPEG-LS .80/.81 = none. With cp313 wheels: pylibjpeg added for all JPEG / J2K / JPEG-LS.
- R7's own `pip install` (heliosli code dataset) = pydicom 3.0.1 + timm 1.0.25 only; does not add decoders.
- CPU decode cost per study (4-core Kaggle CPU, this 135-slice study): Raptor U 0.7 s, RLE 3.8 s, J2K lossless 13.0 s via
  Pillow (7.3 s with the openjpeg wheel), JPEG Lossless 3.5 s with libjpeg; fleet U 1.2 s, J2K 16.1 s (9.2 s with wheel).
  A J2K-heavy hidden set would be a runtime risk, not a drop risk. Hidden V224 walls (~3 h) look like uncompressed data (GUESS).
- Competition CSVs carry no transfer-syntax column (test.csv = StudyInstanceUID only; test_series.csv = StudyInstanceUID,
  SeriesInstanceUID, Fluid_Sensitive, Fat_Suppression, Anatomical_Plane). Data page (API, data-description) only says:
  "Series come in a mix of transfer syntaxes (uncompressed Explicit VR Little Endian, JPEG Lossless, JPEG 2000, Implicit VR
  Little Endian)." No counts; "about 1300 studies in the test set". Saved in `comp_pages.txt`.

## What a V224-F20-J would be (not built here; main session decides)
- One cell before stage 1: `pip install --no-index --no-deps --target /tmp/_dec --find-links <glorysing dir>` of the 3 cp313
  wheels after asserting their PyPI sha256 (constants in `kernel_src/driver_template.py`, PYPI_SHA), then
  `os.environ['PYTHONPATH'] = '/tmp/_dec' + os.pathsep + old`; fail-closed import check. Raptor subprocesses, the R7
  controller env (CODE + PYTHONPATH) and the fleet workers all inherit it (same pattern as b1-ctrl).
- Proven here at reader level: uncompressed / implicit / RLE / J2K-lossless outputs unchanged (byte-identical), JPEG Lossless
  recovered exactly. Not proven: end-to-end submission identity on the 3 public studies (needs one GPU commit).
- Value (MODEL): ~0 on the public LB (exposure there ~0); private insurance only if the private split has JPEG Lossless
  series, which the census and the fleet gate say is rare. Costs one dataset input (public) and one GPU commit.

## Files (this dir)
`build_jc.py` (assembles the kernel; embeds the V224-F20 fastread B64 and R7 SOURCES with sha asserts), `kernel_src/`
(`driver_template.py`, `probe_jc.py`), `jll.py` (JPEG Lossless encoder for fixtures), `kernel_jc/` (pushed v2,
rsna_jpeg_check.py sha256 ed82d2b3ea5304fd...), `out_jc_v1/`, `out_jc_v2/` (jpeg_check_result.json, jpeg_check_table.md,
kernel log), `push_jc_v1.log`, `push_jc_v2.log`, `wait_jc.sh`, `comp_pages.txt` (competition pages via API),
`pulled/b1ctrl*` (read-only pull of mkhamzanov/rsna-b1-ctrl + its decoder_receipt: public test 15/15 series explicit LE),
`src/` (extracted stage code: raptor_fastread.py, R7 sources + mainline decode modules, fleet_common.py / fleet_worker.py).
v1 note: its decoder install failed (py3.13 vs cp312 wheels) so J2K fixtures were not built and the W phases were
stock; its fleet "implicit differs" was a fixture artefact (hash UIDs re-ordered same-view candidates), fixed in v2.
