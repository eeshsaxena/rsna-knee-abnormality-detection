# Visible test series: DICOM sequence headers (competition test data, 3 studies; no OAI)
import os, glob, csv, sys
import pydicom
D = '/home/training/rsna_comp/data'
rows = list(csv.DictReader(open(D + '/test_series.csv')))
for r in rows:
    p = os.path.join(D, 'test_series', r['StudyInstanceUID'], r['SeriesInstanceUID'])
    fs = sorted(glob.glob(p + '/*.dcm'))
    if not fs:
        print(r['StudyInstanceUID'][-8:], r['SeriesInstanceUID'][-8:], 'NO FILES'); continue
    ds = pydicom.dcmread(fs[len(fs) // 2], stop_before_pixels=True)
    g = lambda k: str(getattr(ds, k, ''))
    print(r['StudyInstanceUID'][-8:], r['SeriesInstanceUID'][-8:], r['Anatomical_Plane'][:3], 'F%s/FS%s' % (r['Fluid_Sensitive'], r['Fat_Suppression']),
          '| n', len(fs), '|', g('SeriesDescription'), '| TR', g('RepetitionTime')[:7], 'TE', g('EchoTime')[:6], '|', g('ScanningSequence'), '|', g('ScanOptions'), '|', g('Manufacturer')[:12])
