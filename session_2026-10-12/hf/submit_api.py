import sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
try:
    r = api.competition_submit_code(file_name='submission.csv', message='HV heliosli Raptor+S6 verbatim, PREREG_10-12 item 1, runtime probe',
                                    competition='rsna-knee-abnormality-detection', kernel='eeshsaxena/rsna-sub-hv', kernel_version=3)
    print('OK', r)
except Exception as e:
    print('ERR', type(e).__name__, e)
    resp = getattr(e, 'response', None)
    if resp is not None:
        print('BODY', resp.status_code, resp.text[:2000])
