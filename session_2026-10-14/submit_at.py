# submit_at.py KERNEL VERSION "MESSAGE": submit one gated kernel version via the python API (prints ref or the error body).
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
k, v, msg = sys.argv[1], int(sys.argv[2]), sys.argv[3]
try:
    r = api.competition_submit_code(file_name='submission.csv', message=msg, competition='rsna-knee-abnormality-detection', kernel=k, kernel_version=v)
    print('OK', k, v, r)
except Exception as e:
    print('ERR', k, v, type(e).__name__, e); resp = getattr(e, 'response', None)
    if resp is not None: print('BODY', resp.status_code, resp.text[:2000])
