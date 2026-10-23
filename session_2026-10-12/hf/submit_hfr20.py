from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
try:
    r = api.competition_submit_code(file_name='submission.csv', competition='rsna-knee-abnormality-detection',
        message='HF-R20: H (Raptor+S6 compact-v1) .8 + K14 R fleet (C+DT+E+E2, 20) .2, PREREG_10-12b rule 2 (last HF test)',
        kernel='eeshsaxena/rsna-sub-hfr20', kernel_version=1)
    print('OK', r)
except Exception as e:
    resp = getattr(e, 'response', None); print('ERR', e, resp.text[:500] if resp is not None else '')
