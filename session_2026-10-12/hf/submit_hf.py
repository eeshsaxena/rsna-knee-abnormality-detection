import sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
for k, v, msg in [('eeshsaxena/rsna-sub-hf20', 2, 'HF20: H (heliosli Raptor+S6, compact-v1) .8 + our fleet D (5 TTR224) .2 rank blend, PREREG_10-12'),
                  ('eeshsaxena/rsna-sub-hf30', 1, 'HF30: H (heliosli Raptor+S6, compact-v1) .7 + our fleet D (5 TTR224) .3 rank blend, PREREG_10-12')]:
    try:
        r = api.competition_submit_code(file_name='submission.csv', message=msg, competition='rsna-knee-abnormality-detection', kernel=k, kernel_version=v)
        print('OK', k, v, r)
    except Exception as e:
        resp = getattr(e, 'response', None); print('ERR', k, v, e, resp.text[:500] if resp is not None else '')
