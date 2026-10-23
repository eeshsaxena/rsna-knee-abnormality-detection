import time, sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
ref = int(sys.argv[1]); t0 = time.time()
while True:
    try:
        subs = api.competition_submissions('rsna-knee-abnormality-detection')
        s = next((x for x in subs if int(getattr(x, 'ref', 0) or 0) == ref), None)
        st = str(getattr(s, 'status', '?')); sc = getattr(s, 'public_score', None) or getattr(s, 'publicScore', None)
        print(time.strftime('%H:%M:%S'), ref, st, sc, flush=True)
        if s is not None and ('COMPLETE' in st.upper() or 'ERROR' in st.upper()):
            break
    except Exception as e:
        print(time.strftime('%H:%M:%S'), 'poll error', e, flush=True)
    if time.time() - t0 > 10.5 * 3600: break
    time.sleep(600)
