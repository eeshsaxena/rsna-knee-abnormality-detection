# watch_reads.py (10-16, Claude): read-only poll of our submissions; exits (printing a line) as soon as any watched ref leaves PENDING.
import time, sys
from kaggle.api.kaggle_api_extended import KaggleApi
W = set(sys.argv[1:])
a = KaggleApi(); a.authenticate()
while True:
    try:
        subs = a.competition_submissions('rsna-knee-abnormality-detection')
        done = [s for s in subs if str(getattr(s, 'ref', '')) in W and 'PENDING' not in str(getattr(s, 'status', ''))]
        if done:
            for s in subs[:6]:
                print(getattr(s, 'ref', None), getattr(s, 'status', None), getattr(s, 'public_score', None), str(getattr(s, 'description', ''))[:50], flush=True)
            sys.exit(0)
    except Exception as e:
        print('poll error', e, flush=True)
    time.sleep(600)
