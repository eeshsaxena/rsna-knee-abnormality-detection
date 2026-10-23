# subs_status.py (10-21, plan1022 track). READ-ONLY: lists our competition submissions (status / date / public score / desc prefix).
import collections
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
subs = list(api.competition_submissions('rsna-knee-abnormality-detection'))
cnt = collections.Counter(str(s.status) for s in subs)
print('n', len(subs), dict(cnt))
for s in subs[:30]:
    print(s.ref, s.date, str(s.status), s.public_score, (s.description or '')[:40])
