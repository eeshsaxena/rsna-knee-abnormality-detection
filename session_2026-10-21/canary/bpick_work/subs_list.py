# read-only: list our competition submissions (no submit). RESEARCH ONLY.
import json, sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
subs = list(api.competition_submissions('rsna-knee-abnormality-detection'))
rows = []
for s in subs:
    d = {k: str(getattr(s, k, None)) for k in ('ref', 'date', 'status', 'public_score', 'private_score', 'description', 'error_description', 'file_name')}
    rows.append(d)
json.dump(rows, open(sys.argv[1], 'w'), indent=1)
for d in rows[:25]:
    print(d['ref'], d['date'], d['status'], d['public_score'], (d['description'] or '')[:60], d['error_description'][:60] if d['error_description'] else '')
