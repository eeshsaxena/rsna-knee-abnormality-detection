# bytes_check.py (checker, 10-21): READ-ONLY. All RSNA submissions with total_bytes, score, status; plus day counts + limits.
import datetime, collections
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
C = 'rsna-knee-abnormality-detection'
print('fetched_utc', datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'))
subs = api.competition_submissions(C, page_size=200) or []
print('ALL_COUNT', len(subs))
days = collections.Counter()
for s in sorted(subs, key=lambda s: str(s.date)):
    days[str(s.date)[:10]] += 1
    print('ROW', s.ref, str(s.date)[:19], str(s.status).split('.')[-1], 'pub=%s' % s.public_score,
          'bytes=%s' % getattr(s, 'total_bytes', None), 'by=%s' % getattr(s, 'submitted_by', ''), (s.description or '')[:34].replace('\n', ' '))
print('DAYS', sorted(days.items())[-6:])
lim = api.competition_get_submission_limits(C) if hasattr(api, 'competition_get_submission_limits') else None
print('LIMITS', lim)
