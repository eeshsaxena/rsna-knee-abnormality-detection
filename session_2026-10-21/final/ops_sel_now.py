# ops_sel_now.py (10-21, Claude, ops track): READ-ONLY. Lists the team's CURRENTLY SELECTED final submissions (SubmissionGroup.SELECTED)
# and looks up a few named ids in the full list. No submit, no push, no download.
# Usage: py ops_sel_now.py [id ...]   default ids = swide 57464326, R6 57248084
import sys, datetime
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_enums import SubmissionGroup
C = 'rsna-knee-abnormality-detection'
api = KaggleApi(); api.authenticate()
want = [int(x) for x in sys.argv[1:]] or [57464326, 57248084]
row = lambda s: '%s %s %s pub=%s desc=%s' % (s.ref, str(s.date)[:19], str(s.status).split('.')[-1], s.public_score, (s.description or '')[:40])
sel = api.competition_submissions(C, group=SubmissionGroup.SUBMISSION_GROUP_SELECTED, page_size=50) or []
print('fetched_utc', datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'))
print('SELECTED_COUNT', len(sel))
for s in sel: print('SELECTED', row(s))
allsubs = api.competition_submissions(C, page_size=200) or []
print('ALL_COUNT', len(allsubs))
for w in want:
    hit = [s for s in allsubs if int(s.ref) == w]
    print('ID', w, row(hit[0]) if hit else 'NOT FOUND')
ok = sorted(int(s.ref) for s in sel) == sorted(want) if len(sys.argv) > 1 else None
print('SELECTION_MATCHES_ARGS', ok)
