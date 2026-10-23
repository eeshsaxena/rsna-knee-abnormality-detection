# sel_validate.py (checker, 10-21): READ-ONLY. Validates that the SELECTED-group list call can return non-zero,
# by running it on every competition the account has entered (incl. finished ones with known final picks).
import datetime
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_enums import SubmissionGroup
api = KaggleApi(); api.authenticate()
print('fetched_utc', datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'))
slugs = []
for grp in ('entered',):
    tok = None
    for _ in range(5):
        r = api.competitions_list(group=grp, page_size=50, page_token=tok)
        comps = getattr(r, 'competitions', None) or []
        for c in comps:
            ref = str(c.ref).rstrip('/').split('/')[-1]
            slugs.append((ref, str(getattr(c, 'deadline', ''))[:19]))
        tok = getattr(r, 'next_page_token', None)
        if not tok: break
print('ENTERED', len(slugs))
for s, dl in slugs:
    try:
        sel = api.competition_submissions(s, group=SubmissionGroup.SUBMISSION_GROUP_SELECTED, page_size=50) or []
        n = len(sel)
        ids = ' '.join(str(x.ref) for x in sel[:5])
    except Exception as e:
        n = 'ERR'; ids = type(e).__name__
    print('COMP', s, 'deadline', dl, 'SELECTED', n, ids)
