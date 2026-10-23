# ops_subs_now.py (10-21, Claude, ops track): READ-ONLY list of our team's competition submissions (status / score / date / desc prefix).
# No submit, no push, no download. Prints the attribute names once so we know whether a 'selected' flag is exposed.
import datetime, sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
subs = list(api.competition_submissions('rsna-knee-abnormality-detection'))
print('n', len(subs), 'fetched_utc', datetime.datetime.utcnow().isoformat(timespec='seconds'))
s0 = subs[0]
print('attrs', sorted(a for a in dir(s0) if not a.startswith('_') and not callable(getattr(s0, a, None)))[:60])
for s in subs[:int(sys.argv[1]) if len(sys.argv) > 1 else 16]:
    g = lambda *ns: next((getattr(s, n) for n in ns if getattr(s, n, None) is not None), None)
    print(g('ref', 'id'), str(g('date', 'date_submitted'))[:19], str(g('status')), g('public_score', 'publicScore'),
          g('private_score', 'privateScore'), 'sel=%s' % g('is_selected', 'selected', 'isSelected'), (g('description') or '')[:28])
