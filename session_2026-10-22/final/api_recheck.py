# api_recheck.py (10-21, Claude, recheck track): READ-ONLY Kaggle API audit. No submit, no push, no select, no download.
# Lists all team submissions (to subs_recheck.csv), the SELECTED group, submission limits, the r6pw20 kernel status + metadata,
# and looks up the named ids. Prints everything to stdout.
import csv, datetime, sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_enums import SubmissionGroup
C = 'rsna-knee-abnormality-detection'
SLUG = 'eeshsaxena/rsna-sub-v224r6flips6fpw20j'
NAMED = [('swide', 57464326), ('R6', 57248084), ('pw25', 57463742), ('r6pw20', 57463719), ('spl', 57464296),
         ('salt', 57464911), ('pw30', 57425970), ('pw20', 57425269), ('eoa20', 57413368), ('eoa10', 57413380), ('c4', 57343435)]
api = KaggleApi(); api.authenticate()
now = datetime.datetime.now(datetime.timezone.utc)
print('fetched_utc', now.isoformat(timespec='seconds'))
subs = api.competition_submissions(C, page_size=200) or []
print('ALL_COUNT', len(subs))
s0 = subs[0]
attrs = sorted(a for a in dir(s0) if not a.startswith('_') and not callable(getattr(s0, a, None)))
print('SUB_ATTRS', attrs)
fields = [a for a in ('ref', 'date', 'status', 'public_score', 'private_score', 'submitted_by', 'team_name', 'file_name',
                      'description', 'url', 'total_bytes') if a in attrs]
with open(sys.argv[1] if len(sys.argv) > 1 else 'subs_recheck.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(fields)
    for s in subs: w.writerow([str(getattr(s, a, '')) for a in fields])
row = lambda s: '%s | %s | %s | pub=%s | by=%s | %s' % (s.ref, str(s.date)[:19], str(s.status).split('.')[-1], s.public_score,
                                                     getattr(s, 'submitted_by', '?'), (s.description or '')[:70])
for nm, i in NAMED:
    hit = [s for s in subs if int(s.ref) == i]
    print('NAMED', nm, row(hit[0]) if hit else '%s NOT FOUND' % i)
# per UTC day counts (last 4 days)
days = {}
for s in subs:
    d = str(s.date)[:10]; days[d] = days.get(d, 0) + 1
for d in sorted(days)[-5:]: print('UTC_DAY', d, days[d])
print('TODAY_UTC', now.strftime('%Y-%m-%d'), days.get(now.strftime('%Y-%m-%d'), 0))
print('LATEST5'); [print('  ', row(s)) for s in subs[:5]]
st_counts = {}
for s in subs: k = str(s.status).split('.')[-1]; st_counts[k] = st_counts.get(k, 0) + 1
print('STATUS_COUNTS', st_counts)
# best public
sc = sorted(((float(s.public_score), int(s.ref)) for s in subs if s.public_score not in (None, '')), reverse=True)
print('TOP_PUBLIC', sc[:8])
# selected
sel = api.competition_submissions(C, group=SubmissionGroup.SUBMISSION_GROUP_SELECTED, page_size=50) or []
print('SELECTED_COUNT', len(sel))
for s in sel: print('SELECTED', row(s))
# limits
try:
    lim = api.competition_get_submission_limits(C)
    print('LIMITS', {a: getattr(lim, a) for a in dir(lim) if not a.startswith('_') and not callable(getattr(lim, a, None))})
except Exception as e: print('LIMITS_ERR', repr(e)[:200])
# kernel
try:
    ks = api.kernels_status(SLUG)
    print('KERNEL_STATUS', {a: str(getattr(ks, a)) for a in dir(ks) if not a.startswith('_') and not callable(getattr(ks, a, None))})
except Exception as e: print('KERNEL_STATUS_ERR', repr(e)[:200])
try:
    kl = api.kernels_list(user='eeshsaxena', search='v224r6flips6fpw20j', page_size=20) or []
    for k in kl:
        d = {a: str(getattr(k, a)) for a in dir(k) if not a.startswith('_') and not callable(getattr(k, a, None))}
        print('KERNEL_META', {x: d[x] for x in d if x in ('ref', 'title', 'last_run_time', 'current_version_number', 'version_number',
              'is_private', 'enable_gpu', 'enable_internet', 'kernel_type', 'language', 'author', 'slug', 'id')})
        print('KERNEL_META_KEYS', sorted(d))
except Exception as e: print('KERNEL_LIST_ERR', repr(e)[:200])
