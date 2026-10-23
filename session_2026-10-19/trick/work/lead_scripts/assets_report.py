import json, pandas as pd
W = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/work/'
d = json.load(open(W + 'lead_meta/user_assets_top959.json'))
cr = pd.read_csv(W + 'lead_crossers_since1015.csv')[['teamId', 'grp', 'first959_snap', 'prev_score']]
ok = {u: r for u, r in d.items() if 'ds_err' not in r}
print('users with dataset list OK:', len(ok), 'of', len(d))
rows = []
for u, r in ok.items():
    for x in r['datasets']:
        rows.append(dict(user=u, rank=r['rank'], team=r['team'], teamId=r['teamId'], **x))
D = pd.DataFrame(rows)
D['updated'] = pd.to_datetime(D.updated, errors='coerce', format='mixed')
D = D[D.updated >= '2026-10-01']
kw = 'knee|rsna|mri|acl|menisc|label|ckpt|weight|fold|teacher|soft|oof|effusion|synov|coatnet|convnext|raptor'
D = D[D.title.str.contains(kw, case=False) | D.ref.str.contains(kw, case=False) | (D['size'].fillna(0) > 5e7)]
D = D.merge(cr, on='teamId', how='left')
pd.set_option('display.width', 260)
print(D.sort_values(['rank', 'updated'])[['rank', 'team', 'grp', 'ref', 'updated', 'dl', 'size', 'lic']].to_string(index=False, max_colwidth=48))
teams_ok = pd.Series({r['teamId']: 1 for r in ok.values()})
print('teams fully/partly covered:', len(teams_ok))
print('teams with >=1 recent knee-ish public dataset:', D.teamId.nunique())
