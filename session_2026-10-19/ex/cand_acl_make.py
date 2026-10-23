# Fills G74P (gold74 pairwise 2x2 counts among ACL, MM, LM, Eff, MOA, LOA, PFOA) and G74J4 (gold74 4-label joint cells),
# both local RSNA aggregates from cand_acl_local.json, into cand_acl_box_template.py -> cand_acl_box_run.py (piped via ssh stdin).
import json
E = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/ex/'
loc = json.load(open(E + 'cand_acl_local.json'))
src = open(E + 'cand_acl_box_template.py', encoding='utf-8').read()
marker = "H = os.path.expanduser('~/oai_build')"
assert marker in src
src = src.replace(marker, marker + '\nG74P = ' + json.dumps(loc['G74P']) + '\nG74J4 = ' + json.dumps(loc['G74J4']), 1)
open(E + 'cand_acl_box_run.py', 'w', encoding='utf-8').write(src)
print('ok', len(loc['G74P']), sum(loc['G74J4']))
