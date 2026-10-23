# submit_a18.py (10-18, Claude): PREREG_10-15 A18 + addendum (Eesh's explicit go 10-18 ~12:15). Submits each slate probe as soon as its
# gate file ends with 'PROBE_CHECK PASS 16/16'; skips on FAIL; idempotent (checks submits.log AND the Kaggle list); stops at 05:20 IST 10-19.
import time, datetime, os, sys
from kaggle.api.kaggle_api_extended import KaggleApi
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); D = os.path.join(R, 'session_2026-10-18', 'a18')
LOG = os.path.join(R, 'session_2026-10-14', 'submits.log')
SLATE = [('c4', 'v224r6a25c4j', 'A18-C4: a25 best 57316395 + table PF OA .5, Bakers .5, Lat Men .4, Fracture .65 (stacked)'),
         ('fr65', 'v224r6a25fr65j', 'A18-FR65: a25 best 57316395 + Fracture Raptor weight .65 (one line)'),
         ('a15', 'v224r6a15j', 'A18-A15: a25 best 57316395 + A224 .25 -> .15 (one line)'),
         ('c4a15', 'v224r6a15c4j', 'A18-C4A15: a25 best 57316395 + c4 table + A224 .15 (stacked, 2 lines)'),
         ('pf50', 'v224r6a25pf50j', 'A18-PF50: a25 best 57316395 + PF OA Raptor weight .6 -> .5 (one line)')]
STOP = datetime.datetime(2026, 10, 19, 5, 20)
api = KaggleApi(); api.authenticate()
def say(m):
    line = '[%s IST] %s' % (datetime.datetime.now().strftime('%m-%d %H:%M:%S'), m); print(line, flush=True)
    open(LOG, 'a', encoding='utf-8').write(line + '\n')
def submitted_descs():
    try: return [s.description or '' for s in api.competition_submissions('rsna-knee-abnormality-detection')]
    except Exception as e: print('list err', e, flush=True); return None
done = set()
while datetime.datetime.now() < STOP and len(done) < len(SLATE):
    for key, tag, desc in SLATE:
        if key in done: continue
        g = os.path.join(D, 'gate_%s_v1.txt' % tag)
        if not os.path.exists(g): continue
        last = open(g, encoding='utf-8', errors='replace').read().strip().splitlines()[-1:] or ['']
        if 'PROBE_CHECK' not in last[0]: continue
        if 'PASS 16/16' not in last[0]:
            say('A18 %s GATE FAIL -> not submitted: %s' % (key, last[0][:120])); done.add(key); continue
        descs = submitted_descs()
        if descs is None: continue
        if any(d.startswith(desc.split(':')[0] + ':') for d in descs) or ('rsna-sub-' + tag) in open(LOG, encoding='utf-8', errors='replace').read():
            print('already submitted', key, flush=True); done.add(key); continue
        try:
            r = api.competition_submit_code(file_name='submission.csv', message=desc, competition='rsna-knee-abnormality-detection',
                                            kernel='eeshsaxena/rsna-sub-' + tag, kernel_version=1)
            say('A18 %s SUBMITTED eeshsaxena/rsna-sub-%s v1: %s' % (key, tag, r)); done.add(key)
        except Exception as e:
            resp = getattr(e, 'response', None); body = resp.text[:300] if resp is not None else ''
            say('A18 %s SUBMIT ERR %s %s %s' % (key, type(e).__name__, str(e)[:150], body))
            if 'limit' in (str(e) + body).lower(): done.add(key)
    time.sleep(120)
say('A18 submitter finished: done %s' % sorted(done))
