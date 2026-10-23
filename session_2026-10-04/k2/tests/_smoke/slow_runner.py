# K2e test runner (smoke only): the shipped worker, with fleet_common.build_study sleeping 30 s on ONE study
import os, sys, time, runpy
sys.path.insert(0, os.environ['FLEET_DIR'])
import fleet_common as fc
_orig, SLOW = fc.build_study, os.environ['SLOW_SID']
def build_study(df, root, sid, *a, **k):
    if sid == SLOW:
        time.sleep(30)
    return _orig(df, root, sid, *a, **k)
fc.build_study = build_study
w = os.environ['FLEET_DIR'] + '/fleet_worker.py'
sys.argv = [w]
runpy.run_path(w, run_name='__main__')
