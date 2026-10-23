import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_lib import patch
K2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
patch(f'{K2}/code/fleet_worker.py', [
("""#   K2c 'loop_min' (main loop only, before the bench) saved next to 'minutes'
""",
"""#   K2c 'loop_min' (main loop only, before the bench) saved next to 'minutes'
#   K2d cv2.setNumThreads(1): 2 workers x 4 decode threads already fill 4 vCPU (the prep kernel that measured
#       0.68 s/study did the same); output identical (row-parallel resize, single-threaded JPEG codec), guarded by the
#       shard-0 PARITY re-encode (exit 4) on Kaggle and the box A0 OOF parity
#   K2e per-study decode timeout (FLEET_STUDY_TIMEOUT_S, default 300 s; a healthy study takes ~2-6 s per thread): a
#       decode thread stuck on one file skips THAT study (row NaN, NV -1, counted in n_fail and stats decode_timeout)
#       instead of blocking the worker until the cell kills it after the wall; the cell's 2 % coverage gate decides
"""),
("""from concurrent.futures import ThreadPoolExecutor
""",
"""from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout
try:                                                                         # K2d
    import cv2; cv2.setNumThreads(1)
except Exception:
    pass
STUDY_TIMEOUT_S = float(os.environ.get('FLEET_STUDY_TIMEOUT_S', 300))       # K2e
"""),
("""            i, st, stats = f.result()
""",
"""            try:                                                            # K2e: a stuck decode skips ONE study
                i, st, stats = f.result(timeout=max(5.0, min(STUDY_TIMEOUT_S, DEADLINE + 60 - time.time())))
            except _FutTimeout:
                n_done += 1; n_fail += 1; STATS['decode_timeout'] = STATS.get('decode_timeout', 0) + 1
                print(f'{TAG} decode timeout: one study skipped (row stays NaN, NV -1)', flush=True)
                continue
"""),
])
