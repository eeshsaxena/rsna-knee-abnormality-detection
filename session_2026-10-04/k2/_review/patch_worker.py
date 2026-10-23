import os
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fleet_worker_patched.py')
s = open(p, encoding='utf-8').read()
a = "from concurrent.futures import ThreadPoolExecutor\n"
assert a in s
s = s.replace(a, "from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout\n"
                 "try:                                                                         # K2d\n"
                 "    import cv2; cv2.setNumThreads(1)        # 2 workers x 4 decode threads on 4 vCPU (prep kernel did the same)\n"
                 "except Exception:\n"
                 "    pass\n"
                 "STUDY_TIMEOUT_S = float(os.environ.get('FLEET_STUDY_TIMEOUT_S', 300))       # K2e\n")
b = "            i, st, stats = f.result()\n"
assert b in s
s = s.replace(b, "            try:                                                            # K2e: a stuck decode skips ONE study\n"
                 "                i, st, stats = f.result(timeout=max(5.0, min(STUDY_TIMEOUT_S, DEADLINE + 60 - time.time())))\n"
                 "            except _FutTimeout:\n"
                 "                n_done += 1; n_fail += 1; STATS['decode_timeout'] = STATS.get('decode_timeout', 0) + 1\n"
                 "                print(f'{TAG} decode timeout: one study skipped (row stays NaN, NV -1)', flush=True)\n"
                 "                continue\n")
open(p, 'w', encoding='utf-8').write(s)
print('ok')
