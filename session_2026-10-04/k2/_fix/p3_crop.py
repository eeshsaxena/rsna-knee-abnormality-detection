import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_lib import patch
K2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
patch(f'{K2}/cells/k2_crop_post_cell.py', [
("""# Each group is fail-loud on its own: a file exists only if every stage it depends on finished cleanly. The fleet
# stage's files and submission.csv are never touched here.
""",
"""# Each group is fail-loud on its own: a file exists only if every stage it depends on finished cleanly. The fleet
# stage's files and submission.csv are never touched here.
# C6 after the CF writer has read the sandbox, every submission*.csv the K1 cell left in /kaggle/working/_k2_crop is
#    renamed k1out*.csv: those are parent copies (crop failed) or P+C blends under K2's own names (submission.csv,
#    submission_w20.csv), so none may be picked in the submit dialog. The only submittable files are top level.
"""),
("""                except BaseException:
                    pass
    try:
        rec['stage_min'] = (time.time() - globals().get('_k2c_t0', time.time())) / 60
""",
"""                except BaseException:
                    pass
    try:                                                              # C6 sandbox outputs never look submittable
        ren = []
        for f in (sorted(os.listdir(_K2C_SAND)) if os.path.isdir(_K2C_SAND) else []):
            if f.startswith('submission') and f.endswith('.csv'):
                os.replace(f'{_K2C_SAND}/{f}', f'{_K2C_SAND}/k1out{f[len("submission"):]}'); ren.append(f)
        rec['sandbox_renamed'] = ren
    except BaseException as e:
        rec['sandbox_rename'] = f'failed {e!r}'
    try:
        rec['stage_min'] = (time.time() - globals().get('_k2c_t0', time.time())) / 60
"""),
])
