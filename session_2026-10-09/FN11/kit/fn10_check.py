#!/usr/bin/env python3
"""fn10_check.py (2026-10-09, Claude): the commit / A0 gate for the K10 parent-free kernels FN10 (rsna-sub-k10fn, E + D)
and FN10-3 (rsna-sub-k10fn3, C + D + E). It is fn_check.py (kfn kit, UNCHANGED, imported) with one thing set from the
build record: the FN kernel slug = the single MODE 'FN' kernel listed in the given FN_BUILD.json (fn_check.SLUG['fn'] was
the constant 'rsna-sub-k9fn'). Every check, message and exit code is fn_check's; the summary word stays FNCHECK.
10-09: also accepts FN10-3-N70 (rsna-sub-k10fn3-n70, F .30 / N .70); its weights come from its FN_BUILD.json. To pin the
weights on the command line use fnw_check.py --expect_w3 0.30,0.70 (fn_check unchanged + weight checks).

  python fn10_check.py --work DIR --fn_build SP/k10/build_fn/FN_BUILD.json --kernel fn [every fn_check option]
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
sys.path.insert(0, HERE)
import fn_check                                                        # noqa: E402
from fn_check import *                                                 # noqa: E402,F401,F403  (the box harness uses fn_check's names)


def set_slug(fn_build):
    kb = json.load(open(fn_build))
    fn = [s for s, k in kb['kernels'].items() if k.get('mode') == 'FN']
    assert len(fn) == 1, f'{fn_build}: want exactly one MODE FN kernel, got {fn}'
    assert fn[0] in ('rsna-sub-k10fn', 'rsna-sub-k10fn3', 'rsna-sub-k10fn3-n70'), f'{fn[0]} is not a K10 FN kernel'   # N70 slug added 10-09
    fn_check.SLUG['fn'] = fn[0]
    return fn[0]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    set_slug(argv[argv.index('--fn_build') + 1])
    return fn_check.main(argv)


if __name__ == '__main__':
    sys.exit(main())
