#!/bin/bash
# fail-before / pass-after (10-15 review S6R-2): a px_build worker that dies OUTSIDE the per-study try (here: torch cannot be
# imported under a 600 MB address-space limit, as an OOM / import failure would) must make px_build_s6.py exit non-zero.
# CPU only, cores 28-39, a scratch --out in /tmp. Usage: test_px_build_exit.sh [PX_BUILD]  ->  PX_BUILD_EXIT_TEST PASS|FAIL
PXB="${1:-$HOME/s6seed/px_build_s6.py}"
T="$(mktemp -d /tmp/s6pxb.XXXXXX)"; echo "1.2.3.4" > "$T/uids.txt"; mkdir -p "$T/data"
( ulimit -v 600000; exec taskset -c 28-39 nice -n 19 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$HOME/miniconda3/envs/kmimic/bin/python" -u "$PXB" --data "$T/data" --split train --uids "$T/uids.txt" --src "$HOME/s6seed/src" \
    --code "$HOME/s6seed/code" --anchor "$HOME/s6seed/weights/anatomy.pt" --out "$T/out" --procs 1 --threads 1 --tmp "$T/tmp" --mem-cap-gb 0 ) > "$T/log" 2>&1
rc=$?
grep -E 'all workers exited|WORKER_FAILURE|BUILD_END' "$T/log"
died=0; grep -qE 'all workers exited: \[[^0]' "$T/log" && died=1
echo "PX_BUILD_EXIT_TEST $([ $died = 1 ] && [ $rc != 0 ] && echo PASS || echo FAIL) (worker died $died, px_build rc $rc, script $(sha256sum "$PXB" | cut -c1-12), log $T/log)"
