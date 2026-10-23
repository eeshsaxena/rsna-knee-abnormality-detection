#!/bin/bash
# fake_py.sh (rental_kc tests only): the mock's PYI. `fake_py.sh fleet_train.py <argv>` runs tests/fake_trainer.py (no GPU);
# anything else runs the box rsna python unchanged (setup.sh parity probe, lane receipts).
P=$HOME/miniconda3/envs/rsna/bin/python
if [ "${1:-}" = fleet_train.py ]; then exec $P $HOME/rkc/src/tests/fake_trainer.py "$@"; fi   # keeps fleet_train.py in the cmdline (stop_member checks it)
exec $P "$@"
