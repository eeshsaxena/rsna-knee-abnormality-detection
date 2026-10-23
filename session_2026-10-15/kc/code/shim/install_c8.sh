#!/bin/bash
# install_c8.sh -- BOX_PLAN_1015 C8, run by MAIN after review: copy the KC trainer files into ~/fast_build/code2 as NEW files
# (kc_common.py, kc_g1.py, fleet_train_kc.py; no existing file is edited), refuse to overwrite a different existing copy,
# and append their md5s to ~/kc/code/RECEIPT.txt. Idempotent (an identical existing copy is accepted).
set -eu
SRC=${KC_C8_SRC:-$HOME/kc/code}; CODE=${KC_C8_CODE_DIR:-$HOME/fast_build/code2}; REC=${KC_C8_REC:-$HOME/kc/code/RECEIPT.txt}
for f in kc_common.py kc_g1.py fleet_train_kc.py; do
  m=$(md5sum < "$SRC/$f" | cut -c1-32)
  if [ -e "$CODE/$f" ]; then
    [ "$(md5sum < "$CODE/$f" | cut -c1-32)" = "$m" ] || { echo "REFUSED: $CODE/$f exists with another md5"; exit 2; }
    echo "C8 $f already present (md5 $m)"
  else
    cp -p "$SRC/$f" "$CODE/$f.c8tmp" && mv -n "$CODE/$f.c8tmp" "$CODE/$f"
    echo "$(date '+%F %T') C8 copied $f md5 $m -> $CODE/" | tee -a "$REC"
  fi
done
