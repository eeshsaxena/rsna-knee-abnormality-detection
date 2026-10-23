#!/bin/bash
# uninstall_shim.sh -- MAIN only: restore the pinned trainer (md5 3e62434a) from the backup. Never while a KC line is #RUN.
set -eu
CODE=${KC_SHIM_CODE_DIR:-$HOME/fast_build/code2}
TGT=$CODE/${KC_SHIM_TARGET_NAME:-fleet_train.py}
BAK=${KC_SHIM_BAK:-$HOME/kc/code/fleet_train.py.bak_3e62434a}
REC=${KC_SHIM_REC:-$HOME/kc/code/RECEIPT.txt}
[ "$(md5sum < "$BAK" | cut -c1-32)" = 3e62434adea4fd8971af7c96e0465773 ] || { echo "REFUSED: backup md5 wrong"; exit 2; }
cp -p "$BAK" "$TGT.kcunshim.tmp" && mv -f "$TGT.kcunshim.tmp" "$TGT"
echo "$(date '+%F %T') shim REMOVED: trainer md5 now $(md5sum < "$TGT" | cut -c1-32)" | tee -a "$REC"
