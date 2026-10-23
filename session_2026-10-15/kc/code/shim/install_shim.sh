#!/bin/bash
# install_shim.sh -- BOX_PLAN_1015 M4, run by MAIN only, AFTER (1) PREREG_TEACHER Amendment A3 (shim) is filed and
# (2) install_c8.sh. Replaces the trainer in ~/fast_build/code2 by the shimmed text ~/kc/code/shim/ft_shimmed.py.txt
# (= the pinned source + the 5-line --kc dispatch block before the first import; patch: fleet_train_shim.patch).
# md5 3e62434a -> 8d12515f. Backup kept at ~/kc/code/fleet_train.py.bak_3e62434a. Temp file + mv in the same dir: the
# running T_a f3 process is unaffected (Python compiled its source at launch). Rollback: uninstall_shim.sh.
set -eu
CODE=${KC_SHIM_CODE_DIR:-$HOME/fast_build/code2}
TGT=$CODE/${KC_SHIM_TARGET_NAME:-fleet_train.py}
SHIMMED=${KC_SHIM_TEXT:-$HOME/kc/code/shim/ft_shimmed.py.txt}
BAK=${KC_SHIM_BAK:-$HOME/kc/code/fleet_train.py.bak_3e62434a}
REC=${KC_SHIM_REC:-$HOME/kc/code/RECEIPT.txt}
WANT_PRE=3e62434adea4fd8971af7c96e0465773; WANT_POST=8d12515f237d204ad63ca93bd74b36b9
pre=$(md5sum < "$TGT" | cut -c1-32)
if [ "$pre" = "$WANT_POST" ]; then echo "shim already installed (md5 $pre)"; exit 0; fi
[ "$pre" = "$WANT_PRE" ] || { echo "REFUSED: target md5 $pre != $WANT_PRE"; exit 2; }
[ "$(md5sum < "$SHIMMED" | cut -c1-32)" = "$WANT_POST" ] || { echo "REFUSED: shimmed text md5 drift"; exit 2; }
for f in fleet_train_kc.py kc_common.py kc_g1.py; do [ -f "$CODE/$f" ] || { echo "REFUSED: $CODE/$f missing (run install_c8.sh first)"; exit 2; }; done
[ -e "$BAK" ] || cp -p "$TGT" "$BAK"
[ "$(md5sum < "$BAK" | cut -c1-32)" = "$WANT_PRE" ] || { echo "REFUSED: backup $BAK md5 is not $WANT_PRE"; exit 2; }
cp "$SHIMMED" "$TGT.kcshim.tmp" && chmod --reference="$TGT" "$TGT.kcshim.tmp" && mv -f "$TGT.kcshim.tmp" "$TGT"
post=$(md5sum < "$TGT" | cut -c1-32)
[ "$post" = "$WANT_POST" ] || { echo "ERROR: post-install md5 $post != $WANT_POST (restore with uninstall_shim.sh)"; exit 3; }
echo "$(date '+%F %T') M4 shim installed: trainer md5 $pre -> $post (backup $BAK)" | tee -a "$REC"
