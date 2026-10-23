#!/bin/bash
# box_placeholders.sh {status|create|release} [FOLDS..]: the box-queue side of the Vast run (BOX_QUEUE_NOTE.md).
#   status   read only: placeholder dir, .log, queue line state and any box trainer for each fold
#   create   mkdir EMPTY ~/fleet_runs/fF_s1044_ttr_frac75 (kit.env) (refused if the box runs/ran that fold or anything is
#            there). The queue runner then marks that fold's queue line #REFUSED and never trains it.
#   release  rmdir the empty placeholder (partial Vast content is moved aside to *_vastpartial_<time>, never
#            deleted; a COMPLETE run is kept). vastS1044: NO queue line is added (the box queue has no s1044 lines).
# Default FOLDS = state.env FOLDS, else "2 3 4". vast_launch.sh runs create; vast_destroy.sh runs release for every
# fold that was not fetched complete.
SCRIPT_TAG=placeholders; source "$(dirname "$0")/config.sh"
CMD=${1:-status}; shift || true
FOLDS=${*:-$(st_get FOLDS)}; FOLDS=${FOLDS:-1 2 3 4}
for F in $FOLDS; do case $F in 1|2|3|4) ;; *) die "folds must be from 1-4: got $F";; esac; done
box_up || die "box $BOX unreachable"; sync_box_ops || die "could not copy the helpers to the box"
case $CMD in
  status) log "placeholders status [$FOLDS]"; bops ph-status $FOLDS;;
  create) log "placeholders create [$FOLDS]"; bops ph-create $FOLDS;;
  release) log "placeholders release [$FOLDS]"; bops ph-release $FOLDS; rc=$?; exit $rc;;
  *) die "usage: box_placeholders.sh {status|create|release} [FOLDS..]";;
esac
