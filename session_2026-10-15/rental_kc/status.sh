#!/bin/bash
# status.sh (laptop, read only): kit state, spend, box driver / fetch logs, instance progress (via the box route).
SCRIPT_TAG=status; source "$(dirname "$0")/config.sh"
echo "state: instance $(st_get INSTANCE_ID) ($(st_get NGPU) x $(st_get GPU), \$$(st_get DPH)/h) destroyed=$(st_get DESTROYED) | guard $(date -d @$(st_get GUARD_DEADLINE 2>/dev/null || echo 0) '+%m-%d %H:%M' 2>/dev/null)"
sp=$(spent_total 2>/dev/null) && echo "kit spend \$$sp of \$$CAP_USD" || echo "kit spend: unknown (2FA?)"
bx 'echo "--- driver.log"; tail -n 8 ~/rkc/driver.log 2>/dev/null; echo "--- fetch.log"; tail -n 8 ~/rkc/fetch/fetch.log 2>/dev/null
    for f in FINAL_FETCH_OK ABORTED FETCH_GAVE_UP; do [ -e ~/rkc/$f ] && echo "MARK $f $(cat ~/rkc/$f | head -c 200)"; done
    . ~/rkc/run.env 2>/dev/null && [ -n "${I_HOST:-}" ] && ssh -i ~/.ssh/vast_rsna_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o UserKnownHostsFile=~/.ssh/known_hosts_rkc -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -p $I_PORT root@$I_HOST "bash /workspace/rsna/kit/status.sh" 2>/dev/null | sed "s/^/  inst /"'
