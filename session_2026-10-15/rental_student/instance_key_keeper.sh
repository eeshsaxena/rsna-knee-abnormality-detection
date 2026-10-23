#!/bin/bash
# instance_key_keeper.sh (laptop kit root, NOT in remote/ so the packed bundle kit stays unchanged; runs on the instance as /root/key_keeper.sh; launch.sh starts it with setsid right after adding the box key). 10-15 01:20 IST incident on 55960494:
# Vast re-synced /root/.ssh/authorized_keys from the ACCOUNT keys at 00:55 IST (no trailing newline) and dropped the box fetch
# key, so the box fetch loop got "Permission denied (publickey)" until main re-added it by hand. Every 30 s: if the box key
# (/root/.ssh/box_key.pub) is not on a line of its own, end the file with a newline if needed and append it. Public key only;
# instance-local; never touches the account keys; dies with the instance. Log /root/key_keeper.log.
K=/root/.ssh/box_key.pub; A=/root/.ssh/authorized_keys; KEY=$(head -n 1 $K)
case $KEY in ssh-ed25519\ *) ;; *) echo "[$(date -u +%m-%d_%H:%M:%S)] no box key in $K: exit" >> /root/key_keeper.log; exit 1;; esac
echo $$ > /root/key_keeper.pid
while true; do
  if ! grep -qxF "$KEY" $A 2>/dev/null; then
    mkdir -p /root/.ssh; [ -s $A ] && [ -n "$(tail -c 1 $A)" ] && echo >> $A
    echo "$KEY" >> $A; chmod 600 $A
    echo "[$(date -u +%m-%d_%H:%M:%S)] re-added box key" >> /root/key_keeper.log
  fi
  sleep 30
done
