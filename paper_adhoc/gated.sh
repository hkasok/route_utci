#!/usr/bin/env bash
# Run a command only when CPUs are free, one gated job at a time.
#   usage: paper_adhoc/gated.sh NAME NEED_THREADS command args...
# Jobs serialise on a lock file, then wait until (nproc - 1-min load) >= NEED,
# re-checking every 5 min (nproc --all: plain nproc honours OMP_NUM_THREADS). Log of starts/ends: paper_adhoc/gated.log.
set -u
NAME=$1; NEED=$2; shift 2
LOG="$(cd "$(dirname "$0")" && pwd)/gated.log"; LOCK=/tmp/route_utci_gated.lock
exec 9>"$LOCK"
echo "$(date '+%F %T') queued $NAME (needs $NEED threads)" >> "$LOG"
flock 9
while :; do
  free=$(awk -v n="$(nproc --all)" '{printf "%d", n - $1}' /proc/loadavg)
  [ "$free" -ge "$NEED" ] && break
  sleep 300
done
echo "$(date '+%F %T') start $NAME (free $free)" >> "$LOG"
"$@"; rc=$?
echo "$(date '+%F %T') end $NAME rc=$rc" >> "$LOG"
exit $rc
