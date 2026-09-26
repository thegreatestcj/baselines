#!/bin/bash
# Unattended runner + watchdog for the PhysON groups on one machine.
#
#   bash watch_physon.sh mosiv "0_0 0_4 0_5" 0,1,2   # MOSIV scenes on GPUs 0,1,2
#   bash watch_physon.sh omni  "0_3 0_4"     0,1     # OmniPhysGS scenes
#
# Every 5 minutes, each listed scene that is not DONE, not running and not waiting in a queue file
# is (re)launched on the GPU with the most free memory that still has room (MIN_FREE_GB; MOSIV also
# needs no other MOSIV task on that GPU, OmniPhysGS at most MAX_PER_GPU tasks); a running task
# whose log has not changed for STALE_MIN minutes is killed and relaunched. Each scene is
# relaunched at most MAX_ATTEMPTS times. Exits when every scene is DONE (or after 24 h).
# gen_tasks.sh options (MOSIV_PHYSON_CONVERT_ARGS, OMNIPHYSGS_PHYSON_ARGS, PHYSON_*_SUBSET) are
# inherited from the environment, so export them before starting. DRY=1 prints decisions only.
set -u
cd "$(dirname "$0")"
GROUP=${1:?mosiv|omni}; SCENES=${2:?scene ids}; GPUS=${3:?comma-separated gpu ids}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
if [ "$GROUP" = mosiv ]; then
  MIN_FREE_GB=${MIN_FREE_GB:-50}; MAX_PER_GPU=${MAX_PER_GPU:-1}
  STALE_MIN=${STALE_MIN:-150}   # the parameter stage prints nothing during a full 48-frame forward (+ CFL retries)
  SUB=${PHYSON_MO_SUBSET:-multiobject_heterogeneous}; TASKGROUP=mosiv_physon_mo
  done_of() { echo "MOSIV/output/physon/$SUB/$1/DONE"; }
  log_of() { echo "logs/mosiv_physon_${SUB}_$1.log"; }
  tag_of() { echo "mosiv_physon:$SUB/$1"; }
  wrapper_pat() { echo "bash -c .*$SUB/$1[.]json"; }
  any_pat="train_dynamic_MO[.]py -c config/physon/"
else
  MIN_FREE_GB=${MIN_FREE_GB:-30}; MAX_PER_GPU=${MAX_PER_GPU:-2}
  STALE_MIN=${STALE_MIN:-40}    # fit.py logs every step
  SUB=${PHYSON_HET_SUBSET:-singleobject_heterogeneous}; TASKGROUP=omniphysgs_physon_het
  done_of() { echo "OmniPhysGS/outputs/PhysON/$SUB/$1/DONE"; }
  log_of() { echo "logs/omniphysgs_physon_${SUB}_$1.log"; }
  tag_of() { echo "omniphysgs_physon:$SUB/$1"; }
  wrapper_pat() { echo "bash -c .*$SUB/$1/config[.]yaml"; }
  any_pat="fit[.]py --config data/PhysON/"
fi
mkdir -p logs
declare -A attempts assigned launched_at   # assigned[scene]=gpu for tasks this script launched
log() { echo "$(date '+%m-%d %H:%M') $*"; }
free_gb() {
  nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$1" | awk -F', ' '{print int(($1-$2)/1024)}'
}
tasks_on_gpu() {  # tasks of this group bound to GPU $1 (wrappers carry CUDA_VISIBLE_DEVICES)
  local g=$1 n=0 p
  for p in $(pgrep -u "$(id -un)" -f "$any_pat"); do
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$g" && n=$((n + 1))
  done
  echo $n
}
occupancy() {  # max(process-based count, tasks this script placed on the GPU in the last 30 min or still running)
  local g=$1 n m=0 s
  n=$(tasks_on_gpu "$g")
  for s in "${!assigned[@]}"; do
    [ "${assigned[$s]}" = "$g" ] || continue
    if [ $(( $(date +%s) - ${launched_at[$s]:-0} )) -lt 1800 ] || pgrep -u "$(id -un)" -f "$(wrapper_pat "$s")" >/dev/null; then m=$((m + 1)); fi
  done
  [ "$n" -gt "$m" ] && echo "$n" || echo "$m"
}
pick_gpu() {
  local best="" bestfree=0 g f
  for g in ${GPUS//,/ }; do
    f=$(free_gb "$g")
    [ "$f" -ge "$MIN_FREE_GB" ] || continue
    [ "$(occupancy "$g")" -lt "$MAX_PER_GPU" ] || continue
    [ "$f" -gt "$bestfree" ] && { best=$g; bestfree=$f; }
  done
  echo "$best"
}
kill_task() {  # wrapper pid: worker -> wrapper -> python
  local wr=$1 wk; wk=$(ps -o ppid= -p "$wr" | tr -d ' ')
  kill "$wk" 2>/dev/null; pkill -P "$wr" 2>/dev/null; kill "$wr" 2>/dev/null; sleep 4
  for c in $(pgrep -P "$wr" 2>/dev/null); do kill -9 "$c" 2>/dev/null; done
}
launch() {  # scene
  local s=$1 gpu tf
  attempts[$s]=$(( ${attempts[$s]:-0} + 1 ))
  if [ "${attempts[$s]}" -gt "$MAX_ATTEMPTS" ]; then log "GIVE UP $s after $MAX_ATTEMPTS launches"; return; fi
  gpu=$(pick_gpu)
  if [ -z "$gpu" ]; then log "no GPU with room for $s yet"; attempts[$s]=$(( ${attempts[$s]} - 1 )); return; fi
  rm -rf "$(dirname "$(done_of "$s")")/DONE.claim"
  tf=$PWD/tasks_watch_${GROUP}_${s}.txt
  PHYSON_SCENES="$s" bash gen_tasks.sh "$TASKGROUP" > "$tf" 2> "$tf.err"
  [ -s "$tf" ] || { log "gen_tasks produced nothing for $s: $(head -1 "$tf.err")"; attempts[$s]=$(( ${attempts[$s]} - 1 )); return; }
  if [ -n "${DRY:-}" ]; then log "DRY: would launch $s on GPU $gpu"; rm -f "$tf"; return; fi
  TASKS_FILE=$tf setsid nohup bash run_queue.sh "$gpu" 1 >> "logs/watch_${GROUP}_queue.log" 2>&1 < /dev/null &
  assigned[$s]=$gpu; launched_at[$s]=$(date +%s)
  log "LAUNCHED $s on GPU $gpu (attempt ${attempts[$s]}, free $(free_gb "$gpu") GB)"
  sleep 20   # let the worker claim the task before the next placement decision
}

t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt 86400 ]; do
  all_done=1
  for s in $SCENES; do
    [ -f "$(done_of "$s")" ] && { unset "assigned[$s]"; continue; }
    all_done=0
    wr=$(pgrep -u "$(id -un)" -f "$(wrapper_pat "$s")" | head -1)
    if [ -n "$wr" ]; then
      lf=$(log_of "$s")
      if [ -f "$lf" ] && [ $(( ($(date +%s) - $(stat -c %Y "$lf")) / 60 )) -ge "$STALE_MIN" ]; then
        log "STALLED $s (log idle $(( ($(date +%s) - $(stat -c %Y "$lf")) / 60 )) min): killing wrapper $wr"
        [ -z "${DRY:-}" ] && kill_task "$wr"
        launch "$s"
      fi
      continue
    fi
    grep -qs "^$(tag_of "$s")|" tasks*.txt 2>/dev/null && continue   # queued, not yet taken
    launch "$s"
  done
  st=""
  for s in $SCENES; do
    if [ -f "$(done_of "$s")" ]; then st="$st $s:DONE"
    elif [ "$GROUP" = mosiv ]; then f=$(log_of "$s"); st="$st $s:v$(grep -c 'Best params.*velocity' "$f" 2>/dev/null)/p$(grep -c 'Best params.*modulus' "$f" 2>/dev/null)"
    else st="$st $s:$(grep -c '^epoch' "$(log_of "$s")" 2>/dev/null)"; fi
  done
  log "status:$st"
  [ "$all_done" = 1 ] && { log "ALL DONE"; break; }
  [ -n "${DRY:-}" ] && break
  sleep 300
done
