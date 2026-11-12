#!/bin/bash
# run_real.sh <log.out> <SGD_batch.py args...>: SGD_batch.py with a watchdog: no LM step for 300 s (once the LM has
# started) -> native + Python stacks saved (<log>.hang<n>.*), killed, then 'SGD_batch.py resume <NPZ folder>' (3 tries)
# 2026-11-12: moved into git (was D:\Amaury\simu_precision_claude\lm_tuning\run_real.sh); runs the code of the repo it is
# in. GOLD RULE: <log.out> = a .out in the queue folder D:\AMAURY\polMFM_runs\<data set>\<date>_<queue>\ (absolute path).
cd "$(dirname "$(readlink -f "$0")")/.."  # root of the repo that holds this script (tools/run_real.sh)
export MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/home/amaury/exit/envs/env_linux/bin/python
HBPY=tools/hb_any.py
out=$1; shift
args=("$@"); npz=""
for try in 1 2 3 4; do
  export HEARTBEAT=$out.hb; rm -f $HEARTBEAT
  echo "=== $(basename $out) start (try $try) $(date +%F_%T)" >> $out.queue
  $PY $HBPY SGD_batch.py "${args[@]}" >> $out 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 15
    [ -f $HEARTBEAT ] || continue
    grep -q start $HEARTBEAT && continue
    # the heartbeat file can be read while it is rewritten (empty): skip, else the arithmetic error stops this script
    # (2026-11-03: the wrapper of cricket sm2 exited with 1 while the fit went on)
    t=$(cut -d' ' -f1 $HEARTBEAT 2>/dev/null); [[ $t =~ ^[0-9]+$ ]] || continue
    age=$(( $(date +%s) - t ))
    if [ $age -gt 300 ]; then
      echo "=== HANG: no LM step for ${age}s, last: $(cut -d' ' -f2- $HEARTBEAT) $(date +%F_%T)" >> $out.queue
      for t in /proc/$pid/task/*; do echo "$(basename $t) $(cat $t/comm) $(cut -d' ' -f3,14,15 $t/stat) wchan=$(cat $t/wchan)"; done > $out.hang$try.threads
      timeout 180 /home/amaury/dbg/bin/gdb -p $pid -batch -ex "info threads" -ex "thread apply all bt 40" > $out.hang$try.gdb 2>&1
      ~/.local/bin/py-spy dump --native --pid $pid > $out.hang$try.pyspy 2>&1
      kill -USR1 $pid; sleep 3; kill -9 $pid; break
    fi
  done
  wait $pid; code=$?
  echo "=== exit $code $(date +%F_%T)" >> $out.queue
  [ $code -eq 0 ] && break
  npz=$(grep -m1 "Saving in:" $out | sed 's/.*Saving in: *//')
  [ -n "$npz" ] || { echo "=== no NPZ folder to resume, stop" >> $out.queue; break; }
  args=(resume "$npz" --no-figures --post-process)
done
