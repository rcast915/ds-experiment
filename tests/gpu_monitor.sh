#!/usr/bin/env bash
# gpu_monitor.sh — log the GPU's SM clock, power draw, temperature, throttle
# reasons and utilization once a second for the whole job, plus the compute
# processes on the GPU every 5 s (to tell clock drift from contention).
# Timestamps are UTC, as in the benchmarks' output. Runs on the HOST, from
# the submit_*.sh scripts: nvidia-smi is not bound into the writable container.
#
#   pids=$(bash tests/gpu_monitor.sh start results/<gpu>_<label>)   # before the job
#   bash tests/gpu_monitor.sh stop "$pids" results/<gpu>_<label>   # after: kill + summary
#
# Files: <prefix>_gpulog.csv (per-second samples), <prefix>_gpuprocs.csv
# (compute processes), and the summary printed by stop (also <prefix>_gpulog.txt).

cmd="${1:?usage: start <prefix> | stop <pids> <prefix>}"

if ! command -v nvidia-smi >/dev/null; then
    echo "[gpu_monitor] nvidia-smi not found on $(hostname); GPU not logged." >&2
    exit 0
fi

if [[ "$cmd" == start ]]; then
    prefix="${2:?prefix}"
    # Keep only the fields this driver knows (throttle reasons were renamed
    # clocks_event_reasons in newer drivers).
    fields=()
    for f in timestamp index clocks.sm clocks.mem clocks.max.sm power.draw power.limit \
             temperature.gpu utilization.gpu memory.used clocks_event_reasons.active \
             pstate; do
        nvidia-smi --query-gpu="$f" --format=csv,noheader >/dev/null 2>&1 && fields+=("$f")
    done
    if [[ ! " ${fields[*]} " =~ " clocks_event_reasons.active " ]] && \
       nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader >/dev/null 2>&1; then
        fields+=(clocks_throttle_reasons.active)
    fi
    q=$(IFS=,; echo "${fields[*]}")
    TZ=UTC nohup nvidia-smi --query-gpu="$q" --format=csv,nounits -lms 1000 \
        > "${prefix}_gpulog.csv" 2>/dev/null &
    p1=$!
    TZ=UTC nohup nvidia-smi --query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_memory \
        --format=csv,nounits -l 5 > "${prefix}_gpuprocs.csv" 2>/dev/null &
    p2=$!
    echo "[gpu_monitor] logging GPU to ${prefix}_gpulog.csv (fields: $q)" >&2
    echo "$p1 $p2"
    exit 0
fi

if [[ "$cmd" == stop ]]; then
    pids="${2:-}"; prefix="${3:?prefix}"
    [[ -n "$pids" ]] && kill $pids 2>/dev/null
    sleep 1
    csv="${prefix}_gpulog.csv"
    [[ -s "$csv" ]] || { echo "[gpu_monitor] no samples in $csv"; exit 0; }
    {
    echo "GPU monitor summary: $csv"
    # Columns by header name; stats over samples with utilization > 50% (under load).
    awk -F', *' '
        NR==1 { for (i=1;i<=NF;i++) { h=$i; sub(/ \[.*\]/,"",h); col[h]=i }; next }
        function v(name) { return (name in col) ? $(col[name]) : "" }
        {
            n_all++
            u = v("utilization.gpu")+0
            if (u <= 50) next
            n++
            sm[n]=v("clocks.sm")+0; pw[n]=v("power.draw")+0; t[n]=v("temperature.gpu")+0
            r = v("clocks_event_reasons.active"); if (r=="") r = v("clocks_throttle_reasons.active")
            reasons[r]++
            if (n==1) first=v("timestamp"); last=v("timestamp")
            pl=v("power.limit"); mx=v("clocks.max.sm")
        }
        function stats(a, lab, unit,   i,j,x,s) {
            for (i=1;i<=n;i++) s[i]=a[i]
            for (i=2;i<=n;i++) { x=s[i]; for (j=i-1;j>=1 && s[j]>x;j--) s[j+1]=s[j]; s[j+1]=x }
            printf "  %-12s min %8.1f  median %8.1f  max %8.1f %s\n", lab, s[1], s[int((n+1)/2)], s[n], unit
        }
        END {
            printf "  samples      %d total, %d under load (utilization > 50%%), %s .. %s\n", n_all, n, first, last
            if (n==0) exit
            stats(sm, "SM clock", "MHz (max " mx ")")
            stats(pw, "power", "W (limit " pl ")")
            stats(t,  "temperature", "C")
            printf "  throttle / clock-event reasons under load (bitmask: count):\n"
            for (r in reasons) printf "    %s: %d\n", r, reasons[r]
            printf "    (NVML bits: 0x1 idle, 0x2 app clocks, 0x4 SW power cap, 0x8 HW slowdown,\n"
            printf "     0x10 sync boost, 0x20 SW thermal, 0x40 HW thermal, 0x80 HW power brake)\n"
        }' "$csv"
    procs="${prefix}_gpuprocs.csv"
    if [[ -s "$procs" ]]; then
        echo "  compute processes seen on the GPU (pid name: samples):"
        awk -F', *' 'NR>1 && $3 ~ /^[0-9]+$/ { k=$3" "$4; c[k]++ } END { for (k in c) printf "    %s: %d\n", k, c[k]; if (length(c)==0) print "    none" }' "$procs"
    fi
    } | tee "${prefix}_gpulog.txt"
    exit 0
fi
echo "usage: start <prefix> | stop <pids> <prefix>" >&2; exit 1
