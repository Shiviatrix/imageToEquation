#!/usr/bin/env bash
set -e

actionCmd="${1:-start}"
pidFile="hpc.pid"
logDir="logs"
logFile="${logDir}/hpcTraining.log"

mkdir -p "${logDir}"

if command -v sbatch &> /dev/null; then
    echo "Submitting to Slurm cluster via sbatch..."
    sbatch hpcJob.slurm "${@:2}"
    exit 0
fi

case "$actionCmd" in
    start)
        if [ -f "$pidFile" ] && kill -0 "$(cat "$pidFile")" 2>/dev/null; then
            echo "Daemon already running with PID $(cat "$pidFile")"
            exit 0
        fi

        extraArgs="${*:2}"
        if [ -z "$extraArgs" ]; then
            extraArgs="--mode curated --targetPsnr 26.0 --targetLoss 0.015 --maxLayers 36 --patienceSteps 35 --outputDir hpc_transcriptions"
        fi

        echo "Starting training worker with arguments: ${extraArgs}..."
        nohup python3 -u runHpcTrainer.py ${extraArgs} > "${logFile}" 2>&1 &

        workerPid=$!
        echo "$workerPid" > "$pidFile"
        echo "Worker started (PID: $workerPid). Log: ${logFile}"
        ;;

    stop)
        if [ -f "$pidFile" ]; then
            workerPid=$(cat "$pidFile")
            if kill -0 "$workerPid" 2>/dev/null; then
                kill "$workerPid"
                rm -f "$pidFile"
                echo "Stopped worker $workerPid."
            else
                rm -f "$pidFile"
                echo "Worker was not running. Cleared stale PID."
            fi
        else
            echo "No running worker found."
        fi
        ;;

    status)
        if [ -f "$pidFile" ] && kill -0 "$(cat "$pidFile")" 2>/dev/null; then
            echo "Worker is running (PID: $(cat "$pidFile"))"
            tail -n 15 "${logFile}"
        else
            echo "Worker is not running."
        fi
        ;;

    logs)
        tail -f "${logFile}"
        ;;

    *)
        echo "Usage: ./submitHpc.sh [start|stop|status|logs]"
        exit 1
        ;;
esac
