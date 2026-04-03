#!/bin/bash
# Usage:
#     (1) bash start_all.sh
#     (2) CUDA_VISIBLE_DEVICES=0,1,2,3 bash start_all.sh
#
# torch.compile is controlled via config.json: "service": { "compile": true }
# Pre-compile with: PYTHONPATH=. .venv/base/bin/python precompile.py

set -e

export TORCHINDUCTOR_CACHE_DIR=./torch_compile_cache

# ============ Parse script arguments ============
GATEWAY_PROTO="https"
GATEWAY_EXTRA_ARGS=""
for arg in "$@"; do
    case "$arg" in
        --http)
            GATEWAY_PROTO="http"
            GATEWAY_EXTRA_ARGS="--http"
            ;;
    esac
done

# ============ Configuration ============
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="/data/dukai_venvs/realtime/bin/python"

GATEWAY_PORT=$($VENV_PYTHON -c "import sys; sys.path.insert(0,'$PROJECT_DIR'); from config import get_config; print(get_config().gateway_port)" 2>/dev/null || echo "10024")
WORKER_BASE_PORT=$($VENV_PYTHON -c "import sys; sys.path.insert(0,'$PROJECT_DIR'); from config import get_config; print(get_config().worker_base_port)" 2>/dev/null || echo "22400")

# ============ Detect GPUs ============
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    GPU_LIST=$(seq 0 $((NUM_GPUS - 1)) | tr '\n' ',' | sed 's/,$//')
else
    GPU_LIST="$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=$(echo "$GPU_LIST" | tr ',' '\n' | wc -l)
fi

echo "=================================================="
echo "  MiniCPMO45 Service Launcher"
echo "=================================================="
echo "  GPUs: $GPU_LIST ($NUM_GPUS)"
echo "  Gateway: ${GATEWAY_PROTO}://localhost:$GATEWAY_PORT"
echo "  Workers: localhost:$WORKER_BASE_PORT ~ localhost:$((WORKER_BASE_PORT + NUM_GPUS - 1)) (HTTP, internal)"
echo "=================================================="

cd "$PROJECT_DIR"
mkdir -p tmp

# ============ Start Workers ============
WORKER_ADDRS=""
GPU_IDX=0

for GPU_ID in $(echo "$GPU_LIST" | tr ',' ' '); do
    WORKER_PORT=$((WORKER_BASE_PORT + GPU_IDX))

    echo "[Worker $GPU_IDX] Starting on GPU $GPU_ID, port $WORKER_PORT..."

    nohup env CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH=. $VENV_PYTHON worker.py \
        --port $WORKER_PORT \
        --gpu-id $GPU_ID \
        --worker-index $GPU_IDX \
        > "tmp/worker_${GPU_IDX}.log" 2>&1 &

    echo $! > "tmp/worker_${GPU_IDX}.pid"

    if [ -z "$WORKER_ADDRS" ]; then
        WORKER_ADDRS="localhost:$WORKER_PORT"
    else
        WORKER_ADDRS="$WORKER_ADDRS,localhost:$WORKER_PORT"
    fi

    GPU_IDX=$((GPU_IDX + 1))
done

echo ""
echo "Waiting for Workers to load models (~30-90s)..."

print_worker_health() {
    local worker_idx="$1"
    local worker_port="$2"
    local worker_pid_file="tmp/worker_${worker_idx}.pid"
    local worker_log_file="tmp/worker_${worker_idx}.log"

    local worker_pid=""
    if [ -f "$worker_pid_file" ]; then
        worker_pid=$(cat "$worker_pid_file" 2>/dev/null)
    fi

    local pid_state="unknown"
    if [ -n "$worker_pid" ]; then
        if kill -0 "$worker_pid" 2>/dev/null; then
            pid_state="alive(pid=$worker_pid)"
        else
            pid_state="dead(pid=$worker_pid)"
        fi
    fi

    local health_json
    health_json=$(curl -s --max-time 2 "http://localhost:$worker_port/health" 2>/dev/null || true)
    if [ -n "$health_json" ]; then
        local parsed
        parsed=$($VENV_PYTHON -c 'import sys, json
try:
    d = json.load(sys.stdin)
    print(
        "reachable status={status} worker_status={worker_status} model_loaded={model_loaded} "
        "gpu_id={gpu_id} requests={requests} avg_ms={avg_ms}".format(
            status=d.get("status", "?"),
            worker_status=d.get("worker_status", "?"),
            model_loaded=d.get("model_loaded", False),
            gpu_id=d.get("gpu_id", "?"),
            requests=d.get("total_requests", 0),
            avg_ms=round(float(d.get("avg_inference_time_ms", 0.0) or 0.0), 1),
        )
    )
except Exception as e:
    print(f"health-parse-error={e}")' <<< "$health_json" 2>/dev/null || echo "health-parse-error")
        echo "  [Worker $worker_idx:$worker_port] $parsed | $pid_state"
    else
        echo "  [Worker $worker_idx:$worker_port] unreachable | $pid_state"
    fi

    if [ -f "$worker_log_file" ]; then
        local last_log
        last_log=$(tail -n 2 "$worker_log_file" 2>/dev/null | sed 's/^/      log> /')
        if [ -n "$last_log" ]; then
            echo "$last_log"
        fi
    fi
}

sleep 2
MAX_RETRIES=3000
STATUS_INTERVAL=5
LAST_STATUS_RETRY=-1

for i in $(seq 0 $((NUM_GPUS - 1))); do
    eval "WORKER_READY_$i=0"
done

for RETRY in $(seq 0 $MAX_RETRIES); do
    READY_COUNT=0

    for i in $(seq 0 $((NUM_GPUS - 1))); do
        WORKER_PORT=$((WORKER_BASE_PORT + i))
        if curl -s --max-time 2 "http://localhost:$WORKER_PORT/health" 2>/dev/null | $VENV_PYTHON -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('model_loaded') else 1)" 2>/dev/null; then
            eval "IS_READY=\$WORKER_READY_$i"
            if [ "$IS_READY" -eq 0 ]; then
                echo "[Worker $i] Ready on port $WORKER_PORT"
                eval "WORKER_READY_$i=1"
            fi
            READY_COUNT=$((READY_COUNT + 1))
        fi
    done

    if [ "$READY_COUNT" -eq "$NUM_GPUS" ]; then
        echo "All workers are ready."
        break
    fi

    if [ "$RETRY" -eq "$MAX_RETRIES" ]; then
        break
    fi

    if [ $((RETRY % STATUS_INTERVAL)) -eq 0 ] && [ "$LAST_STATUS_RETRY" -ne "$RETRY" ]; then
        ELAPSED_S=$((RETRY * 2))
        echo ""
        echo "[Wait] ${READY_COUNT}/${NUM_GPUS} workers ready after ~${ELAPSED_S}s"
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            WORKER_PORT=$((WORKER_BASE_PORT + i))
            print_worker_health "$i" "$WORKER_PORT"
        done
        echo ""
        LAST_STATUS_RETRY=$RETRY
    fi

    sleep 2
done

for i in $(seq 0 $((NUM_GPUS - 1))); do
    eval "IS_READY=\$WORKER_READY_$i"
    if [ "$IS_READY" -eq 0 ]; then
        WORKER_PORT=$((WORKER_BASE_PORT + i))
        echo "[Worker $i] FAILED to become ready on port $WORKER_PORT. Check tmp/worker_${i}.log"
    fi
done

# ============ Start Gateway ============
echo ""
echo "[Gateway] Starting on port $GATEWAY_PORT..."

nohup env PYTHONPATH=. $VENV_PYTHON gateway.py \
    --port $GATEWAY_PORT \
    --workers "$WORKER_ADDRS" \
    $GATEWAY_EXTRA_ARGS \
    > "tmp/gateway.log" 2>&1 &

echo $! > "tmp/gateway.pid"

sleep 2

CURL_FLAGS=""
if [ "$GATEWAY_PROTO" = "https" ]; then
    CURL_FLAGS="-k"
fi

if curl -s $CURL_FLAGS "${GATEWAY_PROTO}://localhost:$GATEWAY_PORT/health" 2>/dev/null | $VENV_PYTHON -c "import sys,json; json.load(sys.stdin); exit(0)" 2>/dev/null; then
    echo "[Gateway] Ready"
else
    echo "[Gateway] May still be starting. Check tmp/gateway.log"
fi

echo ""
echo "=================================================="
echo "  Service is running!"
echo "  Chat Demo:  ${GATEWAY_PROTO}://localhost:$GATEWAY_PORT"
echo "  Admin:      ${GATEWAY_PROTO}://localhost:$GATEWAY_PORT/admin"
echo "  API Docs:   ${GATEWAY_PROTO}://localhost:$GATEWAY_PORT/docs"
echo "  Workers:    $WORKER_ADDRS"
echo ""
echo "  Logs:"
echo "    Gateway:  tmp/gateway.log"
echo "    Workers:  tmp/worker_*.log"
echo ""
echo "  To stop:"
echo "    kill \$(cat tmp/*.pid 2>/dev/null) 2>/dev/null"
echo "=================================================="
