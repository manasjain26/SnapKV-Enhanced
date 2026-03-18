#!/bin/bash
# SnapKV Enhanced — Parallel LongBench Ablation
# Runs 2 configs concurrently on separate GPUs (for 7B models that fit on 1 GPU).
#
# Usage:
#   bash scripts/run_ablation_parallel.sh <model_name> <dataset|all>
#   bash scripts/run_ablation_parallel.sh mistral-7B-instruct-v0.2 all

MODEL=${1:-"mistral-7B-instruct-v0.2"}
DATASET=${2:-"qasper"}

ALL_DATASETS=(
    "narrativeqa" "qasper" "multifieldqa_en"
    "hotpotqa" "2wikimqa" "musique"
    "gov_report" "qmsum" "multi_news"
    "trec" "triviaqa" "samsum"
    "passage_count" "passage_retrieval_en"
    "lcc" "repobench-p"
)

CONFIGS=(
    "baseline_avgpool"
    "baseline_maxpool"
    "enhanced_weighted_only"
    "enhanced_multiwindow_only"
    "enhanced_spikes_only"
    "enhanced_combined"
)

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-2}
echo "============================================"
echo "SnapKV Enhanced — Parallel Ablation"
echo "Model: $MODEL"
echo "GPUs available: $NUM_GPUS"
echo "============================================"

run_config() {
    local gpu=$1
    local config=$2
    if [ "$DATASET" = "all" ]; then
        for ds in "${ALL_DATASETS[@]}"; do
            echo "[GPU $gpu] config=$config  dataset=$ds"
            CUDA_VISIBLE_DEVICES=$gpu python pred_snap.py \
                --model "$MODEL" \
                --compress_args_path "${config}.json" \
                --dataset "$ds"
        done
    else
        echo "[GPU $gpu] config=$config  dataset=$DATASET"
        CUDA_VISIBLE_DEVICES=$gpu python pred_snap.py \
            --model "$MODEL" \
            --compress_args_path "${config}.json" \
            --dataset "$DATASET"
    fi
}

# Launch configs in batches of NUM_GPUS
for ((i=0; i<${#CONFIGS[@]}; i+=NUM_GPUS)); do
    pids=()
    for ((g=0; g<NUM_GPUS && i+g<${#CONFIGS[@]}; g++)); do
        run_config "$g" "${CONFIGS[$((i+g))]}" &
        pids+=($!)
    done
    # Wait for this batch to finish before starting the next
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
done

echo ""
echo "============================================"
echo "Running evaluation..."
echo "============================================"
for config in "${CONFIGS[@]}"; do
    echo "Evaluating: ${MODEL}${config}"
    python eval.py --model "${MODEL}${config}"
done

echo ""
echo "============================================"
echo "All experiments complete!"
echo "============================================"
