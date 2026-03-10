#!/bin/bash
# SnapKV Enhanced — LongBench Ablation Experiments
# Run each config to compare baseline vs improvements
#
# Usage: bash run_ablation.sh <model_name> <dataset>
# Example: bash run_ablation.sh mistral-7B-instruct-v0.2 qasper
#
# To run ALL datasets: bash run_ablation.sh <model_name> all

MODEL=${1:-"mistral-7B-instruct-v0.2"}
DATASET=${2:-"qasper"}

CONFIGS=(
    "baseline_avgpool"
    "baseline_maxpool"
    "enhanced_weighted_only"
    "enhanced_multiwindow_only"
    "enhanced_spikes_only"
    "enhanced_combined"
)

ALL_DATASETS=(
    "narrativeqa" "qasper" "multifieldqa_en"
    "hotpotqa" "2wikimqa" "musique"
    "gov_report" "qmsum" "multi_news"
    "trec" "triviaqa" "samsum"
    "passage_count" "passage_retrieval_en"
    "lcc" "repobench-p"
)

echo "============================================"
echo "SnapKV Enhanced — Ablation Experiments"
echo "Model: $MODEL"
echo "============================================"

for config in "${CONFIGS[@]}"; do
    echo ""
    echo "--------------------------------------------"
    echo "Running config: $config"
    echo "--------------------------------------------"

    if [ "$DATASET" = "all" ]; then
        for ds in "${ALL_DATASETS[@]}"; do
            echo "  Dataset: $ds"
            python pred_snap.py \
                --model $MODEL \
                --compress_args_path ${config}.json \
                --dataset $ds
        done
    else
        python pred_snap.py \
            --model $MODEL \
            --compress_args_path ${config}.json \
            --dataset $DATASET
    fi
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
