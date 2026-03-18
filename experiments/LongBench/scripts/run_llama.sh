#!/bin/bash
# Run Llama-3.2-1B: baseline vs enhanced_combined on 5 datasets
# Skips existing results. Usage: bash scripts/run_llama.sh [GPU_ID]

GPU=${1:-0}
MODEL="llama-3.2-1b-instruct"
DATASETS=("hotpotqa" "2wikimqa" "gov_report" "passage_retrieval_en" "lcc")
CONFIGS=("baseline_avgpool" "enhanced_combined")

echo "Model: $MODEL  GPU: $GPU"
echo "============================================"

for config in "${CONFIGS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        out_file="pred_e/${MODEL}${config}/${ds}.jsonl"
        if [ -f "$out_file" ] && [ -s "$out_file" ]; then
            echo "SKIP  ${config} / ${ds}"
            continue
        fi
        echo "RUN   ${config} / ${ds}"
        CUDA_VISIBLE_DEVICES=$GPU python pred_snap.py \
            --model "$MODEL" \
            --compress_args_path "${config}.json" \
            --dataset "$ds"
    done
done

echo ""
echo "--- Evaluating ---"
for config in "${CONFIGS[@]}"; do
    python eval.py --model "${MODEL}${config}"
done
echo "Done!"
