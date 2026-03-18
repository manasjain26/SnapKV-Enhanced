#!/bin/bash
# Run Llama-3.2-1B: baseline vs enhanced_combined on REMAINING 11 datasets
# (Complements run_llama.sh which covers hotpotqa, 2wikimqa, gov_report, passage_retrieval_en, lcc)
# Skips existing results. Usage: bash scripts/run_llama_extra.sh [GPU_ID]

GPU=${1:-0}
MODEL="llama-3.2-1b-instruct"
DATASETS=("narrativeqa" "qasper" "multifieldqa_en" "musique" "qmsum" "multi_news" "trec" "triviaqa" "samsum" "passage_count" "repobench-p")
CONFIGS=("baseline_avgpool" "enhanced_combined")

echo "Model: $MODEL  GPU: $GPU  (extra datasets)"
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
