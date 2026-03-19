# SnapKV-Enhanced

Enhanced version of [SnapKV](https://arxiv.org/abs/2404.14469) — an efficient KV cache compression method for long-context LLMs. This fork introduces three improvements over the original algorithm: **weighted pooling**, **multi-window observation**, and **spike protection**, evaluated on the [LongBench](https://github.com/THUDM/LongBench) benchmark.

## Supported Models

| Model | HuggingFace ID | VRAM (FP16) |
|---|---|---|
| Llama-3.2-1B-Instruct | `meta-llama/Llama-3.2-1B-Instruct` | ~2 GB |
| Llama-3.2-3B-Instruct | `meta-llama/Llama-3.2-3B-Instruct` | ~6 GB |
| Mistral-7B-Instruct-v0.2 | `mistralai/Mistral-7B-Instruct-v0.2` | ~14 GB |

## Environment Setup

### 1. Create Conda Environment

```bash
conda create -n snapkv python=3.10 -y
conda activate snapkv
```

### 2. Install PyTorch (CUDA 11.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1, use:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install SnapKV-Enhanced

```bash
git clone https://github.com/manasjain26/SnapKV-Enhanced.git
cd SnapKV-Enhanced
pip install -e .
```

### 4. Install Additional Dependencies

```bash
pip install "datasets<3.0" jieba rouge_score fuzzywuzzy
```

> **Note:** The `datasets` library v3.0+ has breaking changes that cause issues with LongBench data loading. Pin to `<3.0` to avoid errors.

### 5. HuggingFace Authentication

Models like Llama-3.2 require accepting the license on HuggingFace. Set your token:

```bash
export HF_TOKEN="your_huggingface_token"
```

Or log in interactively:

```bash
huggingface-cli login
```

## Experiment Configurations

Two configurations are used for comparison:

| Config | Pooling | Observation Windows | Spike Protection |
|---|---|---|---|
| `baseline_avgpool` | Average | 1 | No |
| `enhanced_combined` | Weighted | 3 | Yes |

Config files are in `experiments/LongBench/config/`.

## Running LongBench Experiments

All commands should be run from the `experiments/LongBench/` directory:

```bash
cd experiments/LongBench
```

### Available Scripts

| Script | Model | Datasets | Description |
|---|---|---|---|
| `run_llama.sh` | Llama-3.2-1B | 5 core | hotpotqa, 2wikimqa, gov_report, passage_retrieval_en, lcc |
| `run_llama_extra.sh` | Llama-3.2-1B | 11 remaining | narrativeqa, qasper, multifieldqa_en, musique, qmsum, multi_news, trec, triviaqa, samsum, passage_count, repobench-p |
| `run_mistral.sh` | Mistral-7B | 5 core | hotpotqa, 2wikimqa, gov_report, passage_retrieval_en, lcc |
| `run_mistral_extra.sh` | Mistral-7B | 11 remaining | Same 11 as llama_extra |

Each script runs both `baseline_avgpool` and `enhanced_combined` configs and **skips existing results**.

### Single GPU

```bash
# Run Llama on all 16 datasets sequentially
bash scripts/run_llama.sh 0
bash scripts/run_llama_extra.sh 0

# Run Mistral on all 16 datasets sequentially
bash scripts/run_mistral.sh 0
bash scripts/run_mistral_extra.sh 0
```

### Manual Run (Single Dataset)

```bash
CUDA_VISIBLE_DEVICES=0 python pred_snap.py \
    --model llama-3.2-1b-instruct \
    --compress_args_path enhanced_combined.json \
    --dataset hotpotqa
```

## Evaluation

Evaluation runs automatically at the end of each script. To evaluate manually:

```bash
cd experiments/LongBench
python eval.py --model "llama-3.2-1b-instructbaseline_avgpool"
python eval.py --model "llama-3.2-1b-instructenhanced_combined"
```

Results are saved to `experiments/LongBench/pred_e/<model><config>/result.json`.

## Project Structure

```
SnapKV-Enhanced/
├── snapkv/
│   └── monkeypatch/
│       ├── monkeypatch.py              # Entry point for applying SnapKV patches
│       ├── snapkv_utils.py             # Core SnapKV algorithm + enhancements
│       ├── llama_hijack_modern.py       # Llama attention patch (transformers >=4.43)
│       └── mistral_hijack_modern.py     # Mistral attention patch (transformers >=4.43)
├── experiments/
│   └── LongBench/
│       ├── pred_snap.py                # Prediction script
│       ├── eval.py                     # Evaluation script
│       ├── config/                     # Model paths, configs, prompts
│       │   ├── baseline_avgpool.json
│       │   ├── enhanced_combined.json
│       │   ├── model2path.json
│       │   └── ...
│       └── scripts/                    # Run scripts
│           ├── run_llama.sh
│           ├── run_llama_extra.sh
│           ├── run_mistral.sh
│           └── run_mistral_extra.sh
└── pyproject.toml
```

## Citation

```bibtex
@article{li2024snapkv,
  title={SnapKV: LLM Knows What You are Looking for Before Generation},
  author={Li, Yuhong and Huang, Yingbing and Yang, Bowen and Venkitesh, Bharat and Locatelli, Acyr and Ye, Hanchen and Cai, Tianle and Lewis, Patrick and Chen, Deming},
  journal={arXiv preprint arXiv:2404.14469},
  year={2024}
}
```
