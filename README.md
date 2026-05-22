# Search-R1-TRL

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Search-R1-TRL implements **Search-R1** reinforcement learning training using **GRPO** (Group Relative Policy Optimization) with BM25 retrieval, following the same retrieval pipeline as the original [Search-R1](https://github.com/duckie/r1-search).

## Overview

This project trains a small LLM to autonomously decide when to search for external knowledge during multi-turn reasoning. The agent learns to:

1. Reason within `<think>` tags when new information arrives
2. Issue queries via `<search>query</search>` to retrieve top results from a BM25 index
3. Provide final answers inside `<answer>answer</answer>` tags
4. Perform multiple search rounds until confident

### Training Flow

```
Model generates → <search> query  → BM25 search → <information> results </information>
       ↓
Model generates → <answer> answer → EM reward → GRPO policy update (KL penalty)
```

## Features

- **BM25 Retrieval**: Uses Pyserini/Lucene for document retrieval (same as original Search-R1)
- **Multi-turn Interaction**: Model can search multiple times before answering
- **GRPO Training**: Group Relative Policy Optimization with reference model KL penalty
- **EM Reward**: Exact-match reward on extracted answer with format bonus
- **Dataset Support**: NQ, HotpotQA, TriviaQA out-of-the-box + custom JSONL format

## Prerequisites

- **Java JDK** (required by Pyserini)
- **GPU** with CUDA (training runs on one or multiple GPUs)
- Python 3.10+

## Installation

```bash
# Install Java (Ubuntu/Debian)
sudo apt-get install default-jdk -y
export JAVA_HOME=/usr/lib/jvm/default-java

# Clone and install dependencies
git clone https://github.com/your-username/agentic_search_train.git
cd agentic_search_train
pip install -r requirements.txt
```

## Quick Start

### Step 1: Download corpus & build BM25 index

```bash
# Download wiki-18 corpus from HuggingFace and build BM25 index
python build_index.py --mode download --save_dir ./index
```

Or build from your own corpus:

```bash
python build_index.py --mode build --corpus_path ./data/wiki.jsonl --save_dir ./index
```

### Step 2: Prepare training data

```bash
# Supported datasets: nq, hotpotqa, triviaqa
python prepare_data.py --dataset nq --local_dir ./data/nq_search
python prepare_data.py --dataset hotpotqa --local_dir ./data/hotpotqa_search

# Custom dataset (JSONL with "question" and "answer"/"answers" fields)
python prepare_data.py --dataset custom --data_file questions.jsonl --local_dir ./data/custom
```

### Step 3: Train

```bash
bash train_grpo.sh
```

Or run directly:

```bash
python train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --data_path data/nq_search \
    --bm25_index_path ./index/bm25 \
    --bm25_corpus_path ./index/wiki-18.jsonl \
    --output_dir output/search-r1 \
    --num_generations 5 \
    --num_epochs 3 \
    --learning_rate 1e-6 \
    --max_turns 2 \
    --temperature 1.0 \
    --kl_loss_coef 0.001 \
    --save_steps 100
```

## Project Structure

```
agentic_search_train/
├── build_index.py      # Download corpus (HuggingFace) & build Pyserini BM25 index
├── search_env.py       # BM25Retriever wrapper (LuceneSearcher)
├── prepare_data.py     # QA dataset downloader & converter (NQ/HotpotQA/TriviaQA/custom)
├── train_grpo.py       # GRPO training: multi-turn generation + reward + policy update
├── train_grpo.sh       # One-click training script
├── requirements.txt    # Python dependencies
└── README.md
```

## Configuration

### Key hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--model_name_or_path` | `Qwen/Qwen2.5-3B` | Base model for training |
| `--num_generations` | `5` | Number of rollouts per query (GRPO group size) |
| `--max_turns` | `2` | Maximum search rounds per query |
| `--max_response_length` | `500` | Max tokens per generation turn |
| `--max_obs_length` | `500` | Max tokens for retrieved context |
| `--kl_loss_coef` | `0.001` | KL penalty coefficient |
| `--temperature` | `1.0` | Sampling temperature |
| `--retriever_topk` | `3` | Number of documents retrieved per query |
| `--gradient_checkpointing` | `True` | Save VRAM during training |

### Reward function

The reward is computed on the model's response text:

- **1.0** — Extracted answer matches ground truth (Exact Match)
- **0.1** — Valid `<answer>...</answer>` format but incorrect content
- **0.05–0.08** — Partial format bonus (attempted but incomplete tags)
- **0.0** — No valid format detected

## Supported Datasets

| Dataset | Source | Train | Test |
|---|---|---|---|
| NQ | `RUC-NLPIR/FlashRAG_datasets` (nq) | train | test |
| HotpotQA | `RUC-NLPIR/FlashRAG_datasets` (hotpotqa) | train | dev |
| TriviaQA | `RUC-NLPIR/FlashRAG_datasets` (triviaqa) | train | test |
| Custom | Local JSONL (`question` + `answer`) | 90% | 10% |

## Multi-GPU

Set `CUDA_VISIBLE_DEVICES` in `train_grpo.sh` to control which GPUs are used:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## License

MIT
