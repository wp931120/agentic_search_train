#!/bin/bash
# Search-R1 GRPO training with Pyserini BM25

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

BASE_MODEL='Qwen/Qwen2.5-0.5B-Instruct'
EXPERIMENT_NAME=search-r1-trl-bm25-qwen2.5-3b-em
DATA_DIR='data/hotpotqa_search'

BM25_INDEX='./index/bm25/bm25'
BM25_CORPUS='./index/wiki-18.jsonl'

# ============================================================================
# Step 0: Prepare data (run once)
# ============================================================================
# python3 prepare_data.py --dataset nq --local_dir $DATA_DIR --template_type base

# ============================================================================
# Step 1: Download corpus + build BM25 index (run once)
# ============================================================================
# python3 build_index.py --mode download --save_dir ./index

# ============================================================================
# Step 2: Train
# ============================================================================
python3 train_grpo.py \
    --model_name_or_path $BASE_MODEL \
    --data_path $DATA_DIR \
    --bm25_index_path $BM25_INDEX \
    --bm25_corpus_path $BM25_CORPUS \
    --retriever_topk 3 \
    --output_dir output/$EXPERIMENT_NAME \
    --max_samples 100 \
    --num_generations 2 \
    --learning_rate 1e-6 \
    --lr_warmup_ratio 0.1 \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_prompt_length 4096 \
    --max_response_length 500 \
    --max_obs_length 500 \
    --max_turns 2 \
    --temperature 1.0 \
    --kl_loss_coef 0.001 \
    --attn_implementation sdpa \
    --save_steps 100 \
    2>&1 | tee $EXPERIMENT_NAME.log
