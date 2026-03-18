#!/bin/bash
export HF_HOME=/home/aadivyar/.cache/huggingface
export TRANSFORMERS_CACHE=/home/aadivyar/.cache/huggingface
export TMPDIR=/tmp

CUDA_VISIBLE_DEVICES=2 python -m evaluations.gsm_symbolic.cli \
    --run-dir outputs/generated-csd/latest \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --device cuda \
    --limit 3 \
    --max-steps 1024 \
    --vocab-size 3000 \
    --debug-delimiters
