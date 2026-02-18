#!/usr/bin/env bash
set -euo pipefail



llama-server -m "${MODELS_DIR}/snowflake-arctic-embed-l-v2.0.F16.gguf" \
  --port 8081 --embeddings --ctx-size 8192 \
  --batch-size 2048 --ubatch-size 1024 \
  > embeddings.log 2>&1 &

# 2) Translation
#llama-server -m "${MODELS_DIR}/EuroLLM-9B-Instruct.Q3_K_M.gguf" \
#  --port 8082 --ctx-size 4096 \
#  > translation.log 2>&1 &

# 3) Complex NLP
#llama-server -m "${MODELS_DIR}/mistral-7b-instruct-v0.2.Q4_K_M.gguf" \
#  --port 8080 --ctx-size 8192 \
#  > nlp.log 2>&1 &

echo "Started llama-server processes:"
echo "  NLP         http://localhost:8080/v1  (log: nlp.log)"
echo "  Embeddings  http://localhost:8081/v1  (log: embeddings.log)"
echo "  Translation http://localhost:8082/v1  (log: translation.log)"
echo
echo "Stop them with:"
echo "  kill $(jobs -p)"
