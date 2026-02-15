#!/usr/bin/env bash
set -euo pipefail

scripts=(
  stage_00_setup.py
  stage_01_ingest.py
  stage_02_parse.py
  stage_03_metadata.py
  stage_04_mentions.py
  stage_05_embeddings.py
  stage_06_taxonomy.py
  stage_07_novelty.py
  stage_08_events.py
  stage_09_outputs.py
  stage_10_timeline.py
  stage_11_validation.py
  export_pipeline_data.py
)

for s in "${scripts[@]}"; do
  echo "==> Running: $s"
  python3 "$s"
done

echo "All stages completed successfully."