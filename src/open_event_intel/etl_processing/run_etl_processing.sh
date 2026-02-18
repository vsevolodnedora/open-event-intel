#!/usr/bin/env bash
set -euo pipefail

# Directory this script lives in (works no matter where you run it from)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------
# Defaults (can be overridden by CLI flags)
# ----------------------------
# If RUN_ID is "NONE" (or empty), auto-generate a SHA-256 hex run id.
RUN_ID="NONE"
RUN_ID_SALT="OPEN_EVENT_INTEL"

CONFIG_DIR="config/"
SOURCE_DB="database/preprocessed_posts.db"
WORKING_DB="database/processed_posts.db"
OUTPUT_DIR="output/processed/"
LOG_DIR="output/processed/logs/"

# Embeddings stage (stage_05_embeddings) defaults (can be overridden by CLI flags)
EMBEDDING_MODEL="arctic-embed"
EMBEDDING_MODEL_BASE_URL="http://localhost:8081/v1"

# Export script (new IO) defaults
EXPORT_OUT="docs/etl_data/"
EXPORT_RUNS=5
EXPORT_SKIP_TRACES=false
EXPORT_COMPLETED_ONLY=false
EXPORT_VERBOSE=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Pipeline options:
  --run-id RUN_ID                 (use "NONE" to auto-generate)
  --config-dir PATH               (default: ${CONFIG_DIR})
  --source-db PATH                (default: ${SOURCE_DB})
  --working-db PATH               (default: ${WORKING_DB})
  --output-dir PATH               (default: ${OUTPUT_DIR})
  --log-dir PATH                  (default: ${LOG_DIR})

Embeddings options (stage_05_embeddings.py only):
  --embedding-model NAME          (default: ${EMBEDDING_MODEL})
  --embedding-model-base-url URL  (default: ${EMBEDDING_MODEL_BASE_URL})

Export options (export_pipeline_data.py):
  --export-out PATH               (default: ${EXPORT_OUT})
  --export-runs N                 (default: ${EXPORT_RUNS})
  --export-skip-traces            (default: off)
  --export-completed-only         (default: off)
  --export-verbose, -v            (default: off)

  -h, --help

Example:
  $(basename "$0") --run-id NONE --log-dir output/processed/logs/ --export-out docs/etl_data/ -v
EOF
}

# ----------------------------
# Helpers
# ----------------------------

# Normalize paths:
# - absolute paths stay absolute
# - relative paths become relative to SCRIPT_DIR
make_abs() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s\n' "$SCRIPT_DIR/$p"
  fi
}

# Generate run id:
# SHA-256 hex digest of: datetime.now().isoformat() + RUN_ID_SALT
generate_run_id() {
  RUN_ID_SALT="$RUN_ID_SALT" python3 - <<'PY'
import os, hashlib, datetime
salt = os.environ.get("RUN_ID_SALT", "OPEN_EVENT_INTEL")
seed = datetime.datetime.now().isoformat()
print(hashlib.sha256((seed + salt).encode("utf-8")).hexdigest())
PY
}

# ----------------------------
# Parse args
# ----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:?Missing value for --run-id}"
      shift 2
      ;;
    --config-dir)
      CONFIG_DIR="${2:?Missing value for --config-dir}"
      shift 2
      ;;
    --source-db)
      SOURCE_DB="${2:?Missing value for --source-db}"
      shift 2
      ;;
    --working-db)
      WORKING_DB="${2:?Missing value for --working-db}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?Missing value for --output-dir}"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="${2:?Missing value for --log-dir}"
      shift 2
      ;;

    # Embeddings (stage_05 only)
    --embedding-model)
      EMBEDDING_MODEL="${2:?Missing value for --embedding-model}"
      shift 2
      ;;
    --embedding-model-base-url)
      EMBEDDING_MODEL_BASE_URL="${2:?Missing value for --embedding-model-base-url}"
      shift 2
      ;;

    # Export (new IO)
    --export-out)
      EXPORT_OUT="${2:?Missing value for --export-out}"
      shift 2
      ;;
    --export-runs)
      EXPORT_RUNS="${2:?Missing value for --export-runs}"
      shift 2
      ;;
    --export-skip-traces)
      EXPORT_SKIP_TRACES=true
      shift
      ;;
    --export-completed-only)
      EXPORT_COMPLETED_ONLY=true
      shift
      ;;
    --export-verbose|-v)
      EXPORT_VERBOSE=true
      shift
      ;;

    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# ----------------------------
# Auto-generate RUN_ID when requested
# ----------------------------
# (Case-insensitive check for NONE; also treat legacy placeholder "RUN_ID" as "generate one".)
run_id_upper="$(printf '%s' "${RUN_ID:-}" | tr '[:lower:]' '[:upper:]')"
if [[ -z "${RUN_ID:-}" || "$run_id_upper" == "NONE" || "${RUN_ID:-}" == "RUN_ID" ]]; then
  RUN_ID="$(generate_run_id)"
  echo "==> Auto-generated RUN_ID: $RUN_ID"
fi

# Hard guard: ensure we ended up with a 64-char SHA256 hex string
if ! [[ "$RUN_ID" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "ERROR: RUN_ID must be a 64-character SHA256 hex string; got: '$RUN_ID'" >&2
  exit 1
fi

# ----------------------------
# Ensure output directories exist (safe even if stages also create them)
# ----------------------------
mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$EXPORT_OUT"

## ----------------------------
## Apply path normalization (relative -> relative to SCRIPT_DIR)
## ----------------------------
#CONFIG_DIR="$(make_abs "$CONFIG_DIR")"
#SOURCE_DB="$(make_abs "$SOURCE_DB")"
#WORKING_DB="$(make_abs "$WORKING_DB")"
#OUTPUT_DIR="$(make_abs "$OUTPUT_DIR")"
#LOG_DIR="$(make_abs "$LOG_DIR")"
#EXPORT_OUT="$(make_abs "$EXPORT_OUT")"

# ----------------------------
# Pipeline stages (old IO)
# ----------------------------
common_args=(
  --run-id "$RUN_ID"
  --config-dir "$CONFIG_DIR"
  --source-db "$SOURCE_DB"
  --working-db "$WORKING_DB"
  --output-dir "$OUTPUT_DIR"
  --log-dir "$LOG_DIR"
)

pipeline_scripts=(
  stage_00_setup/stage_00_setup.py
  stage_01_ingest/stage_01_ingest.py
  stage_02_parse/stage_02_parse.py
  stage_03_metadata/stage_03_metadata.py
  stage_04_mentions/stage_04_mentions.py
  stage_05_embeddings/stage_05_embeddings.py
  stage_06_taxonomy/stage_06_taxonomy.py
  stage_07_novelty/stage_07_novelty.py
  stage_08_events/stage_08_events.py
  stage_09_outputs/stage_09_outputs.py
  stage_10_timeline/stage_10_timeline.py
  stage_11_validation/stage_11_validation.py
)

for rel_script in "${pipeline_scripts[@]}"; do
  script_path="$SCRIPT_DIR/$rel_script"

  # Only stage_05_embeddings expects these extra flags; do NOT pass them to other stages.
  stage_args=( "${common_args[@]}" )
  if [[ "$rel_script" == "stage_05_embeddings/stage_05_embeddings.py" ]]; then
    stage_args+=( --embedding_model "$EMBEDDING_MODEL" --embedding_model_base_url "$EMBEDDING_MODEL_BASE_URL" )
  fi

  echo "==> Running: python3 $script_path ${stage_args[*]}"
  python3 "$script_path" "${stage_args[@]}"
done

# ----------------------------
# Export stage
# export_pipeline_data.py expects: --db --out --runs [--skip-traces] [--completed-only] [-v]
# ----------------------------
export_script="$SCRIPT_DIR/export_etl_pipeline_data.py"
export_args=(
  --db "$WORKING_DB"
  --out "$EXPORT_OUT"
  --runs "$EXPORT_RUNS"
)

$EXPORT_SKIP_TRACES && export_args+=(--skip-traces)
$EXPORT_COMPLETED_ONLY && export_args+=(--completed-only)
$EXPORT_VERBOSE && export_args+=(-v)

echo "==> Running: python3 $export_script ${export_args[*]}"
python3 "$export_script" "${export_args[@]}"

echo "All stages completed successfully."
