#!/bin/bash

# Navigate to the directory where the script is located (optional safety)
cd "$(dirname "$0")"

# Relative path to the actual prompts/definitions in the sibling private data repo.
# Resolved relative to this script's directory (src/tkg/):
#   src/tkg -> src -> repo-root -> parent dir holding nlp_news_summary_data
TARGET="../../../nlp_news_summary_data/prompts_and_definitions"
LINK_NAME="prompts_and_definitions"

ln -sfn "$TARGET" "$LINK_NAME"

echo "Symlink created: $LINK_NAME → $TARGET"

# Verify that the symlink exists and its target is accessible
if [ -L "$LINK_NAME" ] && [ -e "$LINK_NAME" ]; then
  echo "Verification OK: '$LINK_NAME' is a symlink and its target exists."
else
  echo "Verification FAILED: '$LINK_NAME' is broken or not a symlink." >&2
  exit 1
fi