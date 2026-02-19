#!/bin/bash

# Navigate to the directory where the script is located (optional safety)
cd "$(dirname "$0")"

# Relative path to the actual config
TARGET="../../../nlp_news_summary_data/etl_config/config.yaml"
LINK_NAME="etl_config/config.yaml"
ln -sf "$TARGET" "$LINK_NAME"
echo "Symlink created: $LINK_NAME → $TARGET"
