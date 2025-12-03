#!/bin/bash

# Navigate to the directory where the script is located (optional safety)
cd "$(dirname "$0")"

# Relative path to the actual database
TARGET="../../nlp_news_summary_data/database/scraped_posts.db"
LINK_NAME="scraped_posts.db"
ln -sf "$TARGET" "$LINK_NAME"
echo "Symlink created: $LINK_NAME → $TARGET"


TARGET="../../nlp_news_summary_data/database/preprocessed_posts.db"
LINK_NAME="preprocessed_posts.db"
ln -sf "$TARGET" "$LINK_NAME"
echo "Symlink created: $LINK_NAME → $TARGET"