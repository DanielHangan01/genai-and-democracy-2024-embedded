#!/bin/bash
# Directory containing JSON files
dir="./dataset/tr_news/"

# Loop through all JSON files
find "$dir" -type f -name "*.json" | while read -r file; do
  # Check if maintext is null
  if jq -e '.maintext == null' "$file" > /dev/null; then
    echo "Deleting $file"
    rm "$file"
  fi
done