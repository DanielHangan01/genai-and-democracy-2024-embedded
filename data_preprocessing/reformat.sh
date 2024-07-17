#!/bin/bash
# Directory containing JSON files
dir="/home/daniel/workspace/github.com/danielhangan01/genai-and-democracy-2024-embedded/data_preprocessing/dataset/ro_news"

# Loop through all JSON files
find "$dir" -type f -name "*.json" | while read -r file; do
  # Process the JSON file and overwrite it with the modified content
  jq '{title, timestamp: .date_publish, content: .maintext}' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
done