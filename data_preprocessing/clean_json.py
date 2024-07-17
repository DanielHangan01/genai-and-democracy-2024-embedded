import json

# Input and output file paths
input_file = '/home/daniel/workspace/github.com/danielhangan01/genai-and-democracy-2024-embedded/data_preprocessing/ro_combined.json'
output_file = '/home/daniel/workspace/github.com/danielhangan01/genai-and-democracy-2024-embedded/data_preprocessing/test_ro_news.json'

# Function to clean content field of each JSON object
def clean_content(content):
    if content is None:
        return None
    # Remove \n and \" from content
    cleaned_content = content.replace('\n', ' ').replace('\"', '”')
    return cleaned_content

# List to hold cleaned JSON objects
cleaned_json_objects = []

# Read the combined JSON file
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

    # Iterate over each JSON object in the array
    for obj in data:
        # Clean the content field
        obj['content'] = clean_content(obj['content'])
        
        # Append cleaned object to list
        cleaned_json_objects.append(obj)

# Write cleaned JSON objects to output file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(cleaned_json_objects, f, ensure_ascii=False, indent=2)

print(f"Cleaned JSON saved to {output_file}")


