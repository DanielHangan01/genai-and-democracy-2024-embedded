# This file will be called once the setup is complete. 
# It should perform all the preprocessing steps that are necessary to run the project. 
# All important arguments will be passed via the command line.
# The input files will adhere to the format specified in datastructure/input-file.json

import json
from os.path import join, split as split_path
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("./finetuned-four-epoch-multi-qa-mpnet-base-dot-v1")

# TODO Implement the preprocessing steps here
def handle_input_file(file_location, output_path):
    with open(file_location) as f:
        data = json.load(f)
        article = []
        title = data.get('title', '') or ''
        content = data.get('content', '') or ''
        timestamp = data.get('timestamp', '') or ''
        article.append(f"{title} {content} {timestamp}")
    
    transformed_data = model.encode(article, convert_to_tensor=True)
    transformed_data_list = transformed_data.cpu().numpy().tolist()
    output_data = {"transformed_representation": transformed_data_list}
    
    file_name = split_path(file_location)[-1]
    with open(join(output_path, file_name), "w") as f:
        json.dump(output_data, f)
    

# This is a useful argparse-setup, you probably want to use in your project:
import argparse
parser = argparse.ArgumentParser(description='Preprocess the data.')
parser.add_argument('--input', type=str, help='Path to the input data.', required=True, action="append")
parser.add_argument('--output', type=str, help='Path to the output directory.', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    files_inp = args.input
    files_out = args.output
    
    for file_location in files_inp:
        handle_input_file(file_location, files_out)

 