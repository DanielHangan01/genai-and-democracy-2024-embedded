# This file will be executed when a user wants to query your project.
import argparse
from os.path import join
import json
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

model = SentenceTransformer("./finetuned-four-epoch-multi-qa-mpnet-base-dot-v1")

# TODO Implement the inference logic here
def handle_user_query(query, query_id, output_path):
    encoded_query = model.encode(query)
    
    if isinstance(encoded_query, np.ndarray):
        encoded_query = torch.tensor(encoded_query)
    
    encoded_query_list = encoded_query.cpu().numpy().tolist()
    
    output_data = {"generated_query": encoded_query_list}
    with open(join(output_path, f"{query_id}.json"), "w") as f:
        json.dump(output_data, f)

# TODO OPTIONAL
# This function is optional for you
# You can use it to interfer with the default ranking of your system.
#
# If you do embeddings, this function will simply compute the cosine-similarity
# and return the ordering and scores
def rank_articles(generated_queries, article_representations):
    """
    This function takes as arguments the generated / augmented user query, as well as the
    transformed article representations.
    
    It needs to return a list of shape (M, 2), where M <= #article_representations.
    Each tuple contains [index, score], where index is the index in the article_repr array.
    The list need already be ordered by score. Higher is better, between 0 and 1.
    
    An empty return list indicates no matches.
    """
    # Convert lists to tensors
    generated_queries = torch.tensor(generated_queries, dtype=torch.float32)
    article_representations = torch.tensor(article_representations, dtype=torch.float32)
    
    # Calculate cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(generated_queries.unsqueeze(0), article_representations, dim=1)
    
    # Get top k results
    k = 1
    top_results = torch.topk(cos_sim, k=k)
    
    # Create the result list
    results = [(index.item(), score.item()) for index, score in zip(top_results.indices, top_results.values)]
    
    return results


""" if True:
    #handle_user_query("What is Big Combo?", "1", "output")
    with open("output/1.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
        generated_query = data.get("generated_query")
    with open("output/article_1.json", 'r', encoding='utf-8') as file:
        article_representations = json.load(file)

    print(rank_articles(generated_query,article_representations))

    exit(0)
    
exit(0) """

# This is a sample argparse-setup, you probably want to use in your project:
parser = argparse.ArgumentParser(description='Run the inference.')
parser.add_argument('--query', type=str, help='The user query.', required=True, action="append")
parser.add_argument('--query_id', type=str, help='The IDs for the queries, in the same order as the queries.', required=True, action="append")
parser.add_argument('--output', type=str, help='Path to the output directory.', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    queries = args.query
    query_ids = args.query_id
    output = args.output
    
    assert len(queries) == len(query_ids), "The number of queries and query IDs must be the same."
    
    for query, query_id in zip(queries, query_ids):
        handle_user_query(query, query_id, output)
    