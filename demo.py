import json
import os
import time
import torch
from sentence_transformers import SentenceTransformer

def load_articles(filename):
    """
    Load articles from a JSON file.
    
    Args:
    - filename (str): Path to the JSON file.
    
    Returns:
    - list: List of concatenated article strings (title + timestamp + content).
            Returns None if there was an error loading the file.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if not isinstance(data, list):
                raise TypeError(f"Expected JSON array in {filename}, got {type(data)}")
            
            articles = []
            for article in data:
                if not isinstance(article, dict):
                    raise TypeError(f"Expected JSON object for article in {filename}, got {type(article)}")
                
                title = article.get('title', '') or ''
                timestamp = article.get('timestamp', '') or ''
                content = article.get('content', '') or ''
                articles.append(f"{title} {timestamp} {content} ")
            
            return articles
    except Exception as e:
        print(f"An error occurred while loading {filename}: {e}")
        return None

def combine_articles():
    """
    Combine articles from multiple JSON files.
    
    Returns:
    - list: Combined list of articles from all files.
            Returns an empty list if any file loading fails.
    """
    en_articles = load_articles('./data_preprocessing/dataset/test_en_news.json')
    tr_articles = load_articles('./data_preprocessing/dataset/test_tr_news.json')
    ro_articles = load_articles('./data_preprocessing/dataset/test_ro_news.json')

    # Check if any article loading failed
    if None in (en_articles, tr_articles, ro_articles):
        return []

    # Combine articles from all languages
    all_articles = en_articles + tr_articles + ro_articles
    return all_articles

def load_model():
    """
    Load the SentenceTransformer model.
    
    Returns:
    - SentenceTransformer model object or None if loading fails.
    """
    try:
        model = SentenceTransformer("./finetuned-four-epoch-multi-qa-mpnet-base-dot-v1")
        return model
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None

def embed(model, all_articles):
    """
    Encode articles using the SentenceTransformer model.
    
    Args:
    - model: SentenceTransformer model object.
    - all_articles (list): List of article strings.
    
    Returns:
    - torch.Tensor: Tensor of embeddings for all articles.
    """
    try:
        all_embeddings = model.encode(all_articles, convert_to_tensor=True)
        return all_embeddings
    except Exception as e:
        print(f"An error occurred while creating embeddings: {e}")
        return None

def save_embeddings(embeddings, filepath):
    """
    Save embeddings to a file.
    
    Args:
    - embeddings (torch.Tensor): Tensor of embeddings.
    - filepath (str): Path to the file where embeddings will be saved.
    """
    try:
        torch.save(embeddings, filepath)
        print(f"Embeddings saved to {filepath}.")
    except Exception as e:
        print(f"An error occurred while saving embeddings: {e}")

def load_embeddings(filepath):
    """
    Load embeddings from a file.
    
    Args:
    - filepath (str): Path to the file where embeddings are saved.
    
    Returns:
    - torch.Tensor: Tensor of loaded embeddings or None if loading fails.
    """
    try:
        embeddings = torch.load(filepath)
        print(f"Embeddings loaded from {filepath}.")
        return embeddings
    except Exception as e:
        print(f"An error occurred while loading embeddings: {e}")
        return None

def semantic_search(model, query, embeddings, articles, top_k=5):
    """
    Perform semantic search using embeddings.
    
    Args:
    - model: SentenceTransformer model object.
    - query (str): Query string.
    - embeddings (torch.Tensor): Tensor of embeddings for articles.
    - articles (list): List of article strings.
    - top_k (int): Number of top results to return.
    
    Returns:
    - list: List of tuples (article, similarity_score) for top results.
    """
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_sim = torch.nn.functional.cosine_similarity(query_embedding, embeddings)
    top_results = torch.topk(cos_sim, k=top_k)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append((articles[idx], float(score)))

    return results

def main():
    embeddings_path = "./data_preprocessing/dataset/embeddings.pt"
    print("Loading the model...")
    model = load_model()
    if not model:
        print("Failed to load the model. Exiting.")
        return
    print("Model has been loaded.")
    
    print("Loading the articles...")
    all_articles = combine_articles()
    if not all_articles:
        print("Failed to load articles. Exiting.")
        return
    print("Articles have been loaded.")
    
    if os.path.exists(embeddings_path):
        print("Loading existing embeddings...")
        all_embeddings = load_embeddings(embeddings_path)
    else:
        print("Creating the embeddings...")
        start_time = time.time()
        all_embeddings = embed(model, all_articles)
        end_time = time.time()
        if all_embeddings is None:
            print("Failed to create embeddings. Exiting.")
            return
        save_embeddings(all_embeddings, embeddings_path)
        print(f"Time taken to create embeddings: {end_time - start_time:.2f} seconds.")
    
    print("Embeddings have been loaded/created.")

    while True:
        query = input("Enter your query: ")
        if not query.strip():
            print("Empty query entered. Please enter a valid query.")
            continue
        results = semantic_search(model, query, all_embeddings, all_articles)
        print("Top Results:")
        for result in results:
            print(result)

if __name__ == "__main__":
    main()
