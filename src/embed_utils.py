"""
Filename: embed_utils.py
Description: Loads a transformer model and provides functions to embed song lyrics as vectors for Pinecone search.
Supports optional caching for batch embedding.
"""

from transformers import AutoTokenizer, AutoModel
import torch
import pickle
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

def embed_lyrics(text_list, batch_size=16, cache_path=None):
    # Only use cache if explicitly provided and for batch uploads
    use_cache = cache_path and len(text_list) > 1
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            embeddings = pickle.load(f)
        print(f"Loaded embeddings from {cache_path}")
        if len(embeddings) == len(text_list):
            return embeddings
        else:
            print("Cache size mismatch, regenerating embeddings...")

    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = model(**encoded_input)
        batch_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        embeddings.extend(batch_embeddings.cpu().numpy())

    if use_cache:
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"Saved embeddings to {cache_path}")

    return embeddings