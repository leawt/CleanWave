"""
Filename: upload_dataset.py
Description: Reads a JSON file of labeled song lyrics, generates embeddings, and uploads them to a Pinecone index.
Recreates the index with the correct dimension and batches uploads to avoid size limits.
"""

import json
import os
from pinecone import Pinecone
from pinecone_utils import recreate_index  # Import the function
from embed_utils import embed_lyrics

with open("data/song_lyrics.json", "r") as f:
    dataset = json.load(f)

lyrics = [item["lyrics"] for item in dataset]
labels = [item["fcc_label"] for item in dataset]

# Recreate the index with the correct dimension
recreate_index(dimension=384)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "llama-text-embed-v2-index"
index = pc.Index(INDEX_NAME)

embeddings = embed_lyrics(lyrics)

def preprocess_lyrics(lyrics, max_length=2000):
    # Remove duplicate lines
    lines = lyrics.splitlines()
    seen = set()
    unique_lines = []
    for line in lines:
        line = line.strip()
        if line and line not in seen:
            unique_lines.append(line)
            seen.add(line)
    processed = " ".join(unique_lines)
    # truncate 
    return processed[:max_length]

records = [
    {
        "id": str(i),
        "values": embeddings[i],
        "metadata": {
            "text": preprocess_lyrics(lyrics[i]), 
            "label": labels[i]
        }
    }
    for i in range(len(lyrics))
]

BATCH_SIZE = 100
for i in range(0, len(records), BATCH_SIZE):
    batch = records[i:i+BATCH_SIZE]
    index.upsert(batch)
    print(f"Uploaded batch {i//BATCH_SIZE + 1} ({len(batch)} records)")

print(f"Uploaded {len(lyrics)} lyrics with FCC labels to Pinecone.")