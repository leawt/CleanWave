"""
Filename: pinecone_utils.py
Description: Utility functions for Pinecone index management.
Provides functions to create, delete, and access the Pinecone index for FCC lyrics classification.
"""

import os
from pinecone import Pinecone, ServerlessSpec

INDEX_NAME = "llama-text-embed-v2-index"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def recreate_index(dimension=384):
    # Delete if exists
    if INDEX_NAME in pc.list_indexes().names():
        pc.delete_index(INDEX_NAME)
        print(f"Deleted existing index '{INDEX_NAME}'.")
    # Create with correct dimension
    pc.create_index(
        name=INDEX_NAME,
        dimension=dimension,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    print(f"Created index '{INDEX_NAME}' with dimension {dimension}.")

def get_index():
    return pc.Index(INDEX_NAME)