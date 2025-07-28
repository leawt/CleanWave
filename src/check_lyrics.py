"""
Filename: app.py
Description: Streamlit web app for CleanWave FCC Lyrics Classifier.
Allows users to input song lyrics and classifies them as 'safe', 'indecent', 'obscene', or 'profane'
based on FCC guidelines. Displays color-coded verdicts and similarity scores using Pinecone vector search.
"""

from embed_utils import embed_lyrics
from pinecone_utils import get_index

FCC_CATEGORIES = ["safe", "indecent", "obscene", "profane"]

def check_lyrics_safety(input_lyrics, top_k=5):
    index = get_index()
    # Embed the input lyrics
    vector = embed_lyrics([input_lyrics])[0]

    scores = {}
    matches = {}

    for category in FCC_CATEGORIES:
        result = index.query(
            vector=vector.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter={"label": category}
        )
        matches_list = result["matches"]
        if matches_list:
            avg_score = sum([m["score"] for m in matches_list]) / len(matches_list)
        else:
            avg_score = 0
        scores[category] = avg_score
        matches[category] = matches_list

    verdict = max(scores, key=scores.get)

    return {
        "verdict": verdict,
        "scores": scores,
        "matches": matches
    }