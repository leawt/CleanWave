# CleanWave
_Checking if a song is radio ready._

[CleanWave](https://cleanwave.streamlit.app/) is a semantic lyric classification tool built to help radio DJs and content moderators determine whether a song is compliant with [FCC broadcasting standards](https://www.fcc.gov/consumers/guides/obscene-indecent-and-profane-broadcasts). By leveraging Pinecone, sentence-transformers, and Streamlit, CleanWave avoids rigid keyword filtering and instead uses semantic similarity to make nuanced classification decisions.

When lyrics are pasted into the app, CleanWave classifies them into one of four FCC categories — Safe, Profane, Indecent, or Obscene — enabling users to make informed broadcasting decisions with transparency and speed.

### FCC Definitions
- **Obscene**: Appeals to prurient interest, depicts sexual conduct in a patently offensive way, and lacks serious artistic, political, or scientific value.
- **Indecent**: Describes sexual or excretory acts or organs in a patently offensive way but does not meet the full obscenity test.
- **Profane**: Contains grossly offensive language (e.g., curse words, slurs).
- **Safe**: Contains no objectionable content.

The idea for this app came from my time as a DJ at my college radio station (90.1 FM KZSU!). Whenever I was picking songs on the fly or taking requests from callers, I’d end up frantically scrubbing through lyrics to make sure nothing violated FCC rules. Profane songs were easy to spot, but the lines between _indecent_ and _obscene_ were way blurrier. I used to think, “I should build a tool for this,” but quickly realized it would take a huge amount of data to match songs 1:1 and would still need some serious NLP to handle the gray areas.

With Pinecone and semantic search, I was finally able to get around those problems. Instead of trying to directly label every single song, CleanWave uses embeddings and vector search to understand the meaning behind the lyrics and classify them accordingly.

---

## How It Works:
1. **Paste lyrics** into the [Streamlit web app](https://cleanwave.streamlit.app/) (or run it locally-- see [Quickstart](#quickstart) instructions below).
2. **Lyrics are embedded** using a transformer model.
3. **Embeddings are compared** to a Pinecone vector database of labeled lyrics.
4. **Semantic similarity scores** are calculated for each FCC category.
5. **FCC verdict** is displayed with similarity score details.

---

## Key Challenges Addressed

### Problem 1: FCC Rules Are Ambiguous  
The boundaries between "indecent" and "obscene" can be subtle and context-specific. Traditional profanity filters fail because they rely on literal word matching without nuance.

**Solution:**  
CleanWave uses **semantic similarity search** powered by Pinecone to evaluate the meaning behind lyrics — capturing ambiguity that keyword lists miss.

### Problem 2: Data Is Limited and Hard to Label  
Training a classifier would require tens of thousands of labeled examples — expensive and time-consuming, especially with fuzzy labels.

**Solution:**  
CleanWave avoids heavy model training by using **Pinecone as a zero-shot classifier**. By embedding a small, high-quality labeled dataset, new inputs are classified by finding the most semantically similar examples in vector space.

---

## Workflow

1. **Data Collection & Labeling**  
   - Uses [this Kaggle dataset of 60,000 Spotify songs](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs).  
   - Labels are applied using profanity word lists, regex, and manual inspection.

2. **Embedding Lyrics**  
   - Each song’s lyrics are embedded using [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) to capture contextual meaning.

3. **Vector Storage with Pinecone**  
   - Labeled embeddings are uploaded to Pinecone to support high-speed, scalable similarity search.

4. **Inference and Classification**  
   - On user input, lyrics are embedded and compared to labeled examples in each FCC category using Pinecone.  
   - The most semantically similar group is selected as the **FCC Verdict**, with similarity scores and examples shown.

5. **UI**  
   - A simple [Streamlit](https://cleanwave.streamlit.app/) web app interface for users to paste lyrics and view results.

---

## Quickstart

1. **Clone this repo**
    ```
    git clone https://github.com/yourusername/CleanWave.git
    cd CleanWave
    ```

2. **Set up environment**
    ```
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. **Upload embeddings to Pinecone**
    ```
    python src/upload_dataset.py
    ```

4. **Run the app**
    ```
    streamlit run src/app.py
    ```
---

## Example Output

```
FCC Verdict: INDECENT

Similarity Scores:
- Obscene: 0.52
- Indecent: 0.85
- Profane: 0.76
- Safe: 0.34
```

---

## Requirements

- Python 3.8+
- Pinecone API Key (set as `PINECONE_API_KEY`)
- Open access to [Genius.com](https://genius.com) if collecting new lyrics

---

## Created By
Lea Wang-Tomic 
