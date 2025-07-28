"""
Filename: app.py
Description: Streamlit web app for CleanWave FCC Lyrics Classifier.
Allows users to input song lyrics and classifies them as 'safe', 'indecent', 'obscene', or 'profane'
based on FCC guidelines. Displays color-coded verdicts and similarity scores using Pinecone vector search.
"""

import streamlit as st
from check_lyrics import check_lyrics_safety

st.set_page_config(page_title="CleanWave: FCC Lyrics Classifier")
st.title("CleanWave")
st.subheader("FCC-Compliant Lyrics Screener")

st.markdown("Paste in the full lyrics of a song below, and we’ll check if it contains **obscene**, **indecent**, or **profane** content under FCC guidelines.")

with st.expander("What do these FCC categories mean?"):
    st.markdown("""
- **Obscene**: Graphic sexual conduct that appeals to prurient interest and lacks artistic, literary, political, or scientific value.
- **Indecent**: Sexually suggestive or excretory content that is patently offensive, but not obscene.
- **Profane**: Grossly offensive language, including vulgar or culturally sensitive terms.
- **Safe**: No objectionable or restricted content.
""")

lyrics_input = st.text_area("Enter your song lyrics here:", height=300)

def score_to_color(score):
    # Green (low) to yellow (mid) to red (high)
    r = int(255 * score)
    g = int(255 * (1 - score))
    b = 50
    return f"rgb({r},{g},{b})"

if st.button("Classify Lyrics"):
    if not lyrics_input.strip():
        st.warning("Please enter lyrics to analyze.")
    else:
        result = check_lyrics_safety(lyrics_input)
        verdict = result['verdict'].upper()
        verdict_color = "#27ae60" if verdict == "SAFE" else "#e74c3c"

        st.markdown(
            f"### FCC Verdict: <span style='color:{verdict_color}; font-weight:bold'>{verdict}</span>",
            unsafe_allow_html=True
        )

        st.markdown("#### Similarity Scores:")
        for category, score in result["scores"].items():
            color = score_to_color(score)  # color depends on score value
            st.markdown(
                f"- <span style='color:black; font-weight:bold'>{category.capitalize()}</span>: "
                f"<span style='color:{color}; font-weight:bold'>{score:.4f}</span>",
                unsafe_allow_html=True
            )

        with st.expander("📖 See Matched Examples for Each Category"):
            for category, matches in result["matches"].items():
                st.markdown(f"**{category.capitalize()} Matches:**")
                for match in matches:
                    text = match['metadata']['text']
                    st.write(f"> {text[:150]}...")
