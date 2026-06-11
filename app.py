import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

st.set_page_config(page_title="Soundclass", page_icon="♪", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400;1,700&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,600;1,8..60,300;1,8..60,400&display=swap');

:root {
    --navy:        #0B1930;
    --navy-mid:    #122040;
    --navy-light:  #1A2F55;
    --navy-border: #1E3560;
    --offwhite:    #F5F0E8;
    --offwhite-dim:#C8C0B0;
    --offwhite-muted: #8A8070;
    --gold:        #C9A84C;
    --gold-dim:    rgba(201,168,76,0.12);
    --red-accent:  #C43B2A;
}

html, body, [class*="css"], .stApp {
    font-family: 'Source Serif 4', Georgia, serif !important;
    background-color: var(--navy) !important;
    color: var(--offwhite) !important;
}

#MainMenu, footer, .stDeployButton { display: none !important; visibility: hidden !important; }

.main .block-container {
    padding: 0 3rem 4rem 3rem !important;
    max-width: 1200px !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--navy); }
::-webkit-scrollbar-thumb { background: var(--navy-border); border-radius: 2px; }

.soundclass-masthead {
    border-bottom: 3px double var(--offwhite-dim);
    padding: 2.5rem 0 1.5rem;
    margin-bottom: 0.5rem;
    position: relative;
}
.soundclass-eyebrow {
    font-family: 'Source Serif 4', serif;
    font-size: 0.6rem;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: var(--offwhite-muted);
    margin-bottom: 0.6rem;
}
.soundclass-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(3.5rem, 7vw, 6rem);
    font-weight: 900;
    font-style: italic;
    color: var(--offwhite);
    letter-spacing: -0.03em;
    line-height: 0.95;
    margin: 0;
}
.soundclass-subtitle {
    font-family: 'Source Serif 4', serif;
    font-style: italic;
    font-size: 0.85rem;
    color: var(--offwhite-muted);
    margin-top: 0.75rem;
    letter-spacing: 0.02em;
}
.soundclass-rule {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-top: 1.2rem;
}
.soundclass-rule-line {
    flex: 1;
    height: 1px;
    background: var(--offwhite-dim);
}
.soundclass-rule-ornament {
    font-family: 'Playfair Display', serif;
    color: var(--gold);
    font-size: 1.1rem;
}

.search-label {
    font-family: 'Source Serif 4', serif;
    font-size: 0.6rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--offwhite-muted);
    margin-bottom: 0.5rem;
    display: block;
}

.mode-label {
    font-family: 'Source Serif 4', serif;
    font-size: 0.6rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--gold);
    margin: 1.5rem 0 0.5rem;
    display: block;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--navy-border);
}

[data-testid="stTextInput"] input {
    background: var(--navy-mid) !important;
    border: 1px solid var(--navy-border) !important;
    border-radius: 0px !important;
    color: var(--offwhite) !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.15rem !important;
    font-style: italic !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    letter-spacing: 0.01em;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px var(--gold-dim) !important;
    outline: none !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: var(--offwhite-muted) !important;
    font-style: italic !important;
}
[data-testid="stTextInput"] label { display: none !important; }

[data-testid="stSelectbox"] > div > div {
    background: var(--navy-mid) !important;
    border: 1px solid var(--navy-border) !important;
    border-radius: 0px !important;
    color: var(--offwhite) !important;
    font-family: 'Source Serif 4', serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stSelectbox"] label {
    color: var(--offwhite-muted) !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.3em !important;
    text-transform: uppercase !important;
    font-family: 'Source Serif 4', serif !important;
}

[data-testid="stButton"] button {
    background: var(--offwhite) !important;
    color: var(--navy) !important;
    border: none !important;
    border-radius: 0px !important;
    font-family: 'Source Serif 4', serif !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.25em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 2rem !important;
    transition: background 0.18s ease, color 0.18s ease;
    cursor: pointer !important;
}
[data-testid="stButton"] button:hover {
    background: var(--gold) !important;
    color: var(--navy) !important;
}

.result-header {
    font-family: 'Source Serif 4', serif;
    font-size: 0.58rem;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: var(--offwhite-muted);
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.result-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--navy-border);
}

.song-card {
    background: var(--navy-mid);
    border: 1px solid var(--navy-border);
    border-left: 3px solid var(--gold);
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.18s ease, background 0.18s ease;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.song-card:hover {
    background: var(--navy-light) !important;
    border-left-color: var(--offwhite) !important;
}
.song-card-number {
    font-family: 'Playfair Display', serif;
    font-size: 0.65rem;
    color: var(--gold);
    letter-spacing: 0.1em;
    min-width: 2rem;
    font-style: italic;
}
.song-card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--offwhite);
    flex: 1;
    padding: 0 1rem;
}
.song-card-artist {
    font-family: 'Source Serif 4', serif;
    font-style: italic;
    font-size: 0.8rem;
    color: var(--offwhite-muted);
}

.input-song-display {
    background: var(--navy-light);
    border: 1px solid var(--navy-border);
    border-top: 3px solid var(--offwhite);
    padding: 1.25rem 1.5rem;
    margin: 1.5rem 0 0.5rem;
}
.input-song-label {
    font-size: 0.55rem;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: var(--offwhite-muted);
    font-family: 'Source Serif 4', serif;
    margin-bottom: 0.4rem;
}
.input-song-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 900;
    font-style: italic;
    color: var(--offwhite);
    line-height: 1.1;
}
.input-song-artist {
    font-family: 'Source Serif 4', serif;
    font-style: italic;
    font-size: 0.85rem;
    color: var(--gold);
    margin-top: 0.3rem;
}

.ai-insight-box {
    background: var(--navy-mid);
    border: 1px solid var(--navy-border);
    border-left: 3px solid var(--gold);
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
    font-family: 'Source Serif 4', serif;
    font-style: italic;
    font-size: 0.9rem;
    color: var(--offwhite-dim);
    line-height: 1.75;
}
.ai-insight-label {
    font-size: 0.55rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--gold);
    font-style: normal;
    margin-bottom: 0.5rem;
    display: block;
    font-family: 'Source Serif 4', serif;
}

.stats-bar {
    display: flex;
    gap: 2rem;
    padding: 1rem 0;
    border-top: 1px solid var(--navy-border);
    border-bottom: 1px solid var(--navy-border);
    margin: 1rem 0;
}
.stat-item { text-align: center; }
.stat-number {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--offwhite);
    display: block;
    line-height: 1;
}
.stat-label {
    font-size: 0.55rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--offwhite-muted);
    font-family: 'Source Serif 4', serif;
    margin-top: 0.3rem;
    display: block;
}

[data-testid="stAlert"] {
    background: var(--navy-mid) !important;
    border: 1px solid var(--navy-border) !important;
    border-left: 3px solid var(--red-accent) !important;
    border-radius: 0 !important;
    color: var(--offwhite) !important;
}

[data-testid="stSpinner"] > div { border-top-color: var(--gold) !important; }

[data-baseweb="popover"] ul,
[data-baseweb="menu"] {
    background: var(--navy-mid) !important;
    border: 1px solid var(--navy-border) !important;
    border-radius: 0 !important;
}
[data-baseweb="menu"] li { color: var(--offwhite) !important; font-family: 'Source Serif 4', serif !important; }
[data-baseweb="menu"] li:hover { background: var(--navy-light) !important; }

[data-testid="stRadio"] label {
    color: var(--offwhite) !important;
    font-family: 'Source Serif 4', serif !important;
    font-size: 0.85rem !important;
}

@keyframes fade-in {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate-in { animation: fade-in 0.4s ease forwards; }
</style>
""", unsafe_allow_html=True)

# ── Masthead ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="soundclass-masthead animate-in">
    <div class="soundclass-eyebrow">Vol. I &nbsp;·&nbsp; Lyrics Intelligence &nbsp;·&nbsp; 57,650 Songs</div>
    <h1 class="soundclass-title">Soundclass</h1>
    <p class="soundclass-subtitle">A lyrical recommendation engine &mdash; find what sounds like what you already love.</p>
    <div class="soundclass-rule">
        <div class="soundclass-rule-line"></div>
        <div class="soundclass-rule-ornament">✦</div>
        <div class="soundclass-rule-line"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Backend ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading the archive...")
def load_data_and_model():
    df = pd.read_csv("spotify_millsongdata.csv")
    df = df.sample(5000, random_state=42).reset_index(drop=True)
    df["song"] = df["song"].str.strip()
    df["text"] = df["text"].fillna("")

    stemmer = PorterStemmer()

    def tokenize(txt):
        try:
            tokens = nltk.word_tokenize(txt.lower())
            return " ".join(stemmer.stem(w) for w in tokens if w.isalpha())
        except:
            return txt.lower()

    df["processed"] = df["text"].apply(tokenize)
    vectorizer = TfidfVectorizer(analyzer="word", stop_words="english", max_features=8000)
    tfidf_matrix = vectorizer.fit_transform(df["processed"])
    return df, vectorizer, tfidf_matrix

def recommend(song_name, df, vectorizer, tfidf_matrix, n=9):
    matches = df[df["song"].str.lower() == song_name.lower()]
    if matches.empty:
        partial = df[df["song"].str.lower().str.contains(song_name.lower(), na=False)]
        if partial.empty:
            return None, None
        idx = partial.index[0]
    else:
        idx = matches.index[0]

    song_vec = tfidf_matrix[idx]
    scores = cosine_similarity(song_vec, tfidf_matrix).flatten()
    scores[idx] = 0
    top_indices = np.argsort(scores)[::-1][:n]

    results = []
    for i in top_indices:
        results.append({
            "song": df.iloc[i]["song"],
            "artist": df.iloc[i]["artist"],
            "score": round(float(scores[i]), 3)
        })

    return results, df.iloc[idx]

def mood_to_song(description, df):
    client = Groq(api_key=GROQ_API_KEY)
    song_list = df[["song", "artist"]].drop_duplicates().head(300)
    songs_text = "\n".join([f"{r['song']} by {r['artist']}" for _, r in song_list.iterrows()])

    prompt = f"""You are a music expert with deep knowledge of song lyrics and themes.

A user has described what they want to listen to:
"{description}"

Here is a sample of songs available in the archive:
{songs_text}

Your task: pick the ONE song from this list that best matches the user's description based on likely lyrical themes, mood, and emotional content. Return ONLY the exact song title as it appears in the list, nothing else. No explanation, no artist name, just the song title."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30
    )
    return response.choices[0].message.content.strip()

def explain_recommendations(input_song, results, description=None):
    client = Groq(api_key=GROQ_API_KEY)
    result_list = "\n".join([f"{r['song']} by {r['artist']}" for r in results[:5]])

    if description:
        context = f'The user described: "{description}". This led to the anchor song: "{input_song["song"]}" by {input_song["artist"]}.'
    else:
        context = f'The user searched for: "{input_song["song"]}" by {input_song["artist"]}.'

    prompt = f"""You are Soundclass, a lyrical intelligence engine.

{context}

The top 5 recommendations are:
{result_list}

In 3-4 sentences, explain what lyrical or thematic thread connects these songs to the anchor track. Be specific about themes, mood, or imagery. Write in a refined, editorial tone."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

df, vectorizer, tfidf_matrix = load_data_and_model()

# ── Stats Bar ─────────────────────────────────────────────────────────────────
total_artists = df["artist"].nunique()
st.markdown(f"""
<div class="stats-bar">
    <div class="stat-item">
        <span class="stat-number">{len(df):,}</span>
        <span class="stat-label">Songs in Archive</span>
    </div>
    <div class="stat-item">
        <span class="stat-number">{total_artists:,}</span>
        <span class="stat-label">Artists</span>
    </div>
    <div class="stat-item">
        <span class="stat-number">9</span>
        <span class="stat-label">Recommendations</span>
    </div>
    <div class="stat-item">
        <span class="stat-number">TF&#8209;IDF</span>
        <span class="stat-label">Method</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Search Mode Toggle ────────────────────────────────────────────────────────
st.markdown('<span class="mode-label">Search Mode</span>', unsafe_allow_html=True)
mode = st.radio("mode", ["Search by song title", "Describe a mood or feeling"], label_visibility="collapsed", horizontal=True)

song_input = ""
mood_input = ""
search = False
description_used = None

if mode == "Search by song title":
    st.markdown('<span class="search-label">Song Title</span>', unsafe_allow_html=True)
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        song_input = st.text_input("song", placeholder="e.g. Bohemian Rhapsody, Waiting For The Man...", label_visibility="collapsed")
    with col_btn:
        st.markdown("<div style='height:0.1rem'></div>", unsafe_allow_html=True)
        search = st.button("Find")

    st.markdown("<div style='margin-top:1.5rem'>", unsafe_allow_html=True)
    artists = sorted(df["artist"].unique().tolist())
    selected_artist = st.selectbox("Browse by Artist", ["— Select an artist —"] + artists)
    if selected_artist != "— Select an artist —":
        artist_songs = df[df["artist"] == selected_artist]["song"].tolist()
        selected_song = st.selectbox("Select a song", ["— Select a song —"] + artist_songs)
        if selected_song != "— Select a song —":
            song_input = selected_song
            search = True
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown('<span class="search-label">Describe What You Want to Hear</span>', unsafe_allow_html=True)
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        mood_input = st.text_input("mood", placeholder="e.g. something melancholic about lost love and empty streets...", label_visibility="collapsed")
    with col_btn:
        st.markdown("<div style='height:0.1rem'></div>", unsafe_allow_html=True)
        search = st.button("Find")

# ── Results ───────────────────────────────────────────────────────────────────
if search:
    if mode == "Describe a mood or feeling" and mood_input.strip():
        with st.spinner("Finding the right song for your mood..."):
            matched_title = mood_to_song(mood_input.strip(), df)
            song_input = matched_title
            description_used = mood_input.strip()

    if song_input.strip():
        with st.spinner("Searching the archive..."):
            results, input_song = recommend(song_input.strip(), df, vectorizer, tfidf_matrix)

        if results is None:
            st.error(f'No song matching "{song_input}" found in the archive. Try a different title.')
        else:
            st.markdown(f"""
            <div class="input-song-display animate-in">
                <div class="input-song-label">{'Matched From Your Mood' if description_used else 'Now Playing'}</div>
                <div class="input-song-name">{input_song['song']}</div>
                <div class="input-song-artist">{input_song['artist']}</div>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Analyzing lyrical connections..."):
                insight = explain_recommendations(input_song, results, description_used)

            st.markdown(f"""
            <div class="ai-insight-box animate-in">
                <span class="ai-insight-label">✦ Soundclass Intelligence</span>
                {insight}
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="result-header">Recommended Tracks</div>', unsafe_allow_html=True)

            for i, r in enumerate(results, 1):
                st.markdown(f"""
                <div class="song-card animate-in">
                    <span class="song-card-number">0{i}</span>
                    <span class="song-card-title">{r['song']}</span>
                    <span class="song-card-artist">{r['artist']}</span>
                </div>
                """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    border-top: 1px solid #1E3560;
    margin-top: 3rem;
    padding-top: 1.25rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
">
    <span style="font-family:'Playfair Display',serif; font-style:italic; font-size:1rem; color:#8A8070;">Soundclass</span>
    <span style="font-family:'Source Serif 4',serif; font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase; color:#8A8070;">
        Lyrics Intelligence &nbsp;·&nbsp; TF-IDF + Groq Llama 3.3
    </span>
</div>
""", unsafe_allow_html=True)