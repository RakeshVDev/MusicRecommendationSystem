import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import os

# Spotify creds (keep as you have)
CLIENT_ID = "3c8ae8a30406489f93d9d27f180e29a9"
CLIENT_SECRET = "5b0000f6be494188af765155e619a2a9"
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_song_album_cover_url(song_name, artist_name):
    try:
        query = f"track:{song_name} artist:{artist_name}"
        res = sp.search(q=query, type='track', limit=1)
        items = res.get('tracks', {}).get('items', [])
        if items:
            imgs = items[0].get('album', {}).get('images', [])
            if imgs:
                return imgs[0]['url']
    except Exception as e:
        print("Spotify error:", e)
    return "https://via.placeholder.com/300x300.png?text=No+Image"

# ---------- robust TF-IDF builder/loader ----------
def prepare_tfidf_matrix(df, tfidf_path='tfidf_matrix.npz', vectorizer_path='tfidf_vectorizer.pkl', text_col_candidates=None):
    """
    Returns: (tfidf_matrix (sparse), vectorizer, text_col_used)
    Tries to load from disk if present; otherwise builds from df[text_col] or combined text.
    """
    # Try loading cached versions
    if os.path.exists(tfidf_path) and os.path.exists(vectorizer_path):
        try:
            matrix = sparse.load_npz(tfidf_path)
            vectorizer = pickle.load(open(vectorizer_path, 'rb'))
            print("Loaded cached TF-IDF matrix.")
            return matrix, vectorizer, None
        except Exception as e:
            print("Could not load TF-IDF cache:", e)

    # choose text column
    if text_col_candidates is None:
        text_col_candidates = ['lyrics', 'text', 'description', 'features', 'combined']
    text_col = None
    for c in text_col_candidates:
        if c in df.columns:
            text_col = c
            break

    if text_col is None:
        # fallback: combine columns commonly available for songs
        parts = []
        for c in ['song', 'artist', 'album', 'genre']:
            if c in df.columns:
                parts.append(df[c].fillna('').astype(str))
        if not parts:
            raise ValueError("No text-like columns found to build TF-IDF. Add 'lyrics' or similar column.")
        combined = parts[0].copy()
        for p in parts[1:]:
            combined += " " + p
        df['__combined__'] = combined
        text_col = '__combined__'

    texts = df[text_col].fillna('').astype(str)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    matrix = vectorizer.fit_transform(texts)

    # Save cached versions
    try:
        sparse.save_npz(tfidf_path, matrix)
        pickle.dump(vectorizer, open(vectorizer_path, 'wb'))
        print("Saved TF-IDF matrix to disk.")
    except Exception as e:
        print("Could not save TF-IDF cache:", e)

    return matrix, vectorizer, text_col

# Recommend using on-the-fly similarity (no NxN)
def recommend(song, df, tfidf_matrix, top_n=5):
    if song not in df['song'].values:
        st.warning("Selected song not found.")
        return [], []
    idx = int(df[df['song'] == song].index[0])
    # compute similarity for this item only: returns shape (1, N)
    sim_vector = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    # get pairs (index, score) and sort
    sim_scores = list(enumerate(sim_vector))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    filtered = [pair for pair in sim_scores if pair[0] != idx]
    top = filtered[:top_n]
    recommended_names = [df.iloc[i]['song'] for i, _ in top]
    # get posters if present otherwise use spotify
    poster_col = None
    for c in ['poster', 'poster_link', 'image', 'album_image']:
        if c in df.columns:
            poster_col = c
            break
    posters = []
    for i, _ in top:
        artist = df.iloc[i].get('artist', '')
        if poster_col:
            posters.append(df.iloc[i][poster_col])
        else:
            posters.append(get_song_album_cover_url(df.iloc[i]['song'], artist))
    return recommended_names, posters

# ---------- Streamlit app ----------
st.header('Music Recommender System')
df = pickle.load(open('df.pkl','rb'))

# prepare tfidf (will build if not cached)
tfidf_matrix, vectorizer, text_col_used = prepare_tfidf_matrix(df)

music_list = df['song'].values
selected_song = st.selectbox("Type or select a song from the dropdown", music_list)

if st.button('Show Recommendation'):
    rec_names, rec_posters = recommend(selected_song, df, tfidf_matrix, top_n=5)
    # pad to 5
    while len(rec_names) < 5:
        rec_names.append("N/A")
        rec_posters.append("https://via.placeholder.com/300x300.png?text=No+Image")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.text(rec_names[i])
            st.image(rec_posters[i], use_column_width='always')


st.markdown(
    """
    <div style="text-align:center; margin-top:40px; font-size:22px; 
                font-weight:900; color:#1DB954; letter-spacing:1px;">
        DEVELOPED BY <span style="color:#FFFFFF; background:#1DB954; padding:5px 12px; border-radius:8px;">
        RAKESH & RAHUL
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
