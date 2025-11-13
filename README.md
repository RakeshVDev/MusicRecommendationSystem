# Music_Recommender_System

🎵 Music Recommender System

An intelligent Machine Learning–based Music Recommendation System built with Python, Streamlit, and Spotify API.
This project recommends songs similar to a user-selected track using TF-IDF vectorization and cosine similarity on song metadata or lyrics.

🧠 Project Overview

The system analyzes the textual content (song titles, artists, albums, or lyrics) to understand similarity between songs.
When a user selects a song, the app dynamically computes cosine similarity in real-time and recommends the top 5 similar songs along with album covers fetched using the Spotify API.

⚙️ Features

✅ Real-time music recommendations based on similarity
✅ Integration with Spotify API to display album cover images
✅ Fast TF-IDF–based text similarity (no heavy model training)
✅ Streamlit web app for interactive user experience
✅ Dark UI theme with clean 5-column layout for recommendations

🧩 Tech Stack
Component	Technology Used
Language	Python 3.11
Framework	Streamlit
Machine Learning	TF-IDF Vectorizer, Cosine Similarity
API	Spotify API (Spotipy Library)
Libraries	numpy, pandas, scikit-learn, spotipy, streamlit
Data Source	spotify_millsongdata.csv (or your custom dataset)
🚀 Installation & Setup
1️⃣ Clone or Download this Repository
git clone https://github.com/RakeshVDev/music-recommender.git
cd music-recommender

2️⃣ Create and Activate a Virtual Environment
python -m venv .venv
.venv\Scripts\activate  # on Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit App
python -m streamlit run app.py


Then open the URL shown (usually http://localhost:8501
) in your browser.

📁 Project Structure
📦 MUSICPROJECT
│
├── app.py                # Streamlit main app
├── df.pkl                # Preprocessed song dataset
├── tfidf_matrix.npz      # Cached TF-IDF features
├── tfidf_vectorizer.pkl  # Saved TF-IDF vectorizer
├── similarity.pkl        # Optional precomputed similarity
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
└── .venv/                # Virtual environment (optional)

🧮 How it Works

Loads the dataset (df.pkl) containing songs and artists.

Builds or loads a TF-IDF matrix (text features of songs).

When a user selects a song, the app computes cosine similarity with all songs.

Returns the Top 5 most similar songs and fetches their album covers via the Spotify API.

🖼️ UI Preview
Screenshot	Description
🎧
	Streamlit web app showing top recommended songs and album covers.
🧰 Example Code Snippet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
matrix = tfidf.fit_transform(df['lyrics'].fillna(''))

index = df[df['song'] == selected_song].index[0]
similarity = cosine_similarity(matrix[index:index+1], matrix).flatten()
top_indices = similarity.argsort()[-6:-1][::-1]
recommended_songs = df.iloc[top_indices]['song']

💡 Future Improvements

🔹 Add audio feature–based recommendations using Spotify track embeddings
🔹 Support personalized recommendations using user history
🔹 Include genre-based filters or mood-based clustering
🔹 Integrate deep learning models for richer embeddings (e.g., BERT-based lyric similarity)

👨‍💻 Contribution

Developed & Maintained by:
💼 RkTech & Team
🚀 Innovation through AI & ML

If you’d like to contribute, feel free to fork this repository and submit a pull request.

For RUN -

CMD - python -m pip install spotipy

Run - python -m streamlit run app.py


📜 License

This project is open-source and available under the MIT License.