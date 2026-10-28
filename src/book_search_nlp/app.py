"""
app.py — Application Streamlit pour le moteur de recherche BookFinder.

Lance l'interface web qui permet de rechercher des livres par description
en comparant TF-IDF et Sentence Embeddings.

Usage :
    streamlit run src/book_search_nlp/app.py

Auteur : BOUDERNA EL MAHDI
"""

import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from nltk.stem import PorterStemmer
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ── Configuration ────────────────────────────────────────────────────────────

# L'app cherche les fichiers d'index dans nlp_outputs/ (généré par le notebook)
BASE = Path(__file__).resolve().parent.parent.parent / "nlp_outputs"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STEMMER = PorterStemmer()
STOPWORDS = set(ENGLISH_STOP_WORDS) - {"no", "not", "never"}


# ── Prétraitement de la requête ──────────────────────────────────────────────

def preprocess_query(text: str) -> str:
    """Prépare la requête pour TF-IDF (minuscules, stemming, suppression stopwords)."""
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    return " ".join(
        STEMMER.stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1
    )


# ── Chargement des données ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement des index...")
def load_indices():
    """Charge le corpus, le vectoriseur TF-IDF et les embeddings."""
    # Cherche books_indexed.csv ou indexed_books.csv (le notebook peut créer l'un ou l'autre)
    books_path = BASE / "books_indexed.csv"
    if not books_path.exists():
        books_path = BASE / "indexed_books.csv"

    books = pd.read_csv(books_path, dtype={"book_id": str}).fillna("")
    vectorizer = joblib.load(BASE / "tfidf_vectorizer.joblib")
    tfidf_matrix = load_npz(BASE / "tfidf_matrix.npz")
    inverted_index = tfidf_matrix.tocsc()
    embeddings = np.load(BASE / "semantic_embeddings.npy")

    # Vérifier la cohérence des tailles
    assert len(books) == tfidf_matrix.shape[0] == len(embeddings), (
        "Les index ne sont pas synchronisés. Relancer le notebook pour les reconstruire."
    )

    return books, vectorizer, tfidf_matrix, inverted_index, embeddings


@st.cache_resource(show_spinner="Chargement du modèle NLP...")
def load_encoder():
    """Charge le modèle Sentence-Transformers (une seule fois)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


# ── Fonctions de recherche ───────────────────────────────────────────────────

def _normalize_title(title: str) -> str:
    """Normalise un titre en minuscules sans ponctuation pour détecter les rééditions."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(title).lower())).strip()


def search_tfidf(query, vectorizer, tfidf_matrix, inverted_index, books, k=10):
    """Recherche TF-IDF avec index inversé + similarité cosinus (sans doublons de titre)."""
    q = vectorizer.transform([preprocess_query(query)])
    if q.nnz == 0:
        return []

    posting_lists = [
        inverted_index.indices[inverted_index.indptr[t]:inverted_index.indptr[t + 1]]
        for t in q.indices
    ]
    candidates = np.unique(np.concatenate(posting_lists))
    scores = (tfidf_matrix[candidates] @ q.T).toarray().ravel()
    order = np.argsort(-scores)

    results = []
    seen_titles = set()
    for idx_pos in order:
        if scores[idx_pos] <= 0:
            break
        doc_idx = int(candidates[idx_pos])
        title_key = _normalize_title(books.iloc[doc_idx]["title"])
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            results.append((doc_idx, float(scores[idx_pos])))
            if len(results) >= k:
                break
    return results


def search_embeddings(query, embeddings, model, books, k=10):
    """Recherche sémantique par produit scalaire sur embeddings normalisés (sans doublons de titre)."""
    v = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ v
    top_indices = np.argsort(-scores)

    results = []
    seen_titles = set()
    for doc_idx in top_indices:
        title_key = _normalize_title(books.iloc[int(doc_idx)]["title"])
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            results.append((int(doc_idx), float(scores[doc_idx])))
            if len(results) >= k:
                break
    return results


# ── Interface Streamlit ──────────────────────────────────────────────────────

st.set_page_config(page_title="BookFinder NLP", layout="wide")
st.title("BookFinder — Moteur de Recherche de Livres")

books, vectorizer, tfidf_matrix, inverted_index, embeddings = load_indices()
st.write(f"Corpus : **{len(books):,} livres** indexés")

query = st.text_input(
    "Décris le livre que tu recherches (en anglais)",
    "a fantasy adventure with dragons and magic",
)
method = st.radio("Méthode de recherche", ["TF-IDF", "Embeddings"], horizontal=True)
k = st.slider("Nombre de résultats", min_value=1, max_value=15, value=10)

if st.button("Rechercher", type="primary"):
    if not query.strip():
        st.warning("Saisis une requête pour continuer.")
    else:
        with st.spinner("Recherche en cours..."):
            time.sleep(5)
            # Lancer la recherche
            if method == "TF-IDF":
                results = search_tfidf(query, vectorizer, tfidf_matrix, inverted_index, books, k)
            else:
                model = load_encoder()
                results = search_embeddings(query, embeddings, model, books, k)

        if not results:
            st.warning("Aucun livre trouvé pour cette requête.")

        for rank, (idx, score) in enumerate(results, start=1):
            b = books.iloc[idx]
            st.markdown(f"### {rank}. {b['title']}")
            st.write(f"**Score cosinus : {score:.4f}**")
            st.write(str(b["description"])[:750])
            url = str(b.get("url", ""))
            if url.startswith("https://"):
                st.link_button("Voir sur Goodreads", url)
            st.divider()
