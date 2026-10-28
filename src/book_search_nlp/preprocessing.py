"""
preprocessing.py — Nettoyage et prétraitement du texte pour le moteur BookFinder.

Pipeline :
    1. clean_text   : Décode les entités HTML, supprime les balises, normalise les espaces.
    2. preprocess    : Passe en minuscules, tokenise (mots uniquement), retire les stopwords
                       (en gardant no/not/never), et applique un stemming de Porter.
    3. get_stats     : Renvoie des statistiques sur le corpus nettoyé.

Auteur : BOUDERNA EL MAHDI
"""

import html
import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ── Outils partagés ──────────────────────────────────────────────────────────

_stemmer = PorterStemmer()

# On garde les négations (no, not, never) car elles changent le sens.
STOPWORDS = set(ENGLISH_STOP_WORDS) - {"no", "not", "never"}


# ── Fonctions principales ────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Décode les entités HTML, supprime les balises et normalise les espaces."""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text: str) -> str:
    """Tokenise, retire les stopwords et applique le stemming de Porter.

    Conçu pour préparer le texte avant la vectorisation TF-IDF.
    Pour les embeddings (Sentence-Transformers), utiliser `clean_text` suffit
    car le modèle exploite le contexte complet de la phrase.
    """
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", clean_text(text).lower())
    return " ".join(
        _stemmer.stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1
    )


def get_stats(descriptions: "pd.Series") -> dict:
    """Calcule des statistiques sur les descriptions nettoyées (après `preprocess`).

    Parameters
    ----------
    descriptions : pd.Series
        Colonne de texte déjà passée par `preprocess`.

    Returns
    -------
    dict avec clés : mean_tokens, median_tokens, vocab_size
    """
    lengths = descriptions.str.split().str.len()
    vocab = set(" ".join(descriptions).split())
    return {
        "mean_tokens": round(float(lengths.mean()), 1),
        "median_tokens": int(lengths.median()),
        "vocab_size": len(vocab),
    }
