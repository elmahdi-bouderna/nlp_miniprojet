"""
semantic_search.py — Recherche sémantique par embeddings de phrases.

Utilise le modèle léger `all-MiniLM-L6-v2` (Sentence-BERT) pour encoder
chaque description en un vecteur dense de 384 dimensions, puis recherche
les K voisins les plus proches par similarité cosinus.

Auteur : BOUDERNA EL MAHDI
"""

from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ── Construction de l'index ──────────────────────────────────────────────────

def build_embeddings(texts: list[str], output_dir: Path, batch_size: int = 32) -> np.ndarray:
    """Encode les textes avec Sentence-Transformers et sauvegarde le résultat.

    Parameters
    ----------
    texts : list[str]
        Textes bruts (pas besoin de les prétraiter — le modèle Transformer
        travaille sur le texte complet).
    output_dir : Path
        Dossier où sauvegarder `semantic_embeddings.npy`.
    batch_size : int
        Taille du lot pour l'encodage.

    Returns
    -------
    np.ndarray de forme (n_docs, 384)
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "semantic_embeddings.npy", embeddings)

    return embeddings


def load_embeddings(output_dir: Path) -> np.ndarray:
    """Charge les embeddings depuis le disque."""
    return np.load(output_dir / "semantic_embeddings.npy")


# ── Index de voisins proches ─────────────────────────────────────────────────

def build_nn_index(embeddings: np.ndarray, n_neighbors: int = 15) -> NearestNeighbors:
    """Construit un index NearestNeighbors (distance cosinus, brute-force CPU).

    Parameters
    ----------
    embeddings : np.ndarray
        Matrice (n_docs, dim) normalisée.
    n_neighbors : int
        Nombre maximum de voisins à rechercher.

    Returns
    -------
    NearestNeighbors ajusté
    """
    nn_index = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn_index.fit(embeddings)
    return nn_index


# ── Recherche ────────────────────────────────────────────────────────────────

def search(query: str, nn_index: NearestNeighbors, model=None, k: int = 10) -> list[tuple[int, float]]:
    """Recherche les K livres les plus proches sémantiquement.

    Parameters
    ----------
    query : str
        Requête en langage naturel.
    nn_index : NearestNeighbors
        Index de voisins proches ajusté sur les embeddings du corpus.
    model : SentenceTransformer | None
        Modèle pour encoder la requête. Chargé automatiquement si None.
    k : int
        Nombre de résultats.

    Returns
    -------
    list[tuple[int, float]]
        Paires (index_du_livre, score_cosinus), triées par score décroissant.
    """
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    q = model.encode([query], normalize_embeddings=True)
    distances, indices = nn_index.kneighbors(q, n_neighbors=k)

    # Similarité cosinus = 1 − distance cosinus
    scores = 1.0 - distances[0]
    return [(int(idx), float(score)) for idx, score in zip(indices[0], scores)]
