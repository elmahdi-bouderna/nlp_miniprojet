"""
prepare_dataset.py — Acquisition et nettoyage du corpus de livres Goodreads.

Lit le fichier compressé `goodreads_books_young_adult.json.gz`, filtre les livres
anglais ayant un résumé suffisamment long (≥ 30 mots), retire les doublons
et les balises HTML, puis exporte un CSV propre prêt pour le pipeline NLP.

Source : UCSD Book Graph / Goodreads Young Adult (Wan & McAuley).

Usage :
    python -m book_search_nlp.prepare_dataset

Auteur : BOUDERNA EL MAHDI
"""

import gzip
import json
import re
from pathlib import Path

import pandas as pd

# ── Chemins par défaut ───────────────────────────────────────────────────────

# Le dossier racine du package (src/book_search_nlp/)
_PKG_DIR = Path(__file__).resolve().parent

RAW_SOURCE = _PKG_DIR / "data" / "raw" / "goodreads_books_young_adult.json.gz"
OUTPUT_CSV = _PKG_DIR / "data" / "books.csv"

MIN_WORDS = 30
MAX_BOOKS = 10_000


# ── Fonctions ────────────────────────────────────────────────────────────────

def _strip_html(text: str) -> str:
    """Supprime les balises HTML et normalise les espaces."""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare(
    source: Path = RAW_SOURCE,
    output: Path = OUTPUT_CSV,
    max_books: int = MAX_BOOKS,
    min_words: int = MIN_WORDS,
) -> pd.DataFrame:
    """Lit le JSON compressé, filtre et exporte le corpus nettoyé.

    Parameters
    ----------
    source : Path
        Chemin du fichier .json.gz brut.
    output : Path
        Chemin du CSV de sortie.
    max_books : int
        Nombre maximal de livres à conserver.
    min_words : int
        Nombre minimal de mots dans la description.

    Returns
    -------
    pd.DataFrame avec colonnes : book_id, title, description, url
    """
    books = []
    seen = set()

    with gzip.open(source, "rt", encoding="utf-8") as f:
        for line in f:
            book = json.loads(line)

            book_id = book.get("book_id")
            title = book.get("title", "").strip()
            description = _strip_html(book.get("description", ""))

            if (
                book_id not in seen
                and title
                and len(description.split()) >= min_words
                and book.get("language_code") == "eng"
            ):
                books.append({
                    "book_id": book_id,
                    "title": title,
                    "description": description,
                    "url": book.get("url", ""),
                })
                seen.add(book_id)

            if len(books) >= max_books:
                break

    df = pd.DataFrame(books)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Corpus sauvegardé : {len(df)} livres → {output}")
    return df


# ── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = prepare()
    print(df.head())
