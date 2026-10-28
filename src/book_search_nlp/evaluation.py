"""
evaluation.py — Métriques de recherche d'information (Precision@K, Recall@K, MAP@K).

Ce module évalue la qualité des résultats de recherche en comparant les livres
retournés à un jeu de vérité terrain (ground truth) annoté manuellement.

Auteur : BOUDERNA EL MAHDI
"""

from pathlib import Path

import pandas as pd


# ── Métriques unitaires ──────────────────────────────────────────────────────

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Proportion de résultats pertinents parmi les K premiers."""
    hits = [1 if rid in relevant_ids else 0 for rid in retrieved_ids[:k]]
    return sum(hits) / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Proportion de documents pertinents retrouvés parmi les K premiers."""
    hits = [1 if rid in relevant_ids else 0 for rid in retrieved_ids[:k]]
    return sum(hits) / len(relevant_ids) if relevant_ids else 0.0


def average_precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Average Precision (AP) sur les K premiers résultats.

    Pour chaque rang où un document pertinent apparaît, on calcule la
    précision cumulée à ce rang, puis on moyenne.
    """
    hits = [1 if rid in relevant_ids else 0 for rid in retrieved_ids[:k]]
    correct = 0
    ap = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            correct += 1
            ap += correct / rank
    denominator = min(len(relevant_ids), k)
    return ap / denominator if denominator > 0 else 0.0


# ── Évaluation complète ─────────────────────────────────────────────────────

def evaluate(
    search_fn,
    books: "pd.DataFrame",
    pool_path: Path,
    k: int = 10,
    method_name: str = "method",
) -> pd.DataFrame:
    """Évalue une fonction de recherche sur toutes les requêtes du pool d'annotations.

    Parameters
    ----------
    search_fn : callable(query, k) -> list[tuple[int, float]]
        Fonction de recherche retournant des paires (index, score).
    books : pd.DataFrame
        DataFrame du corpus (doit contenir la colonne `book_id`).
    pool_path : Path
        Chemin vers le fichier CSV d'annotations (colonnes :
        query_id, query, book_id, relevance).
    k : int
        Nombre de résultats à évaluer.
    method_name : str
        Nom de la méthode (pour le rapport).

    Returns
    -------
    pd.DataFrame avec colonnes : query_id, methode, Precision@K, Recall@K, MAP@K
    """
    pool = pd.read_csv(pool_path, dtype=str).fillna("")
    rows = []

    for qid, group in pool.groupby("query_id"):
        # Ignorer les requêtes non annotées
        if not group["relevance"].isin(["0", "1"]).all():
            continue

        relevant = set(group.loc[group["relevance"] == "1", "book_id"])
        if not relevant:
            continue

        query = group["query"].iloc[0]
        results = search_fn(query, k)
        retrieved = [str(books.iloc[idx]["book_id"]) for idx, _ in results][:k]

        rows.append({
            "query_id": qid,
            "methode": method_name,
            f"Precision@{k}": precision_at_k(retrieved, relevant, k),
            f"Recall@{k}": recall_at_k(retrieved, relevant, k),
            f"MAP@{k}": average_precision_at_k(retrieved, relevant, k),
        })

    return pd.DataFrame(rows)


def compare_methods(
    search_fns: dict,
    books: "pd.DataFrame",
    pool_path: Path,
    k: int = 10,
    output_dir: Path | None = None,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Compare plusieurs méthodes de recherche et produit un rapport.

    Parameters
    ----------
    search_fns : dict[str, callable]
        Dictionnaire {nom_methode: search_fn}.
    books : pd.DataFrame
        DataFrame du corpus.
    pool_path : Path
        Chemin vers les annotations.
    k : int
        Nombre de résultats.
    output_dir : Path | None
        Si fourni, sauvegarde les résultats en CSV.

    Returns
    -------
    tuple : (summary_df, detail_df)
        summary_df : moyennes par méthode.
        detail_df  : détail par requête et méthode.
    """
    all_metrics = []
    for name, fn in search_fns.items():
        metrics = evaluate(fn, books, pool_path, k, name)
        all_metrics.append(metrics)

    if not all_metrics:
        return pd.DataFrame(), pd.DataFrame()

    detail = pd.concat(all_metrics, ignore_index=True)
    score_cols = [c for c in detail.columns if c.startswith(("Precision", "Recall", "MAP"))]
    summary = detail.groupby("methode", as_index=False)[score_cols].mean()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        detail.to_csv(output_dir / "evaluation_per_query.csv", index=False)
        summary.to_csv(output_dir / "evaluation_summary.csv", index=False)

    return summary, detail
