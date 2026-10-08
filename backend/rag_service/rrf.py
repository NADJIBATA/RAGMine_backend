"""
Reciprocal Rank Fusion (RRF)

Référence : Cormack, Clarke & Buettcher (SIGIR 2009)
            "Reciprocal Rank Fusion outperforms Condorcet and
             individual Rank Learning Methods"

Formule :
    RRF_score(doc) = Σ  1 / (k + rank_i)
                    i∈retrievers

    k = 60  (valeur standard, réduit l'impact des premiers rangs)

Principe :
    - Chaque retriever vote pour ses documents via leur rang
    - Un document bien classé dans PLUSIEURS retrievers
      obtient un score RRF élevé
    - Les scores absolus (cosinus, BM25) ne sont PAS comparés
      → seul le rang compte → pas de problème d'échelle
"""

from backend.shared.logger import setup_logger
from backend.shared.config import get_settings

logger = setup_logger("rrf")
settings  = get_settings()


# ─────────────────────────────────────────────────────────────────
#   CONSTANTE
# ─────────────────────────────────────────────────────────────────

RRF_K = settings.rrf_k          # valeur standard de la littérature


# ─────────────────────────────────────────────────────────────────
#   IDENTIFIER UN CHUNK DE FAÇON UNIQUE
# ─────────────────────────────────────────────────────────────────

def _get_chunk_id(chunk: dict) -> str:
    """
    Construit un identifiant unique et stable pour un chunk.

    Priorité :
      1. chunk_id  → ID Qdrant (présent dans BM25Retriever)
      2. Clé composite source + project_id + identifiant métier
         → fallback si chunk_id absent (résultats dense)

    Le même chunk doit produire le même ID
    qu'il vienne du retriever dense ou BM25.
    """

    # ── 1. ID Qdrant direct ───────────────────────────────────────
    qdrant_id = chunk.get("chunk_id", "")
    if qdrant_id:
        return qdrant_id

    # ── 2. Clé composite par type de source ──────────────────────
    source     = chunk.get("source", "")
    project_id = chunk.get("project_id", "")

    if source in ("wiki", "wiki_attachment"):
        title   = chunk.get("title") or chunk.get("wiki_title", "")
        section = chunk.get("section", "")
        idx     = chunk.get("chunk_index", "")
        return f"{source}::{project_id}::{title}::{section}::{idx}"

    if source in ("issue", "issue_attachment"):
        issue_id = chunk.get("issue_id", "")
        idx      = chunk.get("chunk_index", "")
        return f"{source}::{issue_id}::{idx}"

    if source == "project":
        return f"project::{project_id}"

    # ── 3. Fallback générique ─────────────────────────────────────
    url = chunk.get("url", "")
    idx = chunk.get("chunk_index", "")
    return f"{source}::{project_id}::{url}::{idx}"


# ─────────────────────────────────────────────────────────────────
#   RECIPROCAL RANK FUSION
# ─────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k:            int = RRF_K,
    session_id:   str = None,
) -> list[dict]:
    """
    Fusionne N listes de résultats classés via RRF.

    Args:
        ranked_lists : liste de listes de chunks
                       ex: [dense_results, bm25_results]
                       chaque liste est déjà triée par score décroissant
        k            : constante RRF (défaut=60)
        session_id   : pour les logs

    Returns:
        Liste fusionnée triée par score RRF décroissant.
        Chaque chunk contient :
          - rrf_score        : score RRF final
          - rrf_rank_dense   : rang dans la liste dense (si présent)
          - rrf_rank_bm25    : rang dans la liste BM25  (si présent)
          - rrf_sources      : retrievers qui ont retourné ce chunk
    """

    # ── Validation ────────────────────────────────────────────────
    if not ranked_lists:
        logger.warning(f"reciprocal_rank_fusion | listes vides | session={session_id}")
        return []

    ranked_lists = [lst for lst in ranked_lists if lst]     # supprimer listes vides
    if not ranked_lists:
        return []

    # ── Accumuler les scores RRF ──────────────────────────────────
    rrf_scores:   dict[str, float] = {}   # chunk_id → score RRF cumulé
    chunk_store:  dict[str, dict]  = {}   # chunk_id → chunk dict (référence)
    rank_details: dict[str, dict]  = {}   # chunk_id → {dense: rank, bm25: rank}

    # Noms des retrievers pour les logs et métadonnées
    retriever_names = [f"retriever_{i}" for i in range(len(ranked_lists))]

    # Cas standard : 2 listes → nommage explicite
    if len(ranked_lists) == 2:
        retriever_names = ["dense", "bm25"]
    elif len(ranked_lists) == 3:
        retriever_names = ["dense", "bm25", "extra"]

    for retriever_idx, ranked_list in enumerate(ranked_lists):
        retriever_name = retriever_names[retriever_idx]

        for rank, chunk in enumerate(ranked_list, start=1):
            chunk_id  = _get_chunk_id(chunk)
            rrf_score = 1.0 / (k + rank)

            # ── Accumuler le score ────────────────────────────────
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score

            # ── Stocker le chunk (première occurrence) ────────────
            if chunk_id not in chunk_store:
                chunk_store[chunk_id]  = chunk.copy()
                rank_details[chunk_id] = {}

            # ── Enregistrer le rang par retriever ─────────────────
            rank_details[chunk_id][retriever_name] = rank

    # ── Trier par score RRF décroissant ───────────────────────────
    sorted_ids = sorted(
        rrf_scores.keys(),
        key=     lambda cid: rrf_scores[cid],
        reverse= True,
    )

    # ── Construire la liste finale ────────────────────────────────
    fused = []
    for cid in sorted_ids:
        chunk = chunk_store[cid].copy()
        ranks = rank_details[cid]

        # ── Ajouter les métadonnées RRF ───────────────────────────
        chunk["rrf_score"]       = round(rrf_scores[cid], 6)
        chunk["rrf_rank_dense"]  = ranks.get("dense",  None)
        chunk["rrf_rank_bm25"]   = ranks.get("bm25",   None)
        chunk["rrf_sources"]     = list(ranks.keys())

        # ── Remplacer le score principal par le score RRF ─────────
        # (pour compatibilité avec le reranker qui lit "score")
        chunk["score_original"]  = chunk.get("score", 0.0)
        chunk["score"]           = chunk["rrf_score"]

        fused.append(chunk)

    # ── Logs ──────────────────────────────────────────────────────
    nb_dense    = len(ranked_lists[0]) if len(ranked_lists) > 0 else 0
    nb_bm25     = len(ranked_lists[1]) if len(ranked_lists) > 1 else 0
    nb_overlap  = sum(
        1 for cid in rrf_scores
        if len(rank_details[cid]) > 1
    )

    logger.info(
        f"reciprocal_rank_fusion | session={session_id} | "
        f"dense={nb_dense} + bm25={nb_bm25} → fused={len(fused)} | "
        f"overlap={nb_overlap} chunks en commun | "
        f"k={k}"
    )

    if fused:
        top = fused[0]
        logger.debug(
            f"reciprocal_rank_fusion | top1 | "
            f"rrf_score={top['rrf_score']} | "
            f"sources={top['rrf_sources']} | "
            f"rank_dense={top['rrf_rank_dense']} | "
            f"rank_bm25={top['rrf_rank_bm25']} | "
            f"chunk_id={_get_chunk_id(top)}"
        )

    return fused


# ─────────────────────────────────────────────────────────────────
#   ANALYSE DES RÉSULTATS RRF  (utile pour l'évaluation)
# ─────────────────────────────────────────────────────────────────

def analyze_fusion(fused: list[dict], top_k: int = 5) -> dict:
    """
    Analyse les résultats RRF pour comprendre
    la contribution de chaque retriever.

    Utile pour la section expérimentale de la mémoire.

    Returns:
        dict avec :
          - top_k_sources     : pour chaque chunk du top-k,
                                quels retrievers l'ont trouvé
          - only_dense        : chunks trouvés UNIQUEMENT par dense
          - only_bm25         : chunks trouvés UNIQUEMENT par BM25
          - both              : chunks trouvés par les DEUX
          - contribution      : % de contribution par retriever
    """
    if not fused:
        return {}

    top      = fused[:top_k]
    only_dense  = 0
    only_bm25   = 0
    both        = 0

    top_k_sources = []
    for chunk in top:
        sources = chunk.get("rrf_sources", [])
        top_k_sources.append({
            "chunk_id":      _get_chunk_id(chunk),
            "rrf_score":     chunk.get("rrf_score", 0),
            "sources":       sources,
            "rank_dense":    chunk.get("rrf_rank_dense"),
            "rank_bm25":     chunk.get("rrf_rank_bm25"),
        })

        if sources == ["dense"]:
            only_dense += 1
        elif sources == ["bm25"]:
            only_bm25  += 1
        elif len(sources) > 1:
            both       += 1

    total = len(top)
    return {
        "top_k_sources":  top_k_sources,
        "only_dense":     only_dense,
        "only_bm25":      only_bm25,
        "both":           both,
        "contribution": {
            "dense_only_pct": round(only_dense / total * 100, 1) if total else 0,
            "bm25_only_pct":  round(only_bm25  / total * 100, 1) if total else 0,
            "both_pct":       round(both        / total * 100, 1) if total else 0,
        },
    }