"""
BM25 Retriever — construit un index BM25 à partir des payloads Postgres.

Stratégie :
  - Au premier appel  → charge TOUS les chunks depuis Postgres (pagination keyset)
  - Met en cache l'index en mémoire
  - Rafraîchissement manuel ou automatique après N minutes
"""

import re
import time
import json
import asyncio
import unicodedata
import asyncpg
from rank_bm25 import BM25Okapi

from backend.shared.config import get_settings
from backend.shared.logger import setup_logger

settings = get_settings()
logger   = setup_logger("bm25_retriever")

COLLECTION_NAME   = settings.collection_name
CACHE_TTL_SECONDS = getattr(settings, "bm25_cache_ttl", 86400)
BATCH_SIZE        = settings.bm25_batch_size


class BM25Retriever:
    """
    Index BM25 construit sur les payloads Postgres.

    Attributs internes :
        _chunks      : liste brute des payloads (dict)
        _corpus      : liste de tokens (list[list[str]])
        _bm25        : instance BM25Okapi
        _built_at    : timestamp de la dernière construction
        _lock        : asyncio.Lock pour éviter les builds concurrents
    """

    def __init__(self):
        self.pool = None
        self._chunks:   list[dict]       = []
        self._corpus:   list[list[str]]  = []
        self._bm25:     BM25Okapi | None = None
        self._built_at: float            = 0.0
        self._lock                       = asyncio.Lock()

    # ─────────────────────────────────────────
    #   CONNEXION POOL
    # ─────────────────────────────────────────

    async def connect(self):
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                dsn=settings.database_url,
                min_size=2,
                max_size=20,   
            )


    async def close(self):
        if self.pool:
            await self.pool.close()
            self.pool = None

    # ─────────────────────────────────────────
    #   TOKENISATION
    # ─────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Tokenisation simple et robuste :
          1. Normalisation Unicode (accents)
          2. Minuscules
          3. Séparation sur les non-alphanumériques
          4. Suppression des tokens vides et <= 1 char
        """
        if not text:
            return []

        # Normalisation Unicode → supprime les accents
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")

        text   = text.lower()
        tokens = re.split(r"[^a-z0-9]+", text)
        tokens = [t for t in tokens if len(t) > 1]

        return tokens

    # ─────────────────────────────────────────
    #   CONSTRUIRE LE TEXTE D'UN CHUNK
    # ─────────────────────────────────────────

    @staticmethod
    def _build_chunk_text(payload: dict) -> str:
        """
        Concatène les champs textuels importants du payload
        pour maximiser la couverture BM25.

        Champs pondérés (répétition = boost) :
          - title / subject / name  → répétés 3x  (forte valeur informative)
          - project_name            → répété 2x
          - text                    → 1x (contenu principal)
          - champs secondaires      → 1x
        """
        parts = []

        # ── Titre / Sujet (boost x3) ──────────────────────────────
        for field in ("title", "subject", "name", "wiki_title"):
            value = (payload.get(field) or "").strip()
            if value:
                parts.extend([value] * 3)
                break

        # ── Projet (boost x2) ─────────────────────────────────────
        project_name = (payload.get("project_name") or "").strip()
        if project_name:
            parts.extend([project_name] * 2)

        # ── Contenu principal ─────────────────────────────────────
        text = (payload.get("text") or "").strip()
        if text:
            parts.append(text)

        # ── Champs secondaires ────────────────────────────────────
        for field in (
            "section", "component", "status",
            "priority", "tracker", "assigned_to_name",
            "author_name", "filename",
        ):
            value = (payload.get(field) or "").strip()
            if value:
                parts.append(value)

        return " ".join(parts)

    # ─────────────────────────────────────────
    #   CHARGER LES CHUNKS DEPUIS POSTGRES
    # ─────────────────────────────────────────

    async def _load_chunks_from_db(
        self,
        user_projects: list[str] | None = None,
    ) -> list[dict]:
        """
        Charge tous les chunks depuis Postgres.
        Si user_projects est fourni → filtre sur project_id.
        """
        where_sql = ""
        params    = []

        if user_projects:
            where_sql = "WHERE project_id = ANY($1)"
            params.append(user_projects)

        chunks = []
        total  = 0

        logger.info("_load_chunks_from_db | début du chargement Postgres")

        # Pagination par keyset (id) pour éviter les OFFSET coûteux
        last_id = None
        while True:
            if last_id is None:
                query = f"""
                    SELECT id, text, payload
                    FROM {COLLECTION_NAME}
                    {where_sql}
                    ORDER BY id
                    LIMIT {BATCH_SIZE};
                """
                batch_params = params
            else:
                cursor_clause = (
                    f"AND id > ${len(params) + 1}" if where_sql
                    else f"WHERE id > ${len(params) + 1}"
                )
                query = f"""
                    SELECT id, text, payload
                    FROM {COLLECTION_NAME}
                    {where_sql} {cursor_clause}
                    ORDER BY id
                    LIMIT {BATCH_SIZE};
                """
                batch_params = params + [last_id]

            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *batch_params)

            if not rows:
                break

            for row in rows:
                payload = row["payload"] or {}
                if isinstance(payload, str):
                    payload = json.loads(payload)
                payload["text"] = row["text"] or payload.get("text", "")
                payload["chunk_id"] = str(row["id"])  # nom conservé pour compat RRF
                chunks.append(payload)

            last_id = rows[-1]["id"]
            total  += len(rows)

            logger.debug(
                f"_load_chunks_from_db | batch={len(rows)} | total={total}"
            )

            if len(rows) < BATCH_SIZE:
                break

        logger.info(f"_load_chunks_from_db | {total} chunks chargés")
        return chunks

    # ─────────────────────────────────────────
    #   CONSTRUIRE L'INDEX BM25
    # ─────────────────────────────────────────

    async def build_index(
        self,
        user_projects: list[str] | None = None,
        force:         bool             = False,
    ) -> None:
        """
        Construit (ou reconstruit) l'index BM25.

        - Si l'index existe et n'est pas expiré → skip (sauf force=True)
        - Thread-safe grâce au Lock asyncio
        """
        async with self._lock:

            # ── Vérifier si rebuild nécessaire ────────────────────
            age = time.time() - self._built_at
            if self._bm25 is not None and not force and age < CACHE_TTL_SECONDS:
                logger.debug(
                    f"build_index | index valide | "
                    f"age={age:.0f}s / ttl={CACHE_TTL_SECONDS}s"
                )
                return

            logger.info(
                f"build_index | construction index BM25 | "
                f"force={force} | age={age:.0f}s"
            )

            # ── Charger les chunks ─────────────────────────────────
            chunks = await self._load_chunks_from_db(None)

            if not chunks:
                logger.warning("build_index | aucun chunk disponible")
                return

            # ── Tokeniser ─────────────────────────────────────────
            corpus = []
            for chunk in chunks:
                chunk_text = self._build_chunk_text(chunk)
                tokens     = self._tokenize(chunk_text)
                corpus.append(tokens)

            # ── Construire BM25Okapi ───────────────────────────────
            self._chunks   = chunks
            self._corpus   = corpus
            self._bm25     = BM25Okapi(corpus)
            self._built_at = time.time()

            logger.info(
                f"build_index | index BM25 prêt | "
                f"{len(chunks)} chunks | "
                f"vocab={len(self._bm25.idf)} termes"
            )

    # ─────────────────────────────────────────
    #   RECHERCHE BM25
    # ─────────────────────────────────────────

    async def search(
        self,
        query:         str,
        user_projects: list[str] | None = None,
        top_k:         int              = 20,
        min_score:     float            = 0.0,
        session_id:    str              = None,
    ) -> list[dict]:

        # ── S'assurer que l'index est construit (index global) ────────
        await self.build_index()  # plus de user_projects ici

        if self._bm25 is None:
            logger.warning(f"search | index BM25 vide | session={session_id}")
            return []

        # ── Tokeniser la requête ──────────────────────────────────────
        query_tokens = self._tokenize(query)

        if not query_tokens:
            logger.warning(
                f"search | requête vide après tokenisation | session={session_id}"
            )
            return []

        # ── Filtrer les indices autorisés AVANT le scoring ────────────
        if user_projects is not None:
            user_projects_set = set(user_projects)
            allowed_indices = [
                i for i, chunk in enumerate(self._chunks)
                if chunk.get("project_id") in user_projects_set
            ]
        else:
            allowed_indices = list(range(len(self._chunks)))

        if not allowed_indices:
            logger.info(
                f"search | aucun chunk autorisé pour ces projets | "
                f"session={session_id} | projects={user_projects}"
            )
            return []

        # ── Scorer uniquement les chunks autorisés ────────────────────
        scores = self._bm25.get_scores(query_tokens)

        scored_indices = [
            (i, float(scores[i]))
            for i in allowed_indices
            if float(scores[i]) > min_score
        ]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        top_indices = scored_indices[:top_k]

        if not top_indices:
            logger.info(
                f"search | aucun résultat BM25 | "
                f"query='{query[:60]}' | session={session_id}"
            )
            return []

        # ── Normaliser les scores ─────────────────────────────────────
        max_score = top_indices[0][1] if top_indices else 1.0
        if max_score == 0:
            max_score = 1.0

        # ── Construire les résultats ──────────────────────────────────
        results = []
        for idx, raw_score in top_indices:
            payload    = self._chunks[idx]
            norm_score = raw_score / max_score
            results.append({
                # ── Scores ────────────────────────────────────────
                "score":        round(norm_score, 4),
                "bm25_score":   round(raw_score,  4),

                # ── Contenu ───────────────────────────────────────
                "text":         payload.get("text",          ""),

                # ── Identification ────────────────────────────────
                "chunk_id":   payload.get("chunk_id",    ""),
                "source":       payload.get("source",        ""),
                "project_id":   payload.get("project_id",    ""),
                "project_name": payload.get("project_name",  payload.get("name", "")),
                "url":          payload.get("url",           ""),

                # ── Wiki ──────────────────────────────────────────
                "title":        payload.get("title",         ""),
                "section":      payload.get("section",       ""),
                "h1":           payload.get("h1",            ""),
                "h2":           payload.get("h2",            ""),
                "h3":           payload.get("h3",            ""),
                "wiki_title":   payload.get("wiki_title",    ""),

                # ── Issue ─────────────────────────────────────────
                "issue_id":     payload.get("issue_id",      ""),
                "subject":      payload.get("subject",       ""),
                "component":    payload.get("component",     ""),
                "status":       payload.get("status",        ""),
                "priority":     payload.get("priority",      ""),
                "tracker":      payload.get("tracker",       ""),
                "assigned_to_name": payload.get("assigned_to_name", ""),
                "assigned_to_id":   payload.get("assigned_to_id",   ""),
                "done_ratio":   payload.get("done_ratio",    0),

                # ── Attachment ────────────────────────────────────
                "filename":     payload.get("filename",      ""),
                "content_type": payload.get("content_type",  ""),
                "has_content":  payload.get("has_content",   False),

                # ── Auteur et dates ───────────────────────────────
                "author_name":  payload.get("author_name",   ""),
                "author_id":    payload.get("author_id",     ""),
                "updated_on":   payload.get("updated_on",    ""),
                "created_on":   payload.get("created_on",    ""),

                # ── Stats chunk ───────────────────────────────────
                "word_count":   payload.get("word_count",    0),
                "chunk_index":  payload.get("chunk_index",   0),
                "total_chunks": payload.get("total_chunks",  1),
            })

        logger.info(
            f"search | BM25 | session={session_id} | "
            f"query='{query[:60]}' | "
            f"{len(results)} résultats | "
            f"allowed_chunks={len(allowed_indices)} | "
            f"best_score={results[0]['bm25_score']:.3f}"
        )

        return results

    # ─────────────────────────────────────────
    #   FORCER LE REFRESH DE L'INDEX
    # ─────────────────────────────────────────

    async def refresh(
        self,
        user_projects: list[str] | None = None,
    ) -> None:
        """Force la reconstruction complète de l'index BM25."""
        logger.info("refresh | reconstruction forcée de l'index BM25")
        await self.build_index(user_projects=user_projects, force=True)

    # ─────────────────────────────────────────
    #   STATS DE L'INDEX
    # ─────────────────────────────────────────

    def stats(self) -> dict:
        """Retourne les statistiques de l'index courant."""
        return {
            "built":       self._bm25 is not None,
            "nb_chunks":   len(self._chunks),
            "vocab_size":  len(self._bm25.idf) if self._bm25 else 0,
            "built_at":    self._built_at,
            "age_seconds": round(time.time() - self._built_at, 1),
            "ttl_seconds": CACHE_TTL_SECONDS,
        }