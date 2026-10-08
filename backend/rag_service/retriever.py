import asyncio
import time
import json
import asyncpg
from pgvector.asyncpg import register_vector

from backend.shared.qwen3_client    import Qwen3Client
from backend.shared.reranker_client import RerankerClient
from backend.rag_service.bm25_retriever  import BM25Retriever
from backend.rag_service.rrf             import reciprocal_rank_fusion, analyze_fusion
from backend.shared.config          import get_settings
from backend.shared.logger          import setup_logger

settings = get_settings()
logger   = setup_logger("retriever")

COLLECTION_NAME = settings.collection_name
VECTOR_SIZE     = settings.vector_size
TOP_K           = settings.rag_top_k
MIN_SCORE       = settings.rag_min_score

TOP_K_FETCH     = settings.top_k_fetch
TOP_K_FINAL     = settings.top_k_final


USE_HALFVEC = VECTOR_SIZE > 2000

if USE_HALFVEC:
    DISTANCE_EXPR = (
        f"(embedding::halfvec({VECTOR_SIZE}) <=> $1::halfvec({VECTOR_SIZE}))"
    )
    logger.info(
        f"Retriever | VECTOR_SIZE={VECTOR_SIZE} > 2000 → "
        f"utilisation du cast halfvec pour bénéficier de l'index HNSW"
    )
else:
    DISTANCE_EXPR = "(embedding <=> $1)"

# ── Modes hybride ──────────────────────────────────────────────────────────────
MODE_DENSE  = "dense"
MODE_BM25   = "bm25"
MODE_HYBRID = "hybrid"

# ── Templates de formatage par source ─────────────────────────────────────────
SOURCE_TEMPLATES = {

    "wiki": lambda r: (
        f"[Wiki - score: {r['score']}]\n"
        f"Titre   : {r.get('title',      '')}\n"
        f"Section : {r.get('section',    '')}\n"
        f"Projet_id  : {r.get('project_id', '')}\n"
        f"Projet_name  : {r.get('project_name', '')}\n"
        f"Auteur  : {r.get('author_name', '')} ({r.get('author_id', '')})\n"
        f"Modifié : {r.get('updated_on', '')}\n"
        f"URL     : {r.get('url',        '')}"
    ),

    "wiki_attachment": lambda r: (
        f"[Wiki Attachment - score: {r['score']}]\n"
        f"Fichier : {r.get('filename',    '')}\n"
        f"Type    : {r.get('content_type','')}\n"
        f"Wiki    : {r.get('wiki_title',  '')}\n"
        f"Projet_id  : {r.get('project_id',  '')}\n"
        f"Projet_name  : {r.get('project_name', '')}\n"
        f"URL     : {r.get('url',         '')}"
    ),

    "issue": lambda r: (
        f"[Ticket #{r.get('issue_id','')} - score: {r['score']}]\n"
        f"Sujet     : {r.get('subject',     '')}\n"
        f"Composant : {r.get('component',   '')}\n"
        f"Projet_id    : {r.get('project_id',  '')}\n"
        f"Projet_name  : {r.get('project_name', '')}\n"
        f"Statut    : {r.get('status',      '')}\n"
        f"Priorité  : {r.get('priority',    '')}\n"
        f"Assigné   : {r.get('assigned_to_name', '')} ({r.get('assigned_to_id', '')})\n"
        f"URL       : {r.get('url',         '')}"
    ),

    "issue_attachment": lambda r: (
        f"[Ticket Attachment - score: {r['score']}]\n"
        f"Fichier : {r.get('filename',    '')}\n"
        f"Type    : {r.get('content_type','')}\n"
        f"Ticket  : #{r.get('issue_id',  '')}\n"
        f"Projet_id  : {r.get('project_id', '')}\n"
        f"Projet_name  : {r.get('project_name', '')}\n"
        f"URL     : {r.get('url',        '')}"
    ),

    "project": lambda r: (
        f"[Projet - score: {r['score']}]\n"
        f"Projet  : {r.get('project_id', '')}\n"
        f"Nom     : {r.get('name',       '')}\n"
        f"URL     : {r.get('url',        '')}"
    ),
}

ALL_SOURCES = list(SOURCE_TEMPLATES.keys())


class Retriever:

    def __init__(self):
        self.pool      = None
        self.llm       = Qwen3Client()
        self.reranker  = RerankerClient()
        self.bm25      = BM25Retriever()

    # ─────────────────────────────────────────
    #   CONNEXION POOL (avec codec pgvector)
    # ─────────────────────────────────────────

    async def connect(self):
        if self.pool is None:

            # ── 1. S'assurer que l'extension existe (idempotent) ──────
            try:
                tmp_conn = await asyncpg.connect(dsn=settings.database_url)
            except Exception as e:
                logger.error(f"connect | connexion initiale échouée | {e}")
                raise

            try:
                await tmp_conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            except Exception as e:
                logger.error(f"connect | création extension vector échouée | {e}")
                raise
            finally:
                try:
                    await tmp_conn.close()
                except Exception as e:
                    logger.error(f"connect | fermeture connexion temporaire échouée | {e}")

            # ── 2. Créer le pool avec le codec vector ──────────────────
            async def init_conn(conn):
                try:
                    await register_vector(conn)
                except Exception as e:
                    logger.error(f"connect | register_vector échoué | {e}")
                    raise

            try:
                self.pool = await asyncpg.create_pool(
                    dsn=settings.database_url,
                    min_size=2,
                    max_size=20,
                    init=init_conn,
                )
            except Exception as e:
                logger.error(f"connect | création du pool échouée | {e}")
                raise

        # ── 3. Connecter aussi le pool du BM25Retriever
        try:
            await self.bm25.connect()
        except Exception as e:
            logger.error(f"connect | bm25.connect échoué | {e}")
            raise


    async def close(self):
        if self.pool:
            try:
                await self.pool.close()
            except Exception as e:
                logger.error(f"close | fermeture du pool échouée | {e}")
            finally:
                self.pool = None

        # ── Fermer aussi le pool du BM25Retriever
        try:
            await self.bm25.close()
        except Exception as e:
            logger.error(f"close | bm25.close échoué | {e}")


    # ─────────────────────────────────────────
    #   PERMISSIONS UTILISATEUR
    # ─────────────────────────────────────────

    async def get_user_projects(self, api_key: str) -> list[str]:
        from backend.shared.redmine_client import RedmineClient

        try:
            client   = RedmineClient(api_key=api_key)
            projects = await client.get_paginated("/projects.json")
        except Exception as e:
            logger.error(f"get_user_projects | erreur Redmine | {e}")
            return []

        project_ids = []
        try:
            for project in projects:
                identifier = project.get("identifier")
                pid        = project.get("id")

                if identifier:
                    project_ids.append(str(identifier))
                if pid:
                    project_ids.append(str(pid))
        except Exception as e:
            logger.error(f"get_user_projects | parsing erreur | {e}")
            return []

        logger.info(
            f"get_user_projects | {len(projects)} projets | "
            f"{len(project_ids)} identifiants"
        )
        return project_ids

    # ─────────────────────────────────────────
    #   RERANKING
    # ─────────────────────────────────────────

    async def _rerank(
        self,
        query:      str,
        chunks:     list[dict],
        top_k:      int = TOP_K_FINAL,
        session_id: str = None,
    ) -> list[dict]:

        logger.info(
            f"_rerank | session={session_id} | "
            f"{len(chunks)} chunks → top {top_k}"
        )

        try:
            ranked = await self.reranker.rerank_chunks(
                query=  query,
                chunks= chunks,
                top_k=  top_k,
            )
        except Exception as e:
            logger.error(f"_rerank | erreur reranker | session={session_id} | {e}")
            return chunks[:top_k]

        if ranked:
            logger.info(
                f"_rerank | session={session_id} | "
                f"best={ranked[0].get('rerank_score', 0)} | "
                f"worst={ranked[-1].get('rerank_score', 0)}"
            )

        return ranked

    # ─────────────────────────────────────────
    #   CONSTRUIRE LE FILTRE SQL
    # ─────────────────────────────────────────
    # Dans le systeme actuel on fais pas du filtrage suite a la discussion avec Lucas et Leo
    def _build_filter(
        self,
        user_projects: list[str],
        source:        str  = None,
        project_id:    str  = None,
        component:     str  = None,
        status:        str  = None,
        assigned_to:   str  = None,
        tracker:       str  = None,
        priority:      str  = None,
        session_id:    str  = None,
    ) -> tuple[str, list] | None:
        """
        Retourne (where_clause_sql, params) ou None si accès refusé.
        where_clause_sql commence par "AND ..." ou est vide.
        """
        try:
            clauses = []
            params  = []
            idx     = 1  # $1 réservé au vecteur de recherche dans la requête finale

            def add(field, value):
                nonlocal idx
                idx += 1
                clauses.append(f"{field} = ${idx}")
                params.append(value)

            if project_id:
                if project_id in user_projects:
                    add("project_id", str(project_id))
                else:
                    logger.warning(
                        f"_build_filter | project_id '{project_id}' "
                        f"non accessible | session={session_id}"
                    )
                    return None
            else:
                if user_projects:
                    idx += 1
                    clauses.append(f"project_id = ANY(${idx})")
                    params.append(user_projects)

            if source:
                add("source", source)
            if component:
                add("component", component)
            if status:
                add("status", status)
            if assigned_to:
                add("assigned_to", assigned_to)
            if tracker:
                add("tracker", tracker)
            if priority:
                add("priority", priority)

            where_sql = ("AND " + " AND ".join(clauses)) if clauses else ""
            return where_sql, params
        except Exception as e:
            logger.error(f"_build_filter | erreur | session={session_id} | {e}")
            return None

    # ─────────────────────────────────────────
    #   FORMATER UN RÉSULTAT
    # ─────────────────────────────────────────

    def _format_result_header(self, result: dict, index: int) -> str:
        source   = result.get("source", "")
        template = SOURCE_TEMPLATES.get(source)

        try:
            if template:
                return f"[Source {index}] " + template(result)

            return (
                f"[Source {index} - {source} - score: {result.get('score', 0)}]\n"
                f"Projet : {result.get('project_id', '')}\n"
                f"URL    : {result.get('url', '')}"
            )
        except Exception as e:
            logger.error(f"_format_result_header | erreur | index={index} | {e}")
            return f"[Source {index} - {source}]"

    # ─────────────────────────────────────────
    #   FORMATER LES RÉSULTATS POSTGRES
    # ─────────────────────────────────────────

    def _format_search_results(self, rows) -> list[dict]:
        """
        Formate les lignes Postgres en dicts standardisés.
        Chaque row a: id, text, score (distance/similarité), payload (JSONB), + colonnes dédiées.
        """
        formatted = []
        for r in rows:
            try:
                payload = r["payload"] or {}
                if isinstance(payload, str):
                    payload = json.loads(payload)

                formatted.append({
                    "score":        round(r["score"], 4),
                    "text":         r["text"] or payload.get("text", ""),

                    "chunk_id":   str(r["id"]),   # on garde le nom pour compat RRF
                    "source":       payload.get("source",        ""),
                    "project_id":   payload.get("project_id",    ""),
                    "project_name": payload.get("project_name",  payload.get("name")),
                    "url":          payload.get("url",           ""),

                    "title":        payload.get("title",         ""),
                    "section":      payload.get("section",       ""),
                    "h1":           payload.get("h1",            ""),
                    "h2":           payload.get("h2",            ""),
                    "h3":           payload.get("h3",            ""),
                    "wiki_title":   payload.get("wiki_title",    ""),

                    "issue_id":     payload.get("issue_id",      ""),
                    "subject":      payload.get("subject",       ""),
                    "component":    payload.get("component",     ""),
                    "comment_by":   payload.get("comment_by",    ""),
                    "comment_date": payload.get("comment_date",  ""),
                    "status":       payload.get("status",        ""),
                    "priority":     payload.get("priority",      ""),
                    "tracker":      payload.get("tracker",       ""),
                    "assigned_to_name": payload.get("assigned_to_name", ""),
                    "assigned_to_id":   payload.get("assigned_to_id",   ""),
                    "done_ratio":   payload.get("done_ratio",    0),

                    "filename":     payload.get("filename",      ""),
                    "content_type": payload.get("content_type",  ""),
                    "has_content":  payload.get("has_content",   False),

                    "author_name":  payload.get("author_name",   ""),
                    "author_id":    payload.get("author_id",     ""),
                    "updated_on":   payload.get("updated_on",    ""),
                    "created_on":   payload.get("created_on",    ""),

                    "word_count":   payload.get("word_count",    0),
                    "chunk_index":  payload.get("chunk_index",   0),
                    "total_chunks": payload.get("total_chunks",  1),
                })
            except Exception as e:
                logger.error(f"_format_search_results | ligne ignorée | {e}")
                continue
        return formatted

    # ─────────────────────────────────────────
    #   RECHERCHE DENSE (original)
    # ─────────────────────────────────────────

    async def search(
        self,
        query:        str,
        api_key:      str,
        project_id:   str   = None,
        source:       str   = None,
        component:    str   = None,
        status:       str   = None,
        assigned_to:  str   = None,
        tracker:      str   = None,
        priority:     str   = None,
        top_k:        int   = TOP_K_FINAL,
        min_score:    float = MIN_SCORE,
        session_id:   str   = None,
        rerank:       bool  = True,
    ) -> list[dict]:

        try:
            user_projects = await self.get_user_projects(api_key)
        except Exception as e:
            logger.error(f"search | get_user_projects échoué | session={session_id} | {e}")
            return []

        if not user_projects:
            logger.warning(f"search | aucun projet accessible | session={session_id}")
            return []

        filter_result = self._build_filter(
            user_projects=user_projects,
            source=       source,
            project_id=   project_id,
            component=    component,
            status=       status,
            assigned_to=  assigned_to,
            tracker=      tracker,
            priority=     priority,
            session_id=   session_id,
        )

        if filter_result is None:
            return []

        where_sql, filter_params = filter_result

        try:
            query_vector = await self.llm.embed(query)
        except Exception as e:
            logger.error(f"search | embed erreur | {str(e)} | session={session_id}")
            return []

        fetch_limit = TOP_K_FETCH if rerank else top_k

        sql = f"""
            SELECT id, text, payload,
                   1 - {DISTANCE_EXPR} AS score
            FROM {COLLECTION_NAME}
            WHERE 1=1 {where_sql}
            AND 1 - {DISTANCE_EXPR} >= {min_score}
            ORDER BY {DISTANCE_EXPR}
            LIMIT {fetch_limit};
        """

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(sql, query_vector, *filter_params)
        except Exception as e:
            logger.error(f"search | postgres erreur | {str(e)} | session={session_id}")
            return []

        formatted = self._format_search_results(rows)

        if not formatted:
            logger.info(f"search | aucun résultat | session={session_id} | query='{query[:60]}'")
            return []

        if rerank and len(formatted) > 1:
            try:
                formatted = await self._rerank(query=query, chunks=formatted, top_k=top_k, session_id=session_id)
            except Exception as e:
                logger.error(f"search | rerank échoué | session={session_id} | {e}")
                formatted = formatted[:top_k]
        else:
            formatted = formatted[:top_k]

        logger.info(f"search | dense | session={session_id} | {len(formatted)} résultats")

        return formatted

    # ─────────────────────────────────────────
    #   RECHERCHE DENSE INTERNE (sans rerank)
    # ─────────────────────────────────────────

    async def _search_dense_raw(
        self,
        query:         str,
        user_projects: list[str],
        project_id:    str   = None,
        source:        str   = None,
        top_k:         int   = TOP_K_FETCH,
        min_score:     float = MIN_SCORE,
        session_id:    str   = None,
    ) -> list[dict]:

        filter_result = self._build_filter(
            user_projects=user_projects,
            source=       source,
            project_id=   project_id,
            session_id=   session_id,
        )
        if filter_result is None:
            return []

        where_sql, filter_params = filter_result

        try:
            query_vector = await self.llm.embed(query)
        except Exception as e:
            logger.error(f"_search_dense_raw | embed erreur | {str(e)} | session={session_id}")
            return []

        sql = f"""
            SELECT id, text, payload,
                   1 - {DISTANCE_EXPR} AS score
            FROM {COLLECTION_NAME}
            WHERE 1=1 {where_sql}
            AND 1 - {DISTANCE_EXPR} >= {min_score}
            ORDER BY {DISTANCE_EXPR}
            LIMIT {top_k};
        """

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(sql, query_vector, *filter_params)
        except Exception as e:
            logger.error(f"_search_dense_raw | postgres erreur | {str(e)} | session={session_id}")
            return []

        formatted = self._format_search_results(rows)

        logger.info(f"_search_dense_raw | session={session_id} | {len(formatted)} chunks")
        return formatted

    # ─────────────────────────────────────────
    #   RECHERCHE HYBRIDE
    # ─────────────────────────────────────────

    async def search_hybrid(
        self,
        query:        str,
        api_key:      str,
        project_id:   str   = None,
        source:       str   = None,
        top_k:        int   = TOP_K_FINAL,
        min_score:    float = MIN_SCORE,
        session_id:   str   = None,
        mode:         str   = settings.rag_mode,
        rerank:       bool  = True,
    ) -> tuple[list[dict], dict]:
        """
        Recherche hybride configurable.

        Args:
            mode  : "dense" | "bm25" | "hybrid"
            rerank: True → applique le reranker neuronal

        Returns:
            (results, meta)
            results : liste de chunks triés
            meta    : infos de debug (nb résultats par retriever,
                      analyse RRF, latences)
        """
        meta = {
            "mode":         mode,
            "rerank":       rerank,
            "nb_dense":     0,
            "nb_bm25":      0,
            "nb_fused":     0,
            "nb_final":     0,
            "rrf_analysis": {},
        }

        # ── 1. Permissions (une seule fois) ───────────────────────
        try:
            user_projects = await self.get_user_projects(api_key)
        except Exception as e:
            logger.error(f"search_hybrid | get_user_projects échoué | session={session_id} | {e}")
            return [], meta

        if not user_projects:
            logger.warning(
                f"search_hybrid | aucun projet accessible | session={session_id}"
            )
            return [], meta

        # ── 2. Filtrer BM25 sur les projets accessibles ───────────
        bm25_projects = user_projects
        if project_id:
            if project_id in user_projects:
                bm25_projects = [project_id]
            else:
                logger.warning(
                    f"search_hybrid | project_id '{project_id}' "
                    f"non accessible | session={session_id}"
                )
                return [], meta

        # ══════════════════════════════════════════════════════════
        #   MODE DENSE SEUL
        # ══════════════════════════════════════════════════════════
        if mode == MODE_DENSE:
            t0 = time.time()
            try:
                dense_results = await self._search_dense_raw(
                    query=         query,
                    user_projects= user_projects,
                    project_id=    project_id,
                    source=        source,
                    top_k=         TOP_K_FETCH if rerank else top_k,
                    min_score=     min_score,
                    session_id=    session_id,
                )
            except Exception as e:
                logger.error(f"search_hybrid | dense échoué | session={session_id} | {e}")
                dense_results = []

            meta["nb_dense"]   = len(dense_results)
            meta["t_dense_ms"] = round((time.time() - t0) * 1000, 1)

            if rerank and len(dense_results) > 1:
                t1 = time.time()
                try:
                    dense_results = await self._rerank(
                        query=      query,
                        chunks=     dense_results,
                        top_k=      top_k,
                        session_id= session_id,
                    )
                except Exception as e:
                    logger.error(f"search_hybrid | rerank dense échoué | session={session_id} | {e}")
                    dense_results = dense_results[:top_k]
                meta["t_rerank_ms"] = round((time.time() - t1) * 1000, 1)
            else:
                dense_results = dense_results[:top_k]

            meta["nb_final"] = len(dense_results)
            return dense_results, meta

        # ══════════════════════════════════════════════════════════
        #   MODE BM25 SEUL
        # ══════════════════════════════════════════════════════════
        elif mode == MODE_BM25:
            t0 = time.time()
            try:
                bm25_results = await self.bm25.search(
                    query=         query,
                    user_projects= bm25_projects,
                    top_k=         TOP_K_FETCH if rerank else top_k,
                    min_score=     0.0,
                    session_id=    session_id,
                )
            except Exception as e:
                logger.error(f"search_hybrid | bm25 échoué | session={session_id} | {e}")
                bm25_results = []

            meta["nb_bm25"]   = len(bm25_results)
            meta["t_bm25_ms"] = round((time.time() - t0) * 1000, 1)

            if rerank and len(bm25_results) > 1:
                t1 = time.time()
                try:
                    bm25_results = await self._rerank(
                        query=      query,
                        chunks=     bm25_results,
                        top_k=      top_k,
                        session_id= session_id,
                    )
                except Exception as e:
                    logger.error(f"search_hybrid | rerank bm25 échoué | session={session_id} | {e}")
                    bm25_results = bm25_results[:top_k]
                meta["t_rerank_ms"] = round((time.time() - t1) * 1000, 1)
            else:
                bm25_results = bm25_results[:top_k]

            meta["nb_final"] = len(bm25_results)
            return bm25_results, meta

        # ══════════════════════════════════════════════════════════
        #   MODE HYBRIDE  (dense + BM25 + RRF + rerank)
        # ══════════════════════════════════════════════════════════
        elif mode == MODE_HYBRID:

            # ── Dense + BM25 en parallèle ─────────────────────────
            t0 = time.time()

            dense_task = self._search_dense_raw(
                query=         query,
                user_projects= user_projects,
                project_id=    project_id,
                source=        source,
                top_k=         TOP_K_FETCH,
                min_score=     min_score,
                session_id=    session_id,
            )

            bm25_task = self.bm25.search(
                query=         query,
                user_projects= bm25_projects,
                top_k=         TOP_K_FETCH,
                min_score=     0.0,
                session_id=    session_id,
            )

            try:
                dense_results, bm25_results = await asyncio.gather(
                    dense_task,
                    bm25_task,
                    return_exceptions=True,
                )
            except Exception as e:
                logger.error(f"search_hybrid | gather échoué | session={session_id} | {e}")
                dense_results, bm25_results = [], []

            if isinstance(dense_results, Exception):
                logger.error(f"search_hybrid | dense échoué | session={session_id} | {dense_results}")
                dense_results = []
            if isinstance(bm25_results, Exception):
                logger.error(f"search_hybrid | bm25 échoué | session={session_id} | {bm25_results}")
                bm25_results = []

            meta["nb_dense"]      = len(dense_results)
            meta["nb_bm25"]       = len(bm25_results)
            meta["t_retrieve_ms"] = round((time.time() - t0) * 1000, 1)

            logger.info(
                f"search_hybrid | session={session_id} | "
                f"dense={len(dense_results)} | bm25={len(bm25_results)}"
            )

            # ── Vérifier qu'on a au moins une liste ───────────────
            if not dense_results and not bm25_results:
                logger.warning(
                    f"search_hybrid | aucun résultat | session={session_id}"
                )
                return [], meta

            # ── RRF ───────────────────────────────────────────────
            t1 = time.time()

            try:
                fused = reciprocal_rank_fusion(
                    ranked_lists= [dense_results, bm25_results],
                    session_id=   session_id,
                )
            except Exception as e:
                logger.error(f"search_hybrid | RRF échoué | session={session_id} | {e}")
                fused = dense_results or bm25_results

            meta["nb_fused"] = len(fused)
            meta["t_rrf_ms"] = round((time.time() - t1) * 1000, 1)

            # ── Analyse RRF (pour la mémoire) ─────────────────────
            try:
                meta["rrf_analysis"] = analyze_fusion(fused, top_k=top_k)
            except Exception as e:
                logger.error(f"search_hybrid | analyze_fusion échoué | session={session_id} | {e}")
                meta["rrf_analysis"] = {}

            logger.info(
                f"search_hybrid | RRF | session={session_id} | "
                f"{len(fused)} chunks fusionnés"
            )

            # ── Reranking ─────────────────────────────────────────
            if rerank and len(fused) > 1:
                t2 = time.time()
                try:
                    fused = await self._rerank(
                        query=      query,
                        chunks=     fused,
                        top_k=      top_k,
                        session_id= session_id,
                    )
                except Exception as e:
                    logger.error(f"search_hybrid | rerank fused échoué | session={session_id} | {e}")
                    fused = fused[:top_k]
                meta["t_rerank_ms"] = round((time.time() - t2) * 1000, 1)
            else:
                fused = fused[:top_k]

            meta["nb_final"] = len(fused)

            logger.info(
                f"search_hybrid | done | session={session_id} | "
                f"mode={mode} | rerank={rerank} | "
                f"final={len(fused)} chunks"
            )

            return fused, meta

        else:
            logger.error(
                f"search_hybrid | mode inconnu '{mode}' | session={session_id}"
            )
            return [], meta

    # ─────────────────────────────────────────
    #   RECHERCHE AVEC CONTEXTE FORMATÉ
    # ─────────────────────────────────────────

    async def search_and_format(
        self,
        query:       str,
        api_key:     str,
        project_id:  str   = None,
        source:      str   = None,
        component:   str   = None,
        status:      str   = None,
        assigned_to: str   = None,
        tracker:     str   = None,
        priority:    str   = None,
        top_k:       int   = TOP_K_FINAL,
        min_score:   float = MIN_SCORE,
        session_id:  str   = None,
        rerank:      bool  = True,
    ) -> str:
        """
        Méthode originale — comportement inchangé.
        Utilisée par le service RAG en production.
        """
        try:
            results = await self.search(
                query=       query,
                api_key=     api_key,
                project_id=  project_id,
                source=      source,
                component=   component,
                status=      status,
                assigned_to= assigned_to,
                tracker=     tracker,
                priority=    priority,
                top_k=       top_k,
                min_score=   min_score,
                session_id=  session_id,
                rerank=      rerank,
            )
        except Exception as e:
            logger.error(f"search_and_format | search échoué | session={session_id} | {e}")
            return ""

        if not results:
            logger.info(
                f"search_and_format | session={session_id} | "
                f"aucun résultat pour '{query[:60]}'"
            )
            return ""

        parts = []
        for i, r in enumerate(results, 1):
            try:
                header       = self._format_result_header(r, i)
                text         = r.get("text", "")
                rerank_score = r.get("rerank_score")

                if rerank_score is not None:
                    header += f"\nRerank score : {rerank_score}"

                parts.append(f"{header}\n\n{text}")
            except Exception as e:
                logger.error(f"search_and_format | erreur formatage source {i} | {e}")
                continue

        context = "\n\n---\n\n".join(parts)

        logger.info(
            f"search_and_format | session={session_id} | "
            f"query='{query[:60]}' | "
            f"{len(parts)} sources | {len(context)} chars | "
            f"rerank={rerank}"
        )

        return context
