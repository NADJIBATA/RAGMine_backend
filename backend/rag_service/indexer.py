import gc
import asyncpg
from pgvector.asyncpg import register_vector
import json
from backend.shared.qwen3_client   import Qwen3Client
from backend.shared.redmine_client import RedmineClient
from backend.shared.config         import get_settings
from backend.shared.logger         import setup_logger
from backend.rag_service.chunker import (
    build_wiki_chunks,
    build_issue_chunks,
    build_project_chunks,
    compute_chunk_hash,
)

settings = get_settings()
logger   = setup_logger("indexer")

# ── Config Qdrant ──────────────────────────
COLLECTION_NAME = settings.collection_name
VECTOR_SIZE     = settings.vector_size
BATCH_SIZE      = settings.batch_size


class Indexer:

    def __init__(self, api_key: str = None):
        self.pool = None
        self.embed   = Qwen3Client()
        self.redmine = RedmineClient(api_key=api_key)
        self.api_key = api_key

    # ─────────────────────────────────────────
    #   CONNEXION (pool + extension pgvector)
    # ─────────────────────────────────────────

    async def connect(self, dsn: str = None) -> None:
        """
        1. Crée l'extension pgvector via une connexion simple (sans codec).
        2. Crée le pool avec le codec vector enregistré sur chaque connexion.

         Ne crée PAS la table ni les index : appeler explicitement
        `init_collection()` après `connect()` si besoin (idempotent).
        """
        dsn = dsn or settings.database_url

        # ── 1. Créer l'extension AVANT tout pool avec codec vector ────
        try:
            tmp_conn = await asyncpg.connect(dsn=dsn)
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

        # ── 2. Créer le pool (le codec vector peut maintenant s'enregistrer) ──
        async def init_conn(conn):
            try:
                await register_vector(conn)
            except Exception as e:
                logger.error(f"connect | register_vector échoué | {e}")
                raise

        try:
            self.pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=2,
                max_size=20,
                init=init_conn,
            )
        except Exception as e:
            logger.error(f"connect | création du pool échouée | {e}")
            raise

    async def init_collection(self) -> None:
        """
        Crée (si besoin) la table + les index. 
        Doit être appelé après `connect()`.
        """
        if self.pool is None:
            raise RuntimeError("init_collection() appelé avant connect()")

        try:
            async with self.pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {COLLECTION_NAME} (
                        id           TEXT PRIMARY KEY,
                        embedding    vector({VECTOR_SIZE}),
                        text         TEXT,
                        project_id   TEXT,
                        source       TEXT,
                        issue_id     TEXT,
                        status       TEXT,
                        assigned_to  TEXT,
                        component    TEXT,
                        tracker      TEXT,
                        priority     TEXT,
                        title        TEXT,
                        author       TEXT,
                        updated_on   TEXT,
                        payload      JSONB
                    );
                """)
        except Exception as e:
            logger.error(f"init_collection | création table/extension échouée | {e}")
            raise

        await self._create_payload_indexes()

    async def close(self) -> None:
        if self.pool:
            try:
                await self.pool.close()
            except Exception as e:
                logger.error(f"close | fermeture du pool échouée | {e}")
            finally:
                self.pool = None

    # ─────────────────────────────────────────
    #   INDEX PAYLOAD
    # ─────────────────────────────────────────

    async def _create_payload_indexes(self) -> None:
        indexes_sql = [
            f"CREATE INDEX IF NOT EXISTS idx_project_id ON {COLLECTION_NAME} (project_id);",
            f"CREATE INDEX IF NOT EXISTS idx_source ON {COLLECTION_NAME} (source);",
            f"CREATE INDEX IF NOT EXISTS idx_issue_id ON {COLLECTION_NAME} (issue_id);",
            f"CREATE INDEX IF NOT EXISTS idx_status ON {COLLECTION_NAME} (status);",
            f"CREATE INDEX IF NOT EXISTS idx_payload_gin ON {COLLECTION_NAME} USING GIN (payload);",
        ]
        try:
            async with self.pool.acquire() as conn:
                for sql in indexes_sql:
                    try:
                        await conn.execute(sql)
                    except Exception as e:
                        logger.error(f"_create_payload_indexes | échec sur '{sql}' | {e}")

                # ── Index vectoriel : HNSW limité à 2000 dims → cast halfvec au-delà ──
                # (doit correspondre à DISTANCE_EXPR dans retriever.py)
                try:
                    if VECTOR_SIZE > 2000:
                        index_expr = f"(embedding::halfvec({VECTOR_SIZE})) halfvec_cosine_ops"
                    else:
                        index_expr = "embedding vector_cosine_ops"
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_embedding
                        ON {COLLECTION_NAME}
                        USING hnsw ({index_expr});
                    """)
                    logger.info(f"_create_payload_indexes | index vectoriel créé | {index_expr}")
                except Exception as e:
                    # Ne bloque jamais le démarrage de l'app
                    logger.error(f"_create_payload_indexes | index vectoriel échoué | {e}")
        except Exception as e:
            logger.error(f"_create_payload_indexes | acquisition connexion échouée | {e}")

    # ─────────────────────────────────────────
    #   INDEXER DES CHUNKS PAR BATCH
    # ─────────────────────────────────────────


    @staticmethod
    def _sanitize_text(value: str) -> str:
        """
        Supprime les caractères NUL (0x00) invalides pour Postgres.
        """
        if not value:
            return value
        try:
            return value.replace("\x00", "")
        except Exception as e:
            logger.error(f"_sanitize_text | erreur | {e}")
            return value

    @staticmethod
    def _sanitize_dict(data: dict) -> dict:
        """
        Nettoie récursivement toutes les valeurs texte d'un dict.
        """
        if not isinstance(data, dict):
            return data

        cleaned = {}
        try:
            for key, value in data.items():
                if isinstance(value, str):
                    cleaned[key] = value.replace("\x00", "")
                elif isinstance(value, dict):
                    cleaned[key] = Indexer._sanitize_dict(value)
                elif isinstance(value, list):
                    cleaned[key] = [
                        v.replace("\x00", "") if isinstance(v, str)
                        else Indexer._sanitize_dict(v) if isinstance(v, dict)
                        else v
                        for v in value
                    ]
                else:
                    cleaned[key] = value
        except Exception as e:
            logger.error(f"_sanitize_dict | erreur | {e}")
            return data
        return cleaned



    async def index_chunks(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0

        total_indexed = 0
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_start: batch_start + BATCH_SIZE]
            try:
                for chunk in batch:
                    chunk["text"] = self._sanitize_text(chunk.get("text", ""))
                    chunk["metadata"] = self._sanitize_dict(chunk.get("metadata", {}))
            except Exception as e:
                logger.error(f"index_chunks | sanitize erreur | {e}")
                continue

            texts = [c["text"] for c in batch]

            try:
                embeddings = await self.embed.embed_batch(texts)
            except Exception as e:
                logger.error(f"index_chunks | embed erreur | {e}")
                continue

            rows = []
            try:
                for chunk, embedding in zip(batch, embeddings):
                    point_id = compute_chunk_hash(chunk["text"], chunk["metadata"])
                    meta = chunk["metadata"]
                    rows.append((
                        point_id, embedding, chunk["text"],
                        meta.get("project_id"), meta.get("source"),
                        meta.get("issue_id"), meta.get("status"),
                        meta.get("assigned_to_name"), meta.get("component"),
                        meta.get("tracker"), meta.get("priority"),
                        meta.get("title") or meta.get("subject"),
                        meta.get("author_name"),
                        meta.get("updated_on"),
                        json.dumps(meta, ensure_ascii=False, default=str),
                    ))
            except Exception as e:
                logger.error(f"index_chunks | construction des rows échouée | {e}")
                continue

            try:
                async with self.pool.acquire() as conn:
                    await conn.executemany(f"""
                        INSERT INTO {COLLECTION_NAME}
                        (id, embedding, text, project_id, source, issue_id,
                         status, assigned_to, component, tracker, priority,
                         title, author, updated_on, payload)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
                        ON CONFLICT (id) DO UPDATE SET
                            embedding    = EXCLUDED.embedding,
                            text         = EXCLUDED.text,
                            project_id   = EXCLUDED.project_id,
                            source       = EXCLUDED.source,
                            issue_id     = EXCLUDED.issue_id,
                            status       = EXCLUDED.status,
                            assigned_to  = EXCLUDED.assigned_to,
                            component    = EXCLUDED.component,
                            tracker      = EXCLUDED.tracker,
                            priority     = EXCLUDED.priority,
                            title        = EXCLUDED.title,
                            author       = EXCLUDED.author,
                            updated_on   = EXCLUDED.updated_on,
                            payload      = EXCLUDED.payload;
                    """, rows)
                total_indexed += len(rows)
            except Exception as e:
                logger.error(f"index_chunks | upsert erreur | {e}")

        return total_indexed

    # ─────────────────────────────────────────
    #   SUPPRIMER PAR PROJET
    # ─────────────────────────────────────────

    async def delete_by_project(self, project_id: str) -> None:
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    f"DELETE FROM {COLLECTION_NAME} WHERE project_id = $1;",
                    str(project_id),
                )
        except Exception as e:
            logger.error(f"delete_by_project | erreur | project_id={project_id} | {e}")

    # ─────────────────────────────────────────
    #   SUPPRIMER PAR SOURCE
    # ─────────────────────────────────────────

    async def delete_by_source(self, source: str, project_id: str = None) -> None:
        try:
            if project_id:
                query = f"DELETE FROM {COLLECTION_NAME} WHERE source=$1 AND project_id=$2;"
                params = (source, str(project_id))
            else:
                query = f"DELETE FROM {COLLECTION_NAME} WHERE source=$1;"
                params = (source,)
            async with self.pool.acquire() as conn:
                await conn.execute(query, *params)
        except Exception as e:
            logger.error(
                f"delete_by_source | erreur | source={source} | "
                f"project_id={project_id} | {e}"
            )

    # ─────────────────────────────────────────
    #   SUPPRIMER UN TICKET SPÉCIFIQUE
    # ─────────────────────────────────────────

    async def delete_issue(self, issue_id: str) -> None:
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    f"DELETE FROM {COLLECTION_NAME} WHERE issue_id=$1;", str(issue_id)
                )
        except Exception as e:
            logger.error(f"delete_issue | erreur | issue_id={issue_id} | {e}")

    # ─────────────────────────────────────────
    #   SUPPRIMER UNE PAGE WIKI SPÉCIFIQUE
    # ─────────────────────────────────────────

    async def delete_wiki_page(self, project_id: str, title: str) -> None:
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    f"DELETE FROM {COLLECTION_NAME} WHERE project_id=$1 AND title=$2;",
                    str(project_id), title,
                )
        except Exception as e:
            logger.error(
                f"delete_wiki_page | erreur | project_id={project_id} | "
                f"title={title} | {e}"
            )

    # ─────────────────────────────────────────
    #   INDEXER LE WIKI D'UN PROJET
    # ─────────────────────────────────────────

    async def index_project_wiki(self, project_id: str) -> int:
        logger.info(f"index_project_wiki | start | project_id={project_id}")

        await self.delete_by_source("wiki",            project_id)
        await self.delete_by_source("wiki_attachment", project_id)

        total = 0

        try:
            wiki_index = await self.redmine.get(
                f"/projects/{project_id}/wiki/index.json"
            )
        except Exception as e:
            logger.error(f"index_project_wiki | appel Redmine échoué | {e}")
            return 0

        if "error" in wiki_index:
            logger.warning(
                f"index_project_wiki | wiki inaccessible | "
                f"project={project_id} | {wiki_index.get('message')}"
            )
            return 0

        try:
            project_data = await self.redmine.get(f"/projects/{project_id}.json")
        except Exception as e:
            logger.error(f"index_project_wiki | appel projet échoué | {e}")
            project_data = {"error": str(e)}

        if "error" in project_data:
            logger.warning(
                f"index_project_wiki | projet inaccessible | "
                f"project={project_id} | {project_data.get('message')}"
            )
            project_name = ""
        else:
            project_infos = project_data.get("project", {})
            project_name  = project_infos.get("name", "")

        wiki_pages = wiki_index.get("wiki_pages", [])

        logger.info(
            f"index_project_wiki | {len(wiki_pages)} pages | "
            f"project={project_id}"
        )

        for page in wiki_pages:
            title = page.get("title", "")

            logger.info(f"index_project_wiki | traitement | title={title}")

            try:
                page_data = await self.redmine.get(
                    f"/projects/{project_id}/wiki/{title}.json",
                    params={"include": "attachments"},
                )
            except Exception as e:
                logger.error(
                    f"index_project_wiki | appel page échoué | title={title} | {e}"
                )
                continue

            if "error" in page_data:
                logger.warning(
                    f"index_project_wiki | page inaccessible | "
                    f"title={title} | {page_data.get('message')}"
                )
                continue

            wiki_page = page_data.get("wiki_page", {})

            if not wiki_page:
                continue

            try:
                chunks = await build_wiki_chunks(
                    wiki_page=  wiki_page,
                    project_name= project_name,
                    project_id= project_id,
                    api_key=    self.api_key,
                )
            except Exception as e:
                logger.error(
                    f"index_project_wiki | build_wiki_chunks échoué | "
                    f"title={title} | {e}"
                )
                continue

            indexed = await self.index_chunks(chunks)
            total  += indexed

            logger.info(
                f"index_project_wiki | title={title} | "
                f"{indexed} chunks indexés"
            )

            del chunks, wiki_page, page_data
            gc.collect()

        logger.info(
            f"index_project_wiki | done | "
            f"project={project_id} | {total} chunks total"
        )
        return total

    # ─────────────────────────────────────────
    #   INDEXER LES TICKETS D'UN PROJET
    # ─────────────────────────────────────────

    async def index_project_issues(self, project_id: str) -> int:
        logger.info(f"index_project_issues | start | project_id={project_id}")

        await self.delete_by_source("issue",            project_id)
        await self.delete_by_source("issue_attachment", project_id)

        total     = 0
        issue_ids = []

        offset = 0
        limit  = 100

        while True:
            try:
                result = await self.redmine.get(
                    "/issues.json",
                    params={
                        "project_id": project_id,
                        "status_id":  "*",
                        "limit":      limit,
                        "offset":     offset,
                    },
                )
            except Exception as e:
                logger.error(f"index_project_issues | appel Redmine échoué | {e}")
                break

            if "error" in result:
                logger.warning(
                    f"index_project_issues | erreur Redmine | "
                    f"{result.get('message')}"
                )
                break

            issues = result.get("issues", [])
            if not issues:
                break

            for issue in issues:
                issue_project_id = issue.get("project", {}).get("id")
                if str(issue_project_id) != str(project_id):
                    continue
                issue_ids.append(str(issue.get("id", "")))

            total_count  = result.get("total_count", 0)
            offset      += len(issues)

            logger.info(
                f"index_project_issues | collecte IDs | "
                f"{offset}/{total_count} | project={project_id}"
            )

            if offset >= total_count:
                break

        logger.info(
            f"index_project_issues | "
            f"{len(issue_ids)} tickets à indexer | "
            f"project={project_id}"
        )

        for i, issue_id in enumerate(issue_ids):

            try:
                issue_data = await self.redmine.get(
                    f"/issues/{issue_id}.json",
                    params={"include": "journals,attachments,relations,children"},
                )
            except Exception as e:
                logger.error(
                    f"index_project_issues | appel ticket échoué | "
                    f"#{issue_id} | {e}"
                )
                continue

            if "error" in issue_data:
                logger.warning(
                    f"index_project_issues | ticket inaccessible | "
                    f"#{issue_id} | {issue_data.get('message')}"
                )
                continue

            issue = issue_data.get("issue", {})
            if not issue:
                continue

            try:
                chunks  = await build_issue_chunks(
                    issue=   issue,
                    api_key= self.api_key,
                )
            except Exception as e:
                logger.error(
                    f"index_project_issues | build_issue_chunks échoué | "
                    f"#{issue_id} | {e}"
                )
                continue

            indexed  = await self.index_chunks(chunks)
            total   += indexed

            logger.info(
                f"index_project_issues | "
                f"{i + 1}/{len(issue_ids)} | "
                f"#{issue_id} | {indexed} chunks | "
                f"project={project_id}"
            )

            del chunks, issue, issue_data
            gc.collect()

        logger.info(
            f"index_project_issues | done | "
            f"project={project_id} | {total} chunks total"
        )
        return total

    # ─────────────────────────────────────────
    #   INDEXER LES INFOS D'UN PROJET
    # ─────────────────────────────────────────

    async def index_project_info(self, project_id: str) -> int:
        logger.info(f"index_project_info | start | project_id={project_id}")

        await self.delete_by_source("project", project_id)

        try:
            project_data = await self.redmine.get(
                f"/projects/{project_id}.json",
                params={"include": "trackers,issue_categories,enabled_modules"},
            )
        except Exception as e:
            logger.error(f"index_project_info | appel Redmine échoué | {e}")
            return 0

        if "error" in project_data:
            logger.warning(
                f"index_project_info | projet inaccessible | "
                f"{project_data.get('message')}"
            )
            return 0

        project = project_data.get("project", {})

        if not project:
            return 0

        try:
            chunks  = build_project_chunks(project)
        except Exception as e:
            logger.error(f"index_project_info | build_project_chunks échoué | {e}")
            return 0

        indexed = await self.index_chunks(chunks)

        del chunks, project, project_data
        gc.collect()

        logger.info(
            f"index_project_info | done | "
            f"project={project_id} | {indexed} chunks"
        )
        return indexed

    # ─────────────────────────────────────────
    #   INDEXER UN TICKET SPÉCIFIQUE
    # ─────────────────────────────────────────

    async def index_single_issue(self, issue_id: str) -> int:

        await self.delete_issue(issue_id)

        try:
            issue_data = await self.redmine.get(
                f"/issues/{issue_id}.json",
                params={"include": "journals,attachments,relations,children"},
            )
        except Exception as e:
            logger.error(f"index_single_issue | appel Redmine échoué | #{issue_id} | {e}")
            return 0

        if "error" in issue_data:
            logger.warning(
                f"index_single_issue | ticket inaccessible | "
                f"#{issue_id} | {issue_data.get('message')}"
            )
            return 0

        issue = issue_data.get("issue", {})
        if not issue:
            return 0

        try:
            chunks  = await build_issue_chunks(
                issue=   issue,
                api_key= self.api_key,
            )
        except Exception as e:
            logger.error(f"index_single_issue | build_issue_chunks échoué | #{issue_id} | {e}")
            return 0

        indexed = await self.index_chunks(chunks)

        del chunks, issue, issue_data
        gc.collect()

        logger.info(f"index_single_issue | done | #{issue_id} | {indexed} chunks")
        return indexed

    # ─────────────────────────────────────────
    #   INDEXER UNE PAGE WIKI SPÉCIFIQUE
    # ─────────────────────────────────────────

    async def index_single_wiki_page(self, project_id: str, title: str) -> int:
        logger.info(
            f"index_single_wiki_page | start | "
            f"project={project_id} | title={title}"
        )

        await self.delete_wiki_page(project_id, title)

        try:
            page_data = await self.redmine.get(
                f"/projects/{project_id}/wiki/{title}.json",
                params={"include": "attachments"},
            )
        except Exception as e:
            logger.error(
                f"index_single_wiki_page | appel Redmine échoué | "
                f"title={title} | {e}"
            )
            return 0

        if "error" in page_data:
            logger.warning(
                f"index_single_wiki_page | page inaccessible | "
                f"title={title} | {page_data.get('message')}"
            )
            return 0

        wiki_page = page_data.get("wiki_page", {})

        if not wiki_page:
            return 0

        try:
            project_data = await self.redmine.get(f"/projects/{project_id}.json")
        except Exception as e:
            logger.error(
                f"index_single_wiki_page | appel projet échoué | {e}"
            )
            project_data = {"error": str(e)}

        if "error" in project_data:
            logger.warning(
                f"index_single_wiki_page | projet inaccessible | "
                f"{project_data.get('message')}"
            )
            project_name = ""
        else:
            project_name = project_data.get("project", {}).get("name", "")

        try:
            chunks  = await build_wiki_chunks(
                wiki_page=  wiki_page,
                project_name = project_name,
                project_id= project_id,
                api_key=    self.api_key,
            )
        except Exception as e:
            logger.error(
                f"index_single_wiki_page | build_wiki_chunks échoué | "
                f"title={title} | {e}"
            )
            return 0

        indexed = await self.index_chunks(chunks)

        del chunks, wiki_page, page_data
        gc.collect()

        logger.info(f"index_single_wiki_page | done | title={title} | {indexed} chunks")
        return indexed

    # ─────────────────────────────────────────
    #   INDEXER UN PROJET COMPLET
    # ─────────────────────────────────────────

    async def index_project(self, project_id: str) -> dict:
        logger.info(f"index_project | start | project_id={project_id}")

        try:
            total_project = await self.index_project_info(project_id)
        except Exception as e:
            logger.error(f"index_project | index_project_info échoué | {e}")
            total_project = 0

        try:
            total_wiki    = await self.index_project_wiki(project_id)
        except Exception as e:
            logger.error(f"index_project | index_project_wiki échoué | {e}")
            total_wiki = 0

        try:
            total_issues  = await self.index_project_issues(project_id)
        except Exception as e:
            logger.error(f"index_project | index_project_issues échoué | {e}")
            total_issues = 0

        total = total_project + total_wiki + total_issues

        logger.info(
            f"index_project | done | project={project_id} | "
            f"project={total_project} | wiki={total_wiki} | "
            f"issues={total_issues} | total={total}"
        )

        return {
            "success":       True,
            "project_id":    project_id,
            "total_project": total_project,
            "total_wiki":    total_wiki,
            "total_issues":  total_issues,
            "total":         total,
        }

    # ─────────────────────────────────────────
    #   INDEXER TOUS LES PROJETS
    # ─────────────────────────────────────────

    async def index_all_projects(self) -> dict:
        logger.info("index_all_projects | start")

        try:
            projects = await self.redmine.get_paginated("/projects.json")
        except Exception as e:
            logger.error(f"index_all_projects | appel Redmine échoué | {e}")
            return {
                "success":  False,
                "message":  str(e),
                "projects": [],
                "total":    0,
            }

        if not projects:
            logger.warning("index_all_projects | aucun projet trouvé")
            return {
                "success":  False,
                "message":  "Aucun projet trouvé",
                "projects": [],
                "total":    0,
            }

        logger.info(f"index_all_projects | {len(projects)} projets à indexer")

        results = []

        for project in projects:
            project_id = str(project.get("id", ""))

            try:
                result = await self.index_project(project_id)
                results.append(result)

            except Exception as e:
                logger.error(
                    f"index_all_projects | erreur | "
                    f"project={project_id} | {str(e)}"
                )
                results.append({
                    "success":    False,
                    "project_id": project_id,
                    "error":      str(e),
                    "total":      0,
                })

            gc.collect()

        total = sum(r.get("total", 0) for r in results)
        ok    = sum(1 for r in results if r.get("success"))

        logger.info(
            f"index_all_projects | done | "
            f"{ok}/{len(results)} projets | {total} chunks total"
        )

        return {
            "success":  True,
            "projects": results,
            "total":    total,
            "indexed":  ok,
            "failed":   len(results) - ok,
        }

    # ─────────────────────────────────────────
    #   HELPERS - RÉCUPÉRER CE QUI EXISTE DÉJÀ
    # ─────────────────────────────────────────

    async def _get_indexed_issue_ids(self, project_id: str) -> set[str]:
        indexed_ids = set()
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    f"SELECT DISTINCT issue_id FROM {COLLECTION_NAME} "
                    f"WHERE project_id=$1 AND source='issue' AND issue_id IS NOT NULL;",
                    str(project_id),
                )
        except Exception as e:
            logger.error(f"_get_indexed_issue_ids | erreur | project_id={project_id} | {e}")
            return indexed_ids

        for r in rows:
            indexed_ids.add(str(r["issue_id"]))
        return indexed_ids

    async def _get_indexed_wiki_titles(self, project_id: str) -> set[str]:
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    f"SELECT DISTINCT title FROM {COLLECTION_NAME} "
                    f"WHERE project_id=$1 AND source='wiki' AND title IS NOT NULL;",
                    str(project_id),
                )
        except Exception as e:
            logger.error(f"_get_indexed_wiki_titles | erreur | project_id={project_id} | {e}")
            return set()
        return {str(r["title"]) for r in rows}

    async def _get_indexed_issue_updated_at(self, project_id: str) -> dict[str, str]:
        indexed = {}
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    f"SELECT issue_id, MAX(updated_on) AS updated_on FROM {COLLECTION_NAME} "
                    f"WHERE project_id=$1 AND source='issue' AND issue_id IS NOT NULL "
                    f"GROUP BY issue_id;",
                    str(project_id),
                )
        except Exception as e:
            logger.error(
                f"_get_indexed_issue_updated_at | erreur | project_id={project_id} | {e}"
            )
            return indexed

        for r in rows:
            indexed[str(r["issue_id"])] = r["updated_on"]
        return indexed

    async def _get_indexed_wiki_updated_at(self, project_id: str) -> dict[str, str]:
        indexed = {}
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    f"SELECT title, MAX(updated_on) AS updated_on FROM {COLLECTION_NAME} "
                    f"WHERE project_id=$1 AND source='wiki' AND title IS NOT NULL "
                    f"GROUP BY title;",
                    str(project_id),
                )
        except Exception as e:
            logger.error(
                f"_get_indexed_wiki_updated_at | erreur | project_id={project_id} | {e}"
            )
            return indexed

        for r in rows:
            indexed[str(r["title"])] = r["updated_on"]
        return indexed


    async def _get_indexed_project_info_updated_at(
        self,
        project_id: str,
    ) -> str | None:
        """
        Retourne la date updated_on du chunk 'project' déjà indexé,
        ou None si le projet n'a jamais été indexé.
        """
        try:
            async with self.pool.acquire() as conn:
                return await conn.fetchval(
                    f"""
                    SELECT MAX(updated_on)
                    FROM {COLLECTION_NAME}
                    WHERE project_id = $1
                      AND source = 'project';
                    """,
                    str(project_id),
                )
        except Exception as e:
            logger.error(
                f"_get_indexed_project_info_updated_at | erreur | "
                f"project_id={project_id} | {e}"
            )
            return None



    # ─────────────────────────────────────────
    #   DELTA ISSUES - NOUVEAUX + MODIFIÉS
    # ─────────────────────────────────────────

    async def index_project_issues_delta(
        self,
        project_id:    str,
        detect_updates: bool = True,
    ) -> dict:
        logger.info(
            f"index_project_issues_delta | start | "
            f"project={project_id} | detect_updates={detect_updates}"
        )

        try:
            if detect_updates:
                indexed_state = await self._get_indexed_issue_updated_at(project_id)
            else:
                indexed_ids   = await self._get_indexed_issue_ids(project_id)
                indexed_state = {iid: "exists" for iid in indexed_ids}
        except Exception as e:
            logger.error(f"index_project_issues_delta | état indexé introuvable | {e}")
            indexed_state = {}

        offset      = 0
        limit       = 100
        all_issues  = []

        while True:
            try:
                result = await self.redmine.get(
                    "/issues.json",
                    params={
                        "project_id": project_id,
                        "status_id":  "*",
                        "limit":      limit,
                        "offset":     offset,
                    },
                )
            except Exception as e:
                logger.error(f"index_project_issues_delta | appel Redmine échoué | {e}")
                break

            if "error" in result:
                logger.warning(
                    f"index_project_issues_delta | erreur Redmine | "
                    f"{result.get('message')}"
                )
                break

            issues = result.get("issues", [])
            if not issues:
                break

            for issue in issues:
                issue_project_id = issue.get("project", {}).get("id")
                if str(issue_project_id) == str(project_id):
                    all_issues.append(issue)

            total_count  = result.get("total_count", 0)
            offset      += len(issues)

            if offset >= total_count:
                break

        logger.info(
            f"index_project_issues_delta | "
            f"{len(all_issues)} tickets Redmine | "
            f"{len(indexed_state)} déjà indexés | "
            f"project={project_id}"
        )

        to_index  = []
        skipped   = 0

        for issue in all_issues:
            issue_id   = str(issue.get("id", ""))
            updated_on = issue.get("updated_on", "")

            if issue_id not in indexed_state:
                to_index.append((issue_id, "new"))

            elif detect_updates:
                indexed_date = indexed_state[issue_id]
                if updated_on > indexed_date:
                    to_index.append((issue_id, "updated"))
                else:
                    skipped += 1
            else:
                skipped += 1

        logger.info(
            f"index_project_issues_delta | "
            f"{len(to_index)} à indexer | "
            f"{skipped} inchangés | "
            f"project={project_id}"
        )

        total_indexed = 0
        count_new     = 0
        count_updated = 0

        for i, (issue_id, reason) in enumerate(to_index):

            if reason == "updated":
                await self.delete_issue(issue_id)

            try:
                issue_data = await self.redmine.get(
                    f"/issues/{issue_id}.json",
                    params={"include": "journals,attachments,relations,children"},
                )
            except Exception as e:
                logger.error(
                    f"index_project_issues_delta | appel ticket échoué | "
                    f"#{issue_id} | {e}"
                )
                continue

            if "error" in issue_data:
                logger.warning(
                    f"index_project_issues_delta | inaccessible | "
                    f"#{issue_id} | {issue_data.get('message')}"
                )
                continue

            issue = issue_data.get("issue", {})
            if not issue:
                continue

            try:
                chunks  = await build_issue_chunks(issue=issue, api_key=self.api_key)
            except Exception as e:
                logger.error(
                    f"index_project_issues_delta | build_issue_chunks échoué | "
                    f"#{issue_id} | {e}"
                )
                continue

            indexed = await self.index_chunks(chunks)
            total_indexed += indexed

            if reason == "new":
                count_new += 1
            else:
                count_updated += 1

            logger.info(
                f"index_project_issues_delta | "
                f"{i + 1}/{len(to_index)} | "
                f"#{issue_id} [{reason}] | {indexed} chunks"
            )

            del chunks, issue, issue_data
            gc.collect()

        result = {
            "project_id":    project_id,
            "new":           count_new,
            "updated":       count_updated,
            "skipped":       skipped,
            "total_indexed": total_indexed,
        }

        logger.info(
            f"index_project_issues_delta | done | "
            f"project={project_id} | "
            f"new={count_new} | updated={count_updated} | "
            f"skipped={skipped} | chunks={total_indexed}"
        )
        return result

    # ─────────────────────────────────────────
    #   DELTA WIKI - NOUVELLES + MODIFIÉES
    # ─────────────────────────────────────────

    async def index_project_wiki_delta(
        self,
        project_id:     str,
        detect_updates: bool = True,
    ) -> dict:
        logger.info(
            f"index_project_wiki_delta | start | "
            f"project={project_id} | detect_updates={detect_updates}"
        )

        try:
            if detect_updates:
                indexed_state = await self._get_indexed_wiki_updated_at(project_id)
            else:
                indexed_titles = await self._get_indexed_wiki_titles(project_id)
                indexed_state  = {t: "exists" for t in indexed_titles}
        except Exception as e:
            logger.error(f"index_project_wiki_delta | état indexé introuvable | {e}")
            indexed_state = {}

        try:
            wiki_index = await self.redmine.get(
                f"/projects/{project_id}/wiki/index.json"
            )
        except Exception as e:
            logger.error(f"index_project_wiki_delta | appel Redmine échoué | {e}")
            return {
                "project_id": project_id,
                "new": 0, "updated": 0,
                "skipped": 0, "total_indexed": 0,
            }

        if "error" in wiki_index:
            logger.warning(
                f"index_project_wiki_delta | wiki inaccessible | "
                f"project={project_id} | {wiki_index.get('message')}"
            )
            return {
                "project_id": project_id,
                "new": 0, "updated": 0,
                "skipped": 0, "total_indexed": 0,
            }

        wiki_pages = wiki_index.get("wiki_pages", [])

        try:
            project_data = await self.redmine.get(f"/projects/{project_id}.json")
        except Exception as e:
            logger.error(f"index_project_wiki_delta | appel projet échoué | {e}")
            project_data = {"error": str(e)}

        if "error" in project_data:
            logger.warning(
                f"index_project_wiki_delta | projet inaccessible | "
                f"{project_data.get('message')}"
            )
            project_name = ""
        else:
            project_name = project_data.get("project", {}).get("name", "")

        to_index = []
        skipped  = 0

        for page in wiki_pages:
            title      = page.get("title", "")
            updated_on = page.get("updated_on", "")

            if title not in indexed_state:
                to_index.append((title, "new"))

            elif detect_updates:
                indexed_date = indexed_state[title]
                if updated_on > indexed_date:
                    to_index.append((title, "updated"))
                else:
                    skipped += 1
            else:
                skipped += 1

        logger.info(
            f"index_project_wiki_delta | "
            f"{len(to_index)} à indexer | "
            f"{skipped} inchangées | "
            f"project={project_id}"
        )

        total_indexed = 0
        count_new     = 0
        count_updated = 0

        for i, (title, reason) in enumerate(to_index):

            if reason == "updated":
                await self.delete_wiki_page(project_id, title)

            try:
                page_data = await self.redmine.get(
                    f"/projects/{project_id}/wiki/{title}.json",
                    params={"include": "attachments"},
                )
            except Exception as e:
                logger.error(
                    f"index_project_wiki_delta | appel page échoué | "
                    f"title={title} | {e}"
                )
                continue

            if "error" in page_data:
                logger.warning(
                    f"index_project_wiki_delta | inaccessible | "
                    f"title={title} | {page_data.get('message')}"
                )
                continue

            wiki_page = page_data.get("wiki_page", {})
            if not wiki_page:
                continue

            try:
                chunks  = await build_wiki_chunks(
                    wiki_page=    wiki_page,
                    project_name= project_name,
                    project_id=   project_id,
                    api_key=      self.api_key,
                )
            except Exception as e:
                logger.error(
                    f"index_project_wiki_delta | build_wiki_chunks échoué | "
                    f"title={title} | {e}"
                )
                continue

            indexed = await self.index_chunks(chunks)
            total_indexed += indexed

            if reason == "new":
                count_new += 1
            else:
                count_updated += 1

            logger.info(
                f"index_project_wiki_delta | "
                f"{i + 1}/{len(to_index)} | "
                f"title={title} [{reason}] | {indexed} chunks"
            )

            del chunks, wiki_page, page_data
            gc.collect()

        result = {
            "project_id":    project_id,
            "new":           count_new,
            "updated":       count_updated,
            "skipped":       skipped,
            "total_indexed": total_indexed,
        }

        logger.info(
            f"index_project_wiki_delta | done | "
            f"project={project_id} | "
            f"new={count_new} | updated={count_updated} | "
            f"skipped={skipped} | chunks={total_indexed}"
        )
        return result



    # ─────────────────────────────────────────
    #   DELTA PROJECT INFO - NOUVELLES + MODIFIÉES
    # ─────────────────────────────────────────

    async def index_project_info_delta(
        self,
        project_id:     str,
        detect_updates: bool = True,
    ) -> dict:
        """
        Indexe les infos d'un projet uniquement si nouvelles ou modifiées.
        """
        logger.info(
            f"index_project_info_delta | start | "
            f"project={project_id} | detect_updates={detect_updates}"
        )

        # ── Date déjà indexée dans Postgres ────────────────────────────────────
        try:
            indexed_date = await self._get_indexed_project_info_updated_at(project_id)
        except Exception as e:
            logger.error(
                f"index_project_info_delta | lecture date indexée échouée | {e}"
            )
            indexed_date = None

        # ── Infos actuelles depuis Redmine ──────────────────────────────────────
        try:
            project_data = await self.redmine.get(
                f"/projects/{project_id}.json",
                params={"include": "trackers,issue_categories,enabled_modules"},
            )
        except Exception as e:
            logger.error(f"index_project_info_delta | appel Redmine échoué | {e}")
            return {
                "project_id":    project_id,
                "new": 0, "updated": 0,
                "skipped": 0, "total_indexed": 0,
            }

        if "error" in project_data:
            logger.warning(
                f"index_project_info_delta | projet inaccessible | "
                f"project={project_id} | {project_data.get('message')}"
            )
            return {
                "project_id":    project_id,
                "new": 0, "updated": 0,
                "skipped": 0, "total_indexed": 0,
            }

        project = project_data.get("project", {})
        if not project:
            return {
                "project_id":    project_id,
                "new": 0, "updated": 0,
                "skipped": 0, "total_indexed": 0,
            }

        updated_on = project.get("updated_on", "")

        # ── Décider quoi faire ───────────────────────────────────────────────────
        if indexed_date is None:
            reason = "new"
        elif detect_updates and updated_on > indexed_date:
            reason = "updated"
        else:
            reason = "skipped"

        logger.info(
            f"index_project_info_delta | project={project_id} | reason={reason}"
        )

        if reason == "skipped":
            del project, project_data
            gc.collect()
            return {
                "project_id":    project_id,
                "new":           0,
                "updated":       0,
                "skipped":       1,
                "total_indexed": 0,
            }

        # ── Réindexer ────────────────────────────────────────────────────────────
        await self.delete_by_source("project", project_id)

        try:
            chunks  = build_project_chunks(project)
        except Exception as e:
            logger.error(
                f"index_project_info_delta | build_project_chunks échoué | "
                f"project={project_id} | {e}"
            )
            return {
                "project_id":    project_id,
                "new": 0, "updated": 0,
                "skipped": 0, "total_indexed": 0,
            }

        indexed = await self.index_chunks(chunks)

        del chunks, project, project_data
        gc.collect()

        result = {
            "project_id":    project_id,
            "new":           1 if reason == "new" else 0,
            "updated":       1 if reason == "updated" else 0,
            "skipped":       0,
            "total_indexed": indexed,
        }

        logger.info(
            f"index_project_info_delta | done | "
            f"project={project_id} | reason={reason} | {indexed} chunks"
        )
        return result

    # ─────────────────────────────────────────
    #   DELTA PROJET COMPLET
    # ─────────────────────────────────────────

    async def index_project_delta(
        self,
        project_id:     str,
        detect_updates: bool = True,
    ) -> dict:
        logger.info(
            f"index_project_delta | start | "
            f"project={project_id} | detect_updates={detect_updates}"
        )

        try:
            project_result = await self.index_project_info_delta(
                project_id=     project_id,
                detect_updates= detect_updates,
            )
        except Exception as e:
            logger.error(f"index_project_delta | index_project_info_delta échoué | {e}")
            project_result = {"new": 0, "updated": 0, "skipped": 0, "total_indexed": 0}

        try:
            wiki_result = await self.index_project_wiki_delta(
                project_id=     project_id,
                detect_updates= detect_updates,
            )
        except Exception as e:
            logger.error(f"index_project_delta | index_project_wiki_delta échoué | {e}")
            wiki_result = {"new": 0, "updated": 0, "skipped": 0, "total_indexed": 0}

        try:
            issues_result = await self.index_project_issues_delta(
                project_id=     project_id,
                detect_updates= detect_updates,
            )
        except Exception as e:
            logger.error(f"index_project_delta | index_project_issues_delta échoué | {e}")
            issues_result = {"new": 0, "updated": 0, "skipped": 0, "total_indexed": 0}


        total = (
            project_result["total_indexed"]
            + wiki_result["total_indexed"]
            + issues_result["total_indexed"]
        )

        result = {
            "success":        True,
            "project_id":     project_id,
            "detect_updates": detect_updates,
            "project": project_result, 
            "wiki": wiki_result,
            "issues": issues_result,
            "total_indexed": total,
        }

        logger.info(
            f"index_project_delta | done | "
            f"project={project_id} | "
            f"project(reason={'new' if project_result['new'] else ('updated' if project_result['updated'] else 'skipped')}) | "
            f"wiki(new={wiki_result['new']} upd={wiki_result['updated']} "
            f"skip={wiki_result['skipped']}) | "
            f"issues(new={issues_result['new']} upd={issues_result['updated']} "
            f"skip={issues_result['skipped']}) | "
            f"total_chunks={total}"
        )
        return result


    # ─────────────────────────────────────────
    #   DELTA TOUS LES PROJETS
    # ─────────────────────────────────────────

    async def index_all_projects_delta(
        self,
        detect_updates: bool = True,
    ) -> dict:
        logger.info(f"index_all_projects_delta | start | detect_updates={detect_updates}")

        try:
            projects = await self.redmine.get_paginated("/projects.json")
        except Exception as e:
            logger.error(f"index_all_projects_delta | appel Redmine échoué | {e}")
            return {
                "success":  False,
                "message":  str(e),
                "projects": [],
                "total":    0,
            }

        if not projects:
            logger.warning("index_all_projects_delta | aucun projet trouvé")
            return {
                "success":  False,
                "message":  "Aucun projet trouvé",
                "projects": [],
                "total":    0,
            }

        logger.info(f"index_all_projects_delta | {len(projects)} projets à vérifier")

        results       = []
        total_new     = 0
        total_updated = 0
        total_skipped = 0
        total_chunks  = 0

        for project in projects:
            project_id = str(project.get("id", ""))

            try:
                result = await self.index_project_delta(
                    project_id=     project_id,
                    detect_updates= detect_updates,
                )
                results.append(result)

                total_new     += (
                    result["project"]["new"]
                    + result["wiki"]["new"]
                    + result["issues"]["new"]
                )
                total_updated += (
                    result["project"]["updated"]
                    + result["wiki"]["updated"]
                    + result["issues"]["updated"]
                )
                total_skipped += (
                    result["project"]["skipped"]
                    + result["wiki"]["skipped"]
                    + result["issues"]["skipped"]
                )
                total_chunks  += result["total_indexed"]

            except Exception as e:
                logger.error(
                    f"index_all_projects_delta | erreur | "
                    f"project={project_id} | {str(e)}"
                )
                results.append({
                    "success":    False,
                    "project_id": project_id,
                    "error":      str(e),
                })

            gc.collect()

        ok = sum(1 for r in results if r.get("success"))

        logger.info(
            f"index_all_projects_delta | done | "
            f"{ok}/{len(results)} projets | "
            f"new={total_new} | updated={total_updated} | "
            f"skipped={total_skipped} | chunks={total_chunks}"
        )

        return {
            "success":       True,
            "detect_updates": detect_updates,
            "projects":      results,
            "total_new":     total_new,
            "total_updated": total_updated,
            "total_skipped": total_skipped,
            "total_chunks":  total_chunks,
            "indexed":       ok,
            "failed":        len(results) - ok,
        }

    # ─────────────────────────────────────────
    #   STATISTIQUES
    # ─────────────────────────────────────────

    async def get_stats(self) -> dict:
        try:
            async with self.pool.acquire() as conn:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {COLLECTION_NAME};")
            return {
                "success": True,
                "collection": COLLECTION_NAME,
                "total_vectors": count,
                "vector_size": VECTOR_SIZE,
            }
        except Exception as e:
            logger.error(f"get_stats | erreur | {e}")
            return {"success": False, "error": str(e)}
