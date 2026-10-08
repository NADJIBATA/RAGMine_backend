from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing   import Optional

from backend.shared.logger          import setup_logger
from backend.shared.config          import get_settings
from backend.rag_service.indexer  import Indexer
from backend.rag_service.retriever import Retriever
# from backend.rag_service.indexer_params   import IndexerParams
# from backend.rag_service.retriver_params import RetrieverParams

settings  = get_settings()
logger    = setup_logger("rag_service")
app       = FastAPI(title="RAGmine - RAG Service")
retriever = Retriever()
# retriever_params = RetrieverParams()

# ─────────────────────────────────────────────
#   MODELES PYDANTIC
# ─────────────────────────────────────────────

class IndexProjectRequest(BaseModel):
    project_id: str
    api_key:    str

class IndexAllRequest(BaseModel):
    api_key: str

class IndexSingleIssueRequest(BaseModel):
    issue_id: str
    api_key:  str

class IndexSingleWikiRequest(BaseModel):
    project_id: str
    title:      str
    api_key:    str

class SearchRequest(BaseModel):
    query:       str
    api_key:     str
    project_id:  Optional[str]   = None
    source:      Optional[str]   = None
    component:   Optional[str]   = None
    status:      Optional[str]   = None
    assigned_to: Optional[str]   = None
    tracker:     Optional[str]   = None
    priority:    Optional[str]   = None
    top_k:       int             = 5
    min_score:   float           = 0.5
    mode:        str             = "hybrid"

class SearchFormatRequest(SearchRequest):
    pass


class IndexDeltaProjectRequest(BaseModel):
    project_id:     str
    api_key:        str
    detect_updates: bool = True   # False = nouveaux seulement

class IndexDeltaAllRequest(BaseModel):
    api_key:        str
    detect_updates: bool = True


# ─────────────────────────────────────────────
#   HELPER — exécuter une tâche de fond et
#   fermer proprement le pool de l'indexer
# ─────────────────────────────────────────────

async def _run_indexer_task(indexer: Indexer, coro_name: str, *args) -> None:
    """
    Exécute une méthode de l'indexer en tâche de fond puis
    ferme systématiquement le pool Postgres associé,
    même en cas d'erreur.
    """
    try:
        method = getattr(indexer, coro_name)
        await method(*args)
    except Exception as e:
        logger.error(f"_run_indexer_task | {coro_name} | erreur | {str(e)}")
    finally:
        await indexer.close()


# ─────────────────────────────────────────────
#   EVENTS APPLICATION (au niveau module !)
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """
    1. Initialise le schéma Postgres complet (extension pgvector +
       table + index) via un Indexer temporaire.
    2. Connecte le Retriever (dense + BM25).
    """
    logger.info("startup | initialisation du schéma Postgres")
    init_indexer = Indexer()
    try:
        await init_indexer.connect()
        await init_indexer.init_collection()
    finally:
        await init_indexer.close()

    logger.info("startup | connexion du retriever (dense)")
    await retriever.connect()

    logger.info("startup | connexion du retriever BM25")
    await retriever.bm25.connect()

    # ── Pré-construction de l'index BM25 (évite la latence au 1er call) ──
    logger.info("startup | construction initiale de l'index BM25")
    try:
        await retriever.bm25.build_index()
    except Exception as e:
        logger.error(f"startup | échec construction index BM25 | {e}")#


@app.on_event("shutdown")
async def shutdown():
    logger.info("shutdown | fermeture du retriever")
    await retriever.close()
    await retriever.bm25.close()


# ─────────────────────────────────────────────
#   HEALTH
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "rag_service"}


# ─────────────────────────────────────────────
#   INIT COLLECTION
# ─────────────────────────────────────────────

@app.post("/init")
async def init_collection():
    """
    Initialise la collection Postgres (table + index payload + vecteur).
    """
    logger.info("init_collection | demandé")
    indexer = Indexer()
    try:
        await indexer.connect()
        await indexer.init_collection()
        return {
            "success": True,
            "message": "Collection Postgres initialisée avec index payload",
        }
    except Exception as e:
        logger.error(f"init_collection | erreur | {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await indexer.close()


# ─────────────────────────────────────────────
#   INDEXATION PROJET (BACKGROUND)
# ─────────────────────────────────────────────

@app.post("/index/project")
async def index_project(
    request:          IndexProjectRequest,
    background_tasks: BackgroundTasks,
):
    """
    Lance l'indexation complète d'un projet en arrière-plan.
    """
    logger.info(f"index_project | project_id={request.project_id}")

    indexer = Indexer(api_key=request.api_key)

    try:
        await indexer.connect()
    except Exception as e:
        await indexer.close()
        raise HTTPException(
            status_code=500,
            detail=f"Impossible d'initialiser Postgres : {str(e)}",
        )

    background_tasks.add_task(
        _run_indexer_task, indexer, "index_project", request.project_id
    )

    return {
        "success":    True,
        "message":    f"Indexation du projet '{request.project_id}' lancée",
        "project_id": request.project_id,
        "mode":       "background",
    }


# ─────────────────────────────────────────────
#   INDEXATION PROJET (SYNCHRONE)
# ─────────────────────────────────────────────

@app.post("/index/project/sync")
async def index_project_sync(request: IndexProjectRequest):
    """
    Indexe un projet de manière synchrone.
    """
    logger.info(f"index_project_sync | project_id={request.project_id}")

    indexer = Indexer(api_key=request.api_key)

    try:
        await indexer.connect()
    except Exception as e:
        await indexer.close()
        raise HTTPException(
            status_code=500,
            detail=f"Impossible d'initialiser Postgres : {str(e)}",
        )

    try:
        result = await indexer.index_project(request.project_id)
        return result
    except Exception as e:
        logger.error(f"index_project_sync | erreur | {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await indexer.close()


# ─────────────────────────────────────────────
#   INDEXATION WIKI SEULEMENT
# ─────────────────────────────────────────────

@app.post("/index/project/wiki")
async def index_project_wiki(
    request:          IndexProjectRequest,
    background_tasks: BackgroundTasks,
):
    """
    Indexe uniquement le wiki d'un projet.
    """
    logger.info(f"index_project_wiki | project_id={request.project_id}")

    indexer = Indexer(api_key=request.api_key)

    try:
        await indexer.connect()
    except Exception as e:
        await indexer.close()
        raise HTTPException(
            status_code=500,
            detail=f"Impossible d'initialiser Postgres : {str(e)}",
        )

    background_tasks.add_task(
        _run_indexer_task, indexer, "index_project_wiki", request.project_id
    )

    return {
        "success":    True,
        "message":    f"Indexation wiki '{request.project_id}' lancée",
        "project_id": request.project_id,
        "mode":       "background",
    }


# ─────────────────────────────────────────────
#   INDEXATION TICKETS SEULEMENT
# ─────────────────────────────────────────────

@app.post("/index/project/issues")
async def index_project_issues(
    request:          IndexProjectRequest,
    background_tasks: BackgroundTasks,
):
    """
    Indexe uniquement les tickets d'un projet.
    """
    logger.info(f"index_project_issues | project_id={request.project_id}")

    indexer = Indexer(api_key=request.api_key)

    try:
        await indexer.connect()
    except Exception as e:
        await indexer.close()
        raise HTTPException(
            status_code=500,
            detail=f"Impossible d'initialiser Postgres : {str(e)}",
        )

    background_tasks.add_task(
        _run_indexer_task, indexer, "index_project_issues", request.project_id
    )

    return {
        "success":    True,
        "message":    f"Indexation tickets '{request.project_id}' lancée",
        "project_id": request.project_id,
        "mode":       "background",
    }


# ─────────────────────────────────────────────
#   INDEXATION INCRÉMENTALE — TICKET
# ─────────────────────────────────────────────

@app.post("/index/issue")
async def index_single_issue(
    request:          IndexSingleIssueRequest,
    background_tasks: BackgroundTasks,
):
    """
    Réindexe un ticket spécifique.
    """
    logger.info(f"index_single_issue | issue_id={request.issue_id}")

    indexer = Indexer(api_key=request.api_key)

    try:
        await indexer.connect()
    except Exception as e:
        await indexer.close()
        raise HTTPException(
            status_code=500,
            detail=f"Impossible d'initialiser Postgres : {str(e)}",
        )

    background_tasks.add_task(
        _run_indexer_task, indexer, "index_single_issue", request.issue_id
    )

    return {
        "success":  True,
        "message":  f"Réindexation ticket #{request.issue_id} lancée",
        "issue_id": request.issue_id,
        "mode":     "background",
    }


# ─────────────────────────────────────────────
#   INDEXATION INCRÉMENTALE — WIKI PAGE
# ─────────────────────────────────────────────

@app.post("/index/wiki/page")
async def index_single_wiki_page(
    request:          IndexSingleWikiRequest,
    background_tasks: BackgroundTasks,
):
    """
    Réindexe une page wiki spécifique.
    """
    logger.info(
        f"index_single_wiki_page | "
        f"project={request.project_id} | title={request.title}"
    )

    indexer = Indexer(api_key=request.api_key)

    try:
        await indexer.connect()
    except Exception as e:
        await indexer.close()
        raise HTTPException(
            status_code=500,
            detail=f"Impossible d'initialiser Postgres : {str(e)}",
        )

    background_tasks.add_task(
        _run_indexer_task,
        indexer,
        "index_single_wiki_page",
        request.project_id,
        request.title,
    )

    return {
        "success":    True,
        "message":    f"Réindexation wiki '{request.title}' lancée",
        "project_id": request.project_id,
        "title":      request.title,
        "mode":       "background",
    }


# ─────────────────────────────────────────────
#   INDEXATION TOUS LES PROJETS
# ─────────────────────────────────────────────

@app.post("/index/all")
async def index_all(
    request:          IndexAllRequest,
    background_tasks: BackgroundTasks,
):
    """
    Indexe tous les projets accessibles.
    """
    logger.info("index_all | lancement")

    indexer = Indexer(api_key=request.api_key)

    try:
        await indexer.connect()
    except Exception as e:
        await indexer.close()
        raise HTTPException(
            status_code=500,
            detail=f"Impossible d'initialiser Postgres : {str(e)}",
        )

    background_tasks.add_task(_run_indexer_task, indexer, "index_all_projects")

    return {
        "success": True,
        "message": "Indexation de tous les projets lancée",
        "mode":    "background",
    }


# ─────────────────────────────────────────────
#   SUPPRIMER INDEX PROJET
# ─────────────────────────────────────────────

@app.delete("/index/project/{project_id}")
async def delete_project_index(
    project_id: str,
    request:    IndexAllRequest,
):
    """
    Supprime tous les chunks d'un projet.
    """
    logger.info(f"delete_project_index | project_id={project_id}")

    indexer = Indexer(api_key=request.api_key)

    try:
        await indexer.connect()
        await indexer.delete_by_project(project_id)
        return {
            "success":    True,
            "message":    f"Index projet '{project_id}' supprimé",
            "project_id": project_id,
        }
    except Exception as e:
        logger.error(f"delete_project_index | erreur | {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await indexer.close()


# ─────────────────────────────────────────────
#   SUPPRIMER INDEX TICKET
# ─────────────────────────────────────────────

@app.delete("/index/issue/{issue_id}")
async def delete_issue_index(
    issue_id: str,
    request:  IndexAllRequest,
):
    """
    Supprime tous les chunks d'un ticket.
    """
    logger.info(f"delete_issue_index | issue_id={issue_id}")

    indexer = Indexer(api_key=request.api_key)

    try:
        await indexer.connect()
        await indexer.delete_issue(issue_id)
        return {
            "success":  True,
            "message":  f"Index ticket #{issue_id} supprimé",
            "issue_id": issue_id,
        }
    except Exception as e:
        logger.error(f"delete_issue_index | erreur | {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await indexer.close()


# ─────────────────────────────────────────────
#   RECHERCHE SÉMANTIQUE
# ─────────────────────────────────────────────

@app.post("/search")
async def search(request: SearchRequest):
    logger.info(
        f"search | query='{request.query[:60]}' | "
        f"project={request.project_id} | "
        f"source={request.source} | "
        f"mode={request.mode}"
    )

    try:
        if request.mode in ("bm25", "hybrid"):
            results, meta = await retriever.search_hybrid(
                query=       request.query,
                api_key=     request.api_key,
                project_id=  request.project_id,
                source=      request.source,
                top_k=       request.top_k,
                min_score=   request.min_score,
                mode=        request.mode,
                rerank=      True,
            )
        else:
            results = await retriever.search(
                query=       request.query,
                api_key=     request.api_key,
                project_id=  request.project_id,
                source=      request.source,
                component=   request.component,
                status=      request.status,
                assigned_to= request.assigned_to,
                tracker=     request.tracker,
                priority=    request.priority,
                top_k=       request.top_k,
                min_score=   request.min_score,
            )
            meta = {"mode": "dense", "rerank": True}

        return {
            "success": True,
            "query":   request.query,
            "results": results,
            "total":   len(results),
            "mode":    request.mode,
            "meta":    meta,
        }

    except Exception as e:
        logger.error(f"search | erreur | {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
#   RECHERCHE CONTEXTE FORMATÉ
# ─────────────────────────────────────────────

@app.post("/search/context")
async def search_context(request: SearchFormatRequest):
    logger.info(
        f"search_context | query='{request.query[:60]}' | "
        f"mode={request.mode}"
    )

    try:
        if request.mode in ("bm25", "hybrid"):
            results, meta = await retriever.search_hybrid(
                query=      request.query,
                api_key=    request.api_key,
                project_id= request.project_id,
                source=     request.source,
                top_k=      request.top_k,
                min_score=  request.min_score,
                mode=       request.mode,
                rerank=     True,
            )

            if not results:
                context = ""
            else:
                parts = []
                for i, r in enumerate(results, 1):
                    header       = retriever._format_result_header(r, i)
                    text         = r.get("text", "")
                    rerank_score = r.get("rerank_score")
                    rrf_score    = r.get("rrf_score")

                    if rerank_score is not None:
                        header += f"\nRerank score : {rerank_score}"
                    if rrf_score is not None:
                        header += f"\nRRF score    : {rrf_score}"

                    parts.append(f"{header}\n\n{text}")

                context = "\n\n---\n\n".join(parts)

        else:
            context = await retriever.search_and_format(
                query=       request.query,
                api_key=     request.api_key,
                project_id=  request.project_id,
                source=      request.source,
                component=   request.component,
                status=      request.status,
                assigned_to= request.assigned_to,
                tracker=     request.tracker,
                priority=    request.priority,
                top_k=       request.top_k,
                min_score=   request.min_score,
            )

        return {
            "success": True,
            "query":   request.query,
            "context": context,
            "empty":   len(context) == 0,
            "mode":    request.mode,
        }

    except Exception as e:
        logger.error(f"search_context | erreur | {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
#   DELTA — PROJET COMPLET (BACKGROUND)
# ─────────────────────────────────────────────

@app.post("/index/project/delta")
async def index_project_delta(
    request:          IndexDeltaProjectRequest,
    background_tasks: BackgroundTasks,
):
    """
    Indexation incrémentale d'un projet en arrière-plan.
    """
    logger.info(
        f"index_project_delta | project_id={request.project_id} | "
        f"detect_updates={request.detect_updates}"
    )

    indexer = Indexer(api_key=request.api_key)

    try:
        await indexer.connect()
    except Exception as e:
        await indexer.close()
        raise HTTPException(
            status_code=500,
            detail=f"Impossible d'initialiser Postgres : {str(e)}",
        )

    background_tasks.add_task(
        _run_indexer_task,
        indexer,
        "index_project_delta",
        request.project_id,
        request.detect_updates,
    )

    return {
        "success":        True,
        "message":        f"Indexation delta projet '{request.project_id}' lancée",
        "project_id":     request.project_id,
        "detect_updates": request.detect_updates,
        "mode":           "background",
    }


# ─────────────────────────────────────────────
#   DELTA — PROJET COMPLET (SYNCHRONE)
# ─────────────────────────────────────────────

@app.post("/index/project/delta/sync")
async def index_project_delta_sync(request: IndexDeltaProjectRequest):
    """
    Indexation incrémentale d'un projet (synchrone).
    """
    logger.info(
        f"index_project_delta_sync | project_id={request.project_id} | "
        f"detect_updates={request.detect_updates}"
    )

    indexer = Indexer(api_key=request.api_key)

    try:
        await indexer.connect()
    except Exception as e:
        await indexer.close()
        raise HTTPException(
            status_code=500,
            detail=f"Impossible d'initialiser Postgres : {str(e)}",
        )

    try:
        result = await indexer.index_project_delta(
            project_id=     request.project_id,
            detect_updates= request.detect_updates,
        )
        return result
    except Exception as e:
        logger.error(f"index_project_delta_sync | erreur | {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await indexer.close()


# ─────────────────────────────────────────────
#   DELTA — WIKI SEULEMENT (BACKGROUND)
# ─────────────────────────────────────────────

@app.post("/index/project/wiki/delta")
async def index_project_wiki_delta(
    request:          IndexDeltaProjectRequest,
    background_tasks: BackgroundTasks,
):
    """
    Indexation incrémentale du wiki d'un projet.
    """
    logger.info(
        f"index_project_wiki_delta | project_id={request.project_id} | "
        f"detect_updates={request.detect_updates}"
    )

    indexer = Indexer(api_key=request.api_key)

    try:
        await indexer.connect()
    except Exception as e:
        await indexer.close()
        raise HTTPException(
            status_code=500,
            detail=f"Impossible d'initialiser Postgres : {str(e)}",
        )

    background_tasks.add_task(
        _run_indexer_task,
        indexer,
        "index_project_wiki_delta",
        request.project_id,
        request.detect_updates,
    )

    return {
        "success":        True,
        "message":        f"Indexation delta wiki '{request.project_id}' lancée",
        "project_id":     request.project_id,
        "detect_updates": request.detect_updates,
        "mode":           "background",
    }


# ─────────────────────────────────────────────
#   DELTA — TICKETS SEULEMENT (BACKGROUND)
# ─────────────────────────────────────────────

@app.post("/index/project/issues/delta")
async def index_project_issues_delta(
    request:          IndexDeltaProjectRequest,
    background_tasks: BackgroundTasks,
):
    """
    Indexation incrémentale des tickets d'un projet.
    """
    logger.info(
        f"index_project_issues_delta | project_id={request.project_id} | "
        f"detect_updates={request.detect_updates}"
    )

    indexer = Indexer(api_key=request.api_key)

    try:
        await indexer.connect()
    except Exception as e:
        await indexer.close()
        raise HTTPException(
            status_code=500,
            detail=f"Impossible d'initialiser Postgres : {str(e)}",
        )

    background_tasks.add_task(
        _run_indexer_task,
        indexer,
        "index_project_issues_delta",
        request.project_id,
        request.detect_updates,
    )

    return {
        "success":        True,
        "message":        f"Indexation delta tickets '{request.project_id}' lancée",
        "project_id":     request.project_id,
        "detect_updates": request.detect_updates,
        "mode":           "background",
    }


# ─────────────────────────────────────────────
#   DELTA — TOUS LES PROJETS (BACKGROUND)
# ─────────────────────────────────────────────

@app.post("/index/all/delta")
async def index_all_delta(
    request:          IndexDeltaAllRequest,
    background_tasks: BackgroundTasks,
):
    """
    Indexation incrémentale de tous les projets accessibles.
    """
    logger.info(f"index_all_delta | detect_updates={request.detect_updates}")

    indexer = Indexer(api_key=request.api_key)

    try:
        await indexer.connect()
    except Exception as e:
        await indexer.close()
        raise HTTPException(
            status_code=500,
            detail=f"Impossible d'initialiser Postgres : {str(e)}",
        )

    background_tasks.add_task(
        _run_indexer_task,
        indexer,
        "index_all_projects_delta",
        request.detect_updates,
    )

    return {
        "success":        True,
        "message":        "Indexation delta tous les projets lancée",
        "detect_updates": request.detect_updates,
        "mode":           "background",
    }


# ─────────────────────────────────────────────
#   DELTA — TOUS LES PROJETS (SYNCHRONE)
# ─────────────────────────────────────────────

@app.post("/index/all/delta/sync")
async def index_all_delta_sync(request: IndexDeltaAllRequest):
    """
    Indexation incrémentale de tous les projets (synchrone).
    """
    logger.info(f"index_all_delta_sync | detect_updates={request.detect_updates}")

    indexer = Indexer(api_key=request.api_key)

    try:
        await indexer.connect()
    except Exception as e:
        await indexer.close()
        raise HTTPException(
            status_code=500,
            detail=f"Impossible d'initialiser Postgres : {str(e)}",
        )

    try:
        result = await indexer.index_all_projects_delta(
            detect_updates=request.detect_updates,
        )
        return result
    except Exception as e:
        logger.error(f"index_all_delta_sync | erreur | {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await indexer.close()


# ─────────────────────────────────────────────
#   STATISTIQUES
# ─────────────────────────────────────────────

@app.get("/stats")
async def stats():
    """
    Statistiques de la collection Postgres.
    """
    logger.info("stats | demandé")

    indexer = Indexer()

    try:
        await indexer.connect()
        result = await indexer.get_stats()
        return result

    except Exception as e:
        logger.error(f"stats | erreur | {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await indexer.close()