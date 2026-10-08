import json
import asyncio
import httpx
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import FastAPI, HTTPException, Depends, Request
from backend.shared.qwen3_client               import Qwen3Client
from backend.shared.devstral_client            import DevstralClient
from backend.orchestrator.mcp_client           import MCPClient
from backend.orchestrator.conversation_manager import ConversationManager
from backend.shared.config                     import get_settings
from backend.shared.logger                     import setup_logger
from backend.orchestrator.cancellation         import CancellationManager
from backend.orchestrator.exception import CancelledByUser

import re
import time
import traceback
settings = get_settings()
logger   = setup_logger("agent")

MAX_TOOL_CALLS = settings.max_tool_calls
RAG_MIN_SCORE  = settings.rag_min_score

# ─────────────────────────────────────────────
#   INTENT TYPES
# ─────────────────────────────────────────────]
INTENT_RAG_ONLY = "rag_only"
INTENT_MCP_ONLY = "mcp_only"
INTENT_CHITCHAT = "chitchat"

# ─────────────────────────────────────────────
#   PROMPTS
# ─────────────────────────────────────────────


def build_system_prompt():
    return f"""Tu es RAGmine, un assistant intelligent connecté à Redmine.
            Tu aides les utilisateurs à consulter, analyser et gérer les données Redmine :
            projets, tickets, wiki, temps passé, utilisateurs.
            Redmine URL : {settings.redmine_url}
            Date et heure : {datetime.now(ZoneInfo("Europe/Paris"))}
    
            ═══════════════════════════════════════════
              CONTEXTE RAG
            ═══════════════════════════════════════════
            - Si un contexte RAG est fourni entre <rag_context> et </rag_context> :
              → Utilise-le comme SOURCE PRINCIPALE de réponse
              → Cite TOUJOURS la source : titre wiki, ID ticket, ID projet, lien Redmine
    
    
            ═══════════════════════════════════════════
              RÉSOLUTION DES NOMS
            ═══════════════════════════════════════════  
            - Les outils MCP résolvent automatiquement les noms en IDs :
              → project_name  : nom du projet en texte libre → résolution automatique 
              → project_name est toujours précédé par le mot « projet » ou « project ». 
              → assigned_to_name : nom de l’utilisateur en texte libre → résolution automatique
              → si le nom vaut « moi » ou « mes », utiliser `get_my_account` pour récupérer l’utilisateur courant
              → fixed_version_name : nom de la version → résolution automatique
            - Tu n'as JAMAIS besoin de résoudre les IDs manuellement
            - Passe directement le nom tel que l'utilisateur l'a fourni
    
            ═══════════════════════════════════════════
              STRATÉGIE DE RÉPONSE
            ═══════════════════════════════════════════
            1. Analyse la question précisément
            2. Cherche dans le contexte RAG fourni
            3. utilise les outils MCP dans l'ordre logique si le mode est le mcp_only
            4. Synthétise une réponse claire et structurée
            5. Cite toujours les sources avec les liens direct vers redmine
    
    
            ═══════════════════════════════════════════
              PAGINATION
            ═══════════════════════════════════════════
            - limit=25  (défaut) → aperçu rapide
            - limit=-1           → TOUS les résultats
    
            Utiliser limit=-1 quand :
              → Recherche d'un élément précis dans une liste complète
              → Analyse exhaustive demandée
              → Nombre de résultats inconnu
    
            ═══════════════════════════════════════════
              GESTION DES DATES
            ═══════════════════════════════════════════
            - Plage de dates    → created_from + created_to
            - Date de début     → created_from="YYYY-MM-DD"
            - Date de fin       → created_to="YYYY-MM-DD"
            - created_on        → réservé aux filtres formatés (>=, <=, ><)
            - Même logique pour updated_from / updated_to / updated_on
            - search ne supporte PAS les filtres de dates :
              → Étape 1 : search pour trouver les IDs
              → Étape 2 : list_issues avec IDs + filtres de dates
    
            ═══════════════════════════════════════════
              FORMAT DES RÉPONSES
            ═══════════════════════════════════════════
            - Langue : français par défaut, adapte-toi à la langue de l'utilisateur
            - Structure :
              → Réponse directe à la question en premier
              → Détails et données ensuite
              → Sources et liens en dernier
            - Toujours inclure :
              → IDs des tickets, projets, utilisateurs
              → Liens directs vers les ressources Redmine 
              → Statut, priorité, assigné pour les tickets
            - Ne retourne JAMAIS du JSON brut
            - Utilise des listes à puces et titres pour la lisibilité
    
            ═══════════════════════════════════════════
              RÈGLES ABSOLUES
            ═══════════════════════════════════════════
            ✗ N'invente JAMAIS de données
            ✗ N'utilise PAS d'informations hors RAG et outils MCP
            ✗ Ne retourne PAS de JSON brut
    
            ✗ Ne filtre JAMAIS les résultats des outils MCP par toi-même :
              → Redmine est organisé en projets parents et sous-projets enfants
              → Un ticket d'un sous-projet enfant appartient aussi au projet parent
              → TOUS les résultats retournés par l'outil sont valides et pertinents
              → Tu n'as PAS à vérifier si un ticket "appartient vraiment" au projet demandé sauf avec l'outil de recherche
              → Tu n'as PAS à exclure des résultats basé sur le nom ou le id du projet
              → Le filtrage est fait par l'API Redmine, pas par toi
    
            ✓ Si information introuvable → dis clairement :
              "Je n'ai pas trouvé cette information dans Redmine"
            ✓ Si erreur outil → explique clairement à l'utilisateur
            ✓ Si question ambiguë → demande une clarification
            """
    
INTENT_PROMPT = """Tu es un classificateur d'intention pour un assistant Redmine.

Analyse le message et retourne UN SEUL mot parmi :

rag_only
  → Question qui contient des mots-clés et nécessite une recherche 
  → Question sur la documentation, les procédures, le wiki, l'architecture, guides
  → "comment fonctionne X", "quelle est la procédure Y", "architecture de Z", "Explication d'un concept"


mcp_only
  → Listes de données (tickets, projets, utilisateurs, wiki)
chitchat
  → Seulement si la réponse existe déjà dans la session

Notes :
- Toujours privilégier la recherche via le RAG plutôt que via l’outil MCP Search. 
- Réponds UNIQUEMENT avec le mot clé, rien d'autre.
- impossible de faire appel mcp avec le mode chitchat
"""


class Agent:

    def __init__(self):
        self.llm_devstral = DevstralClient()
        self.embed_qwen3  = Qwen3Client()
        self.mcp          = MCPClient()
        self.conversation = ConversationManager()
        self.cancellation   = CancellationManager()
        self._locks: dict[str, asyncio.Lock] = {}
        self._lock_last_used: dict[str, float] ={}




    # ─────────────────────────────────────────────
    #   HELPER : récupérer l'event d'annulation
    # ─────────────────────────────────────────────

    def _get_cancel_event(self, session_id: str) -> asyncio.Event:
        """Retourne l'event asyncio du CancellationManager."""
        return self.cancellation.get_or_create(session_id)

    def _check_cancelled(self, session_id: str) -> None:
        """
        Lève CancelledByUser si l'utilisateur a demandé l'arrêt.
        À appeler aux points stratégiques du traitement.
        """
        if self.cancellation.is_cancelled(session_id):
            logger.info(f"_check_cancelled | annulation détectée | session={session_id}")
            raise CancelledByUser("Requête annulée par l'utilisateur.")

    
    def cancel(self, session_id: str) -> bool:
        return self.cancellation.cancel(session_id)

    # ─────────────────────────────────────────
    #   LOCK PAR SESSION
    # ─────────────────────────────────────────

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        self._lock_last_used[session_id] = time.time()
        return self._locks.setdefault(session_id, asyncio.Lock())

    # ─────────────────────────────────────────
    #   CLASSIFIER L'INTENTION
    # ─────────────────────────────────────────

    async def _classify_intent(
        self,
        user_message: str,
        session_id:   str,
        history:      list[dict] = None,
        reasoning_effort: str | None = None
    ) -> str:
        try:
            context = ""
            if history:
                recent = [
                    msg for msg in history
                    if msg["role"] in ("user", "assistant")
                ][-settings.max_messages:]

                for msg in recent:
                    role    = "User"      if msg["role"] == "user" else "Assistant"
                    content = str(msg.get("content", ""))[:500]
                    context += f"{role}: {content}\n"

            classify_message = (
                f"Historique récent :\n{context}\n"
                f"Nouveau message : {user_message}"
                if context
                else user_message
            )

            response = await self.llm_devstral.chat_with_tools(
                messages=[
                    {"role": "system", "content": INTENT_PROMPT},
                    {"role": "user",   "content": classify_message},
                ],
                tools=[],
                cancel_event=self._get_cancel_event(session_id),
                reasoning_effort= reasoning_effort,                 
            )

            raw   = response.get("content", "").strip().lower()
            valid = {INTENT_RAG_ONLY, INTENT_MCP_ONLY, INTENT_CHITCHAT}
            intent = INTENT_RAG_ONLY

            for candidate in valid:
                if re.search(rf"\b{re.escape(candidate)}\b", raw):
                    intent = candidate
                    break

            if intent not in valid:
                logger.warning(
                    f"_classify_intent | intent invalide '{intent}' | "
                    f"fallback rag_only | session={session_id}"
                )
                intent = INTENT_RAG_ONLY

            logger.info(
                f"_classify_intent | '{user_message[:60]}' "
                f"→ {intent} | session={session_id}"
                f"reasoning effort of classification: {reasoning_effort}"
            )
            return intent

        # ── CRITIQUE : propager les annulations ──────────────
        except (asyncio.CancelledError, CancelledByUser):
            raise                                              # ← NE PAS avaler

        except Exception as e:
            logger.error(
                f"_classify_intent | erreur | {str(e)} | "
                f"fallback rag_only | session={session_id}"
            )
            return INTENT_RAG_ONLY



    # ─────────────────────────────────────────
    #   RAG SEARCH
    # ─────────────────────────────────────────

    async def _rag_search(
        self,
        query:      str,
        api_key:    str,
        session_id: str,
        project_id: str = None,
    ) -> tuple[str, float, list[dict]]:
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                
                logger.warning(f"_rag_search | TRYING CONNECT API")

                response = await client.post(
                    f"{settings.rag_service_url}/search",
                    json={
                        "query":      query,
                        "api_key":    api_key,
                        "project_id": project_id,
                        "top_k":      settings.rag_top_k,
                        "min_score":  settings.rag_min_score,
                        "mode":       settings.rag_mode,   
                    },
                )

                logger.warning(f"_rag_search | SUCCESS CONNECT API");
    
                if response.status_code != 200:
                    logger.warning(
                        f"_rag_search | {response.status_code} | session={session_id}"
                    )
                    return "", 0.0, []
    
                data    = response.json()
                results = data.get("results", [])
    
                if not results:
                    logger.info(f"_rag_search | aucun résultat | session={session_id}")
                    return "", 0.0, []
    
                best_score = max(r.get("score", 0) for r in results)
    
                context_response = await client.post(
                    f"{settings.rag_service_url}/search/context",
                    json={
                        "query":      query,
                        "api_key":    api_key,
                        "project_id": project_id,
                        "top_k":      settings.rag_top_k,
                        "min_score":  settings.rag_min_score,
                        "mode":       settings.rag_mode,   
                    },
                )
    
                context = ""
                if context_response.status_code == 200:
                    context = context_response.json().get("context", "")
    
                logger.info(
                    f"_rag_search | session={session_id} | "
                    f"mode={settings.rag_mode} |"              
                    f"best_score={best_score:.3f} | "
                    f"{len(results)} résultats | "
                    f"{len(context)} chars"
                )
    
                return context, best_score, results
    
        except Exception as e:
            logger.error(
                f"_rag_search | erreur | session={session_id} | "
                f"type={type(e).__name__} | "          # ← nom de l'exception
                f"message={str(e) or 'vide'} | "       # ← message (si existe)
                f"traceback={traceback.format_exc()}"  # ← stack trace complète
            )

            return "", 0.0, []

    # ─────────────────────────────────────────
    #   RÉCUPÉRER LES SOURCES COMPLÈTES
    # ─────────────────────────────────────────

    async def _fetch_full_sources(
        self,
        rag_results: list[dict],
        api_key:     str,
        session_id:  str,
    ) -> str:
        seen    = set()
        sources = []

        for r in rag_results:
            source = r.get("source", "")

            if source in ("wiki", "wiki_attachment"):
                key = (
                    "wiki",
                    str(r.get("project_name", "")),
                    str(r.get("project_id", "")),
                    str(r.get("title", "") or r.get("wiki_title", "")),
                )
            elif source in ("issue", "issue_attachment"):
                key = ("issue", str(r.get("issue_id", "")))
            elif source == "project":
                key = ("project", str(r.get("project_name", "")),str(r.get("project_id", "")))
            else:
                logger.debug(f"_fetch_full_sources | source inconnue '{source}' | ignorée")
                continue

            if key not in seen:
                seen.add(key)
                sources.append((key, r))

        logger.info(
            f"_fetch_full_sources | session={session_id} | "
            f"{len(rag_results)} chunks → {len(sources)} sources uniques"
        )

        if not sources:
            return ""

        full_contexts = []

        for key, r in sources:
            source_type = key[0]

            try:
                if source_type == "wiki":
                    project_name = key[1]
                    project_id = key[2]
                    title      = key[3]

                    if not project_id or not title:
                        continue

                    result = await self.mcp.call_tool(
                        tool_name=  "get_wiki_page",
                        parameters={"project_name": project_name, "page_title": title},
                        api_key=    api_key,
                    )

                    if result.get("success"):
                        page = result.get("wiki_page", {})
                        full_contexts.append(
                            f"[Wiki complet]\n"
                            f"Titre   : {title}\n"
                            f"Projet  : {project_id}\n"
                            f"Version : {page.get('version', '')}\n"
                            f"Modifié : {page.get('updated_on', '')}\n"
                            f"URL     : {settings.redmine_url}/projects/{project_id}/wiki/{title}\n\n"
                            f"{page.get('text', '')}"
                        )
                    else:
                        full_contexts.append(
                            f"[Wiki partiel - {title}]\n"
                            f"Projet : {project_id}\n"
                            f"URL    : {settings.redmine_url}/projects/{project_id}/wiki/{title}\n\n"
                            f"{r.get('text', '')}"
                        )

                elif source_type == "issue":
                    issue_id = key[1]

                    if not issue_id:
                        continue

                    result = await self.mcp.call_tool(
                        tool_name=  "get_issue_details",
                        parameters={"issue_id": issue_id},
                        api_key=    api_key,
                    )

                    if result.get("success"):
                        issue       = result.get("issue", {})
                        journals    = issue.get("journals", []) or []
                        comments    = []

                        for j in journals:
                            notes = (j.get("notes", "") or "").strip()
                            if notes:
                                author = (j.get("user", {}) or {}).get("name", "")
                                date   = j.get("created_on", "")
                                comments.append(f"  [{author} - {date}]\n  {notes}")

                        comments_text = (
                            "\n\nCommentaires :\n" + "\n\n".join(comments)
                            if comments else ""
                        )

                        full_contexts.append(
                            f"[Ticket complet #{issue_id}]\n"
                            f"Sujet   : {issue.get('subject', '')}\n"
                            f"Projet  : {(issue.get('project', {}) or {}).get('name', '')}\n"
                            f"Tracker : {(issue.get('tracker', {}) or {}).get('name', '')}\n"
                            f"Statut  : {(issue.get('status', {}) or {}).get('name', '')}\n"
                            f"Priorité: {(issue.get('priority', {}) or {}).get('name', '')}\n"
                            f"Assigné : {(issue.get('assigned_to', {}) or {}).get('name', 'Non assigné')}\n"
                            f"Créé le : {issue.get('created_on', '')}\n"
                            f"Modifié : {issue.get('updated_on', '')}\n"
                            f"URL     : {settings.redmine_url}/issues/{issue_id}\n\n"
                            f"Description :\n{issue.get('description', '') or ''}"
                            f"{comments_text}"
                        )
                    else:
                        full_contexts.append(
                            f"[Ticket partiel #{issue_id}]\n"
                            f"URL : {settings.redmine_url}/issues/{issue_id}\n\n"
                            f"{r.get('text', '')}"
                        )

                elif source_type == "project":
                    project_id = key[2]
                    project_name = key[1]

                    if not project_id:
                        continue

                    result = await self.mcp.call_tool(
                        tool_name=  "get_project_details",
                        parameters={"project_name": project_name},
                        api_key=    api_key,
                    )

                    if result.get("success"):
                        project      = result.get("project", {})
                        trackers     = project.get("trackers", []) or []
                        modules      = project.get("enabled_modules", []) or []

                        full_contexts.append(
                            f"[Projet complet]\n"
                            f"Nom     : {project.get('name', '')}\n"
                            f"ID      : {project_id}\n"
                            f"Statut  : {'Actif' if project.get('status', 1) == 1 else 'Archivé'}\n"
                            f"Créé le : {project.get('created_on', '')}\n"
                            f"Trackers: {', '.join(t.get('name', '') for t in trackers)}\n"
                            f"Modules : {', '.join(m.get('name', '') for m in modules)}\n"
                            f"URL     : {settings.redmine_url}/projects/{project_id}\n\n"
                            f"Description :\n{project.get('description', '') or ''}"
                        )
                    else:
                        full_contexts.append(
                            f"[Projet partiel - {project_id}]\n"
                            f"URL : {settings.redmine_url}/projects/{project_id}\n\n"
                            f"{r.get('text', '')}"
                        )
            except (asyncio.CancelledError, CancelledByUser):
                raise

            except Exception as e:
                logger.error(
                    f"_fetch_full_sources | erreur | source={source_type} | key={key} | {str(e)}"
                )
                full_contexts.append(
                    f"[Source partielle - {r.get('source', '')}]\n{r.get('text', '')}"
                )

        if not full_contexts:
            return ""

        context = "\n\n---\n\n".join(full_contexts)

        logger.info(
            f"_fetch_full_sources | session={session_id} | "
            f"{len(full_contexts)} sources complètes | {len(context)} chars"
        )

        return context

    # ─────────────────────────────────────────
    #   INJECTER LE CONTEXTE RAG
    # ─────────────────────────────────────────

    def _inject_rag_context(self, user_message: str, rag_context: str) -> str:
        if not rag_context:
            return user_message
        return (
            f"<rag_context>\n"
            f"{rag_context}\n"
            f"</rag_context>\n\n"
            f"Question : {user_message}"
        )

    # ─────────────────────────────────────────
    #   REFORMULER LA REQUÊTE RAG
    # ─────────────────────────────────────────

    REWRITE_PROMPT = """Tu es un expert en reformulation de requêtes de recherche.

    Ton rôle : reformuler la question de l'utilisateur pour optimiser la recherche documentaire via le rag.
    
    Règles :
    - Corrige les fautes de frappe 
    - Garde le sens original intact
    - Supprime les formules de politesse inutiles ("peux-tu", "est-ce que tu peux" ,"quelles" , "comment", "donnes moi" etc.)
    - Réponds UNIQUEMENT avec la question reformulée, rien d'autre
    - Même langue que la question originale
    """

    async def _rewrite_query(
        self,
        query:        str,
        session_id:   str,
        track_tokens: callable = None,
        reasoning_effort: str | None = None
    ) -> str:
        try:
            response = await self.llm_devstral.chat_with_tools(
                messages=[
                    {"role": "system", "content": self.REWRITE_PROMPT},
                    {"role": "user",   "content": query},
                ],
                tools=[],
                cancel_event=self._get_cancel_event(session_id),
                reasoning_effort= reasoning_effort,
            )
            if track_tokens:
                await track_tokens(response)

            rewritten = response.get("content", "").strip()

            if not rewritten:
                logger.warning(
                    f"_rewrite_query | réponse vide | fallback original | session={session_id}"
                )
                return query

            logger.info(
                f"_rewrite_query | session={session_id}\n"
                f"  original  : {query}\n"
                f"  rewritten : {rewritten}"
                f"reasoning effort of rewriting : ${reasoning_effort}"
            )
            return rewritten

        # ── CRITIQUE : propager les annulations ──────────────
        except (asyncio.CancelledError, CancelledByUser):
            raise                                             

        except Exception as e:
            logger.error(
                f"_rewrite_query | erreur | {str(e)} | "
                f"fallback original | session={session_id}"
            )
            return query



    # ─────────────────────────────────────────
    #   BOUCLE ReAct MCP
    # ─────────────────────────────────────────

    async def _react_loop(
        self,
        history:      list[dict],
        tools:        list[dict],
        api_key:      str,
        session_id:   str,
        track_tokens: callable = None,
        reasoning_effort: str | None = None,   
               
    ) -> tuple[str, list[dict]]:
        tool_calls_log = []
        nb_tool_calls  = 0

        while nb_tool_calls < MAX_TOOL_CALLS:

            # ── Point de contrôle ─────────────────────
            self._check_cancelled(session_id)               # ← NOUVEAU

            llm_response = await self.llm_devstral.chat_with_tools(
                messages=history,
                tools=   tools,
                cancel_event=self._get_cancel_event(session_id),
                reasoning_effort= reasoning_effort,   
            )
            if track_tokens:
                await track_tokens(llm_response)

            logger.info(
                f"_react_loop | session={session_id} | "
                f"type={llm_response['type']} | "
                f"nb_tool_calls={nb_tool_calls} |"
                f"reasoning_effort={reasoning_effort}" 
            )

            if llm_response["type"] == "final_answer":
                return llm_response["content"], tool_calls_log

            if llm_response["type"] == "tool_call":
                tool_calls  = llm_response["tool_calls"]
                raw_content = llm_response.get("raw") or ""

                if not raw_content.strip():
                    tool_names  = [tc["name"] for tc in tool_calls]
                    raw_content = f"[tool_call: {', '.join(tool_names)}]"

                history.append({"role": "assistant", "content": raw_content})

                for tc in tool_calls:
                    # ── Point de contrôle avant chaque outil ──
                    self._check_cancelled(session_id)       # ← NOUVEAU

                    tool_name = tc["name"]
                    tool_args = tc["arguments"]
                    nb_tool_calls += 1

                    logger.info(
                        f"_react_loop | tool={tool_name} | session={session_id}"
                    )

                    tool_result = await self.mcp.call_tool(
                        tool_name=  tool_name,
                        parameters= tool_args,
                        api_key=    api_key,
                        
                    )

                    logger.info(
                        f"_react_loop | tool={tool_name} | result={tool_result}"
                    )

                    tool_calls_log.append({
                        "tool":   tool_name,
                        "params": tool_args,
                        "result": tool_result,
                    })

                    history.append({
                        "role":    "user",
                        "content": (
                            f"Résultat de l'outil {tool_name} :\n"
                            f"{json.dumps(tool_result, ensure_ascii=False)}"
                        ),
                    })

        # MAX atteint
        logger.warning(
            f"_react_loop | MAX_TOOL_CALLS atteint | session={session_id}"
        )

        history.append({
            "role":    "user",
            "content": "Donne-moi une réponse finale basée sur les informations récupérées.",
        })

        self._check_cancelled(session_id)                   # ← NOUVEAU

        llm_response = await self.llm_devstral.chat_with_tools(
            messages=history,
            tools=   tools,
            cancel_event=self._get_cancel_event(session_id),
            reasoning_effort= reasoning_effort,
        )

        final = (
            llm_response.get("content")
            or llm_response.get("raw", "Je n'ai pas pu obtenir une réponse complète.")
        )

        return final, tool_calls_log


    # ─────────────────────────────────────────
    #   POINT D'ENTRÉE PRINCIPAL
    # ─────────────────────────────────────────

    async def chat(
        self,
        session_id:   str,
        user_message: str,
        api_key:      str,
        reasoning_effort: str | None = None,
        forced_intent:    str | None = None
    ) -> dict:
        self._cleanup_old_locks()
        self.cancellation.cleanup()                          # ← NOUVEAU
        lock = self._get_lock(session_id)

        if lock.locked():
            logger.warning(f"chat | session={session_id} | déjà en cours")
            return {
                "success":    False,
                "response":   None,
                "session_id": session_id,
                "tool_calls": [],
                "rag_used":   False,
                "error":      "Une réponse est déjà en cours de traitement.",
            }

        # ── Réinitialiser l'event d'annulation ────
        self.cancellation.reset(session_id)                 # ← NOUVEAU

        async with lock:
            try:
                return await self._process(
                    session_id=   session_id,
                    user_message= user_message,
                    api_key=      api_key,
                    reasoning_effort= reasoning_effort,
                    forced_intent=    forced_intent
                )

            # ── Annulation propre ──────────────────
            except (CancelledByUser, asyncio.CancelledError):      
                logger.info(f"chat | requête annulée | session={session_id}")
                return {
                    "success":    False,
                    "cancelled":  True,
                    "response":   None,
                    "session_id": session_id,
                    "tool_calls": [],
                    "rag_used":   False,
                    "error":      "Requête annulée par l'utilisateur.",
                }


            except httpx.HTTPStatusError as e:
                try:
                    error_body = e.response.json()
                    error_msg  = error_body.get("error", {}).get("message") or str(e)
                except Exception:
                    error_msg = str(e)

                logger.error(
                    f"chat | erreur HTTP | session={session_id} | {error_msg}"
                )
                return {
                    "success":    False,
                    "response":   None,
                    "session_id": session_id,
                    "tool_calls": [],
                    "rag_used":   False,
                    "error":      error_msg,
                }

            except Exception as e:
                logger.error(
                    f"chat | erreur | session={session_id} | {str(e)}"
                )
                return {
                    "success":    False,
                    "response":   None,
                    "session_id": session_id,
                    "tool_calls": [],
                    "rag_used":   False,
                    "error":      str(e),
                }

    # ─────────────────────────────────────────
    #   TRAITEMENT PRINCIPAL
    # ─────────────────────────────────────────

    async def _process(
        self,
        session_id:   str,
        user_message: str,
        api_key:      str,
        reasoning_effort: str | None = None,
        forced_intent:    str | None = None,
    ) -> dict:

        effort = reasoning_effort or settings.reasoning_effort
        async def _track_tokens(llm_response: dict) -> None:
            t_in  = llm_response.get("tokens_input",  0)
            t_out = llm_response.get("tokens_output", 0)
            if t_in or t_out:
                await self.conversation.add_tokens(
                    session_id=    session_id,
                    tokens_input=  t_in,
                    tokens_output= t_out,
                )

        # ── 1. Outils MCP ─────────────────────
        tools   = await self.mcp.format_tools_for_llm()
        history = await self.conversation.get_or_create(session_id)

        if not history:
            history = [{"role": "system", "content": build_system_prompt()}]

        # ── 2. Intent ─────────────────────────
        self._check_cancelled(session_id)      

        use_forced = forced_intent and forced_intent != "auto"
        if use_forced:
            intent = forced_intent
            logger.info(
                f"_process | intent forcé='{intent}' | "
                f"classify_intent ignoré | session={session_id}"
            )
        else:
            intent = await self._classify_intent(
                user_message=     user_message,
                session_id=       session_id,
                history=          history,
                reasoning_effort= "low",
            )
            logger.info(
                f"_process | intent classifié='{intent}' | session={session_id}"
            )

        history.append({"role": "user", "content": user_message})

        rag_context    = ""
        rag_score      = 0.0
        rag_results    = []
        rag_used       = False
        tool_calls_log = []
        final_response = ""

        # ══════════════════════════════════════
        #   CHITCHAT
        # ══════════════════════════════════════
        if intent == INTENT_CHITCHAT:
            self._check_cancelled(session_id)               # ← NOUVEAU

            llm_response = await self.llm_devstral.chat_with_tools(
                messages=history,
                tools=   [],
                cancel_event=self._get_cancel_event(session_id),
                reasoning_effort= effort,                             
            )
            await _track_tokens(llm_response)
            final_response = llm_response.get("content", "")

        # ══════════════════════════════════════
        #   MCP ONLY
        # ══════════════════════════════════════
        elif intent == INTENT_MCP_ONLY:
            self._check_cancelled(session_id)               

            is_new_session = len(
                [m for m in history if m["role"] == "assistant"]
            ) == 0

            if is_new_session:
                ref_tools = [
                    "get_issue_statuses",
                    "get_trackers",
                    "get_priorities",
                    "get_custom_fields",
                ]
                ref_parts = []

                for tool_name in ref_tools:
                    self._check_cancelled(session_id)       
                    try:
                        result = await self.mcp.call_tool(
                            tool_name=  tool_name,
                            parameters= {},
                            api_key=    settings.redmine_api_key,              
                        )
                        ref_parts.append(
                            f"[Référence {tool_name}]\n"
                            f"{json.dumps(result, ensure_ascii=False)}"
                        )
                    except CancelledByUser:
                        raise                               # ← propager
                    except Exception as e:
                        logger.error(
                            f"_process | erreur référence | tool={tool_name} | "
                            f"{str(e)} | session={session_id}"
                        )

                if ref_parts:
                    ref_message = {
                        "role":    "user",
                        "content": (
                            "Références Redmine disponibles pour cette session "
                            "(statuts, trackers, priorités) :\n\n"
                            + "\n\n---\n\n".join(ref_parts)
                            + "\n\nMémorise ces références pour toute la session."
                        ),
                    }
                    history.insert(len(history) - 1, ref_message)

            final_response, tool_calls_log = await self._react_loop(
                history=      history,
                tools=        tools,
                api_key=      api_key,
                session_id=   session_id,
                track_tokens= _track_tokens,
                reasoning_effort= effort, 
            )

        # ══════════════════════════════════════
        #   RAG ONLY
        # ══════════════════════════════════════
        elif intent == INTENT_RAG_ONLY:
            self._check_cancelled(session_id)

            rewritten_query = await self._rewrite_query(
                query=        user_message,
                session_id=   session_id,
                track_tokens= _track_tokens,
                reasoning_effort= "none", 
            )

            self._check_cancelled(session_id)               

            rag_context, rag_score, rag_results = await self._rag_search(
                query=      rewritten_query,
                api_key=    api_key,
                session_id= session_id,
                
            )

            logger.info(f"Context RAG : {rag_context}" )

            if rag_context:
                rag_used = True
                self._check_cancelled(session_id)           

                full_context = await self._fetch_full_sources(
                    rag_results=rag_results,
                    api_key=    api_key,
                    session_id= session_id,
                )

                self._check_cancelled(session_id)           

                context_to_inject = full_context if full_context else rag_context
                enriched          = self._inject_rag_context(
                    user_message, context_to_inject
                )
                messages_with_rag = (
                    history[:-1] + [{"role": "user", "content": enriched}]
                )

                llm_response = await self.llm_devstral.chat_with_tools(
                    messages=messages_with_rag,
                    tools=   [],
                    cancel_event=self._get_cancel_event(session_id),
                    reasoning_effort= effort, 
                )
                await _track_tokens(llm_response)
                final_response = llm_response.get("content", "")

            else:
                if rag_context:
                    rag_used               = True
                    enriched               = self._inject_rag_context(
                        user_message, rag_context
                    )
                    history[-1]["content"] = enriched

                final_response, tool_calls_log = await self._react_loop(
                    history=      history,
                    tools=        tools,
                    api_key=      api_key,
                    session_id=   session_id,
                    track_tokens= _track_tokens,
                    reasoning_effort= effort, 
                )

        # ── Sauvegarde ─────────────────────────
        history.append({"role": "assistant", "content": final_response})
        await self._save_history(session_id, history)

        question_tokens = await self.conversation.reset_tokens(session_id)
        await self.conversation.save_question_tokens(
            session_id=    session_id,
            question=      user_message,
            tokens_input=  question_tokens["tokens_input"],
            tokens_output= question_tokens["tokens_output"],
            tokens_total=  question_tokens["tokens_total"],
            nb_calls=      question_tokens["nb_calls"],
            intent=        intent,
        )

        return {
            "success":         True,
            "response":        final_response,
            "session_id":      session_id,
            "tool_calls":      tool_calls_log,
            "rag_used":        rag_used,
            "intent":          intent,
            "intent_forced":    use_forced, 
            "rag_score":       round(rag_score, 3),
            "reasoning_effort": effort,     
            "tokens_input":    question_tokens["tokens_input"],
            "tokens_output":   question_tokens["tokens_output"],
            "tokens_total":    question_tokens["tokens_total"],
            "tokens_nb_calls": question_tokens["nb_calls"],
            "error":           None,
        }

    # ─────────────────────────────────────────
    #   SAUVEGARDER L'HISTORIQUE
    # ─────────────────────────────────────────

    async def _save_history(self, session_id: str, history: list[dict]) -> None:
        await self.conversation.clear(session_id)
        await self.conversation.add_messages(session_id, history)
        await self.conversation.refresh_ttl(session_id)
        logger.debug(f"_save_history | session={session_id} | {len(history)} messages")

    async def _stop_chat(self,session_id:str) ->None:
        self._locks.pop(session_id)

    # ─────────────────────────────────────────
    #   supprimer les locks des sessions morts
    # ─────────────────────────────────────────

    def _cleanup_old_locks(self) -> int:
        """
        Supprime les locks des sessions inactives depuis plus de TTL secondes.
        TTL aligné sur redis_session_ttl pour cohérence.d
        """
        ttl     = settings.redis_session_ttl  
        now     = time.time()
        expired = [
            sid for sid, last_used in self._lock_last_used.items()
            if now - last_used > ttl
        ]

        for sid in expired:
            self._locks.pop(sid, None)
            self._lock_last_used.pop(sid, None)

        if expired:
            logger.info(
                f"_cleanup_old_locks | "
                f"{len(expired)} locks supprimés | "
                f"{len(self._locks)} restants"
            )

        return len(expired)

    # ─────────────────────────────────────────
    #   VIDER LA SESSION
    # ─────────────────────────────────────────

    async def clear_session(self, session_id: str) -> None:
        await self.conversation.clear(session_id)
        self.cancellation.remove(session_id)                # ← NOUVEAU
        self._locks.pop(session_id, None)
        self._lock_last_used.pop(session_id, None)
        logger.info(f"clear_session | session={session_id}")

    # ─────────────────────────────────────────
    #   RÉCUPÉRER L'HISTORIQUE
    # ─────────────────────────────────────────

    async def get_history(self, session_id: str) -> list[dict]:
        return await self.conversation.get_history(session_id)
