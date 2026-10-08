from backend.shared.logger import setup_logger
from backend.mcp_server.tools.base_tools import BaseTools

logger = setup_logger("search_tools")

# ── Limite stricte pour search ────────────
SEARCH_MAX_LIMIT = 500


class SearchTools(BaseTools):

    async def search(
        self,
        query:        str,
        project_name: str  = None,
        issues:       bool = True,
        news:         bool = True,
        documents:    bool = True,
        changesets:   bool = True,
        wiki_pages:   bool = True,
        messages:     bool = True,
        projects:     bool = True,
        limit:        int  = 25,
        offset:       int  = 0,
    ) -> dict:
        """
        limit est forcé à 25 maximum pour éviter les payloads trop volumineux.
        Si project_name est fourni, utilise /projects/:project_id/search.json
        Sinon, utilise /search.json avec scope=all.
        """

        # ── Limite stricte ────────────────────────────────────────
        if limit == -1 or limit > SEARCH_MAX_LIMIT:
            logger.warning(
                f"search | limit={limit} non autorisé | forcé à {SEARCH_MAX_LIMIT}"
            )
            limit = SEARCH_MAX_LIMIT

        # ── Résolution project_id ─────────────────────────────────
        project_id = None

        if project_name:
            resolved_project = await self._resolve_project_id(project_name)
            if resolved_project:
                project_id = str(resolved_project)
            else:
                return {
                    "success":     False,
                    "error":       "project_not_found",
                    "message":     f"Projet '{project_name}' introuvable dans Redmine.",
                    "results":     [],
                    "total_count": 0,
                }

        # ── Params communs ────────────────────────────────────────
        params = {
            "q": query,
        }

        # ── Types de contenu ──────────────────────────────────────
        if issues:
            params["issues"] = 1
        if news:
            params["news"] = 1
        if documents:
            params["documents"] = 1
        if changesets:
            params["changesets"] = 1
        if wiki_pages:
            params["wiki_pages"] = 1
        if messages:
            params["messages"] = 1
        if projects:
            params["projects"] = 1

        # ── Endpoint & scope selon project_id ─────────────────────
        if project_id:
            endpoint            = f"/projects/{project_id}/search.json"
            params["scope"]     = "subprojects"
        else:
            endpoint            = "/search.json"
            params["scope"]     = "all"

        logger.info(
            f"search | query='{query}' | "
            f"project_name={project_name} | project_id={project_id} | "
            f"endpoint={endpoint}"
        )

        return await self._get(
            endpoint=endpoint,
            params=  params,
            limit=   limit,
            offset=  offset,
            data_key="results",
        )