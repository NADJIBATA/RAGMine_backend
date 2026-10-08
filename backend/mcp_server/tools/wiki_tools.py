from backend.shared.logger import setup_logger
from backend.mcp_server.tools.base_tools import BaseTools

logger = setup_logger("wiki_tools")


class WikiTools(BaseTools):

    # ─────────────────────────────────────────────
    #   LISTER LES PAGES WIKI D'UN PROJET
    # ─────────────────────────────────────────────

    async def list_wiki_pages(
        self,
        project_name: str,
    ) -> dict:
        """
        Retourne la liste de toutes les pages wiki d'un projet.
        """

        # ── Résolution project_id ─────────────────────────────────
        resolved_project = await self._resolve_project_id(project_name)
        if not resolved_project:
            return {
                "success":     False,
                "error":       "project_not_found",
                "message":     f"Projet '{project_name}' introuvable dans Redmine.",
                "wiki_pages":  [],
                "total_count": 0,
            }
        project_id = str(resolved_project)

        logger.info(f"list_wiki_pages | project_id={project_id}")

        result = await self.client.get(
            f"/projects/{project_id}/wiki/index.json", {}
        )

        if "error" in result:
            return self._error(result, "wiki_pages", [])

        return {
            "success":     True,
            "wiki_pages":  result.get("wiki_pages", []),
            "total_count": len(result.get("wiki_pages", [])),
        }


    # ─────────────────────────────────────────────
    #   DETAILS D'UNE PAGE WIKI
    # ─────────────────────────────────────────────

    async def get_wiki_page(
        self,
        project_name:        str,
        page_title:          str,
        version:             int  = None,
        include_attachments: bool = True,
    ) -> dict:
        """
        Retourne le contenu complet d'une page wiki.
        Si version est fourni → retourne cette version spécifique.
        """

        # ── Résolution project_id ─────────────────────────────────
        resolved_project = await self._resolve_project_id(project_name)
        if not resolved_project:
            return {
                "success":   False,
                "error":     "project_not_found",
                "message":   f"Projet '{project_name}' introuvable dans Redmine.",
                "wiki_page": None,
            }
        project_id = str(resolved_project)

        params = {}
        if include_attachments:
            params["include"] = "attachments"

        # ── Endpoint selon version ────────────────────────────────
        if version is not None:
            endpoint = f"/projects/{project_id}/wiki/{page_title}/{version}.json"
            logger.info(
                f"get_wiki_page | project_id={project_id} | "
                f"page={page_title} | version={version}"
            )
        else:
            endpoint = f"/projects/{project_id}/wiki/{page_title}.json"
            logger.info(
                f"get_wiki_page | project_id={project_id} | "
                f"page={page_title} | version=actuelle"
            )

        result = await self.client.get(endpoint, params)

        if "error" in result:
            return self._error(result, "wiki_page", None)

        return {
            "success":   True,
            "wiki_page": result.get("wiki_page"),
        }


    # ─────────────────────────────────────────────
    #   TOUTES LES PAGES WIKI AVEC CONTENU
    # ─────────────────────────────────────────────

    async def get_all_wiki_pages_with_content(
        self,
        project_name: str,
    ) -> dict:
        """
        Retourne toutes les pages wiki avec leur contenu complet.
        Utile pour l'ingestion RAG.
        """

        # ── Résolution project_id ─────────────────────────────────
        resolved_project = await self._resolve_project_id(project_name)
        if not resolved_project:
            return {
                "success":    False,
                "error":      "project_not_found",
                "message":    f"Projet '{project_name}' introuvable dans Redmine.",
                "wiki_pages": [],
            }
        project_id = str(resolved_project)

        logger.info(f"get_all_wiki_pages_with_content | project_id={project_id}")

        # ── Étape 1 : récupérer la liste des pages ────────────────
        index = await self.list_wiki_pages(project_name)

        if not index["success"]:
            return {
                "success":    False,
                "error":      index.get("error"),
                "message":    index.get("message"),
                "wiki_pages": [],
            }

        pages_with_content = []

        # ── Étape 2 : récupérer le contenu de chaque page ─────────
        for page in index["wiki_pages"]:
            page_result = await self.get_wiki_page(
                project_name=project_name,
                page_title=  page["title"],
            )

            if page_result["success"]:
                pages_with_content.append(page_result["wiki_page"])
                logger.info(f"get_all_wiki_pages_with_content | page récupérée | {page['title']}")
            else:
                logger.warning(
                    f"get_all_wiki_pages_with_content | page ignorée | "
                    f"{page['title']} | {page_result.get('message')}"
                )

        return {
            "success":     True,
            "wiki_pages":  pages_with_content,
            "total_count": len(pages_with_content),
        }