
from backend.shared.logger import setup_logger
from backend.mcp_server.tools.base_tools import BaseTools

logger = setup_logger("project_tools")


class ProjectTools(BaseTools):

    # ─────────────────────────────────────────────
    #   LISTER LES PROJETS
    # ─────────────────────────────────────────────

    async def list_projects(
        self,
        status:                        str = "actif",
        created_on:                    str = None,
        created_from:                  str = None,
        created_to:                    str = None,
        updated_on:                    str = None,
        updated_from:                  str = None,
        updated_to:                    str = None,
        include_trackers:              bool = False,
        include_issue_categories:      bool = False,
        include_enabled_modules:       bool = False,
        include_time_entry_activities: bool = False,
        include_issue_custom_fields:   bool = False,
        limit:                         int = 25,
        offset:                        int = 0,
    ) -> dict:
        """
        Retourne les projets accessibles à l'utilisateur.
        Si limit=-1 → récupère tous les projets automatiquement.
        """

        includes = []
        if include_trackers:
            includes.append("trackers")
        if include_issue_categories:
            includes.append("issue_categories")
        if include_enabled_modules:
            includes.append("enabled_modules")
        if include_time_entry_activities:
            includes.append("time_entry_activities")
        if include_issue_custom_fields:
            includes.append("issue_custom_fields")
        
        match status:
            case "*":
                status_id = ""

            case "fermé":
                status_id =5

            case "actif":
                status_id =1

            case "archivé":
                status_id =9

            case _:
                status_id =1
                
        params = {
            "status":status_id
        }
        if includes:
            params["include"] = ",".join(includes)

        self._apply_dates(
            params,
            
            created_on=created_on,
            created_from=created_from,
            created_to=created_to,
            updated_on=updated_on,
            updated_from=updated_from,
            updated_to=updated_to,
        )

        logger.info(f"list_projects | limit={limit} | offset={offset} | include={params.get('include')}")

        return await self._get(
            endpoint="/projects.json",
            params=params,
            limit=limit,
            offset=offset,
            data_key="projects",
        )

    # ─────────────────────────────────────────────
    #   DETAILS D'UN PROJET
    # ─────────────────────────────────────────────

    async def get_project_details(
        self,
        project_name:                  str,
        include_trackers:              bool = True,
        include_issue_categories:      bool = True,
        include_enabled_modules:       bool = True,
        include_time_entry_activities: bool = False,
        include_issue_custom_fields:   bool = False,
    ) -> dict:
        """
        Retourne les détails complets d'un projet.
        """

        # ── Résolution project_id ─────────────────────────────────
        resolved_project = await self._resolve_project_id(project_name)
        if not resolved_project:
            return {
                "success": False,
                "error":   "project_not_found",
                "message": f"Projet '{project_name}' introuvable dans Redmine.",
                "project": None,
            }
        project_id = str(resolved_project)

        includes = []
        if include_trackers:
            includes.append("trackers")
        if include_issue_categories:
            includes.append("issue_categories")
        if include_enabled_modules:
            includes.append("enabled_modules")
        if include_time_entry_activities:
            includes.append("time_entry_activities")
        if include_issue_custom_fields:
            includes.append("issue_custom_fields")

        params = {}
        if includes:
            params["include"] = ",".join(includes)

        logger.info(
            f"get_project_details | project_id={project_id} | "
            f"include={params.get('include')}"
        )

        result = await self.client.get(f"/projects/{project_id}.json", params)

        if "error" in result:
            return self._error(result, "project", None)

        return {
            "success": True,
            "project": result["project"],
        }

    # ─────────────────────────────────────────────
    #   MEMBRES D'UN PROJET
    # ─────────────────────────────────────────────

    async def get_project_members(
        self,
        project_name: str,
        limit:        int = 25,
        offset:       int = 0,
    ) -> dict:
        """
        Retourne la liste des membres d'un projet avec leurs rôles.
        Si limit=-1 → récupère tous les membres automatiquement.
        """

        # ── Résolution project_id ─────────────────────────────────
        resolved_project = await self._resolve_project_id(project_name)
        if not resolved_project:
            return {
                "success":      False,
                "error":        "project_not_found",
                "message":      f"Projet '{project_name}' introuvable dans Redmine.",
                "memberships":  [],
                "total_count":  0,
            }
        project_id = str(resolved_project)

        logger.info(f"get_project_members | project_id={project_id}")

        return await self._get(
            endpoint=f"/projects/{project_id}/memberships.json",
            params=  {},
            limit=   limit,
            offset=  offset,
            data_key="memberships",
        )

    # ─────────────────────────────────────────────
    #   VERSIONS D'UN PROJET
    # ─────────────────────────────────────────────

    async def get_project_versions(
        self,
        project_name: str,
    ) -> dict:
        """
        Retourne les versions (milestones) d'un projet.
        """

        # ── Résolution project_id ─────────────────────────────────
        resolved_project = await self._resolve_project_id(project_name)
        if not resolved_project:
            return {
                "success":  False,
                "error":    "project_not_found",
                "message":  f"Projet '{project_name}' introuvable dans Redmine.",
                "versions": [],
            }
        project_id = str(resolved_project)

        logger.info(f"get_project_versions | project_id={project_id}")

        result = await self.client.get(
            f"/projects/{project_id}/versions.json", {}
        )

        if "error" in result:
            return self._error(result, "versions", [])

        return {
            "success":  True,
            "versions": result.get("versions", []),
        }



    # ─────────────────────────────────────────────
    #   DETAILS D'UNE VERSION
    # ─────────────────────────────────────────────

    async def get_version_details(
        self,
        project_name: str,
        version_name: str,
    ) -> dict:
        """
        Retourne les détails d'une version spécifique via son nom.
        """

        # ── Résolution project_id ─────────────────────────────────
        resolved_project = await self._resolve_project_id(project_name)
        if not resolved_project:
            return {
                "success": False,
                "error":   "project_not_found",
                "message": f"Projet '{project_name}' introuvable dans Redmine.",
                "version": None,
            }
        project_id = str(resolved_project)

        # ── Résolution version_id ─────────────────────────────────
        resolved_version = await self._resolve_version_id(
            project_id=        project_id,
            fixed_version_name=version_name,
        )
        if not resolved_version:
            return {
                "success": False,
                "error":   "version_not_found",
                "message": f"Version '{version_name}' introuvable dans le projet '{project_name}'.",
                "version": None,
            }

        logger.info(
            f"get_version_details | project_id={project_id} | "
            f"version_id={resolved_version}"
        )

        result = await self.client.get(f"/versions/{resolved_version}.json", {})

        if "error" in result:
            return self._error(result, "version", None)

        return {
            "success": True,
            "version": result.get("version"),
        }



    # ─────────────────────────────────────────────
    #   CATEGORIES D'ISSUES D'UN PROJET
    # ─────────────────────────────────────────────

    
    async def get_project_issue_categories(
        self,
        project_name: str,
    ) -> dict:
        """
        Retourne les catégories d'issues disponibles dans un projet.
        """

        # ── Résolution project_id ─────────────────────────────────
        resolved_project = await self._resolve_project_id(project_name)
        if not resolved_project:
            return {
                "success":          False,
                "error":            "project_not_found",
                "message":          f"Projet '{project_name}' introuvable dans Redmine.",
                "issue_categories": [],
            }
        project_id = str(resolved_project)

        logger.info(f"get_project_issue_categories | project_id={project_id}")

        result = await self.client.get(
            f"/projects/{project_id}/issue_categories.json", {}
        )

        if "error" in result:
            return self._error(result, "issue_categories", [])

        return {
            "success":          True,
            "issue_categories": result.get("issue_categories", []),
        }



    # ─────────────────────────────────────────────
    #   FICHIERS D'UN PROJET
    # ─────────────────────────────────────────────

    async def get_project_files(
        self,
        project_name: str,
    ) -> dict:
        """
        Retourne les fichiers publiés dans un projet.
        """

        # ── Résolution project_id ─────────────────────────────────
        resolved_project = await self._resolve_project_id(project_name)
        if not resolved_project:
            return {
                "success": False,
                "error":   "project_not_found",
                "message": f"Projet '{project_name}' introuvable dans Redmine.",
                "files":   [],
            }
        project_id = str(resolved_project)

        logger.info(f"get_project_files | project_id={project_id}")

        result = await self.client.get(
            f"/projects/{project_id}/files.json", {}
        )

        if "error" in result:
            return self._error(result, "files", [])

        return {
            "success": True,
            "files":   result.get("files", []),
        }



    # ─────────────────────────────────────────────
    #   NEWS D'UN PROJET
    # ─────────────────────────────────────────────

    async def get_project_news(
        self,
        project_name: str,
        limit:        int = 25,
        offset:       int = 0,
    ) -> dict:
        """
        Retourne les actualités d'un projet.
        Si limit=-1 → récupère toutes les actualités automatiquement.
        """
    
        # ── Résolution project_id ─────────────────────────────────
        resolved_project = await self._resolve_project_id(project_name)
        if not resolved_project:
            return {
                "success":     False,
                "error":       "project_not_found",
                "message":     f"Projet '{project_name}' introuvable dans Redmine.",
                "news":        [],
                "total_count": 0,
            }
        project_id = str(resolved_project)
    
        logger.info(f"get_project_news | project_id={project_id}")
    
        return await self._get(
            endpoint=f"/projects/{project_id}/news.json",
            params=  {},
            limit=   limit,
            offset=  offset,
            data_key="news",
        )
    
    # ─────────────────────────────────────────────
    #   DETAILS D'UN MEMBERSHIP
    # ─────────────────────────────────────────────

    async def get_membership_details(self, membership_id: int) -> dict:
        """
        Retourne les détails d'un membership spécifique.
        """

        logger.info(f"get_membership_details | membership_id={membership_id}")
        result = await self.client.get(f"/memberships/{membership_id}.json")

        if "error" in result:
            return self._error(result, "membership", None)

        return {
            "success":    True,
            "membership": result.get("membership"),
        }
