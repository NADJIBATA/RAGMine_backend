from backend.shared.logger import setup_logger
from backend.mcp_server.tools.base_tools import BaseTools

logger = setup_logger("time_entry_tools")


class TimeEntryTools(BaseTools):

    # ─────────────────────────────────────────────
    #   LISTER LES ENTREES DE TEMPS
    # ─────────────────────────────────────────────
    async def list_time_entries(
        self,
        project_name: str = None,
        user_name:    str = None,
        issue_id:     int = None,
        activity_id:  int = None,
        from_date:    str = None,
        to_date:      str = None,
        limit:        int = 25,
        offset:       int = 0,
    ) -> dict:
        """
        Retourne les entrées de temps avec filtres optionnels.
        Si limit=-1 → récupère toutes les entrées automatiquement.
        """

     
        project_id = None
        user_id    = None

        # ── Résolution project_id ─────────────────────────────────
        if project_name:
            resolved_project = await self._resolve_project_id(project_name)
            if resolved_project:
                project_id = str(resolved_project)
            else:
                return {
                    "success":      False,
                    "error":        "project_not_found",
                    "message":      f"Projet '{project_name}' introuvable dans Redmine.",
                    "time_entries": [],
                    "total_count":  0,
                }

        # ── Résolution user_id ────────────────────────────────────
        if user_name:
            resolved_user = await self._resolve_user_id(user_name)
            if resolved_user:
                user_id = resolved_user
            else:
                return {
                    "success":      False,
                    "error":        "user_not_found",
                    "message":      f"Utilisateur '{user_name}' introuvable dans Redmine.",
                    "time_entries": [],
                    "total_count":  0,
                }

        params = {}

        if project_id:
            params["project_id"] = project_id
        if user_id:
            params["user_id"] = user_id
        if issue_id:
            params["issue_id"] = issue_id
        if activity_id:
            params["activity_id"] = activity_id
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        logger.info(
            f"list_time_entries | "
            f"project_name={project_name} | project_id={project_id} | "
            f"user_name={user_name} | user_id={user_id} | "
            f"params={params}"
        )

        return await self._get(
            endpoint="/time_entries.json",
            params=  params,
            limit=   limit,
            offset=  offset,
            data_key="time_entries",
        )

    # ─────────────────────────────────────────────
    #   DETAILS D'UNE ENTREE DE TEMPS
    # ─────────────────────────────────────────────

    async def get_time_entry_details(self, time_entry_id: int) -> dict:
        """
        Retourne les détails d'une entrée de temps par son ID.

        Paramètres :
            time_entry_id : ID de l'entrée de temps
        """

        logger.info(f"get_time_entry_details | time_entry_id={time_entry_id}")
        result = await self.client.get(f"/time_entries/{time_entry_id}.json")

        if "error" in result:
            return self._error(result, "time_entry", None)

        return {
            "success":    True,
            "time_entry": result.get("time_entry"),
        }