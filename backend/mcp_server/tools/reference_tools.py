import asyncio
from backend.shared.logger import setup_logger
from backend.mcp_server.tools.base_tools import BaseTools

logger = setup_logger("reference_tools")


class ReferenceTools(BaseTools):

    # ─────────────────────────────────────────────
    #   STATUTS DES ISSUES
    # ─────────────────────────────────────────────

    async def get_issue_statuses(self) -> dict:
        """
        Retourne tous les statuts disponibles pour les issues.

        Sortie :
            formatted string avec statuts ouverts/fermés
        """

        logger.info("get_issue_statuses")
        result = await self.client.get("/issue_statuses.json")

        if "error" in result:
            return self._error(result, "issue_statuses", [])

        statuses = result.get("issue_statuses", [])

        open_statuses   = []
        closed_statuses = []

        for s in statuses:
            sid   = s.get("id", "")
            sname = s.get("name", "")
            entry = f"{sid}={sname}"

            if s.get("is_closed", False):
                closed_statuses.append(entry)
            else:
                open_statuses.append(entry)

        open_str   = ", ".join(open_statuses)   if open_statuses   else "aucun"
        closed_str = ", ".join(closed_statuses) if closed_statuses else "aucun"

        formatted = (
            f"STATUTS DISPONIBLES :\n"
            f"open=tous les ouverts\n"
            f"STATUTS OUVERTS (utiliser pour tickets actifs/en cours) : {open_str}\n\n"
            f"closed=tous les fermés\n"
            f"STATUTS FERMÉS (utiliser pour tickets terminés/archivés) : {closed_str}\n\n"
            f"*=tous les statuts"
        )

        return {
            "success":  True,
            "statuses": formatted,
        }

    # ─────────────────────────────────────────────
    #   TRACKERS
    # ─────────────────────────────────────────────

    async def get_trackers(self) -> dict:
        """
        Retourne tous les trackers disponibles.
        ex: Tâche, Anomalie, Feature...

        Sortie :
            formatted string avec id et nom des trackers
        """

        logger.info("get_trackers")
        result = await self.client.get("/trackers.json")

        if "error" in result:
            return self._error(result, "trackers", [])

        trackers = result.get("trackers", [])

        trackers_list = []
        for t in trackers:
            tid   = t.get("id", "")
            tname = t.get("name", "")
            trackers_list.append(f"{tid}={tname}")

        trackers_str = ", ".join(trackers_list) if trackers_list else "aucun"

        formatted = f"TRACKERS DISPONIBLES : {trackers_str}"

        return {
            "success":  True,
            "trackers": formatted,
        }

    # ─────────────────────────────────────────────
    #   PRIORITES
    # ─────────────────────────────────────────────

    async def get_priorities(self) -> dict:
        """
        Retourne toutes les priorités disponibles pour les issues.

        Sortie :
            formatted string avec priorités et indication de la valeur par défaut
        """

        logger.info("get_priorities")
        result = await self.client.get("/enumerations/issue_priorities.json")

        if "error" in result:
            return self._error(result, "issue_priorities", [])

        issue_priorities = result.get("issue_priorities", [])

        priorities  = []
        default_priority = None

        for p in issue_priorities:
            pid        = p.get("id", "")
            pname      = p.get("name", "")
            is_default = p.get("is_default", False)
            entry      = f"{pid}={pname}"

            if is_default:
                default_priority = entry

            priorities.append(entry)

        priorities_str = ", ".join(priorities) if priorities else "aucune"
        default_str    = default_priority      if default_priority else "aucune"

        formatted = (
            f"PRIORITÉS DISPONIBLES : {priorities_str}\n"
            f"Priorité par défaut   : {default_str}"
        )

        return {
            "success":    True,
            "priorities": formatted,
        }


    # ─────────────────────────────────────────────
    #   ACTIVITES (TIME ENTRIES)
    # ─────────────────────────────────────────────

    async def get_time_entry_activities(self) -> dict:
        """
        Retourne toutes les activités disponibles pour les entrées de temps.

        Sortie :
            id, name, is_default, active
        """

        logger.info("get_time_entry_activities")
        result = await self.client.get("/enumerations/time_entry_activities.json")

        if "error" in result:
            return self._error(result, "time_entry_activities", [])

        return {
            "success":                True,
            "time_entry_activities":  result.get("time_entry_activities", []),
        }

    # ─────────────────────────────────────────────
    #   CATEGORIES DE DOCUMENTS
    # ─────────────────────────────────────────────

    async def get_document_categories(self) -> dict:
        """
        Retourne toutes les catégories de documents disponibles.

        Sortie :
            id, name, is_default, active
        """

        logger.info("get_document_categories")
        result = await self.client.get("/enumerations/document_categories.json")

        if "error" in result:
            return self._error(result, "document_categories", [])

        return {
            "success":            True,
            "document_categories": result.get("document_categories", []),
        }

    # ─────────────────────────────────────────────
    #   ROLES
    # ─────────────────────────────────────────────

    async def get_roles(self) -> dict:
        """
        Retourne tous les rôles disponibles dans Redmine.

        Sortie :
            id, name, assignable
        """

        logger.info("get_roles")
        result = await self.client.get("/roles.json")

        if "error" in result:
            return self._error(result, "roles", [])

        return {
            "success": True,
            "roles":   result.get("roles", []),
        }

    # ─────────────────────────────────────────────
    #   GROUPES
    # ─────────────────────────────────────────────

    async def get_groups(self) -> dict:
        """
        Retourne tous les groupes disponibles dans Redmine.
        Nécessite des droits administrateur.

        Sortie :
            id, name
        """

        logger.info("get_groups")
        result = await self.client.get("/groups.json")

        if "error" in result:
            return self._error(result, "groups", [])

        return {
            "success": True,
            "groups":  result.get("groups", []),
        }

    # ─────────────────────────────────────────────
    #   CHAMPS PERSONNALISES
    # ─────────────────────────────────────────────

    async def get_custom_fields(self) -> dict:
        """
        Retourne tous les champs personnalisés disponibles.
        Nécessite des droits administrateur.
    
        Sortie :
            formatted string avec id et nom des champs personnalisés
        """
    
        logger.info("get_custom_fields")
        result = await self.client.get("/custom_fields.json")
    
        if "error" in result:
            return self._error(result, "custom_fields", [])
    
        custom_fields = result.get("custom_fields", [])
    
        # ── Grouper par customized_type ───────────────────────────
        grouped: dict[str, list[str]] = {}
    
        for cf in custom_fields:
            cid           = cf.get("id", "")
            cname         = cf.get("name", "")
            customized_type = cf.get("customized_type", "other")
            entry         = f"{cid}={cname}"
    
            if customized_type not in grouped:
                grouped[customized_type] = []
    
            grouped[customized_type].append(entry)
    
        # ── Formater ──────────────────────────────────────────────
        lines = ["CHAMPS PERSONNALISÉS DISPONIBLES :"]
    
        for ctype, entries in grouped.items():
            entries_str = ", ".join(entries)
            lines.append(f"{ctype} : {entries_str}")
    
        formatted = "\n".join(lines)
    
        return {
            "success":       True,
            "custom_fields": formatted,
        }
    
    # ─────────────────────────────────────────────
    #   TOUTES LES REFERENCES EN UNE FOIS
    # ─────────────────────────────────────────────

    async def get_all_references(self) -> dict:
        """
        Retourne toutes les références en un seul appel.
        Utilise asyncio.gather pour paralléliser les appels.
        Utile pour initialiser le contexte du LLM.
        """

        logger.info("get_all_references")

        # ── Appels parallèles ─────────────────────
        (
            statuses,
            trackers,
            priorities,
            activities,
            roles,
            groups,
            custom_fields,
            document_categories,
        ) = await asyncio.gather(
            self.get_issue_statuses(),
            self.get_trackers(),
            self.get_priorities(),
            self.get_time_entry_activities(),
            self.get_roles(),
            self.get_groups(),
            self.get_custom_fields(),
            self.get_document_categories(),
        )

        return {
            "success":               True,
            "issue_statuses":        statuses.get("issue_statuses", []),
            "trackers":              trackers.get("trackers", []),
            "issue_priorities":      priorities.get("issue_priorities", []),
            "time_entry_activities": activities.get("time_entry_activities", []),
            "roles":                 roles.get("roles", []),
            "groups":                groups.get("groups", []),
            "custom_fields":         custom_fields.get("custom_fields", []),
            "document_categories":   document_categories.get("document_categories", []),
        }
