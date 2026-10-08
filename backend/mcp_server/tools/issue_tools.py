from backend.shared.logger import setup_logger
from backend.mcp_server.tools.base_tools import BaseTools
import unicodedata


logger = setup_logger("issue_tools")


class IssueTools(BaseTools):

    # ─────────────────────────────────────────────
    #   LISTER LES ISSUES
    # ─────────────────────────────────────────────

    async def list_issues(
        self,
        project_name:       str = None,
        status_id:          str = "*",
        assigned_to_name:   str = None,
        author_name:        str = None,      # ← NOUVEAU
        category_name:      str = None,      # ← NOUVEAU
        subject:            str = None,      # ← NOUVEAU
        tracker_id:         str = None,
        parent_id:          int = None,
        priority_id:        str = None,
        issue_id:           str = None,
        fixed_version_name: str = None,
        custom_field_id:    int = None,
        custom_field_value: str = None,
        subproject_id:      str = None,
        created_on:         str = None,
        created_from:       str = None,
        created_to:         str = None,
        updated_on:         str = None,
        updated_from:       str = None,
        updated_to:         str = None,
        sort:               str = "updated_on:desc",
        include:            str = None,
        limit:              int = 25,
        offset:             int = 0,
    ) -> dict:
        """
        Retourne une liste d'issues avec filtres optionnels.
        Si limit=-1 → récupère toutes les issues automatiquement.
        """

        project_id     = None
        assigned_to_id = None
        author_id      = None      # ← NOUVEAU
        category_id    = None      # ← NOUVEAU
        subject_filter = subject
        # ── Résolution project_id ─────────────────────────────
        if project_name:
            resolved_project = await self._resolve_project_id(project_name)
            if resolved_project:
                project_id = str(resolved_project)
            else:
                return {
                    "success":     False,
                    "error":       "project_not_found",
                    "message":     f"Projet '{project_name}' introuvable dans Redmine.",
                    "issues":      [],
                    "total_count": 0,
                }

        # ── Résolution assigned_to_id ─────────────────────────
        if assigned_to_name:
            resolved_user = await self._resolve_user_id(assigned_to_name)
            if resolved_user:
                assigned_to_id = str(resolved_user)
            else:
                return {
                    "success":     False,
                    "error":       "user_not_found",
                    "message":     f"Utilisateur '{assigned_to_name}' introuvable dans Redmine.",
                    "issues":      [],
                    "total_count": 0,
                }

        # ── Résolution author_id ──────────────────────────────  ← NOUVEAU
        if author_name:
            resolved_author = await self._resolve_user_id(author_name)
            if resolved_author:
                author_id = str(resolved_author)
            else:
                return {
                    "success":     False,
                    "error":       "author_not_found",
                    "message":     f"Auteur '{author_name}' introuvable dans Redmine.",
                    "issues":      [],
                    "total_count": 0,
                }

        # ── Résolution category_id ────────────────────────────  ← NOUVEAU
        if category_name:
            if not project_id:
                logger.warning(
                    "list_issues | category_name fourni sans project_id | ignoré"
                )
            else:
                resolved_category = await self._resolve_category_id(
                    project_id=    project_id,
                    category_name= category_name,
                )
                if resolved_category:
                    category_id = str(resolved_category)
                else:
                    return {
                        "success":     False,
                        "error":       "category_not_found",
                        "message":     f"Catégorie '{category_name}' introuvable dans le projet '{project_id}'.",
                        "issues":      [],
                        "total_count": 0,
                    }

        params = [
            ("sort",      sort),
        ]

        if status_id == "*":
            params.append(("f[]",            "status_id"))
            params.append(("op[status_id]",  "*"))         
        elif status_id == "open":
            params.append(("f[]",            "status_id"))
            params.append(("op[status_id]",  "o"))          
        elif status_id == "closed":
            params.append(("f[]",            "status_id"))
            params.append(("op[status_id]",  "c"))          
        else:
            # ID numérique spécifique ex: "2"
            params.append(("f[]",            "status_id"))
            params.append(("op[status_id]",  "="))
            params.append(("v[status_id][]", status_id))

        # ── Filtres simples ───────────────────────────────────────
        if project_id:
            params.append(("project_id", project_id))
        if tracker_id:
            params.append(("tracker_id", tracker_id))
        if parent_id:
            params.append(("parent_id", parent_id))
        if issue_id:
            params.append(("issue_id", issue_id))
        if priority_id:
            params.append(("priority_id", priority_id))
        if subproject_id:
            params.append(("subproject_id", subproject_id))
        if include:
            params.append(("include", include))
        if custom_field_id and custom_field_value:
            params.append((f"cf_{custom_field_id}", custom_field_value))

        # ── Filtres avancés f[]/op[]/v[][] ────────────────────────
        # assigned_to
        if assigned_to_id:
            params.append(("f[]",                  "assigned_to_id"))
            params.append(("op[assigned_to_id]",   "="))
            params.append(("v[assigned_to_id][]",  assigned_to_id))

        # author
        if author_id:
            params.append(("f[]",              "author_id"))
            params.append(("op[author_id]",    "="))
            params.append(("v[author_id][]",   author_id))

        # category
        if category_id:
            params.append(("f[]",                "category_id"))
            params.append(("op[category_id]",    "="))
            params.append(("v[category_id][]",   category_id))

        ## subject
        #if subject:
        #    params.append(("f[]",            "subject"))
        #    params.append(("op[subject]",    "~"))
        #    params.append(("v[subject][]",   subject))
        
       
        # ── Dates ─────────────────────────────────────────────────
        self._apply_dates(
            params,              # ← directement la liste de tuples
            created_on=   created_on,
            created_from= created_from,
            created_to=   created_to,
            updated_on=   updated_on,
            updated_from= updated_from,
            updated_to=   updated_to,
        )

        # ── Version ───────────────────────────────────────────────
        if fixed_version_name:
            if not project_id:
                logger.warning("list_issues | fixed_version_name fourni sans project_id | ignoré")
            else:
                version_id = await self._resolve_version_id(
                    project_id=         project_id,
                    fixed_version_name= fixed_version_name,
                )
                if version_id:
                    params.append(("fixed_version_id", version_id))
                else:
                    return {
                        "success":     False,
                        "error":       "version_not_found",
                        "message":     f"Version '{fixed_version_name}' introuvable dans le projet '{project_id}'.",
                        "issues":      [],
                        "total_count": 0,
                    }

        logger.info(f"list_issues | params={params}")

        result  = await self._get(
            endpoint= "/issues.json",
            params=   params,
            limit=    limit,
            offset=   offset,
            data_key= "issues",
        )
        if subject_filter and result.get("success"):
            before = len(result["issues"])

            result["issues"] = [
                issue for issue in result["issues"]
                if self._match_subject(
                    issue_subject= issue.get("subject", ""),
                    query=         subject_filter,
                    threshold=     80,
                )
            ]

            after = len(result["issues"])
            result["total_count"] = after

            logger.info(
                f"list_issues | filtrage subject local "
                f"| query='{subject_filter}' "
                f"| {before} → {after} issues"
            )

        return result
        # ─────────────────────────────────────────────
        #   LISTER LES ISSUES EN RETARD
        # ─────────────────────────────────────────────

    async def list_issues_overdue(
        self,
        project_name:       str = None,
        status_id:          str = "open",
        assigned_to_name:   str = None,
        tracker_id:         str = None,
        parent_id:          int = None,
        priority_id:        str = None,
        issue_id:           str = None,
        fixed_version_name: str = None,
        custom_field_id:    int = None,
        custom_field_value: str = None,
        subproject_id:      str = None,
        created_on:         str = None,
        created_from:       str = None,
        created_to:         str = None,
        updated_on:         str = None,
        updated_from:       str = None,
        updated_to:         str = None,
        sort:               str = "due_date:asc",
        include:            str = None,
        limit:              int = 25,
        offset:             int = 0,
        reference_date:     str = None,
    ) -> dict:
        """
        Retourne les issues en retard :
          - Tickets dont due_date <= ref_date
          - Tickets dont la version cible a une due_date <= ref_date
        Si limit=-1 → récupère toutes les issues automatiquement.
        """
        from datetime import date

        ref_date     = date.fromisoformat(reference_date) if reference_date else date.today()
        ref_date_str = ref_date.isoformat()

 
        project_id     = None
        assigned_to_id = None

        # ── Résolution project_id ─────────────────────────────
        if project_name:
            resolved_project = await self._resolve_project_id(project_name)
            if resolved_project:
                project_id = str(resolved_project)
            else:
                return {
                    "success":     False,
                    "error":       "project_not_found",
                    "message":     f"Projet '{project_name}' introuvable dans Redmine.",
                    "issues":      [],
                    "total_count": 0,
                }

        # ── Résolution assigned_to_id ─────────────────────────
        if assigned_to_name:
            resolved_user = await self._resolve_user_id(assigned_to_name)
            if resolved_user:
                assigned_to_id = str(resolved_user)
            else:
                return {
                    "success":     False,
                    "error":       "user_not_found",
                    "message":     f"Utilisateur '{assigned_to_name}' introuvable dans Redmine.",
                    "issues":      [],
                    "total_count": 0,
                }

        base_params = {
            "status_id": status_id,
            "sort":      sort,
        }

        if project_id:
            base_params["project_id"] = project_id
        if assigned_to_id:
            base_params["assigned_to_id"] = assigned_to_id
        if tracker_id:
            base_params["tracker_id"] = tracker_id
        if parent_id:
            base_params["parent_id"] = parent_id
        if issue_id:
            base_params["issue_id"] = issue_id
        if priority_id:
            base_params["priority_id"] = priority_id
        if subproject_id:
            base_params["subproject_id"] = subproject_id
        if include:
            base_params["include"] = include
        if custom_field_id and custom_field_value:
            base_params[f"cf_{custom_field_id}"] = custom_field_value

        # ── Résolution version ────────────────────────────────
        if fixed_version_name:
            if not project_id:
                logger.warning(
                    "list_issues_overdue | fixed_version_name fourni sans project_id | ignoré"
                )
            else:
                version_id = await self._resolve_version_id(
                    project_id=        project_id,
                    fixed_version_name=fixed_version_name,
                )
                if version_id:
                    base_params["fixed_version_id"] = version_id
                else:
                    return {
                        "success":     False,
                        "error":       "version_not_found",
                        "message":     f"Version '{fixed_version_name}' introuvable dans le projet '{project_id}'.",
                        "issues":      [],
                        "total_count": 0,
                    }

        self._apply_dates(
            base_params,
            created_on=created_on,
            created_from=created_from,
            created_to=created_to,
            updated_on=updated_on,
            updated_from=updated_from,
            updated_to=updated_to,
        )

        logger.info(f"list_issues_overdue | ref_date={ref_date_str} | params={base_params}")

        # ── 1. Tickets avec due_date dépassée ─────────────────
        params_due = {**base_params, "due_date": f"<={ref_date_str}"}
        result_due = await self._get(
            endpoint="/issues.json",
            params=  params_due,
            limit=   limit,
            offset=  offset,
            data_key="issues",
        )
        issues_by_due_date = result_due.get("issues", []) if result_due.get("success") else []

        logger.info(f"list_issues_overdue | due_date | {len(issues_by_due_date)} tickets")

        # ── 2. Tickets via versions en retard ─────────────────
        issues_by_version = []

        if project_id:
            versions_result = await self.client.get(
                f"/projects/{project_id}/versions.json", {}
            )
            versions = versions_result.get("versions", [])

            overdue_versions = [
                v for v in versions
                if v.get("due_date") and v["due_date"] <= ref_date_str
            ]

            logger.info(
                f"list_issues_overdue | versions en retard | "
                f"{len(overdue_versions)}/{len(versions)} | project={project_id}"
            )

            for version in overdue_versions:
                vid          = version["id"]
                version_name = version["name"]
                version_due  = version["due_date"]

                # Si une version spécifique est demandée → skip les autres
                if "fixed_version_id" in base_params:
                    if base_params["fixed_version_id"] != vid:
                        continue

                params_version = {**base_params, "fixed_version_id": vid}
                result_version = await self._get(
                    endpoint="/issues.json",
                    params=  params_version,
                    limit=   limit,
                    offset=  offset,
                    data_key="issues",
                )

                if result_version.get("success"):
                    version_issues = result_version.get("issues", [])
                    logger.info(
                        f"list_issues_overdue | version '{version_name}' "
                        f"(due={version_due}) | {len(version_issues)} tickets"
                    )
                    issues_by_version.extend(version_issues)

        # ── 3. Fusion + dédoublonnage ─────────────────────────
        seen_ids   = set()
        all_issues = []

        for issue in issues_by_due_date + issues_by_version:
            iid = issue.get("id")
            if iid not in seen_ids:
                seen_ids.add(iid)
                all_issues.append(issue)

        logger.info(
            f"list_issues_overdue | total | {len(all_issues)} tickets uniques "
            f"(due_date={len(issues_by_due_date)} + version={len(issues_by_version)})"
        )

        return {
            "success":     True,
            "issues":      all_issues,
            "total_count": len(all_issues),
            "offset":      offset,
            "limit":       limit,
        }
    # ─────────────────────────────────────────────
    #   DETAILS D'UNE ISSUE
    # ─────────────────────────────────────────────

    async def get_issue_details(
        self,
        issue_id:                int,
        include_journals:        bool = True,
        include_attachments:     bool = True,
        include_relations:       bool = True,
        include_watchers:        bool = True,
        include_children:        bool = True,
        include_allowed_statuses: bool = False,
    ) -> dict:
        """
        Retourne les détails complets d'une issue par son ID.
        """

        includes = []
        if include_journals:
            includes.append("journals")
        if include_attachments:
            includes.append("attachments")
        if include_relations:
            includes.append("relations")
        if include_watchers:
            includes.append("watchers")
        if include_children:
            includes.append("children")
        if include_allowed_statuses:
            includes.append("allowed_statuses")

        params = {}
        if includes:
            params["include"] = ",".join(includes)

        logger.info(f"get_issue_details | issue_id={issue_id} | include={params.get('include')}")
        result = await self.client.get(f"/issues/{issue_id}.json", params)

        if "error" in result:
            return self._error(result, "issue", None)

        return {
            "success": True,
            "issue":   result["issue"],
        }

    # ─────────────────────────────────────────────
    #   ISSUES PAR QUERY SAUVEGARDEE
    # ─────────────────────────────────────────────

    async def get_issues_by_query(
        self,
        query_id:           int,
        project_name:       str = None,
        assigned_to_name:   str = None,
        priority_id:        str = None,
        tracker_id:         str = None,
        created_on:         str = None,
        created_from:       str = None,
        created_to:         str = None,
        updated_on:         str = None,
        updated_from:       str = None,
        updated_to:         str = None,
        limit:              int = 25,
        offset:             int = 0,
    ) -> dict:
        """
        Exécute une requête sauvegardée dans Redmine.
        Si limit=-1 → récupère tous les résultats automatiquement.
        """

        project_id     = None
        assigned_to_id = None

        # ── Résolution project_id ─────────────────────────────────
        if project_name:
            resolved_project = await self._resolve_project_id(project_name)
            if resolved_project:
                project_id = str(resolved_project)
            else:
                return {
                    "success":     False,
                    "error":       "project_not_found",
                    "message":     f"Projet '{project_name}' introuvable dans Redmine.",
                    "issues":      [],
                    "total_count": 0,
                }

        # ── Résolution assigned_to_id ─────────────────────────────
        if assigned_to_name:
            resolved_user = await self._resolve_user_id(assigned_to_name)
            if resolved_user:
                assigned_to_id = str(resolved_user)
            else:
                return {
                    "success":     False,
                    "error":       "user_not_found",
                    "message":     f"Utilisateur '{assigned_to_name}' introuvable dans Redmine.",
                    "issues":      [],
                    "total_count": 0,
                }

        params = {
            "query_id": query_id,
        }

        if project_id:
            params["project_id"] = project_id
        if assigned_to_id:
            params["assigned_to_id"] = assigned_to_id
        if priority_id:
            params["priority_id"] = priority_id
        if tracker_id:
            params["tracker_id"] = tracker_id

        self._apply_dates(
            params,
            created_on=created_on,
            created_from=created_from,
            created_to=created_to,
            updated_on=updated_on,
            updated_from=updated_from,
            updated_to=updated_to,
        )

        logger.info(f"get_issues_by_query | query_id={query_id} | params={params}")

        return await self._get(
            endpoint="/issues.json",
            params=  params,
            limit=   limit,
            offset=  offset,
            data_key="issues",
        )


    # ─────────────────────────────────────────────
    #   ISSUES PAR CHAMP PERSONNALISE
    # ─────────────────────────────────────────────

    async def get_issues_by_custom_field(
        self,
        custom_field_id:     int,
        custom_field_value:  str,
        project_name:        str = None,
        assigned_to_name:    str = None,
        fixed_version_name:  str = None,
        priority_id:         str = None,
        tracker_id:          str = None,
        status_id:           str = "*",
        created_on:          str = None,
        created_from:        str = None,
        created_to:          str = None,
        updated_on:          str = None,
        updated_from:        str = None,
        updated_to:          str = None,
        limit:               int = 25,
        offset:              int = 0,
    ) -> dict:
        """
        Filtre les issues par valeur d'un champ personnalisé.
        Si limit=-1 → récupère tous les résultats automatiquement.
        """

        
        project_id     = None
        assigned_to_id = None

        # ── Résolution project_id ─────────────────────────────────
        if project_name:
            resolved_project = await self._resolve_project_id(project_name)
            if resolved_project:
                project_id = str(resolved_project)
            else:
                return {
                    "success":     False,
                    "error":       "project_not_found",
                    "message":     f"Projet '{project_name}' introuvable dans Redmine.",
                    "issues":      [],
                    "total_count": 0,
                }

        # ── Résolution assigned_to_id ─────────────────────────────
        if assigned_to_name:
            resolved_user = await self._resolve_user_id(assigned_to_name)
            if resolved_user:
                assigned_to_id = str(resolved_user)
            else:
                return {
                    "success":     False,
                    "error":       "user_not_found",
                    "message":     f"Utilisateur '{assigned_to_name}' introuvable dans Redmine.",
                    "issues":      [],
                    "total_count": 0,
                }

        params = {
            f"cf_{custom_field_id}": custom_field_value,
            "status_id":             status_id,
        }

        if project_id:
            params["project_id"] = project_id
        if assigned_to_id:
            params["assigned_to_id"] = assigned_to_id
        if priority_id:
            params["priority_id"] = priority_id
        if tracker_id:
            params["tracker_id"] = tracker_id

        # ── Résolution version ────────────────────────────────────
        if fixed_version_name:
            if not project_id:
                logger.warning(
                    "get_issues_by_custom_field | fixed_version_name fourni sans project_id | ignoré"
                )
            else:
                version_id = await self._resolve_version_id(
                    project_id=        project_id,
                    fixed_version_name=fixed_version_name,
                )
                if version_id:
                    params["fixed_version_id"] = version_id
                else:
                    return {
                        "success":     False,
                        "error":       "version_not_found",
                        "message":     f"Version '{fixed_version_name}' introuvable dans le projet '{project_id}'.",
                        "issues":      [],
                        "total_count": 0,
                    }

        self._apply_dates(
            params,
            created_on=created_on,
            created_from=created_from,
            created_to=created_to,
            updated_on=updated_on,
            updated_from=updated_from,
            updated_to=updated_to,
        )

        logger.info(
            f"get_issues_by_custom_field | "
            f"cf_{custom_field_id}={custom_field_value} | params={params}"
        )

        return await self._get(
            endpoint="/issues.json",
            params=  params,
            limit=   limit,
            offset=  offset,
            data_key="issues",
        )


    # ─────────────────────────────────────────────
    #   RELATIONS D'UNE ISSUE
    # ─────────────────────────────────────────────

    async def get_issue_relations(self, issue_id: int) -> dict:
        """
        Retourne toutes les relations d'un ticket.
        """

        logger.info(f"get_issue_relations | issue_id={issue_id}")
        result = await self.client.get(f"/issues/{issue_id}/relations.json")

        if "error" in result:
            return self._error(result, "relations", [])

        return {
            "success":   True,
            "relations": result.get("relations", []),
        }

    # ─────────────────────────────────────────────
    #   DETAILS D'UNE RELATION
    # ─────────────────────────────────────────────

    async def get_relation_details(self, relation_id: int) -> dict:
        """
        Retourne les détails d'une relation spécifique.
        """

        logger.info(f"get_relation_details | relation_id={relation_id}")
        result = await self.client.get(f"/relations/{relation_id}.json")

        if "error" in result:
            return self._error(result, "relation", None)

        return {
            "success":  True,
            "relation": result.get("relation"),
        }