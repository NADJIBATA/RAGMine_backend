from backend.shared.redmine_client import RedmineClient
from backend.mcp_server.tools.utils import _apply_date_filter
from backend.shared.logger import setup_logger
from backend.shared.config import get_settings
import unicodedata
import re
from rapidfuzz import fuzz


settings = get_settings()
logger = setup_logger("base_tools")
class BaseTools:

    def __init__(self, api_key: str = None):
        self.client = RedmineClient(api_key=api_key)

    # ─────────────────────────────────────────────
    #   GESTION DES ERREURS
    # ─────────────────────────────────────────────

    def _error(self, result: dict, key: str, default) -> dict:
        """
        Construit une réponse d'erreur standardisée.

        Paramètres :
            result  : réponse brute du client contenant 'error' et 'message'
            key     : clé de la donnée vide à retourner (ex: 'issues', 'projects')
            default : valeur par défaut (ex: [], None, 0)
        """
        return {
            "success": False,
            "error":   result["error"],
            "message": result["message"],
            key:       default,
        }

    # ─────────────────────────────────────────────
    #   GESTION DES DATES
    # ─────────────────────────────────────────────

    def _apply_dates(
        self,
        params,                  # list of tuples
        created_on:   str = None,
        created_from: str = None,
        created_to:   str = None,
        updated_on:   str = None,
        updated_from: str = None,
        updated_to:   str = None,
    ) -> None:
    
        _apply_date_filter(
            params,
            field=     "created_on",
            exact=     created_on,
            from_date= created_from,
            to_date=   created_to,
        )
        _apply_date_filter(
            params,
            field=     "updated_on",
            exact=     updated_on,
            from_date= updated_from,
            to_date=   updated_to,
        )

    # ─────────────────────────────────────────────
    #   PAGINATION
    # ─────────────────────────────────────────────

    async def _get(
        self,
        endpoint: str,
        params,          # dict OU list of tuples
        limit:    int,
        offset:   int,
        data_key: str,
    ) -> dict:

        # ── Normalisation → toujours list of tuples ──────────────
        if isinstance(params, dict):
            params = list(params.items())

        if limit == -1:
            items = await self.client.get_paginated(endpoint, params)
            return {
                "success":     True,
                data_key:      items,
                "total_count": len(items),
                "offset":      0,
                "limit":       -1,
            }

        # ── GET simple ────────────────────────────────────────────
        params = [(k, v) for k, v in params if k not in ("limit", "offset")]
        params.append(("limit",  limit))
        params.append(("offset", offset))

        result = await self.client.get(endpoint, params)

        if "error" in result:
            return self._error(result, data_key, [])

        return {
            "success":     True,
            data_key:      result.get(data_key, []),
            "total_count": result.get("total_count", 0),
            "offset":      result.get("offset", offset),
            "limit":       result.get("limit", limit),
        }


        # ─────────────────────────────────────────────
        #   UTILISATEUR
        # ─────────────────────────────────────────────

    def _sanitize_user(self, user: dict) -> dict:
        """
        Supprime la clé api_key d'un objet utilisateur pour des raisons de sécurité.
        """
        user.pop("api_key", None)
        return user

    
    # ─────────────────────────────────────────────
    #   RÉSOLUTION VERSION
    # ─────────────────────────────────────────────

    async def _resolve_version_id(
        self,
        project_id:         str,
        fixed_version_name: str,
    ) -> int | None:

        if not project_id or not fixed_version_name:
            return None

        result = await self.client.get(
            f"/projects/{project_id}/versions.json",
            {},
        )

        if "error" in result:
            logger.warning(
                f"_resolve_version_id | erreur API | "
                f"project={project_id} | {result.get('message')}"
            )
            return None

        versions = result.get("versions", [])

        if not versions:
            logger.warning(
                f"_resolve_version_id | aucune version | project={project_id}"
            )
            return None

        search = fixed_version_name.strip()

        for v in versions:
            if v.get("name") == search:
                logger.info(f"_resolve_version_id | match exact | '{search}' → id={v['id']}")
                return v["id"]

        for v in versions:
            if v.get("name", "").lower() == search.lower():
                logger.info(f"_resolve_version_id | match casse | '{search}' → id={v['id']}")
                return v["id"]

        for v in versions:
            if search.lower() in v.get("name", "").lower():
                logger.info(f"_resolve_version_id | match partiel | '{search}' → '{v['name']}' id={v['id']}")
                return v["id"]

        logger.warning(
            f"_resolve_version_id | aucun match | '{search}' | "
            f"disponibles : {[v['name'] for v in versions]}"
        )
        return None

    




    async def _resolve_category_id(
        self,
        project_id: str,
        category_name: str,
    ) -> int | None:
    
        if not project_id or not category_name:
            return None
    
        result = await self.client.get(
            f"/projects/{project_id}/issue_categories.json",
            {},
        )
    
        if "error" in result:
            logger.warning(
                f"_resolve_category_id | erreur API | "
                f"project={project_id} | {result.get('message')}"
            )
            return None
        logger.info(f"resolve_category_id | categories={result}")
        categories = result.get("issue_categories", [])
    
        if not categories:
            logger.warning(
                f"_resolve_category_id | aucune category | project={project_id}"
            )
            return None
    
        search = category_name.strip()
        search_lower = search.lower()
    
        for c in categories:
            if c.get("name") == search:
                logger.info(
                    f"_resolve_category_id | match exact | '{search}' → id={c.get('id')}"
                )
                return c.get("id")
    
        for c in categories:
            if c.get("name", "").lower() == search_lower:
                logger.info(
                    f"_resolve_category_id | match casse | '{search}' → id={c.get('id')}"
                )
                return c.get("id")
    
        for c in categories:
            if search_lower in c.get("name", "").lower():
                logger.info(
                    f"_resolve_category_id | match partiel | "
                    f"'{search}' → '{c.get('name')}' id={c.get('id')}"
                )
                return c.get("id")
    
        logger.warning(
            f"_resolve_category_id | aucun match | '{search}' | "
            f"disponibles : {[c.get('name') for c in categories]}"
        )
    
        return None
    
    # ─────────────────────────────────────────────
    #   RÉSOLUTION PROJET
    # ─────────────────────────────────────────────

    async def _resolve_project_id(
        self,
        project_name: str,
    ) -> int | None:

        all_projects = []
        offset       = 0
        limit        =100

        # ── Récupérer TOUS les projets avec pagination ────────────
        while True:
            result = await self.client.get(
                "/projects.json",
                {"status":"","limit": limit, "offset": offset},
            )

            if "error" in result:
                logger.warning(
                    f"_resolve_project_id | erreur API | {result.get('message')}"
                )
                return None

            projects    = result.get("projects", [])
            total_count = result.get("total_count", 0)

            all_projects.extend(projects)

            logger.debug(
                f"_resolve_project_id | page offset={offset} | "
                f"{len(projects)} projets | total={total_count}"
            )

            # Stop si on a tout récupéré
            if len(all_projects) >= total_count or len(projects) < limit:
                break

            offset += limit

        if not all_projects:
            logger.warning(f"_resolve_project_id | aucun projet | project={project_name}")
            return None

        logger.info(
            f"_resolve_project_id | {len(all_projects)} projets chargés | "
            f"recherche '{project_name}'"
        )

        search = project_name.strip()

        # ── 1. Match exact ────────────────────────────────────────
        for p in all_projects:
            if p.get("name") == search:
                logger.info(f"_resolve_project_id | match exact | '{search}' → id={p['id']}")
                return p["id"]

        # ── 2. Match insensible à la casse ────────────────────────
        for p in all_projects:
            if p.get("name", "").lower() == search.lower():
                logger.info(f"_resolve_project_id | match casse | '{search}' → id={p['id']}")
                return p["id"]

        # ── 3. Match partiel (contient) ───────────────────────────
        for p in all_projects:
            if search.lower() in p.get("name", "").lower():
                logger.info(
                    f"_resolve_project_id | match partiel | "
                    f"'{search}' → '{p['name']}' id={p['id']}"
                )
                return p["id"]

        logger.warning(
            f"_resolve_project_id | aucun match | '{search}' | "
            f"{len(all_projects)} projets disponibles"
        )
        return None

    # ─────────────────────────────────────────────
    #   RÉSOLUTION UTILISATEUR
    # ─────────────────────────────────────────────

    def _get_user_full_name(self, user: dict) -> str:
        """
        Construit le nom complet d'un utilisateur.
        Supporte les deux formats Redmine :
          - { "name": "John Doe" }                        ← format liste issues
          - { "firstname": "John", "lastname": "Doe" }    ← format liste users
        """
        if user.get("name"):
            return user["name"]

        firstname = user.get("firstname", "")
        lastname  = user.get("lastname", "")
        return f"{firstname} {lastname}".strip()

    def _normalize(self, text: str) -> str:
        """
        Normalise un texte pour la comparaison :
        - Minuscules
        - Supprime les accents
        - Supprime les espaces superflus
        """
        # Minuscules
        text = text.lower().strip()
        # Supprimer les accents
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        return text


    def _normalize_subject(self, text: str) -> str:
        """
        Normalise un texte pour la comparaison :
        - minuscules
        - supprime accents
        - supprime caractères spéciaux
        - strip
        """
        text = text.lower().strip()
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


    def _match_subject(self, issue_subject: str, query: str, threshold: int = 70) -> bool:
        """
        Retourne True si le sujet de l'issue matche la query.

        Stratégie multi-niveaux (ordre de priorité) :
        1. Contenance exacte (normalisée)          → score 100
        2. Tous les mots de la query présents      → score 90
        3. Fuzzy partiel (partial_ratio)           → score >= threshold
        4. Token set ratio                         → score >= threshold

        Paramètres :
            issue_subject : sujet du ticket Redmine
            query         : texte recherché
            threshold     : score minimum fuzzy (défaut 70)
        """
        if not issue_subject or not query:
            return False

        norm_subject = self._normalize_subject(issue_subject)
        norm_query   = self._normalize_subject(query)

        # ── Niveau 1 : contenance exacte normalisée ───────────────
        if norm_query in norm_subject:
            return True

        # ── Niveau 2 : tous les mots de la query présents ─────────
        query_words   = norm_query.split()
        subject_words = norm_subject.split()

        if all(
            any(qw in sw for sw in subject_words)
            for qw in query_words
        ):
            return True

        # ── Niveau 3 : fuzzy partial ratio ────────────────────────
        if fuzz.partial_ratio(norm_query, norm_subject) >= threshold:
            return True

        # ── Niveau 4 : token set ratio ────────────────────────────
        if fuzz.token_set_ratio(norm_query, norm_subject) >= threshold:
            return True

        return False
        
    async def _resolve_user_id(
        self,
        user_name: str,
    ) -> int | None:

        all_users = []
        offset    = 0
        limit     = 100

        while True:
            # ── Utiliser api_key admin si fournie ─────────────────
            api_key = settings.redmine_api_key
            if api_key:
                from backend.shared.redmine_client import RedmineClient
                temp_client = RedmineClient(api_key=api_key)
                result      = await temp_client.get(
                    "/users.json",
                    {"limit": limit, "offset": offset},
                )
            else:
                result = await self.client.get(
                    "/users.json",
                    {"limit": limit, "offset": offset},
                )

            if "error" in result:
                logger.warning(
                    f"_resolve_user_id | erreur API | {result.get('message')}"
                )
                # ── Fallback : chercher via /users/current.json ───
                # Si pas d'accès à la liste → essayer de matcher
                # via get_project_members si project connu
                return None

            users       = result.get("users", [])
            total_count = result.get("total_count", 0)

            if not users:
                break

            all_users.extend(users)

            logger.debug(
                f"_resolve_user_id | page offset={offset} | "
                f"{len(users)} users | total={total_count}"
            )

            if len(all_users) >= total_count or len(users) < limit:
                break

            offset += limit

        if not all_users:
            logger.warning(f"_resolve_user_id | aucun utilisateur | user={user_name}")
            return None

        logger.info(
            f"_resolve_user_id | {len(all_users)} utilisateurs chargés | "
            f"recherche '{user_name}'"
        )

        search       = user_name.strip()
        search_norm  = self._normalize(search)
        search_words = search_norm.split()

        # ── 1. Match exact ────────────────────────────────────────
        for u in all_users:
            full_name = self._get_user_full_name(u)
            if full_name == search:
                logger.info(f"_resolve_user_id | match exact | '{search}' → id={u['id']}")
                return u["id"]

        # ── 2. Match normalisé exact ──────────────────────────────
        for u in all_users:
            full_name_norm = self._normalize(self._get_user_full_name(u))
            if full_name_norm == search_norm:
                logger.info(f"_resolve_user_id | match normalisé | '{search}' → '{self._get_user_full_name(u)}' id={u['id']}")
                return u["id"]

        # ── 3. Match login exact ──────────────────────────────────
        for u in all_users:
            login_norm = self._normalize(u.get("login", ""))
            if login_norm == search_norm:
                logger.info(f"_resolve_user_id | match login | '{search}' → login={u.get('login')} id={u['id']}")
                return u["id"]

        # ── 4. Match tous les mots ────────────────────────────────
        for u in all_users:
            full_name_norm = self._normalize(self._get_user_full_name(u))
            if all(word in full_name_norm for word in search_words):
                logger.info(
                    f"_resolve_user_id | match mots | "
                    f"'{search}' → '{self._get_user_full_name(u)}' id={u['id']}"
                )
                return u["id"]

        # ── 5. Match partiel (au moins un mot) ────────────────────
        for u in all_users:
            full_name_norm = self._normalize(self._get_user_full_name(u))
            if any(word in full_name_norm for word in search_words):
                logger.info(
                    f"_resolve_user_id | match partiel | "
                    f"'{search}' → '{self._get_user_full_name(u)}' id={u['id']}"
                )
                return u["id"]

        logger.warning(
            f"_resolve_user_id | aucun match | '{search}' | "
            f"disponibles : {[self._get_user_full_name(u) for u in all_users]}"
        )
        return None
