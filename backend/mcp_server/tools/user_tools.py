from backend.shared.logger import setup_logger
from backend.mcp_server.tools.base_tools import BaseTools

logger = setup_logger("user_tools")


class UserTools(BaseTools):

    # ─────────────────────────────────────────────
    #   DETAILS D'UN UTILISATEUR
    # ─────────────────────────────────────────────

    async def get_user_details(
        self,
        user_name:           str,
        include_memberships: bool = True,
        include_groups:      bool = True,
    ) -> dict:
        """
        Retourne les détails d'un utilisateur par son nom.
        Note : api_key NON exposée pour des raisons de sécurité.
        """

        # ── Résolution user_id ────────────────────────────────────
        resolved_user = await self._resolve_user_id(user_name)
        if not resolved_user:
            return {
                "success": False,
                "error":   "user_not_found",
                "message": f"Utilisateur '{user_name}' introuvable dans Redmine.",
                "user":    None,
            }
        user_id = resolved_user

        includes = []
        if include_memberships:
            includes.append("memberships")
        if include_groups:
            includes.append("groups")

        params = {}
        if includes:
            params["include"] = ",".join(includes)

        logger.info(
            f"get_user_details | user_name={user_name} | "
            f"user_id={user_id} | include={params.get('include')}"
        )

        result = await self.client.get(f"/users/{user_id}.json", params)

        if "error" in result:
            return self._error(result, "user", None)

        if "user" in result:
            return {
                "success": True,
                "user":    self._sanitize_user(result["user"]),
            }

        logger.error(f"get_user_details | réponse inattendue | {result}")
        return {
            "success": False,
            "error":   "unexpected_response",
            "message": "Réponse inattendue de Redmine",
            "user":    None,
        }

    # ─────────────────────────────────────────────
    #   LISTER LES UTILISATEURS
    # ─────────────────────────────────────────────

    async def list_users(
        self,
        status:   int = 1,
        name:     str = None,
        group_id: int = None,
        limit:    int = 25,
        offset:   int = 0,
    ) -> dict:
        """
        Retourne la liste des utilisateurs.
        Nécessite des droits administrateur.
        Si limit=-1 → récupère tous les utilisateurs automatiquement.
        """

        params = {
            "status": status,
        }

        if name:
            params["name"] = name
        if group_id:
            params["group_id"] = group_id

        logger.info(f"list_users | params={params}")

        return await self._get(
            endpoint="/users.json",
            params=params,
            limit=limit,
            offset=offset,
            data_key="users",
        )

    # ─────────────────────────────────────────────
    #   MON COMPTE
    # ─────────────────────────────────────────────

    async def get_my_account(self) -> dict:
        """
        Retourne les informations complètes du compte courant.
        Note : api_key est masquée pour des raisons de sécurité.
        """

        logger.info("get_my_account")
        result = await self.client.get("/my/account.json")

        # ── Erreur retournée par le client ────────
        if "error" in result:
            return self._error(result, "user", None)

        # ── Succès ────────────────────────────────
        if "user" in result:
            return {
                "success": True,
                "user":    self._sanitize_user(result["user"]),
            }

        # ── Réponse inattendue ────────────────────
        logger.error(f"get_my_account | réponse inattendue | {result}")
        return {
            "success": False,
            "error":   "unexpected_response",
            "message": "Réponse inattendue de Redmine",
            "user":    None,
        }