import os
import base64
import httpx
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from backend.shared.config import get_settings
from backend.shared.logger import setup_logger

settings = get_settings()
logger = setup_logger("redmine_client")



def is_retryable(exception):
    if isinstance(exception, httpx.HTTPStatusError):
        if exception.response.status_code == 429:
            return True
        if exception.response.status_code >= 500:
            return True
        return False
    return True

class RedmineClient:

    def __init__(self, api_key: str = None):
        self.base_url = settings.redmine_url.rstrip("/")
        self.headers = {
            "X-Redmine-API-Key": api_key or settings.redmine_api_key,
            "Content-Type": "application/json",
        }

        # ── SSL ──────────────────────────────
        ssl_cert = settings.redmine_ssl_cert_path

        if ssl_cert:
            cert_path = Path(ssl_cert)
        
            if cert_path.exists():
                self.ssl_verify = str(cert_path)
                logger.info(f"SSL | certificat utilisé : {cert_path}")
            else:
                logger.warning(
                    f"SSL | certificat introuvable : {cert_path} | fallback verify=False"
                )
                self.ssl_verify = False
        else:
            logger.warning(f"SSL | certificat introuvable redmine | fallback verify=False")
            self.ssl_verify = False

    # ─────────────────────────────────────────────
    #   GET
    # ─────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception(is_retryable),
    )
    async def get(self, endpoint: str, params = None) -> dict:
        """
        params : dict ou list of tuples
        httpx accepte les deux nativement pour les clés dupliquées (f[], v[][], etc.)
        """
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"GET {url} | params={params}")

        async with httpx.AsyncClient(
            timeout=30.0,
            verify=self.ssl_verify,
        ) as client:
            response = await client.get(
                url,
                headers=self.headers,
                params=params,   
            )

            if response.status_code == 401:
                logger.error(f"401 Unauthorized | Clé API invalide | {url}")
                return {"error": "401", "message": "Clé API invalide ou expirée"}

            elif response.status_code == 403:
                logger.warning(f"403 Forbidden | Accès refusé | {url}")
                return {"error": "403", "message": "Accès refusé ou module non activé"}

            elif response.status_code == 404:
                logger.warning(f"404 Not Found | Ressource introuvable | {url}")
                return {"error": "404", "message": "Ressource introuvable"}

            elif response.status_code >= 500:
                logger.error(f"{response.status_code} {response.text} | {url}")
                response.raise_for_status()

            logger.debug(f"Response {response.status_code} | {url}")
            return response.json()

    # ─────────────────────────────────────────────
    #   GET PAGINATED
    # ─────────────────────────────────────────────

    async def get_paginated(self, endpoint: str, params = None) -> list:
        """
        params : dict ou list of tuples
        """
        # ── Normalisation → toujours list of tuples ──────────────
        if params is None:
            params = []
        elif isinstance(params, dict):
            params = list(params.items())

        # ── Ajout limit/offset ────────────────────────────────────
        params = [(k, v) for k, v in params if k not in ("limit", "offset")]
        params.append(("limit",  100))
        params.append(("offset", 0))

        all_items = []
        offset    = 0

        while True:
            # Met à jour offset dans la liste
            params = [(k, v) for k, v in params if k != "offset"]
            params.append(("offset", offset))

            data = await self.get(endpoint, params)

            if "error" in data:
                logger.warning(f"get_paginated arrêté | {data['message']}")
                break

            items = None
            for value in data.values():
                if isinstance(value, list):
                    items = value
                    break

            if not items:
                break

            all_items.extend(items)
            total   = data.get("total_count", 0)
            offset += len(items)

            logger.info(f"Récupéré {offset}/{total} éléments depuis {endpoint}")

            if offset >= total or len(items) == 0:
                break

        return all_items


    # ─────────────────────────────────────────────
    #   AUTHENTICATE
    # ─────────────────────────────────────────────

    async def authenticate(self, login: str, password: str) -> dict:
        """
        Authentifie un utilisateur avec son login et password Redmine.
        Retourne ses informations et sa clé API si succès.
        """

        credentials = base64.b64encode(f"{login}:{password}".encode()).decode()

        auth_headers = {
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/users/current.json"
        logger.info(f"authenticate | login={login}")

        async with httpx.AsyncClient(
            timeout=30.0,
            verify=self.ssl_verify,
        ) as client:
            response = await client.get(url, headers=auth_headers)

            if response.status_code == 401:
                logger.warning(f"authenticate | échec | login={login}")
                return {
                    "success": False,
                    "error": "401",
                    "message": "Login ou mot de passe incorrect",
                }

            if response.status_code == 403:
                logger.warning(f"authenticate | accès refusé | login={login}")
                return {
                    "success": False,
                    "error": "403",
                    "message": "Compte désactivé ou accès refusé",
                }

            if response.status_code >= 500:
                logger.error(f"authenticate | erreur serveur | {response.status_code} {response.text}")
                return {
                    "success": False,
                    "error": str(response.status_code),
                    "message": "Erreur serveur Redmine",
                }

            user = response.json()["user"]
            logger.info(f"authenticate | succès | user_id={user['id']} | login={user['login']}")

            return {
                "success": True,
                "user_id": user["id"],
                "login": user["login"],
                "firstname": user["firstname"],
                "lastname": user["lastname"],
                "api_key": user["api_key"],
            }