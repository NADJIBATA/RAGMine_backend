import httpx
from backend.shared.config import get_settings
from backend.shared.logger import setup_logger

settings = get_settings()
logger = setup_logger("mcp_client")


class MCPClient:

    def __init__(self):
        self.base_url = settings.mcp_server_url.rstrip("/")
        self._tools_cache: list[dict] | None = None

    # ─────────────────────────────────────────────
    #   RÉCUPÉRER LES OUTILS
    # ─────────────────────────────────────────────

    async def get_tools(self, force_refresh: bool = False) -> list[dict]:
        """
        Récupère la liste des outils depuis le MCP Server.
        Mise en cache pour éviter les appels répétés.

        Retourne la liste brute des outils MCP.
        """

        if self._tools_cache and not force_refresh:
            logger.debug(f"get_tools | cache | {len(self._tools_cache)} outils")
            return self._tools_cache

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/tools/list")
            response.raise_for_status()
            data = response.json()
        self._tools_cache = data.get("tools", [])
        logger.info(f"get_tools | {len(self._tools_cache)} outils récupérés")
        return self._tools_cache

    # ─────────────────────────────────────────────
    #   APPELER UN OUTIL
    # ─────────────────────────────────────────────

    async def call_tool(
        self,
        tool_name:  str,
        parameters: dict,
        api_key:    str,       
    ) -> dict:

        logger.info(f"call_tool | tool={tool_name} | params={parameters}")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/tools/call",
                json={
                    "tool":       tool_name,
                    "parameters": parameters,
                    "api_key":    api_key,      
                }
            )

            if response.status_code == 404:
                logger.error(f"call_tool | outil introuvable | {tool_name}")
                return {
                    "success": False,
                    "error":   "tool_not_found",
                    "message": f"Outil '{tool_name}' introuvable",
                }

            if response.status_code == 422:
                logger.error(f"call_tool | paramètre manquant | {tool_name}")
                return {
                    "success": False,
                    "error":   "missing_parameter",
                    "message": response.json().get("detail", "Paramètre requis manquant"),
                }

            response.raise_for_status()
            result = response.json()

            logger.info(f"call_tool | succès | tool={tool_name}")
            logger.debug(f"call_tool | résultat={str(result)[:200]}")

        return result.get("result", result)

    # ─────────────────────────────────────────────
    #   FORMATER LES OUTILS POUR LE LLM
    # ─────────────────────────────────────────────

    async def format_tools_for_llm(self, force_refresh: bool = False) -> list[dict]:
        """
        Récupère les outils MCP et les convertit
        au format OpenAI / devstral pour le LLM.

        Format retourné :
        [
            {
                "type": "function",
                "function": {
                    "name": "list_issues",
                    "description": "...",
                    "parameters": {
                        "type": "object",
                        "properties": { ... },
                        "required": [ ... ]
                    }
                }
            },
            ...
        ]
        """

        tools = await self.get_tools(force_refresh=force_refresh)
        formatted = []

        for tool in tools:
            properties = {}
            required   = []

            for param_name, param_config in tool.get("parameters", {}).items():
                properties[param_name] = {
                    "type":        param_config.get("type", "string"),
                    "description": param_config.get("description", ""),
                }
                if param_config.get("required", False):
                    required.append(param_name)

            formatted.append({
                "type": "function",
                "function": {
                    "name":        tool["name"],
                    "description": tool["description"].strip(),
                    "parameters": {
                        "type":       "object",
                        "properties": properties,
                        "required":   required,
                    },
                },
            })

        logger.debug(f"format_tools_for_llm | {len(formatted)} outils formatés")
        return formatted

    # ─────────────────────────────────────────────
    #   VIDER LE CACHE
    # ─────────────────────────────────────────────

    def clear_cache(self):
        self._tools_cache = None
        logger.debug("clear_cache | cache vidé")
