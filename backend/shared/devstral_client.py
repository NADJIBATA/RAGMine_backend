import json
import asyncio
import httpx
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from backend.shared.config import get_settings
from backend.shared.logger import setup_logger

settings         = get_settings()
logger           = setup_logger("devstral_client")


def _is_retryable(exception) -> bool:
    # Ne pas retry si annulé
    if isinstance(exception, asyncio.CancelledError):
        return False
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code >= 500
    return True


class DevstralClient:

    def __init__(self):
        self.base_url = settings.vllm_base_url.rstrip("/")
        self.model    = settings.vllm_model_name
        self.headers  = {"Content-Type": "application/json"}

        if settings.vllm_api_key:
            self.headers["Authorization"] = f"Bearer {settings.vllm_api_key}"

        logger.info(
            f"DevstralClient | model={self.model} | "
            f"url={self.base_url}"
        )

    def get_ssl_verify(self) -> str | bool:
        ssl_cert = settings.scaleway_ssl_cert_path
        if ssl_cert:
            cert_path = Path(ssl_cert)
            if cert_path.exists():
                logger.info(f"SSL | certificat utilisé : {cert_path}")
                return str(cert_path)
            logger.warning(f"SSL | certificat introuvable : {cert_path}")
        else:
            logger.warning("SSL | certificat non configuré")
        return False

    # ─────────────────────────────────────────────
    #   EXÉCUTION HTTP ANNULABLE
    # ─────────────────────────────────────────────

    async def _post_cancellable(
        self,
        payload:        dict,
        cancel_event:   asyncio.Event | None = None,
        read_timeout:   float = 180.0,
    ) -> dict:
        """
        Effectue le POST vers vLLM.
        Si cancel_event est déclenché pendant l'attente → annule proprement.
        """
        ssl_verify = self.get_ssl_verify()

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=read_timeout,
                write=30.0,
                pool=5.0,
            ),
            verify=ssl_verify,
        ) as client:

            # ── Créer la tâche HTTP ────────────────────────────────
            http_task = asyncio.create_task(
                client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                )
            )

            # ── Si pas d'event d'annulation : attente simple ───────
            if cancel_event is None:
                response = await http_task
                response.raise_for_status()
                return response.json()

            # ── Surveiller l'annulation en parallèle ──────────────
            async def _watch_cancel() -> None:
                while not http_task.done():
                    await asyncio.sleep(0.3)
                    if cancel_event.is_set():
                        http_task.cancel()
                        logger.info("_post_cancellable | annulation détectée → tâche HTTP annulée")
                        return

            watch_task = asyncio.create_task(_watch_cancel())

            try:
                response = await http_task

            except asyncio.CancelledError:
               
                logger.info("_post_cancellable | CancelledError reçu")
                raise 

            finally:
                
                watch_task.cancel()
                try:
                    await watch_task
                except asyncio.CancelledError:
                    pass

            response.raise_for_status()
            return response.json()

    # ─────────────────────────────────────────────
    #   CHAT SIMPLE
    # ─────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_retryable),
    )
    async def chat(
        self,
        messages:     list[dict],
        temperature:  float = 0.5,
        cancel_event: asyncio.Event | None = None,
        reasoning_effort: str | None = None
    ) -> str:
        effort = reasoning_effort or settings.reasoning_effort
        logger.debug(f"reasoning_effort = ${effort}")
        payload = {
            "model":            self.model,
            "messages":         messages,
            "temperature":      temperature,
        }
        if settings.llm_supports_reasoning_effort:
            payload["reasoning_effort"] = effort

        logger.debug(
            f"chat | messages={len(messages)} | "
            f"reasoning_effort={effort}"
        )

        try:
            data   = await self._post_cancellable(payload, cancel_event, read_timeout=120.0)
            result = data["choices"][0]["message"]["content"]
            logger.debug(f"chat | réponse | {len(result)} caractères")
            return result

        except asyncio.CancelledError:
            logger.info("chat | annulé par l'utilisateur")
            raise

    # ─────────────────────────────────────────────
    #   CHAT AVEC OUTILS
    # ─────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_retryable),
    )
    async def chat_with_tools(
        self,
        messages:     list[dict],
        tools:        list[dict],
        temperature:  float = 0.1,
        cancel_event: asyncio.Event | None = None, 
        reasoning_effort: str | None = None,          
    ) -> dict:
        
        
        effort = reasoning_effort or settings.reasoning_effort
        payload = {
            "model":            self.model,
            "messages":         messages,
            "tools":            tools,
            "tool_choice":      "auto",
            "temperature":      temperature,
        }
        if settings.llm_supports_reasoning_effort:
            payload["reasoning_effort"] = effort

        logger.debug(
            f"chat_with_tools | messages={len(messages)} | "
            f"tools={len(tools)} | "
            f"reasoning_effort={effort}"
        )

        try:
            data = await self._post_cancellable(payload, cancel_event, read_timeout=180.0)

        except asyncio.CancelledError:
            logger.info("chat_with_tools | annulé par l'utilisateur")
            raise  # ← remontera jusqu'à _process → CancelledByUser

        # ── Tokens ────────────────────────────────────────────────
        usage         = data.get("usage", {})
        tokens_input  = usage.get("prompt_tokens",     0)
        tokens_output = usage.get("completion_tokens", 0)
        tokens_total  = usage.get("total_tokens",      0)

        logger.info(
            f"chat_with_tools | tokens | "
            f"input={tokens_input} | "
            f"output={tokens_output} | "
            f"total={tokens_total}"
        )

        message = data["choices"][0]["message"]
        content = message.get("content") or ""

        logger.debug(f"chat_with_tools | réponse brute | {content[:200]}")

        # ── Cas 1 : format OpenAI standard ────────────────────────
        if message.get("tool_calls"):
            tool_calls = []
            for tc in message["tool_calls"]:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append({
                    "name":      tc["function"]["name"],
                    "arguments": args,
                })
            tool_names = [tc["name"] for tc in tool_calls]
            raw        = content if content.strip() else f"[tool_call: {', '.join(tool_names)}]"
            logger.info(f"chat_with_tools | tool_calls OpenAI | {tool_names}")
            return {
                "type":          "tool_call",
                "tool_calls":    tool_calls,
                "content":       None,
                "raw":           raw,
                "tokens_input":  tokens_input,
                "tokens_output": tokens_output,
                "tokens_total":  tokens_total,
            }

        # ── Cas 2 : format devstral [TOOL_CALLS] ──────────────────
        tool_calls = self._parse_devstral_tool_calls(content)
        if tool_calls:
            tool_names = [tc["name"] for tc in tool_calls]
            raw        = content if content.strip() else f"[tool_call: {', '.join(tool_names)}]"
            logger.info(f"chat_with_tools | tool_calls devstral | {tool_names}")
            return {
                "type":          "tool_call",
                "tool_calls":    tool_calls,
                "content":       None,
                "raw":           raw,
                "tokens_input":  tokens_input,
                "tokens_output": tokens_output,
                "tokens_total":  tokens_total,
            }

        # ── Cas 3 : réponse finale ─────────────────────────────────
        logger.info(f"chat_with_tools | réponse finale | {len(content)} chars")
        return {
            "type":          "final_answer",
            "tool_calls":    None,
            "content":       content,
            "raw":           content,
            "tokens_input":  tokens_input,
            "tokens_output": tokens_output,
            "tokens_total":  tokens_total,
        }

    # ─────────────────────────────────────────────
    #   PARSER FORMAT DEVSTRAL
    # ─────────────────────────────────────────────

    def _parse_devstral_tool_calls(self, content: str) -> list | None:
        if "[TOOL_CALLS]" not in content:
            return None

        tool_calls = []
        parts      = content.split("[TOOL_CALLS]")

        for part in parts[1:]:
            part = part.strip()
            if not part:
                continue

            brace_pos = part.find("{")
            if brace_pos == -1:
                logger.warning(f"_parse_devstral_tool_calls | pas de JSON | {part[:100]}")
                continue

            tool_name = part[:brace_pos].strip()
            json_str  = part[brace_pos:].strip()

            depth   = 0
            end_pos = 0
            for i, char in enumerate(json_str):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        break

            json_str = json_str[:end_pos]

            try:
                arguments = json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning(
                    f"_parse_devstral_tool_calls | JSON invalide | "
                    f"{tool_name} | {json_str}"
                )
                arguments = {}

            tool_calls.append({"name": tool_name, "arguments": arguments})
            logger.debug(f"_parse_devstral_tool_calls | {tool_name} | {arguments}")

        return tool_calls if tool_calls else None