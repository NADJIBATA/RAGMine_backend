import gc
import time
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)
from pathlib import Path
import logging
from backend.shared.config import get_settings
from backend.shared.logger import setup_logger

settings = get_settings()
logger   = setup_logger("qwen3_client")


def _is_retryable(exception) -> bool:
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code >= 500
    return True


class Qwen3Client:

    def __init__(self):
        self.base_url = settings.vllm_embedding_base_url.rstrip("/")
        self.model    = settings.vllm_embedding_model
        self.headers  = {"Content-Type": "application/json"}

        if settings.vllm_embedding_api_key:
            self.headers["Authorization"] = f"Bearer {settings.vllm_embedding_api_key}"

        logger.info(
            f"Qwen3Client | init | model={self.model} | url={self.base_url}"
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


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),  
    )
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.debug("embed_batch | liste vide → retour immédiat")
            return []

        logger.debug(
            f"embed_batch | début | "
            f"{len(texts)} textes | "
            f"model={self.model} | "
            f"url={self.base_url}"
        )
        ssl_verify = self.get_ssl_verify()
        payload = {
            "model": self.model,
            "input": texts,
        }

        start = time.perf_counter()

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0),
            verify=ssl_verify,
        ) as client:

            # ── Requête ───────────────────────────
            response = await client.post(
                f"{self.base_url}/v1/embeddings",
                headers=self.headers,
                json=payload,
            )

            duration = time.perf_counter() - start

            # ── Log erreur avant raise ────────────
            if response.status_code != 200:
                logger.error(
                    f"embed_batch | erreur HTTP {response.status_code} | "
                    f"durée={duration:.2f}s | "
                    f"body={response.text[:500]}"
                )
                response.raise_for_status()

            # ── Extraire les vecteurs ─────────────
            items = response.json().get("data", [])
            embeddings = [
                item["embedding"]
                for item in sorted(items, key=lambda x: x.get("index", 0))
            ]

            dimension = len(embeddings[0]) if embeddings else 0

            logger.debug(
                f"embed_batch | succès | "
                f"{len(embeddings)} vecteurs | "
                f"dimension={dimension} | "
                f"durée={duration:.2f}s"
            )

            
            del items, response

        gc.collect()
        return embeddings

    async def embed(self, text: str) -> list[float]:

        logger.debug(
            f"embed | texte={text[:100]}..."
        )

        result = await self.embed_batch([text])

        if not result:
            logger.error("embed | embedding vide retourné")
            raise ValueError("Embedding vide retourné")

        return result[0]