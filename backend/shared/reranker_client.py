import time
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)
import logging
from backend.shared.config import get_settings
from backend.shared.logger import setup_logger
from pathlib import Path

settings = get_settings()
logger   = setup_logger("reranker_client")


def _is_retryable(exception) -> bool:
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code >= 500
    return True


class RerankerClient:

    def __init__(self):
        self.base_url = settings.vllm_reranker_base_url.rstrip("/")
        self.model    = settings.vllm_reranker_model
        self.headers  = {"Content-Type": "application/json"}

        if settings.vllm_reranker_api_key:
            self.headers["Authorization"] = (
                f"Bearer {settings.vllm_reranker_api_key}"
            )

        logger.info(
            f"RerankerClient | init | "
            f"model={self.model} | url={self.base_url}"
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
    async def rerank(
        self,
        query:     str,
        documents: list[str],
        top_k:     int = 5,
    ) -> list[dict]:
        """
        Appelle le endpoint vLLM /v1/score.

        Retourne une liste triée par score décroissant :
        [{"index": int, "relevance_score": float}, ...]
        """
        if not documents:
            logger.debug("rerank | liste vide → retour immédiat")
            return []

        logger.debug(
            f"rerank | début | "
            f"{len(documents)} documents | "
            f"model={self.model} | "
            f"url={self.base_url}"
        )

        use_rerank_api = settings.reranker_api_format == "rerank"

        if use_rerank_api:
            endpoint = "/v1/rerank"
            payload  = {
                "model":     self.model,
                "query":     query,
                "documents": documents,
                "top_n":     top_k,
            }
        else:
            endpoint = "/v1/score"
            payload  = {
                "model":     self.model,
                "text_1":     query,
                "text_2": documents,
                "top_n":     top_k,
            }

        ssl_verify = self.get_ssl_verify()

        start = time.perf_counter()

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=   60.0,
                write=  30.0,
                pool=   5.0,
            ),
            verify=ssl_verify,
        ) as client:

            response = await client.post(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                json=payload,
            )

            duration = time.perf_counter() - start

            if response.status_code != 200:
                logger.error(
                    f"rerank | erreur HTTP {response.status_code} | "
                    f"durée={duration:.2f}s | "
                    f"body={response.text[:500]}"
                )
                response.raise_for_status()

            data = response.json()
            if use_rerank_api:
                # Format /v1/rerank : results[{index, relevance_score}] → même forme que /v1/score
                results = [
                    {"index": r["index"], "score": r.get("relevance_score", 0.0)}
                    for r in data.get("results", [])
                ]
            else:
                results = data.get("data", [])

            logger.debug(
                f"rerank | succès | "
                f"{len(results)} résultats | "
                f"durée={duration:.2f}s"
            )

            return results

    async def rerank_chunks(
        self,
        query:  str,
        chunks: list[dict],
        top_k:  int = 5,
    ) -> list[dict]:
        """
        Reranke une liste de chunks (dicts avec clé "text").

        Retourne les chunks triés par score décroissant
        avec "rerank_score" ajouté à chaque chunk.
        """
        if not chunks:
            return []

        documents = [c.get("text", "") for c in chunks]

        try:
            results = await self.rerank(
                query=     query,
                documents= documents,
                top_k=     top_k,
            )
        except Exception as e:
            logger.error(
                f"rerank_chunks | erreur | {str(e)} | "
                f"fallback ordre cosinus"
            )
            # Fallback : retourner dans l'ordre original
            return chunks[:top_k]

        if not results:
            logger.warning("rerank_chunks | résultats vides | fallback")
            return chunks[:top_k]

        # Associer le score au chunk via l'index
        scored = []
        for item in results:
            idx   = item.get("index", 0)
            score = item.get("score", 0.0)

            if idx < len(chunks):
                c = dict(chunks[idx])
                c["rerank_score"] = round(float(score), 4)
                scored.append(c)

        # Trier par score décroissant
        scored.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        top = scored[:top_k]

        if top:
            logger.info(
                f"rerank_chunks | "
                f"{len(chunks)} → top {len(top)} | "
                f"best={top[0].get('rerank_score', 0)} | "
                f"worst={top[-1].get('rerank_score', 0)}"
            )

        return top