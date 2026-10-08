import json
import redis.asyncio as redis
from backend.shared.config import get_settings
from backend.shared.logger import setup_logger

settings = get_settings()
logger   = setup_logger("conversation_manager")

# ── Nombre max de messages dans l'historique ──────
MAX_MESSAGES = settings.max_messages


class ConversationManager:

    def __init__(self):
        self.redis = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
        )
        self.ttl = settings.redis_session_ttl

    # ─────────────────────────────────────────────
    #   CLÉ REDIS
    # ─────────────────────────────────────────────

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}"

    # ─────────────────────────────────────────────
    #   RÉCUPÉRER L'HISTORIQUE
    # ─────────────────────────────────────────────

    async def get_history(self, session_id: str) -> list[dict]:
        """
        Récupère l'historique complet d'une session.
        Retourne une liste vide si la session n'existe pas.
        """

        key  = self._key(session_id)
        data = await self.redis.get(key)

        if not data:
            logger.debug(f"get_history | session={session_id} | vide")
            return []

        history = json.loads(data)
        logger.debug(f"get_history | session={session_id} | {len(history)} messages")
        return history

    # ─────────────────────────────────────────────
    #   AJOUTER UN MESSAGE
    # ─────────────────────────────────────────────

    async def add_message(
        self,
        session_id: str,
        role:       str,
        content:    str,
    ) -> None:
        """
        Ajoute un message à l'historique.

        role    : "user" | "assistant" | "system" | "tool"
        content : contenu du message
        """

        history = await self.get_history(session_id)

        history.append({
            "role":    role,
            "content": content,
        })

        # ── Limiter la taille de l'historique ────
        # Garder toujours le system prompt (index 0)
        # + les MAX_MESSAGES derniers messages
        if len(history) > MAX_MESSAGES + 1:
            system_messages = [m for m in history if m["role"] == "system"]
            other_messages  = [m for m in history if m["role"] != "system"]
            other_messages  = other_messages[-MAX_MESSAGES:]
            history         = system_messages + other_messages

        key = self._key(session_id)
        await self.redis.setex(key, self.ttl, json.dumps(history))

        logger.debug(f"add_message | session={session_id} | role={role} | {len(history)} messages total")

    # ─────────────────────────────────────────────
    #   AJOUTER PLUSIEURS MESSAGES
    # ─────────────────────────────────────────────

    async def add_messages(
        self,
        session_id: str,
        messages:   list[dict],
    ) -> None:
        """
        Ajoute plusieurs messages d'un coup.
        """
        for message in messages:
            await self.add_message(
                session_id=session_id,
                role=message["role"],
                content=message["content"],
            )

    # ─────────────────────────────────────────────
    #   VIDER L'HISTORIQUE
    # ─────────────────────────────────────────────

    async def clear(self, session_id: str) -> None:
        """
        Vide complètement l'historique d'une session.
        """
        key = self._key(session_id)
        await self.redis.delete(key)
        logger.info(f"clear | session={session_id} | historique vidé")

    # ─────────────────────────────────────────────
    #   VÉRIFIER SI UNE SESSION EXISTE
    # ─────────────────────────────────────────────

    async def exists(self, session_id: str) -> bool:
        """
        Vérifie si une session existe dans Redis.
        """
        key = self._key(session_id)
        return bool(await self.redis.exists(key))

    # ─────────────────────────────────────────────
    #   RAFRAÎCHIR LE TTL
    # ─────────────────────────────────────────────

    async def refresh_ttl(self, session_id: str) -> None:
        """
        Remet le TTL à zéro pour prolonger la session.
        Appelé à chaque interaction utilisateur.
        """
        key = self._key(session_id)
        await self.redis.expire(key, self.ttl)
        logger.debug(f"refresh_ttl | session={session_id} | TTL={self.ttl}s")




    async def get_or_create(self, session_id: str) -> list[dict]:
        """
        Récupère la session si elle existe
        Sinon retourne une liste vide (nouvelle session)
        """
        history = await self.get_history(session_id)

        if history:
            logger.info(f"get_or_create | session existante | {session_id} | {len(history)} messages")
        else:
            logger.info(f"get_or_create | nouvelle session | {session_id}")

        return history


    # ─────────────────────────────────────────────
    #   FERMER LA CONNEXION
    # ─────────────────────────────────────────────

    async def close(self) -> None:
        await self.redis.aclose()
        logger.debug("close | connexion Redis fermée")

        
    # ─────────────────────────────────────────────
    #   TOKENS - CLÉS REDIS
    # ─────────────────────────────────────────────
    
    def _key_tokens(self, session_id: str) -> str:
        return f"tokens:{session_id}"
    
    # ─────────────────────────────────────────────
    #   TOKENS - AJOUTER
    # ─────────────────────────────────────────────
    
    async def add_tokens(
        self,
        session_id:    str,
        tokens_input:  int,
        tokens_output: int,
    ) -> None:
        """
        Accumule les tokens input/output pour une session.
        Appelé après chaque étape LLM d'une même question.
        """
        key  = self._key_tokens(session_id)
        data = await self.redis.get(key)
    
        if data:
            current = json.loads(data)
        else:
            current = {
                "tokens_input":  0,
                "tokens_output": 0,
                "tokens_total":  0,
                "nb_calls":      0,
            }
    
        current["tokens_input"]  += tokens_input
        current["tokens_output"] += tokens_output
        current["tokens_total"]  += tokens_input + tokens_output
        current["nb_calls"]      += 1
    
        await self.redis.setex(key, self.ttl, json.dumps(current))
    
        logger.debug(
            f"add_tokens | session={session_id} | "
            f"+input={tokens_input} +output={tokens_output} | "
            f"cumul input={current['tokens_input']} "
            f"output={current['tokens_output']} "
            f"total={current['tokens_total']} "
            f"nb_calls={current['nb_calls']}"
        )
    
    # ─────────────────────────────────────────────
    #   TOKENS - RÉCUPÉRER
    # ─────────────────────────────────────────────
    
    async def get_tokens(self, session_id: str) -> dict:
        """
        Retourne les tokens cumulés pour une session.
        """
        key  = self._key_tokens(session_id)
        data = await self.redis.get(key)
    
        if not data:
            return {
                "tokens_input":  0,
                "tokens_output": 0,
                "tokens_total":  0,
                "nb_calls":      0,
            }
    
        return json.loads(data)
    
    # ─────────────────────────────────────────────
    #   TOKENS - RESET (après chaque question)
    # ─────────────────────────────────────────────
    
    async def reset_tokens(self, session_id: str) -> dict:
        """
        Récupère les tokens de la question courante
        puis remet le compteur à zéro pour la prochaine.
        Retourne les tokens de la question qui vient de se terminer.
        """
        key     = self._key_tokens(session_id)
        data    = await self.redis.get(key)
    
        if not data:
            return {
                "tokens_input":  0,
                "tokens_output": 0,
                "tokens_total":  0,
                "nb_calls":      0,
            }
    
        tokens = json.loads(data)
        await self.redis.delete(key)
    
        logger.info(
            f"reset_tokens | session={session_id} | "
            f"input={tokens['tokens_input']} | "
            f"output={tokens['tokens_output']} | "
            f"total={tokens['tokens_total']} | "
            f"nb_calls={tokens['nb_calls']}"
        )
        return tokens
    
    # ─────────────────────────────────────────────
    #   TOKENS - HISTORIQUE PAR QUESTION
    # ─────────────────────────────────────────────
    
    def _key_tokens_history(self, session_id: str) -> str:
        return f"tokens_history:{session_id}"
    
    async def save_question_tokens(
        self,
        session_id:    str,
        question:      str,
        tokens_input:  int,
        tokens_output: int,
        tokens_total:  int,
        nb_calls:      int,
        intent:        str = "",
    ) -> None:
        """
        Sauvegarde les tokens d'une question dans l'historique.
        Garde les N dernières questions.
        """
        key  = self._key_tokens_history(session_id)
        data = await self.redis.get(key)
    
        history = json.loads(data) if data else []
    
        history.append({
            "question":      question[:100],   
            "tokens_input":  tokens_input,
            "tokens_output": tokens_output,
            "tokens_total":  tokens_total,
            "nb_calls":      nb_calls,
            "intent":        intent,
        })
    
        
        history = history[-50:]
    
        await self.redis.setex(key, self.ttl, json.dumps(history))
    
        logger.info(
            f"save_question_tokens | session={session_id} | "
            f"question='{question[:60]}' | "
            f"input={tokens_input} output={tokens_output} "
            f"total={tokens_total} nb_calls={nb_calls}"
        )
    
    async def get_tokens_history(self, session_id: str) -> list[dict]:
        """
        Retourne l'historique des tokens par question.
        """
        key  = self._key_tokens_history(session_id)
        data = await self.redis.get(key)
        return json.loads(data) if data else []    
        