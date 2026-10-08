
import asyncio
import time
from backend.shared.logger import setup_logger

logger = setup_logger("cancellation")


class CancellationManager:
    """
    Gère les tokens d'annulation par session.
    Chaque session possède un asyncio.Event :
      - non déclenché  → traitement normal
      - déclenché      → annulation demandée
    """

    def __init__(self):
        self._events:    dict[str, asyncio.Event] = {}
        self._last_used: dict[str, float]         = {}

    # ─────────────────────────────────────────
    #   CRÉER / RÉCUPÉRER UN EVENT
    # ─────────────────────────────────────────

    def get_or_create(self, session_id: str) -> asyncio.Event:
        self._last_used[session_id] = time.time()

        if session_id not in self._events:
            self._events[session_id] = asyncio.Event()
            logger.debug(f"get_or_create | nouvel event | session={session_id}")

        return self._events[session_id]

    # ─────────────────────────────────────────
    #   ANNULER
    # ─────────────────────────────────────────

    def cancel(self, session_id: str) -> bool:
        """
        Déclenche l'annulation.
        Retourne True si un event existait, False sinon.
        """
        event = self._events.get(session_id)

        if event and not event.is_set():
            event.set()
            logger.info(f"cancel | annulation déclenchée | session={session_id}")
            return True

        logger.warning(
            f"cancel | aucun traitement actif ou déjà annulé | session={session_id}"
        )
        return False

    # ─────────────────────────────────────────
    #   RÉINITIALISER (début de requête)
    # ─────────────────────────────────────────

    def reset(self, session_id: str) -> None:
        """
        Remet l'event à zéro avant chaque nouvelle requête.
        """
        self._last_used[session_id] = time.time()

        if session_id in self._events:
            self._events[session_id].clear()
        else:
            self._events[session_id] = asyncio.Event()

        logger.debug(f"reset | event réinitialisé | session={session_id}")

    # ─────────────────────────────────────────
    #   VÉRIFIER SI ANNULÉ
    # ─────────────────────────────────────────

    def is_cancelled(self, session_id: str) -> bool:
        event = self._events.get(session_id)
        return event is not None and event.is_set()

    # ─────────────────────────────────────────
    #   NETTOYER LES SESSIONS EXPIRÉES
    # ─────────────────────────────────────────

    def cleanup(self, ttl: int = 3600) -> int:
        now     = time.time()
        expired = [
            sid for sid, last in self._last_used.items()
            if now - last > ttl
        ]

        for sid in expired:
            self._events.pop(sid, None)
            self._last_used.pop(sid, None)

        if expired:
            logger.info(
                f"cleanup | {len(expired)} sessions supprimées | "
                f"{len(self._events)} restantes"
            )

        return len(expired)

    # ─────────────────────────────────────────
    #   SUPPRIMER UNE SESSION
    # ─────────────────────────────────────────

    def remove(self, session_id: str) -> None:
        self._events.pop(session_id, None)
        self._last_used.pop(session_id, None)
        logger.debug(f"remove | session supprimée | session={session_id}")