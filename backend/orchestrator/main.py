import re
import base64
import secrets
import httpx
import jwt
from pathlib import Path
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Any
from backend.shared.config import get_settings
from backend.shared.logger import setup_logger
from backend.orchestrator.agent import Agent
from backend.orchestrator.conversation_manager import ConversationManager
from backend.orchestrator.cancellation         import CancellationManager
from backend.orchestrator.ldap_service import authenticate_and_fetch_user
from backend.orchestrator.exception import LDAPAuthError


settings     = get_settings()
logger       = setup_logger("orchestrator")
app          = FastAPI(title="RAGmine - Orchestrateur")
agent        = Agent()
conversation = ConversationManager()
security     = HTTPBearer()

# ── CORS ──────────────────────────────────────
# je la change avant la prod , on autorise seulement mon frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=     settings.allowed_origins,
    allow_credentials= True,
    allow_methods=     ["*"],
    allow_headers=     ["*"],
)

REASONING_EFFORTS = {"none", "low", "medium", "high"}
INTENT_MODES      = {"auto", "rag_only", "mcp_only", "chitchat"}
# ─────────────────────────────────────────────
#   MODELES PYDANTIC
# ─────────────────────────────────────────────

class AuthRequest(BaseModel):
    login:    str
    password: str

class ChatRequest(BaseModel):
    message: str
    reasoning_effort: str = settings.reasoning_effort 
    intent: str  = "auto"

class ChatResponse(BaseModel):
    success:    bool
    response:   str | None
    session_id: str
    tool_calls: list[dict[str, Any]] = []
    rag_used:   bool                 = False   
    error:      str | None           = None




# ─────────────────────────────────────────────
#   SSL Verify
# ─────────────────────────────────────────────

def get_ssl_verify():
    """
    Retourne la configuration SSL à utiliser pour httpx.
    """
    ssl_cert = settings.redmine_ssl_cert_path

    if ssl_cert:
        cert_path = Path(ssl_cert)
    
        if cert_path.exists():
            logger.info(f"SSL | certificat utilisé : {cert_path}")
            return str(cert_path)    
            
        else:
            logger.warning(
                f"SSL | certificat introuvable : {cert_path} | fallback verify=False"
            )
            return False
    else:
        logger.warning(f"SSL | certificat redmine introuvable | fallback verify=False")
        return False



# ─────────────────────────────────────────────
#   JWT HELPERS
# ─────────────────────────────────────────────


def create_token(login: str, firstname: str, lastname: str) -> str:
    payload = {
        "sub":       login.upper(),
        "firstname": firstname,
        "lastname":  lastname,
        "iat":       datetime.now(timezone.utc),
        "exp":       datetime.now(timezone.utc) + timedelta(
                         hours=settings.jwt_expire_hours
                     ),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expiré")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token invalide")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    token   = credentials.credentials
    payload = decode_token(token)

    redis_key = f"token:{payload['sub']}"
    stored    = await conversation.redis.get(redis_key)

    if not stored:
        raise HTTPException(status_code=401, detail="Session expirée")

    stored_token = stored.decode("utf-8") if isinstance(stored, bytes) else stored

    if stored_token != token:
        raise HTTPException(status_code=401, detail="Token invalide")

    api_key_raw = await conversation.redis.get(f"apikey:{payload['sub']}")

    # None = clé absente (session expirée) ; "" = accès anonyme (AUTH_MODE=redmine sans api_key)
    if api_key_raw is None:
        raise HTTPException(status_code=401, detail="Session expirée")

    payload["api_key"] = (
        api_key_raw.decode("utf-8") if isinstance(api_key_raw, bytes) else api_key_raw
    )

    return payload


# ─────────────────────────────────────────────
#   HEALTH
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "orchestrator"}

@validator("reasoning_effort")
def validate_reasoning(cls, v):
    if v not in REASONING_EFFORTS:
        raise ValueError(
            f"reasoning_effort invalide '{v}' | "
            f"valeurs autorisées : {REASONING_EFFORTS}"
        )
    return v

@validator("intent")
def validate_intent(cls, v):                                   
    if v not in INTENT_MODES:
        raise ValueError(
            f"intent invalide '{v}' | "
            f"valeurs autorisées : {INTENT_MODES}"
        )
    return v

# ─────────────────────────────────────────────
#   AUTHENTIFICATION
# ─────────────────────────────────────────────

@app.post("/auth")
async def auth(request: AuthRequest):
    login_str = request.login.strip()
    password = request.password
    ssl_verify = get_ssl_verify()

    logger.info(f"auth | login={login_str} | mode={settings.auth_mode}")

    if settings.auth_mode == "redmine":
        firstname, lastname, mail, api_key = await _auth_redmine(login_str, password, ssl_verify)
        return await _open_session(login_str, firstname, lastname, mail, api_key)

    if settings.auth_mode == "local":
        _auth_local(password)
        return await _open_session(login_str, login_str, "", "", settings.redmine_api_key)

    # ─────────────────────────────────────
    # ÉTAPE 1 : Authentification LDAP
    # ─────────────────────────────────────
    try:
        ldap_data = authenticate_and_fetch_user(login_str, password)
    except LDAPAuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"auth | exception LDAP | {str(e)}")
        raise HTTPException(status_code=500, detail="Serveur LDAP inaccessible")

    firstname = ldap_data["firstname"]
    lastname = ldap_data["lastname"]
    mail = ldap_data["mail"]

    # ─────────────────────────────────────
    # ÉTAPE 2 : Récupérer l'api_key Redmine 
    # ─────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=ssl_verify) as client:
            r = await client.get(
                f"{settings.redmine_url}/users.json",
                params={"name": login_str},
                headers={"X-Redmine-API-Key": settings.redmine_api_key},
            )

        if r.status_code != 200:
            raise HTTPException(status_code=500, detail="Erreur Redmine")

        users = r.json().get("users", [])
        matching_user = next(
            (u for u in users if u.get("login", "").upper() == login_str.upper()), None
        )

        if not matching_user:
            raise HTTPException(status_code=404, detail="Utilisateur non trouvé dans Redmine")

        user_id = matching_user["id"]

        async with httpx.AsyncClient(timeout=10.0, verify=ssl_verify) as client:
            r2 = await client.get(
                f"{settings.redmine_url}/users/{user_id}.json",
                headers={"X-Redmine-API-Key": settings.redmine_api_key},
            )

        user_detail = r2.json().get("user", {})
        api_key = user_detail.get("api_key", "")

        if not api_key:
            raise HTTPException(status_code=500, detail="Impossible de récupérer l'api_key Redmine")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"auth | exception Redmine | {str(e)}")
        raise HTTPException(status_code=500, detail="Serveur Redmine inaccessible")

    return await _open_session(login_str, firstname, lastname, mail, api_key)


async def _auth_redmine(login_str: str, password: str, ssl_verify) -> tuple[str, str, str, str]:
    """
    AUTH_MODE=redmine : vérifie login/mot de passe via l'API Redmine (Basic Auth
    sur /my/account.json). Ne nécessite ni LDAP ni clé admin.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=ssl_verify) as client:
            r = await client.get(
                f"{settings.redmine_url.rstrip('/')}/my/account.json",
                auth=(login_str, password),
            )
    except Exception as e:
        logger.error(f"auth | exception Redmine | {str(e)}")
        raise HTTPException(status_code=500, detail="Serveur Redmine inaccessible")

    if r.status_code == 401:
        raise HTTPException(status_code=401, detail="Login ou mot de passe incorrect")
    if r.status_code == 403:
        # Redmine répond 403 (même avec de mauvais identifiants) quand l'API REST est désactivée
        logger.error("auth | Redmine HTTP 403 | API REST désactivée sur ce Redmine → utiliser AUTH_MODE=local")
        raise HTTPException(status_code=500, detail="Authentification API désactivée sur ce Redmine")
    if r.status_code != 200:
        logger.error(f"auth | Redmine HTTP {r.status_code} | {r.text[:200]}")
        raise HTTPException(status_code=500, detail="Erreur Redmine")

    user = r.json().get("user", {})
    # api_key absente (ancienne version Redmine) → accès anonyme aux données publiques
    return (
        user.get("firstname", login_str),
        user.get("lastname", ""),
        user.get("mail", ""),
        user.get("api_key", ""),
    )


def _auth_local(password: str) -> None:
    """
    AUTH_MODE=local : un seul mot de passe partagé (LOCAL_AUTH_PASSWORD), aucun service externe.
    """
    if not settings.local_auth_password:
        logger.error("auth | AUTH_MODE=local mais LOCAL_AUTH_PASSWORD est vide")
        raise HTTPException(status_code=500, detail="LOCAL_AUTH_PASSWORD non configuré")
    if not secrets.compare_digest(password, settings.local_auth_password):
        raise HTTPException(status_code=401, detail="Login ou mot de passe incorrect")


async def _open_session(login_str: str, firstname: str, lastname: str, mail: str, api_key: str) -> dict:
    # ─────────────────────────────────────
    # ÉTAPE 3 : Créer le token + session
    # ─────────────────────────────────────

    token = create_token(login_str, firstname, lastname)

    await conversation.redis.setex(
        f"token:{login_str.upper()}", settings.jwt_expire_hours * 3600, token
    )
    await conversation.redis.setex(
        f"apikey:{login_str.upper()}", settings.jwt_expire_hours * 3600, api_key
    )

    logger.info(f"auth | succès | {login_str} | {firstname} {lastname}")

    return {
        "success": True,
        "token": token,
        "login": login_str,
        "firstname": firstname,
        "lastname": lastname,
        "mail": mail,
    }


# ─────────────────────────────────────────────
#   DÉCONNEXION
# ─────────────────────────────────────────────

@app.post("/logout")
async def logout(user: dict = Depends(get_current_user)):
    login_str = user["sub"]

    await conversation.redis.delete(f"token:{login_str}")
    await conversation.redis.delete(f"apikey:{login_str}")
    await agent.clear_session(login_str)
    await agent.conversation.redis.delete(f"tokens_history:{login_str}")

    logger.info(f"logout | {login_str} | token + apikey révoqués + tokens_history + session vidée")
    return {"success": True, "message": "Déconnecté"}


# ─────────────────────────────────────────────
#   CHAT
# ─────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user:    dict = Depends(get_current_user),
):
    session_id = user["sub"]
    api_key    = user["api_key"]

    logger.info(f"chat | session={session_id} | message={request.message[:100]} | intent={request.intent}")

    try:
        result = await agent.chat(
            session_id=   session_id,
            user_message= request.message,
            api_key=      api_key,
            reasoning_effort= request.reasoning_effort,
            forced_intent= request.intent,
        )
    except Exception as e:
        logger.error(f"chat | erreur inattendue | session={session_id} | {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    if not result["success"]:
        logger.error(
            f"chat | erreur | session={session_id} | "
            f"{result.get('error')}"
        )
        raise HTTPException(
            status_code=500,
            detail=result.get("error", "Erreur interne"),  
        )

    return ChatResponse(
        success=    result["success"],
        response=   result["response"],
        session_id= result["session_id"],
        tool_calls= result["tool_calls"],
        rag_used=   result.get("rag_used", False),
        error=      result["error"],
    )

# ─────────────────────────────────────────────
#   SESSION
# ─────────────────────────────────────────────

@app.get("/session")
async def get_session_info(user: dict = Depends(get_current_user)):
    session_id = user["sub"]
    history    = await agent.get_history(session_id)

    messages = [
        m for m in history
        if m["role"] in ("user", "assistant")
        and not m["content"].startswith("Résultat de")
        and not m["content"].startswith("[TOOL_CALLS]")
        and not m["content"].startswith("<rag_context>")   
    ]

    return {
        "success":        True,
        "session_id":     session_id,
        "exists":         len(history) > 0,
        "total_messages": len(messages),
    }


# ─────────────────────────────────────────────
#   HISTORIQUE
# ─────────────────────────────────────────────

@app.get("/history")
async def get_history(user: dict = Depends(get_current_user)):
    session_id = user["sub"]
    history    = await agent.get_history(session_id)

    readable = [
        m for m in history
        if m["role"] in ("user", "assistant")
        and not m["content"].startswith("Résultat de")
        and not m["content"].startswith("[TOOL_CALLS]")
        and not m["content"].startswith("<rag_context>")  
    ]

    return {
        "success":    True,
        "session_id": session_id,
        "history":    readable,
        "total":      len(readable),
    }


@app.delete("/history")
async def clear_history(user: dict = Depends(get_current_user)):
    session_id = user["sub"]
    await agent.clear_session(session_id)
    return {
        "success":    True,
        "session_id": session_id,
        "message":    f"Session {session_id} vidée",
    }


# ─────────────────────────────────────────────
#   RAG — INDEXATION
# ─────────────────────────────────────────────

@app.post("/rag/index/project")
async def rag_index_project(
    project_id: str,
    user:       dict = Depends(get_current_user),
):
    """
    Lance l'indexation RAG d'un projet en arrière-plan.
    Utilise l'api_key de l'utilisateur connecté.
    """
    api_key    = user["api_key"]
    session_id = user["sub"]

    logger.info(
        f"rag_index_project | session={session_id} | "
        f"project_id={project_id}"
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.rag_service_url}/index/project",
                json={
                    "project_id": project_id,
                    "api_key":    api_key,
                },
            )
            response.raise_for_status()
            return response.json()

    except Exception as e:
        logger.error(f"rag_index_project | erreur | {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/index/project/sync")
async def rag_index_project_sync(
    project_id: str,
    user:       dict = Depends(get_current_user),
):
    """
    Indexe un projet de manière synchrone.
    Attend la fin avant de répondre.
    """
    api_key    = user["api_key"]
    session_id = user["sub"]

    logger.info(
        f"rag_index_project_sync | session={session_id} | "
        f"project_id={project_id}"
    )

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{settings.rag_service_url}/index/project/sync",
                json={
                    "project_id": project_id,
                    "api_key":    api_key,
                },
            )
            response.raise_for_status()
            return response.json()

    except Exception as e:
        logger.error(f"rag_index_project_sync | erreur | {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/index/all")
async def rag_index_all(
    user: dict = Depends(get_current_user),
):
    """
    Lance l'indexation de tous les projets accessibles.
    """
    api_key    = user["api_key"]
    session_id = user["sub"]

    logger.info(f"rag_index_all | session={session_id}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.rag_service_url}/index/all",
                json={"api_key": api_key},
            )
            response.raise_for_status()
            return response.json()

    except Exception as e:
        logger.error(f"rag_index_all | erreur | {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/rag/index/project/{project_id}")
async def rag_delete_project_index(
    project_id: str,
    user:       dict = Depends(get_current_user),
):
    """
    Supprime l'index RAG d'un projet.
    """
    api_key    = user["api_key"]
    session_id = user["sub"]

    logger.info(
        f"rag_delete_project_index | session={session_id} | "
        f"project_id={project_id}"
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"{settings.rag_service_url}/index/project/{project_id}",
                json={"api_key": api_key},
            )
            response.raise_for_status()
            return response.json()

    except Exception as e:
        logger.error(f"rag_delete_project_index | erreur | {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
#   RAG — STATISTIQUES
# ─────────────────────────────────────────────

@app.get("/rag/stats")
async def rag_stats(user: dict = Depends(get_current_user)):
    """
    Retourne les statistiques de l'index RAG Qdrant.
    """
    logger.info(f"rag_stats | session={user['sub']}")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{settings.rag_service_url}/stats"
            )
            response.raise_for_status()
            return response.json()

    except Exception as e:
        logger.error(f"rag_stats | erreur | {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
#   RAG — RECHERCHE DIRECTE
# ─────────────────────────────────────────────

@app.post("/rag/search")
async def rag_search(
    query:      str,
    user:       dict = Depends(get_current_user),
    project_id: str  = None,
    top_k:      int  = 5,
    min_score:  float = 0.4,
):
    """
    Recherche sémantique directe dans le RAG.
    Utile pour tester l'index.
    """
    api_key    = user["api_key"]
    session_id = user["sub"]

    logger.info(
        f"rag_search | session={session_id} | "
        f"query='{query[:60]}'"
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.rag_service_url}/search",
                json={
                    "query":      query,
                    "api_key":    api_key,
                    "project_id": project_id,
                    "top_k":      top_k,
                    "min_score":  min_score,
                },
            )
            response.raise_for_status()
            return response.json()

    except Exception as e:
        logger.error(f"rag_search | erreur | {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



# ─────────────────────────────────────────────
#   CANCEL
# ─────────────────────────────────────────────

@app.post("/chat/cancel")
async def cancel_chat(
    user: dict = Depends(get_current_user),   
):
    """
    Annule la requête en cours pour la session courante.
    """
    session_id = user["sub"]                 
    cancelled  = agent.cancel(session_id)

    logger.info(
        f"cancel_chat | session={session_id} | cancelled={cancelled}"
    )

    return {
        "success":    cancelled,
        "session_id": session_id,
        "message": (
            "Annulation demandée."
            if cancelled
            else "Aucun traitement actif à annuler."
        ),
    }
