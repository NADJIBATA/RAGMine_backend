from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):


    allowed_origins: list[str] = ["http://localhost:3000"]
    # ─── Auth ────────────────────────────────
    # "ldap"    → LDAP + clé admin Redmine (comportement d'origine)
    # "redmine" → login/mot de passe vérifiés directement par Redmine (sans LDAP ni clé admin)
    # "local"   → mot de passe unique LOCAL_AUTH_PASSWORD, Redmine lu avec REDMINE_API_KEY (ou en anonyme)
    auth_mode: str = "ldap"
    local_auth_password: str = ""
    # ─── LDAP ────────────────────────────────
    ldap_host: str 
    ldap_port: int  = 389 
    ldap_base_dn: str
    ldap_service_user: str
    ldap_service_password : str
    # ─── Redmine ────────────────────────────────
    redmine_url:     str
    redmine_api_key: str
    redmine_ssl_cert_path: str
    # ─── vLLM ───────────────────────────────────
    vllm_base_url:        str
    vllm_model_name:      str
    vllm_api_key:         str = ""
    scaleway_ssl_cert_path: str
    # ── vLLM Embedding ────────────────────────
    vllm_embedding_base_url: str 
    vllm_embedding_model:    str 
    vllm_embedding_api_key:  str = ''
    rd_ssl_cert_path:        str
    reasoning_effort:        str = "low"
    # False pour les fournisseurs qui refusent ce paramètre (Groq/llama, gpt-4.1…)
    llm_supports_reasoning_effort: bool = True

    # ─── Qdrant ─────────────────────────────────
    qdrant_host:                str = "qdrant"
    qdrant_port:                int = 6333
    qdrant_collection_issues:   str = "redmine_issues"
    qdrant_collection_wiki:     str = "redmine_wiki"
    qdrant_collection_news:     str = "redmine_news"

    # ─── Redis ──────────────────────────────────
    redis_host:        str = "redis"
    redis_port:        int = 6379
    redis_db:          int = 0
    redis_session_ttl: int = 3600

    # ─── Services ───────────────────────────────
    mcp_server_host:   str = "mcp_server"
    mcp_server_port:   int = 8001
    rag_service_host:  str = "rag_service"
    rag_service_port:  int = 8002
    orchestrator_host: str = "orchestrator"
    orchestrator_port: int = 8000
     
    # ─── Reranker ───────────────────────────────
    vllm_reranker_base_url: str = "http://172.16.29.229:8005"
    vllm_reranker_model:    str = "bge-reranker-v2-m3"
    vllm_reranker_api_key:  str = ""
    # "vllm_score" → vLLM /v1/score | "rerank" → /v1/rerank (Jina, Cohere…)
    reranker_api_format:    str = "vllm_score"
    reranker_max_length: int = 512
    reranker_batch_size: int = 16
    top_k_fetch:       int = 30
    top_k_final:       int = 5
    # ─── RAG ────────────────────────────────────
    rag_top_k:       int   = 5
    rag_min_score:   float = 0.5
    rrf_k:           int = 60          
    bm25_batch_size : int = 1000
    # ─── Chunking ──────────────────────────────
    chunk_size:                int = 900
    chunk_overlap:             int = 150
    max_file_size:             int = 10 * 1024 * 1024
    min_chunk_chars:           int = 200
    min_chunk_words:           int = 15

    # ─── Indexation ──────────────────────────────
    batch_size:                int = 32
    vector_size:               int = 1024
    collection_name:           str = 'ragmine'
    # ─── Ingestion ──────────────────────────────
    ingestion_full_cron:       str = "0 2 * * 0"
    ingestion_incremental_cron: str = "*/15 * * * *"
    # ─── Orchestrator ───────────────────────────
    max_tool_calls:  int   = 10
    max_messages:    int   = 20
    rag_good_score: float = 0.0


    rag_mode: str = "hybrid" 
    # ─── JWT ──────────────────────────────

    jwt_secret:     str = "ragmine_jwt_secret"
    jwt_expire_hours: int = 1
    
    # ─── Postgres ────────────────────────────────
    database_url: str 


    # ─── Logging ────────────────────────────────
    log_level: str = "DEBUG"

    # ─── Environnement ──────────────────────────
    env: str = "development"

    # ─── URLs calculées ─────────────────────────
    @property
    def mcp_server_url(self) -> str:
        return f"http://{self.mcp_server_host}:{self.mcp_server_port}"

    @property
    def rag_service_url(self) -> str:
        return f"http://{self.rag_service_host}:{self.rag_service_port}"
    
    @property
    def reranker_url(self) -> str:                        
        return f"http://{self.reranker_host}:{self.reranker_port}"


    @property
    def orchestrator_url(self) -> str:
        return f"http://{self.orchestrator_host}:{self.orchestrator_port}"

    class Config:
        env_file          = ".env"
        env_file_encoding = "utf-8"
        extra             = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()