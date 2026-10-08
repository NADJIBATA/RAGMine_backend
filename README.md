RAGmine est un application intelligente microservices connectée à Redmine.

Le système combine :

- un orchestrateur connectée directement au LLM "C'est le coeur de l'application"
- un système RAG hybride "BM25 + Qween3"
- un serveur MCP (Model Context Protocol)
- une mémoire conversationnelle Redis
- une base vectorielle dans Qdrant

```text
backend/
├── orchestrateur/
│   ├── Agent.py
│   ├── conversation_manager.py
│   ├── mcp_client.py
│   ├── cancellation.py
│   ├── exception.py
│   └── main.py
├── rag_service/
│   ├── chunker.py
│   ├── indexer.py
│   ├── retriever.py
│   └── main.py
├── mcp/
│   ├── tools/
│   └── main.py
├── shared/
│   ├── config.py
│   ├── logger.py
│   ├── redmine_client.py
│   ├── qwen3_client.py
│   ├── devstral_client.py
│   ├── reranker_client.py
│   ├── bm25_retriever.py
│   └── rrf.py
├── .env
└── docker-compose.yml
```


![Architecture de RAGmine](docs/schema_architecture_agent_ia_redmine_base_donnees.png)