
Le module `backend/orchestrateur` constitue le cœur  de RAGmine.

Il orchestre les interactions entre :

- le LLM
- le systeme RAG
- les outils MCP
- Redis
- les services d’authentification
- la mémoire conversationnelle

L’orchestrateur agit comme un agent intelligent capable :

- de connecter directement avec le LLM
- d’analyser l’intention de la question et choisir dynamiquement entre RAG et outils MCP
- d’exécuter une boucle ReAct
- de maintenir le contexte conversationnel
- de gérer les sessions utilisateur
- de connecter le backend avec le frontend


```text
─ orchestrateur/
  ├── Agent.py
  ├── conversation_manager.py
  ├── mcp_client.py
  ├── cancellation.py
  ├── exception.py
  └── main.py

```

1. Agent.py est le moteur principal de l’orchestrateur.

Il pilote :

  -  la classification d’intention
  -  les appels RAG
  -  les appels MCP
  -  la boucle ReAct
  -  la gestion mémoire


2. conversation_manager.py : Gestion de la mémoire conversationnelle Redis.

Fonctionnalités

  - ajout messages
  - récupération historique
  - nettoyage sessions
  - gestion de TTL 


3. mcp_client.py : Client HTTP vers le serveur MCP.

4. cancellation.py : Gestion des annulations et l'arrêt des appels long.

