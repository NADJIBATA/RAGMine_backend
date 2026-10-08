```text
backend/rag_service/
│
├── chunker.py
├── indexer.py
├── retriver.py
├── indexer_params.py
├── retriver_params.py
└── main.py
```

1. chunker.py
Rôle

Le chunker transforme les données Redmine en documents optimisés pour la recherche IA avec le nettoyage .

Il prépare :
   
   - le texte
   - les métadonnées
   - les sections documentaires
   - les chunks vectorisables



2. indexer.py : Le module Indexer gère l'indexation des données dans Qdrant.

Ses principales fonctions sont :

   - Initialiser Qdrant (collections, index et configuration vectorielle).
   - Générer les embeddings avec Qwen3 Embeddings.
   - Indexer les données par batch pour optimiser les performances et limiter l'utilisation de la mémoire.
   - Indexer les projets, les pages wiki et les tickets, de manière complète ou partielle.
   - Effectuer une indexation incrémentale en comparant le champ updated_on afin de ne traiter que les nouveaux ou les documents modifiés.


3. retriever.py

Le module Retriever gère le moteur de recherche documentaire du système RAG.

Ses principales fonctions sont :

   - Effectuer des recherches vectorielles (Dense) avec les embeddings Qwen3 et Qdrant.
   - Effectuer des recherches lexicales (BM25) basées sur les mots-clés.
   - Combiner les deux approches en mode hybride avec RRF et un reranker neuronal pour améliorer la pertinence.
   - Appliquer les règles de sécurité Redmine (permissions, projets accessibles, filtres).
   - Retourner des résultats enrichis avec les informations utiles (score, source, projet, ticket, auteur, URL).