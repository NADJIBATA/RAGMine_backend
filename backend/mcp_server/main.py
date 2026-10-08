from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any

from backend.shared.logger import setup_logger
from backend.mcp_server.tools.issue_tools     import IssueTools
from backend.mcp_server.tools.project_tools   import ProjectTools
from backend.mcp_server.tools.user_tools      import UserTools
from backend.mcp_server.tools.time_entry_tools import TimeEntryTools
from backend.mcp_server.tools.wiki_tools      import WikiTools
from backend.mcp_server.tools.search_tools    import SearchTools
from backend.mcp_server.tools.reference_tools import ReferenceTools

logger = setup_logger("mcp_server")

app = FastAPI(title="RAGmine - MCP Server")

# ─────────────────────────────────────────────
#   TYPES CASTERS
# ─────────────────────────────────────────────

TYPE_CASTERS = {
    "integer": int,
    "boolean": lambda v: v if isinstance(v, bool) else str(v).lower() == "true",
    "string":  str,
    "number":  float,
}


def _cast_parameters(params: dict, schema: dict) -> dict:
    """
    Cast les paramètres selon les types déclarés dans le registry.
    Ignore les paramètres absents du schema.
    """
    casted = {}
    for key, value in params.items():
        if key in schema:
            expected_type = schema[key].get("type", "string")
            caster = TYPE_CASTERS.get(expected_type, lambda v: v)
            try:
                casted[key] = caster(value)
            except (ValueError, TypeError) as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Paramètre '{key}' invalide : attendu {expected_type} | {str(e)}",
                )
        else:
            casted[key] = value
    return casted


# ─────────────────────────────────────────────
#   REGISTRY DES OUTILS MCP
# ─────────────────────────────────────────────

TOOLS_REGISTRY = {

    # ─────────────────────────────────────────
    #   ISSUES
    # ─────────────────────────────────────────

    "list_issues": {
        "description": """
            But :
            Lister les tickets Redmine avec filtres optionnels (statut, assigné, tracker, dates, projet).

            Quand utiliser :
            - Quand on veut consulter des tickets selon certains critères connus.
            - Quand le project_id est déjà connu.
            - Pour explorer les tickets d'un projet page par page.

            Workflow recommandé :
            - Chaîner avec get_issue_details pour récupérer le détail de chaque ticket si besoin.

            Notes importantes :
            - limit=-1 pour récupérer tous les tickets automatiquement.
            - Évitez de récupérer tous les tickets de l'ensemble des projets, car cela génère un volume de données très important.
            - utilises la toujours avec des filtrage 
            - Différence avec search : list_issues filtre par attributs, search cherche par texte.
        """,
        "parameters": {
            "status_id": {
                "type":        "string",
                "required":    False,
                "description": "ID du statut du ticket.",
            },
            "project_name": {
                "type":        "string",
                "required":    False,
                "description": (
                    "Nom du projet en texte libre."
                ),
            },
            "assigned_to_name": {
                "type":        "string",
                "required":    False,
                "description": (
                    "Nom de l'utilisateur assigné au ticket en texte libre."
                ),
            },
            "author_name": {                                
                "type":        "string",
                "required":    False,
                "description": (
                    "Nom de l'auteur (créateur) du ticket en texte libre. "
                    "Permet de filtrer tous les tickets créés par une personne précise. "
                    "Différent de assigned_to_name : l'auteur est celui qui a ouvert le ticket, "
                    "l'assigné est celui qui doit le traiter."
                ),
            },
            "category_name": {                                
                "type":        "string",
                "required":    False,
                "description": (
                    "Nom de la catégorie du ticket en texte libre. "
                    "⚠️ Nécessite que project_name soit fourni, "
                    "car les catégories sont propres à chaque projet."
                ),
            },
            "subject": {                                     
                "type":        "string",
                "required":    False,
                "description": (
                    "Recherche par mot-clé dans le titre (sujet) du ticket. "
                    "Le filtre est de type 'contient' : tous les tickets dont le titre "
                    "contient la chaîne fournie seront retournés. "
                    "Exemple : subject='facture' retourne tous les tickets dont le titre contient 'facture'. "
                    "Pour une recherche plein texte (titre + description + commentaires), "
                    "préférer l'outil search."
                ),
            },
            "tracker_id": {
                "type":        "string",
                "required":    False,
                "description": "ID du tracker.",
            },
            "priority_id": {
                "type":        "string",
                "required":    False,
                "description": "ID de la priorité.",
            },
            "parent_id": {
                "type":        "integer",
                "required":    False,
                "description": "ID du ticket parent pour filtrer les sous-tickets.",
            },
            "issue_id": {
                "type":        "string",
                "required":    False,
                "description": "ID ou liste d'IDs de tickets séparés par des virgules.",
            },
            "subproject_id": {
                "type":        "string",
                "required":    False,
                "description": "ID du sous-projet.",
            },
            "include": {
                "type":        "string",
                "required":    False,
                "description": "Champs supplémentaires : 'journals,attachments,relations,children,watchers'.",
            },
            "custom_field_id": {
                "type":        "integer",
                "required":    False,
                "description": "ID du champ personnalisé pour filtrer (récupéré via get_custom_fields).",
            },
            "custom_field_value": {
                "type":        "string",
                "required":    False,
                "description": "Valeur exacte du champ personnalisé. ⚠️ custom_field_id ET custom_field_value doivent être fournis ensemble.",
            },
            "created_on": {
                "type":        "string",
                "required":    False,
                "description": "Filtre date déjà formaté ex: >=YYYY-MM-DD | <=YYYY-MM-DD | ><YYYY-MM-DD|YYYY-MM-DD.",
            },
            "created_from": {
                "type":        "string",
                "required":    False,
                "description": "Date de début de création YYYY-MM-DD. Combiné avec created_to → plage. Seul → >=date.",
            },
            "created_to": {
                "type":        "string",
                "required":    False,
                "description": "Date de fin de création YYYY-MM-DD. Combiné avec created_from → plage. Seul → <=date.",
            },
            "updated_on": {
                "type":        "string",
                "required":    False,
                "description": "Filtre date déjà formaté ex: >=YYYY-MM-DD | <=YYYY-MM-DD | ><YYYY-MM-DD|YYYY-MM-DD.",
            },
            "updated_from": {
                "type":        "string",
                "required":    False,
                "description": "Date de début de modification YYYY-MM-DD. Combiné avec updated_to → plage. Seul → >=date.",
            },
            "updated_to": {
                "type":        "string",
                "required":    False,
                "description": "Date de fin de modification YYYY-MM-DD. Combiné avec updated_from → plage. Seul → <=date.",
            },
            "sort": {
                "type":        "string",
                "required":    False,
                "description": "Tri ex: updated_on:desc | created_on:asc | priority:desc.",
            },
            "limit": {
                "type":        "integer",
                "required":    False,
                "description": "Nombre de résultats par page (max 100, défaut 25). -1 pour tout récupérer automatiquement.",
            },
            "offset": {
                "type":        "integer",
                "required":    False,
                "description": "Pagination (offset), défaut 0. Ignoré si limit=-1.",
            },
        },
        "handler": lambda p, k: IssueTools(api_key=k).list_issues(**p),
    },


    "list_issues_overdue": {
      "description": """
            But :
            Lister les tickets Redmine en retard selon deux critères :
              1. Tickets dont la due_date <= aujourd'hui (ou reference_date)
              2. Tickets appartenant à une version de projet dont la due_date est dépassée

            Quand utiliser :
            - Quand on veut identifier les tickets en retard sur un projet ou un assigné.
            Notes importantes :
            - Par défaut, seuls les tickets ouverts sont retournés (status_id='open').
            - reference_date permet de simuler les retards à une date passée ou future (ex: '2024-01-01').
        """,
        "parameters": {
            "reference_date": {
                "type":        "string",
                "required":    False,
                "description": "Date de référence YYYY-MM-DD pour définir le seuil de retard. Défaut = aujourd'hui. Ex: '2024-01-01' pour voir les tickets qui étaient en retard à cette date",
            },
            "status_id": {
                "type":     "string",
                "required": False,
                "description": (
                    "open=tous les ouverts (défaut recommandé pour les retards) "
                ),
            },
            "project_name": {
                "type":        "string",
                "required":    False,
                "description": (
                    "Nom du projet en texte libre "
                ),
            },
            "assigned_to_name": {
                "type":        "string",
                "required":    False,
                "description": (
                    "Nom de l'utilisateur en texte libre"
                ),
            },
            "tracker_id": {
                "type":        "string",
                "required":    False,
                "description": "ID du tracker",
            },
            "priority_id": {
                "type":        "string",
                "required":    False,
                "description": "ID de la priorité ",
            },
            "parent_id": {
                "type":        "integer",
                "required":    False,
                "description": "ID du ticket parent pour filtrer les sous-tickets",
            },
            "issue_id": {
                "type":        "string",
                "required":    False,
                "description": "ID ou liste d'IDs de tickets séparés par des virgules",
            },
            "subproject_id": {
                "type":        "string",
                "required":    False,
                "description": "ID du sous-projet",
            },
            "include": {
                "type":        "string",
                "required":    False,
                "description": "Champs supplémentaires : 'journals,attachments,relations,children,watchers'",
            },
            "custom_field_id": {
                "type":        "integer",
                "required":    False,
                "description": "ID du champ personnalisé pour filtrer (récupéré via get_custom_fields)",
            },
            "custom_field_value": {
                "type":        "string",
                "required":    False,
                "description": "Valeur exacte du champ personnalisé. ⚠️ custom_field_id ET custom_field_value doivent être fournis ensemble",
            },
            "fixed_version_name": {
                "type":        "string",
                "required":    False,
                "description": (
                    "Nom de la version cible (ex: '1.0.0', 'Sprint 3', 'Release 2024'). "
                    "Nécessite project_id. "
                    "Si la version est introuvable → erreur retournée avec les versions disponibles."
                ),
            },
            "created_on": {
                "type":        "string",
                "required":    False,
                "description": "Filtre date déjà formaté ex: >=YYYY-MM-DD | <=YYYY-MM-DD | ><YYYY-MM-DD|YYYY-MM-DD",
            },
            "created_from": {
                "type":        "string",
                "required":    False,
                "description": "Date de début de création YYYY-MM-DD. Combiné avec created_to → plage. Seul → >=date",
            },
            "created_to": {
                "type":        "string",
                "required":    False,
                "description": "Date de fin de création YYYY-MM-DD. Combiné avec created_from → plage. Seul → <=date",
            },
            "updated_on": {
                "type":        "string",
                "required":    False,
                "description": "Filtre date déjà formaté ex: >=YYYY-MM-DD | <=YYYY-MM-DD | ><YYYY-MM-DD|YYYY-MM-DD",
            },
            "updated_from": {
                "type":        "string",
                "required":    False,
                "description": "Date de début de modification YYYY-MM-DD. Combiné avec updated_to → plage. Seul → >=date",
            },
            "updated_to": {
                "type":        "string",
                "required":    False,
                "description": "Date de fin de modification YYYY-MM-DD. Combiné avec updated_from → plage. Seul → <=date",
            },
            "sort": {
                "type":        "string",
                "required":    False,
                "description": "Tri ex: updated_on:desc | created_on:asc | priority:desc",
            },
            "limit": {
                "type":        "integer",
                "required":    False,
                "description": "Nombre de résultats par page (max 100, défaut 25). -1 pour tout récupérer automatiquement",
            },
            "offset": {
                "type":        "integer",
                "required":    False,
                "description": "Pagination (offset), défaut 0. Ignoré si limit=-1",
            },
        },
        "handler": lambda p, k: IssueTools(api_key=k).list_issues_overdue(**p),
    },


    "get_issue_details": {
        "description": """
            But :
            Retourne les détails complets d'un ticket Redmine.
            (id, subject, description, status, priority, tracker, assigned_to, author,
            project, done_ratio, start_date, due_date, created_on, updated_on,
            estimated_hours, spent_hours, journals, attachments, watchers, children)

            Workflow recommandé :
            1. Si l'ID est inconnu → list_issues ou search d'abord.
            2. Récupérer l'ID du ticket voulu.
            3. Appeler get_issue_details avec cet ID.

            Notes importantes :
            - Toujours chaîner après list_issues pour avoir le contenu complet.
        """,
        "parameters": {
            "issue_id": {
                "type":        "integer",
                "required":    True,
                "description": "ID unique du ticket",
            },
            "include_journals": {
                "type":        "boolean",
                "required":    False,
                "description": "Inclure l'historique et les commentaires (défaut: true)",
            },
            "include_attachments": {
                "type":        "boolean",
                "required":    False,
                "description": "Inclure les pièces jointes (défaut: true)",
            },
            "include_relations": {
                "type":        "boolean",
                "required":    False,
                "description": "Inclure les relations avec d'autres tickets (défaut: true)",
            },
            "include_watchers": {
                "type":        "boolean",
                "required":    False,
                "description": "Inclure la liste des observateurs (défaut: true)",
            },
            "include_children": {
                "type":        "boolean",
                "required":    False,
                "description": "Inclure les sous-tickets (défaut: true)",
            },
        },
        "handler": lambda p, k: IssueTools(api_key=k).get_issue_details(**p),
    },

    "get_issue_relations": {
        "description": """
            But :
            Retourne toutes les relations d'un ticket avec d'autres tickets.

            Quand utiliser :
            - On veut comprendre les dépendances entre tickets.
            - On veut savoir si un ticket est bloqué ou bloquant.
            - On veut trouver les doublons ou tickets liés.

            Workflow recommandé :
            1. Chaîner toujours avec get_issue_details pour le détail des tickets liés.

            Notes :
            - Types de relations : relates | duplicates | duplicated | blocks | blocked | precedes | follows | copied_to | copied_from
        """,
        "parameters": {
            "issue_id": {
                "type":        "integer",
                "required":    True,
                "description": "ID unique du ticket",
            },
        },
        "handler": lambda p, k: IssueTools(api_key=k).get_issue_relations(**p),
    },

    "get_relation_details": {
        "description": """
            But :
            Retourne les détails complets d'une relation spécifique entre deux tickets.

            Workflow recommandé :
            1. Appeler get_issue_relations pour lister les relations d'un ticket.
            2. Récupérer l'ID de la relation voulue.
            3. Appeler get_relation_details et get_issue_details pour plus d'informations avec cet ID.

            Notes :
            - Le champ delay est uniquement renseigné pour les relations precedes/follows.
        """,
        "parameters": {
            "relation_id": {
                "type":        "integer",
                "required":    True,
                "description": "ID unique de la relation (récupérable via get_issue_relations)",
            },
        },
        "handler": lambda p, k: IssueTools(api_key=k).get_relation_details(**p),
    },

    #"get_issues_by_query": {
    #    "description": """
    #        But :
    #        Retourne les tickets filtrés par une requête sauvegardée Redmine.
#
    #        Quand utiliser :
    #        - On veut appliquer un filtre personnalisé créé dans l'interface Redmine.
    #    """,
    #    "parameters": {
    #        "query_id": {
    #            "type":        "integer",
    #            "required":    True,
    #            "description": "ID de la requête sauvegardée (visible dans l'URL de la requête)",
    #        },
    #        "project_name": {
    #            "type":        "string",
    #            "required":    False,
    #            "description": (
    #                "Nom du projet en texte libre "
    #            ),
    #        },
    #        "assigned_to_name": {
    #            "type":        "string",
    #            "required":    False,
    #            "description": (
    #                "Nom de l'utilisateur en texte libre"
    #            ),
    #        },
    #        "priority_id": {
    #            "type":        "string",
    #            "required":    False,
    #            "description": "ID de la priorité : Bas (ID:3) | Normal (ID:4) | Urgent (ID:6)",
    #        },
    #        "tracker_id": {
    #            "type":        "string",
    #            "required":    False,
    #            "description": "ID du tracker",
    #        },
    #        "created_on":   { "type": "string", "required": False, "description": "Filtre date formaté ex: >=YYYY-MM-DD" },
    #        "created_from": { "type": "string", "required": False, "description": "Date de début création YYYY-MM-DD" },
    #        "created_to":   { "type": "string", "required": False, "description": "Date de fin création YYYY-MM-DD" },
    #        "updated_on":   { "type": "string", "required": False, "description": "Filtre date formaté ex: >=YYYY-MM-DD" },
    #        "updated_from": { "type": "string", "required": False, "description": "Date de début modification YYYY-MM-DD" },
    #        "updated_to":   { "type": "string", "required": False, "description": "Date de fin modification YYYY-MM-DD" },
    #        "limit": {
    #            "type":        "integer",
    #            "required":    False,
    #            "description": "Nombre de résultats (max 100, défaut 25). -1 pour tout récupérer",
    #        },
    #        "offset": {
    #            "type":        "integer",
    #            "required":    False,
    #            "description": "Pagination, défaut 0",
    #        },
    #    },
    #    "handler": lambda p, k: IssueTools(api_key=k).get_issues_by_query(**p),
    #},

    "get_issues_by_custom_field": {
        "description": """
            But :
            Retourne les tickets filtrés par la valeur d'un champ personnalisé Redmine.

            Workflow recommandé :
            1. Si l'ID du champ est inconnu → get_custom_fields d'abord.
            2. Identifier le champ voulu et sa valeur exacte.
            3. Appeler get_issues_by_custom_field.
            4. Chaîner avec get_issue_details pour le détail de chaque ticket.

            Notes :
            - La valeur custom_field_value doit être EXACTE (sensible à la casse).
        """,
        "parameters": {
            "custom_field_id": {
                "type":        "integer",
                "required":    True,
                "description": "ID du champ personnalisé (récupérable via get_custom_fields)",
            },
            "custom_field_value": {
                "type":        "string",
                "required":    True,
                "description": "Valeur exacte à rechercher dans le champ personnalisé",
            },
            "priority_id":   { "type": "string",  "required": False, "description": "ID de la priorité" },
            "project_name": {
                "type":        "string",
                "required":    False,
                "description": (
                    "Nom du projet en texte libre "
                ),
            },
            "assigned_to_name": {
                "type":        "string",
                "required":    False,
                "description": (
                    "Nom de l'utilisateur en texte libre"
                ),
            },
            "tracker_id":    { "type": "string",  "required": False, "description": "ID du tracker" },
            "status_id": {
                "type":     "string",
                "required": False,
                "description":("ID du statut du ticket."),
            },
            "created_on":    { "type": "string",  "required": False, "description": "Filtre date formaté ex: >=YYYY-MM-DD" },
            "created_from":  { "type": "string",  "required": False, "description": "Date de début création YYYY-MM-DD" },
            "created_to":    { "type": "string",  "required": False, "description": "Date de fin création YYYY-MM-DD" },
            "updated_on":    { "type": "string",  "required": False, "description": "Filtre date formaté ex: >=YYYY-MM-DD" },
            "updated_from":  { "type": "string",  "required": False, "description": "Date de début modification YYYY-MM-DD" },
            "updated_to":    { "type": "string",  "required": False, "description": "Date de fin modification YYYY-MM-DD" },
            "limit":         { "type": "integer", "required": False, "description": "Nombre de résultats (max 100, défaut 25). -1 pour tout récupérer" },
            "offset":        { "type": "integer", "required": False, "description": "Pagination, défaut 0" },
        },
        "handler": lambda p, k: IssueTools(api_key=k).get_issues_by_custom_field(**p),
    },

    # ─────────────────────────────────────────
    #   PROJECTS
    # ─────────────────────────────────────────

    "list_projects": {
        "description": """
            But :
            Liste les projets Redmine accessibles à l'utilisateur courant.
        

            Quand utiliser :
            - On veut découvrir les projets auxquels j’ai accès

            Notes :
            - limit=-1 pour récupérer tous les projets automatiquement.
        """,
        "parameters": {
            "status" :{
                "type":"string",
                "required":False,
                "description":(
                    "status=*       --> pour tous les projets "
                    "status=actif   --> pour seulement les projets actifs"
                    "status=fermé   --> pour seulement les projets fermés "
                    "status=archivé --> pour seulement les projets archivés "
                ),
            },
            "created_on":    { "type": "string",  "required": False, "description": "Filtre date formaté ex: >=YYYY-MM-DD" },
            "created_from":  { "type": "string",  "required": False, "description": "Date de début création YYYY-MM-DD" },
            "created_to":    { "type": "string",  "required": False, "description": "Date de fin création YYYY-MM-DD" },
            "updated_on":    { "type": "string",  "required": False, "description": "Filtre date formaté ex: >=YYYY-MM-DD" },
            "updated_from":  { "type": "string",  "required": False, "description": "Date de début modification YYYY-MM-DD" },
            "updated_to":    { "type": "string",  "required": False, "description": "Date de fin modification YYYY-MM-DD" },
            "limit":         { "type": "integer", "required": False, "description": "Nombre de résultats (max 100, défaut 25). -1 pour tout récupérer" },
            "offset":        { "type": "integer", "required": False, "description": "Pagination, défaut 0. Ignoré si limit=-1" },
        },
        "handler": lambda p, k: ProjectTools(api_key=k).list_projects(**p),
    },

    "get_project_details": {
        "description": """
            But :
            Retourne les détails complets d'un projet Redmine avec son nom.
            (ID, Name ,trackers activés, catégories d'issues, modules activés)
            Quand utiliser :
            recuperer les détails d'un projet avec son nom  
        """,
        "parameters": {
            "project_name": {
                "type":        "string",
                "required":    True,
                "description": (
                    "Nom du projet en texte libre "
                ),
            },
        },
        "handler": lambda p, k: ProjectTools(api_key=k).get_project_details(**p),
    },

    "get_project_members": {
        "description": """
            But :
            Retourne la liste des membres d'un projet avec leurs rôles.

            Quand utiliser :
            - On veut savoir qui travaille sur un projet.
            Notes :
            - limit=-1 pour récupérer tous les membres automatiquement.
            - Pour les détails complets d'un user → get_user_details.
        """,
        "parameters": {
            "project_name": {
                "type":        "string",
                "required":    True,
                "description": (
                    "Nom du projet en texte libre "
                ),
            },
            "limit":  { "type": "integer", "required": False, "description": "Nombre de résultats (max 100, défaut 25). -1 pour tout récupérer" },
            "offset": { "type": "integer", "required": False, "description": "Pagination, défaut 0" },
        },
        "handler": lambda p, k: ProjectTools(api_key=k).get_project_members(**p),
    },

    "get_project_versions": {
        "description": """
            But :
            Retourne les versions (jalons/milestones/releases) d'un projet.

            Notes :
            - Statuts possibles : open | locked | closed
            - Pour les détails d'une version → get_version_details.
        """,
        "parameters": {
            "project_name": {
                "type":        "string",
                "required":    True,
                "description": (
                    "Nom du projet en texte libre "
                ),
            },
        },
        "handler": lambda p, k: ProjectTools(api_key=k).get_project_versions(**p),
    },

    "get_version_details": {
        "description": """
            But :
            Retourne les détails complets d'une version (jalon/milestone) par son ID.
        """,
        "parameters": {
            "project_name": {
                "type":        "string",
                "required":    True,
                "description": (
                    "Nom du projet en texte libre "
                ),
            },
            "version_name": {
                "type":        "string",
                "required":    True,
                "description": "Nom de la version cible (ex: '1.0.0', 'Sprint 3', 'Release 2024').",
            },
        },
        "handler": lambda p, k: ProjectTools(api_key=k).get_version_details(**p),
    },

    #"get_project_issue_categories": {
    #    "description": """
    #        But :
    #        Retourne les catégories d'issues disponibles dans un projet.
    #    """,
    #    "parameters": {
    #        "project_name": {
    #            "type":        "string",
    #            "required":    True,
    #            "description": (
    #                "Nom du projet en texte libre "
    #            ),
    #        },
    #    },
    #    "handler": lambda p, k: ProjectTools(api_key=k).get_project_issue_categories(**p),
    #},

    #"get_project_files": {
    #    "description": """
    #        But :
    #        Retourne les fichiers publiés dans la section Fichiers d'un projet.
#
    #        Notes :
    #        - Pour les pièces jointes d'un ticket → get_issue_details(include_attachments=true)
    #    """,
    #    "parameters": {
    #        "project_name": {
    #            "type":        "string",
    #            "required":    True,
    #            "description": (
    #                "Nom du projet en texte libre "
    #            ),
    #        },
    #    },
    #    "handler": lambda p, k: ProjectTools(api_key=k).get_project_files(**p),
    #},

    #"get_project_news": {
    #    "description": """
    #        But :
    #        Retourne les actualités publiées dans un projet Redmine.
#
    #        Notes :
    #        - Si le module News n'est pas activé → retourne une liste vide sans erreur.
    #        - limit=-1 pour récupérer toutes les actualités automatiquement.
    #    """,
    #    "parameters": {
    #        "project_name": {
    #            "type":        "string",
    #            "required":    True,
    #            "description": (
    #                "Nom du projet en texte libre "
    #            ),
    #        },
    #        "limit":  { "type": "integer", "required": False, "description": "Nombre de résultats (défaut 25). -1 pour tout récupérer" },
    #        "offset": { "type": "integer", "required": False, "description": "Pagination, défaut 0" },
    #    },
    #    "handler": lambda p, k: ProjectTools(api_key=k).get_project_news(**p),
    #},

    "get_membership_details": {
        "description": """
            But :
            Retourne les détails complets d'un membership spécifique par son ID.

            Workflow recommandé :
            1. Appeler get_project_members pour lister les memberships.
            2. Récupérer l'ID du membership voulu.
            3. Appeler get_membership_details avec cet ID.
        """,
        "parameters": {
            "membership_id": {
                "type":        "integer",
                "required":    True,
                "description": "ID unique du membership (récupérable via get_project_members)",
            },
        },
        "handler": lambda p, k: ProjectTools(api_key=k).get_membership_details(**p),
    },

    # ─────────────────────────────────────────
    #   USERS
    # ─────────────────────────────────────────

    "list_users": {
        "description": """
            But :
            Retourne la liste des utilisateurs Redmine.

            Quand utiliser :
            - On veut rechercher un utilisateur par son nom ou statut.
            - On veut obtenir l'ID d'un utilisateur.

            Workflow recommandé :
            1. Appeler list_users sans filtre name (tolérance aux fautes de frappe).
            2. Chercher manuellement l'utilisateur dans les résultats.
            3. Chaîner avec get_user_details si besoin.

            Notes :
            - Nécessite des droits administrateur.
            - Pour l'utilisateur connecté → get_my_account.
            - limit=-1 pour récupérer tous les utilisateurs automatiquement.
        """,
        "parameters": {
            "status":   { "type": "integer", "required": False, "description": "1=actifs | 2=enregistrés | 3=verrouillés | 0=anonymes, défaut 1" },
            "group_id": { "type": "integer", "required": False, "description": "Filtrer par groupe (ID récupérable via get_groups)" },
            "limit":    { "type": "integer", "required": False, "description": "Nombre de résultats (max 100, défaut 25). -1 pour tout récupérer" },
            "offset":   { "type": "integer", "required": False, "description": "Pagination, défaut 0" },
        },
        "handler": lambda p, k: UserTools(api_key=k).list_users(**p),
    },

    "get_my_account": {
        "description": """
            But :
            Retourne les informations complètes du compte utilisateur connecté.

            Quand utiliser :
            - On veut obtenir les informations de l'utilisateur courant.
            - On veut connaître son propre ID utilisateur.

            Notes :
            - La clé API n'est jamais exposée dans la réponse.
            - Pas besoin de droits admin.
        """,
        "parameters": {},
        "handler": lambda p, k: UserTools(api_key=k).get_my_account(**p),
    },

    "get_user_details": {
        "description": """
            But :
            Retourne les détails complets d'un utilisateur par son ID nottament les projets que il est associe .

            Quand utiliser :
            - On veut obtenir les informations d'un utilisateur .
            - on veut connaître les projets associés à un utilisateur (auxquels il est membre).

            Notes : 

            Workflow recommandé :
            1. Si l'ID est inconnu → list_users ou get_project_members d'abord.
            2. Appeler get_user_details avec l'ID.

        """,
        "parameters": {
            "user_name": {
                "type":        "string",
                "required":    True,
                "description": "Nom de l'utilisateur en texte libre",
            },
        },
        "handler": lambda p, k: UserTools(api_key=k).get_user_details(**p),
    },

    # ─────────────────────────────────────────
    #   TIME ENTRIES
    # ─────────────────────────────────────────

    "list_time_entries": {
        "description": """
            But :
            Liste les entrées de temps avec filtres optionnels.

            Quand utiliser :
            - On veut consulter le temps passé sur un projet ou un ticket.
            - On veut filtrer par période, utilisateur ou activité.

            Workflow recommandé :
            1. Si les activités sont inconnues → get_time_entry_activities d'abord.
            2. Appeler list_time_entries avec project_name ET période.

            Notes :
            - limit=-1 pour récupérer toutes les entrées automatiquement.
            - Toujours filtrer par project_name et/ou période pour des résultats pertinents.
        """,
        "parameters": {
            "project_name": {
                "type":        "string",
                "required":    False,
                "description": (
                    "Nom du projet en texte libre "
                ),
            },
            "issue_id":    { "type": "integer", "required": False, "description": "Filtrer par ticket" },
             "user_name": {
                "type":        "string",
                "required":    False,
                "description": "Nom de l'utilisateur en texte libre",
            },
            "activity_id": { "type": "integer", "required": False, "description": "Filtrer par activité : 9=Développement | 10=Analyse | 11=Documentation | 12=Exploitation | 13=Support | 14=Installation | 15=Réunion | 16=Test | 17=Spécifications" },
            "from_date":   { "type": "string",  "required": False, "description": "Date de début YYYY-MM-DD" },
            "to_date":     { "type": "string",  "required": False, "description": "Date de fin YYYY-MM-DD" },
            "limit":       { "type": "integer", "required": False, "description": "Nombre de résultats (max 100, défaut 25). -1 pour tout récupérer" },
            "offset":      { "type": "integer", "required": False, "description": "Pagination, défaut 0" },
        },
        "handler": lambda p, k: TimeEntryTools(api_key=k).list_time_entries(**p),
    },

    "get_time_entry_details": {
        "description": """
            But :
            Retourne les détails complets d'une entrée de temps par son ID.

            Workflow recommandé :
            1. Appeler list_time_entries pour trouver l'ID voulu.
            2. Appeler get_time_entry_details avec cet ID.
        """,
        "parameters": {
            "time_entry_id": {
                "type":        "integer",
                "required":    True,
                "description": "ID unique de l'entrée de temps (récupérable via list_time_entries)",
            },
        },
        "handler": lambda p, k: TimeEntryTools(api_key=k).get_time_entry_details(**p),
    },

    # ─────────────────────────────────────────
    #   WIKI
    # ─────────────────────────────────────────

    "list_wiki_pages": {
        "description": """
            But :
            Liste toutes les pages wiki d'un projet (titres et métadonnées uniquement, sans contenu).

            Workflow recommandé :
            1. Appeler list_wiki_pages pour voir les pages disponibles.
            2. Chaîner avec get_wiki_page pour lire le contenu d'une page précise.
            3. Ou chaîner avec get_all_wiki_pages_with_content pour tout le contenu.

            Notes :
            - Retourne uniquement les titres et métadonnées, pas le contenu.
        """,
        "parameters": {
            "project_name": {
                "type":        "string",
                "required":    True,
                "description": (
                    "Nom du projet en texte libre "
                ),
            },
        },
        "handler": lambda p, k: WikiTools(api_key=k).list_wiki_pages(**p),
    },

    "get_wiki_page": {
        "description": """
            But :
            Retourne le contenu complet d'une page wiki par son titre exact.
            Supporte la récupération d'une révision spécifique via le paramètre version.

            Workflow recommandé :
            1. Si le titre exact est inconnu → list_wiki_pages d'abord.
            2. Appeler get_wiki_page avec le project_name et le titre exact.
            3. Pour une révision spécifique → ajouter version=numero_révision
            Notes :
            - Le titre est sensible à la casse et doit être exact.
            - version=None    → retourne la version actuelle
            - version=X      → retourne la révision numéro X
            - L'historique complet des révisions n'est pas accessible via l'API Redmine
        """,
        "parameters": {
            "project_name": {
                "type":        "string",
                "required":    True,
                "description": (
                    "Nom du projet en texte libre "
                ),
            },
            "page_title": {
                "type":        "string",
                "required":    True,
                "description": "Titre exact de la page wiki (récupérable via list_wiki_pages)",
            },
        },
        "handler": lambda p, k: WikiTools(api_key=k).get_wiki_page(**p),
    },

    #"get_all_wiki_pages_with_content": {
    #    "description": """
    #        But :
    #        Retourne TOUTES les pages wiki d'un projet avec leur contenu complet.
#
    #        Quand utiliser :
    #        - On veut une vue complète de toute la documentation d'un projet.
    #        - search n'a pas retourné de résultats pertinents.
    #        - Ingestion RAG de toute la documentation wiki d'un projet.
#
    #        Notes :
    #        - Peut être lent si le projet a beaucoup de pages wiki.
    #    """,
    #    "parameters": {
    #        "project_name": {
    #            "type":        "string",
    #            "required":    True,
    #            "description": (
    #                "Nom du projet en texte libre "
    #            ),
    #        },
    #    },
    #    "handler": lambda p, k: WikiTools(api_key=k).get_all_wiki_pages_with_content(**p),
    #},

    # ─────────────────────────────────────────
    #   SEARCH
    # ─────────────────────────────────────────

    "search": {
        "description": """
            But :
            Recherche globale dans tous les types de contenu Redmine
            (tickets, wiki, documents, projets, actualités, messages, changesets).

            Quand utiliser :
            - Point d'entrée principal pour trouver rapidement des ressources.
            - Le type de contenu recherché n'est pas connu à l'avance.

            Workflow recommandé :
            1. Appeler search avec le mot-clé.
            2. Analyser les detailles des résultats retournés et identifier ceux qui répondent précisément à la question posée.
            3. Chaîner toujours selon le type retourné :
               issue     → get_issue_details
               wiki-page → get_wiki_page
               project   → get_project_details
               news      → get_project_news
            4. Si aucun résultat → réessayer en autre langue (anglais/français).

            Traitement des résultats :
            - NE PAS retourner tous les résultats bruts à l'utilisateur
            - Filtrer et garder UNIQUEMENT les résultats qui répondent
              directement à la question posée
            - Si plusieurs résultats correspondent → les présenter de manière
              synthétique et organisée

            Notes :
            - limit est toujours fixé à 25
            - Ne jamais utiliser limit=-1 avec search
            - search ne supporte PAS les filtres de dates
        """,
        "parameters": {
            "query": {
                "type":        "string",
                "required":    True,
                "description": "Texte à rechercher",
            },
            "project_name": {
                "type":        "string",
                "required":    False,
                "description": (
                    "Nom du projet en texte libre "
                ),
            },
            "issues":     { "type": "boolean", "required": False, "description": "Rechercher dans les tickets (défaut: true)" },
            "news":       { "type": "boolean", "required": False, "description": "Rechercher dans les actualités (défaut: true)" },
            "documents":  { "type": "boolean", "required": False, "description": "Rechercher dans les documents (défaut: true)" },
            "changesets": { "type": "boolean", "required": False, "description": "Rechercher dans les changesets (défaut: true)" },
            "wiki_pages": { "type": "boolean", "required": False, "description": "Rechercher dans le wiki (défaut: true)" },
            "messages":   { "type": "boolean", "required": False, "description": "Rechercher dans les messages (défaut: true)" },
            "projects":   { "type": "boolean", "required": False, "description": "Rechercher dans les projets (défaut: true)" },
            "limit":      { "type": "integer", "required": False, "description": "Nombre de résultats (défaut 25). -1 pour tout récupérer" },
            "offset":     { "type": "integer", "required": False, "description": "Pagination, défaut 0" },
        },
        "handler": lambda p, k: SearchTools(api_key=k).search(**p),
    },

    # ─────────────────────────────────────────
    #   REFERENCES
    # ─────────────────────────────────────────

    #"get_references": {
    #    "description": """
    #        But :
    #        Retourne toutes les données de référence Redmine en un seul appel parallèle.
    #        (statuts, trackers, priorités, activités, rôles, groupes, champs personnalisés, catégories documents)
#
    #        Quand utiliser :
    #        - En début de session pour initialiser le contexte.
    #        - On a besoin de plusieurs références en même temps.
#
    #        Notes :
    #        - Appeler en premier avant d'utiliser les autres outils.
    #        - Les appels sont parallélisés via asyncio.gather.
    #    """,
    #    "parameters": {},
    #    "handler": lambda p, k: ReferenceTools(api_key=k).get_all_references(**p),
    #},

    "get_issue_statuses": {
        "description": "Retourne uniquement les statuts disponibles pour les tickets.",
        "parameters": {},
        "handler": lambda p, k: ReferenceTools(api_key=k).get_issue_statuses(**p),
    },

    "get_trackers": {
        "description": "Retourne tous les trackers disponibles dans Redmine.",
        "parameters": {},
        "handler": lambda p, k: ReferenceTools(api_key=k).get_trackers(**p),
    },

    "get_priorities": {
        "description": "Retourne toutes les priorités disponibles pour les tickets.",
        "parameters": {},
        "handler": lambda p, k: ReferenceTools(api_key=k).get_priorities(**p),
    },

    "get_time_entry_activities": {
        "description": """
            But :
            Retourne toutes les activités disponibles pour les entrées de temps.

            Quand utiliser :
            - Avant d'appeler list_time_entries avec un filtre activity_id.
        """,
        "parameters": {},
        "handler": lambda p, k: ReferenceTools(api_key=k).get_time_entry_activities(**p),
    },

    #"get_roles": {
    #    "description": "Retourne tous les rôles disponibles dans Redmine.",
    #    "parameters": {},
    #    "handler": lambda p, k: ReferenceTools(api_key=k).get_roles(**p),
    #},

    #"get_groups": {
    #    "description": """
    #        But :
    #        Retourne tous les groupes disponibles dans Redmine.
    #        Nécessite des droits administrateur.
    #        Utiliser avant list_users avec le filtre group_id.
    #    """,
    #    "parameters": {},
    #    "handler": lambda p, k: ReferenceTools(api_key=k).get_groups(**p),
    #},

    "get_custom_fields": {
        "description": """
            Retourne tous les champs personnalisés disponibles dans Redmine.
            Nécessite des droits administrateur.
        """,
        "parameters": {},
        "handler": lambda p, k: ReferenceTools(api_key=k).get_custom_fields(**p),
    },

    #"get_document_categories": {
    #    "description": "Retourne toutes les catégories de documents disponibles.",
    #    "parameters": {},
    #    "handler": lambda p, k: ReferenceTools(api_key=k).get_document_categories(**p),
    #},
}

# ─────────────────────────────────────────────
#   MODELES PYDANTIC
# ─────────────────────────────────────────────

class ToolCallRequest(BaseModel):
    tool:       str
    parameters: dict[str, Any] = {}
    api_key:    str


# ─────────────────────────────────────────────
#   ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    logger.info("Health check demandé")
    return {"status": "ok", "service": "mcp_server"}


@app.get("/tools/list")
async def list_tools():
    """
    Retourne la liste de tous les outils MCP disponibles
    avec leur description et paramètres.
    """
    logger.info("list_tools demandé")

    tools = []
    for name, config in TOOLS_REGISTRY.items():
        tools.append({
            "name":        name,
            "description": config["description"].strip(),
            "parameters":  config["parameters"],
            "returns":     config.get("returns", {}),
            "examples":    config.get("examples", []),
        })

    return {
        "success":     True,
        "tools":       tools,
        "total_count": len(tools),
    }


@app.post("/tools/call")
async def call_tool(request: ToolCallRequest):

    # ── Outil introuvable ─────────────────────
    if request.tool not in TOOLS_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Outil '{request.tool}' introuvable.",
        )

    tool_config = TOOLS_REGISTRY[request.tool]
    schema      = tool_config["parameters"]

    # ── Validation des paramètres requis ──────
    for param_name, param_config in schema.items():
        if param_config.get("required") and param_name not in request.parameters:
            raise HTTPException(
                status_code=422,
                detail=f"Paramètre requis manquant : '{param_name}'",
            )

    # ── Cast des types ────────────────────────
    casted_params = _cast_parameters(request.parameters, schema)

    # ── Appel du handler ──────────────────────
    try:
        result = await tool_config["handler"](casted_params, request.api_key)
        return {
            "success": True,
            "tool":    request.tool,
            "result":  result,
        }

    # ── Erreur inattendue ─────────────────────
    except Exception as e:
        logger.error(f"call_tool | erreur inattendue | tool={request.tool} | {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur inattendue lors de l'exécution de '{request.tool}' : {str(e)}",
        )
