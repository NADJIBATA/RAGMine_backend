from backend.orchestrator.exception import LDAPAuthError
from ldap3 import Server, Connection, ALL, SUBTREE, NTLM
from ldap3.core.exceptions import LDAPBindError, LDAPException
from backend.shared.config import get_settings

from backend.shared.logger import setup_logger

logger = setup_logger("LDAP")
settings = get_settings()

def authenticate_and_fetch_user(username: str, password: str) -> dict:
    """
    Authentifie via LDAP et retourne les infos de l'utilisateur.
    """
    server = Server(settings.ldap_host, port=settings.ldap_port, get_info=ALL)

    # Bind avec compte de service pour chercher le DN utilisateur
    try:
        service_conn = Connection(
            server,
            user=settings.ldap_service_user,
            password=settings.ldap_service_password,
            auto_bind=True,
        )
    except LDAPBindError as e:
        logger.error(f"ldap | échec bind service | {str(e)}")
        raise LDAPAuthError("Erreur de connexion au serveur LDAP")

    search_filter = f"(&(objectClass=user)(objectCategory=person)(sAMAccountName={username}))"

    service_conn.search(
        search_base=settings.ldap_base_dn,
        search_filter=search_filter,
        search_scope=SUBTREE,
        attributes=["givenName", "sn", "mail"],
    )

    if not service_conn.entries:
        service_conn.unbind()
        raise LDAPAuthError("Login ou mot de passe incorrect")

    user_entry = service_conn.entries[0]
    logger.info(f"ldap | user_enty : {user_entry}")
    user_dn = user_entry.entry_dn
    logger.info(f"ldap | user_dn : {user_dn}")

    firstname = str(user_entry.givenName) if user_entry.givenName else ""
    lastname = str(user_entry.sn) if user_entry.sn else ""
    mail = str(user_entry.mail) if user_entry.mail else ""

    service_conn.unbind()

    # Bind avec DN utilisateur + mot de passe → vérification identité
    try:
        user_conn = Connection(server, user=user_dn, password=password, auto_bind=True)
        user_conn.unbind()
    except LDAPBindError:
        raise LDAPAuthError("Login ou mot de passe incorrect")
    except LDAPException as e:
        logger.error(f"ldap | exception bind user | {str(e)}")
        raise LDAPAuthError("Erreur de connexion au serveur LDAP")

    logger.info(f"ldap | authentification réussie | {username}")

    return {
        "login": username,
        "firstname": firstname,
        "lastname": lastname,
        "mail": mail,
    }