class CancelledByUser(Exception):
    """Levée quand l'utilisateur annule une requête en cours."""
    pass

class LDAPAuthError(Exception):
    """erreurs d'authentification LDAP """
    pass