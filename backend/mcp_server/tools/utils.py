

def _format_date_filter(operator: str, *dates: str) -> str:
    """
    Formate un filtre de date pour l'API Redmine.
    ⚠️ PAS d'encodage URL → httpx s'en charge automatiquement.

    Exemples :
        _format_date_filter("><", "2026-03-29", "2026-04-01")
        → "><2026-03-29|2026-04-01"

        _format_date_filter(">=", "2026-03-29")
        → ">=2026-03-29"

        _format_date_filter("<=", "2026-04-01")
        → "<=2026-04-01"
    """

    if operator == "><" and len(dates) == 2:
        return f"><{dates[0]}|{dates[1]}"
    elif operator == ">=":
        return f">={dates[0]}"
    elif operator == "<=":
        return f"<={dates[0]}"
    else:
        return dates[0] if dates else ""


def _apply_date_filter(
    params,           # list of tuples maintenant
    field:     str,
    exact:     str = None,
    from_date: str = None,
    to_date:   str = None,
) -> None:

    if exact:
        if exact.startswith("><"):
            # plage déjà formatée ><date1|date2 → on split
            parts = exact[2:].split("|")
            if len(parts) == 2:
                params.append(("f[]",           field))
                params.append((f"op[{field}]",  "><"))
                params.append((f"v[{field}][]", parts[0]))   # ← valeur 1
                params.append((f"v[{field}][]", parts[1]))   # ← valeur 2
            return
        elif exact.startswith(">="):
            op, value = ">=", exact[2:]
        elif exact.startswith("<="):
            op, value = "<=", exact[2:]
        elif exact.startswith(">"):
            op, value = ">",  exact[1:]
        elif exact.startswith("<"):
            op, value = "<",  exact[1:]
        else:
            op, value = "=",  exact

        params.append(("f[]",           field))
        params.append((f"op[{field}]",  op))
        params.append((f"v[{field}][]", value))

    elif from_date and to_date:
        # ── Plage → deux v[field][] séparés ──────────────────
        params.append(("f[]",           field))
        params.append((f"op[{field}]",  "><"))
        params.append((f"v[{field}][]", from_date))   # ← valeur 1
        params.append((f"v[{field}][]", to_date))     # ← valeur 2

    elif from_date:
        params.append(("f[]",           field))
        params.append((f"op[{field}]",  ">="))
        params.append((f"v[{field}][]", from_date))

    elif to_date:
        params.append(("f[]",           field))
        params.append((f"op[{field}]",  "<="))
        params.append((f"v[{field}][]", to_date))
