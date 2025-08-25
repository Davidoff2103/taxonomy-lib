def normalize_street_name(street: str):
    s = street.split()
    MAPPING = {
        "CL": "Carrer",
        "BJ": "Baixada",
        "PZ": "Plaça",
        "AV": "Avinguda",
        "PJ": "Passatge",
        "PS": "Passeig",
        "RB": "Rambla",
        "TT": "Torrent"
    }

    parts = street.strip().split(maxsplit=1)
    first_word = parts[0]
    rest = parts[1]
    normalized_first = MAPPING.get(first_word, first_word)
    return f"{normalized_first} {rest}".strip()
