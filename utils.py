def normalize_street_name(street: str):
    """
    Normalize the first word of a Catalan street name abbreviation.
    
    This function takes a street name string and replaces the first word if it matches
    a known Catalan street abbreviation. The mapping is based on common abbreviations
    used in Catalan street names and their corresponding full forms.
    
    Parameters:
        street (str): The input street name string to normalize.
    
    Returns:
        str: The normalized street name with the first word replaced if it's an abbreviation.
    """
    s = street.split()  # Split the street name into individual words (not used in final logic)
    
    MAPPING = {
        "CL": "Carrer",
        "BJ": "Baixada",
        "PZ": "Plaça",  # Catalan for "Plaza"
        "AV": "Avinguda",  # Catalan for "Avenue"
        "PJ": "Passatge",  # Catalan for "Passage"
        "PS": "Passeig",  # Catalan for "Promenade"
        "RB": "Rambla",  # A type of pedestrian street in Catalan cities
        "TT": "Torrent"  # A type of natural stream or path in Catalan geography
    }

    # Split the street name into two parts at the first whitespace
    parts = street.strip().split(maxsplit=1)
    first_word = parts[0]  # Extract the first word (street abbreviation)
    rest = parts[1]  # Extract the remaining part of the street name
    
    # Replace the first word with its normalized form using the MAPPING dictionary
    # If the abbreviation isn't in the mapping, keep the original first word
    normalized_first = MAPPING.get(first_word, first_word)
    
    # Combine the normalized first word with the rest of the name and return
    return f"{normalized_first} {rest}".strip()