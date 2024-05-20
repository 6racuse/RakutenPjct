def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """Fonction d'affichage de progress bar, récupérée sur le Notebook du TD1 de Data Sciences de Myriam Tami
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()
        


        
def raw_to_tokens(raw_string, spacy_nlp):
        # Write code for lower-casing
    string = raw_string.lower()

    # Write code to normalize the accents
    string = normalize_accent(string)

    # Write code to tokenize
    spacy_tokens = spacy_nlp(string)
    

    # Write code to remove punctuation tokens and create string tokens
    string_tokens = [token.orth_ for token in spacy_tokens if not token.is_punct if not token.is_stop]
    # Write code to join the tokens back into a single string
    clean_string = " ".join(string_tokens)

    return clean_string


def normalize_accent(string):
    string = string.replace('á', 'a')
    string = string.replace('â', 'a')
    string = string.replace('à', 'a')
    
    string = string.replace('é', 'e')
    string = string.replace('è', 'e')
    string = string.replace('ê', 'e')
    string = string.replace('ë', 'e')

    string = string.replace('î', 'i')
    string = string.replace('ï', 'i')
    string = string.replace('n°','n')
    string = string.replace('ö', 'o')
    string = string.replace('ô', 'o')
    string = string.replace('ò', 'o')
    string = string.replace('ó', 'o')

    string = string.replace('ù', 'u')
    string = string.replace('û', 'u')
    string = string.replace('ü', 'u')

    string = string.replace('ç', 'c')

    return string