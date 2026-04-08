LEMMA_DICT_TELUGU = {
    "నేను": "నేను",  # I (pronoun)
    "నువ్వు": "నువ్వు",  # You (informal)
    "వారు": "వారు",  # They (honorific)
    "ఆమె": "ఆమె",  # She
    "అతను": "అతను",  # He
    "పిల్ల": "పిల్ల",  # child
}


def lemmatize_telugu(token):
    """
    Lemmatize Telugu words using dictionary lookup and rule-based approaches.
    """
    word = token
    
    # 1. Check dictionary first
    if word in LEMMA_DICT_TELUGU:
        return LEMMA_DICT_TELUGU[word]
    
    # 2. Plural to singular: remove plural markers
    # లు is the most common plural suffix
    if word.endswith("లు"):
        base = word[:-2]
        return base
    
    # 3. Remove common verb suffixes
    # -అట (infinitive marker)
    if word.endswith("అట"):
        return word[:-2]
    
    # -ింది (past tense, feminine singular)
    if word.endswith("ింది"):
        return word[:-4]
    
    # -ాడు (past tense, masculine singular)
    if word.endswith("ాడు"):
        return word[:-3]
    
    # -ాను (past tense, first person)
    if word.endswith("ాను"):
        return word[:-3]
    
    # -తున్న (present continuous)
    if word.endswith("తున్న"):
        return word[:-5]
    
    # -ుతుంది (present continuous, formal)
    if word.endswith("ుతుంది"):
        return word[:-6]
    
    # -ుతున్నారు (present continuous, plural/formal)
    if word.endswith("ుతున్నారు"):
        return word[:-9]
    
    # Default: return the word as-is
    return word
