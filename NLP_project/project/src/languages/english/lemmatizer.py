LEMMA_DICT = {
    "went": "go",
    "gone": "go",
    "children": "child",
    "mice": "mouse",
    "better": "good",
    "had": "have",
    "has": "have",
    "was": "be",
    "were": "be",
    "am": "be",
    "am": "be",
    "is": "be",
    "are": "be",
    "being": "be",
    "been": "be",
    "children": "child",
    "grazing": "graze",
    "processed": "process",
    "criteria": "criterion",
    "data": "datum",
    "feet": "foot",
    "teeth": "tooth"
}


def lemmatize(token):
    if not token:
        return ""
        
    orig_word = token
    word = token.lower()

    is_capitalized = orig_word[0].isupper()
    is_upper = orig_word.isupper() and len(orig_word) > 1

    # 1. Check the dictionary first:
    if word in LEMMA_DICT:
        res = LEMMA_DICT[word]
    else:
        # 2. Rule-based approach
        res = word
        # studies -> study
        if word.endswith("ies"):
            res = word[:-3] + "y"
        # running -> run
        elif word.endswith("ing"):
            base = word[:-3]
            if len(base) >= 2 and base[-1] == base[-2] and base[-1] != 's':
                base = base[:-1]
            res = base
        # carried -> carry
        elif word.endswith("ied"):
            res = word[:-3] + "y"
        # played -> play
        elif word.endswith("ed"):
            base = word[:-2]
            if len(base) >= 2 and base[-1] == base[-2] and base[-1] != 's':
                base = base[:-1]
            res = base
        # cats -> cat
        elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
            res = word[:-1]

    # Handle Case restoration (Align with NLTK: follow input case strictly)
    if is_upper:
        return res.upper()
    if is_capitalized:
        # If it was a dictionary hit like "are" -> "be", but input was capitalized "Are", 
        # we capitalize the result "Be".
        return res.capitalize()
    return res

# print(lemmatize("swimming"))
# print(lemmatize("stopped"))
# print(lemmatize("goes"))