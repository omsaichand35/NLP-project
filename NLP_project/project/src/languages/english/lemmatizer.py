LEMMA_DICT = {
    "went": "go",
    "gone": "go",
    "children": "child",
    "mice": "mouse",
    "better": "good"

}


def lemmatize(token):
    word = token.lower()

    # 1. Check the dictionary first:
    if word in LEMMA_DICT:
        return LEMMA_DICT[word]

    # 2. Rule-based approach

    # studies -> study
    if word.endswith("ies"):
        return word[:-3] + "y"

    # running -> run
    if word.endswith("ing"):
        base = word[:-3]

        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base

    # carried -> carry
    if word.endswith("ied"):
        return word[:-3] + "y"

    # played -> play
    if word.endswith("ed"):
        base = word[:-2]

        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base


    # cats -> cat
    if word.endswith("s"):
        return word[:-1]

    #default
    return word

# print(lemmatize("swimming"))
# print(lemmatize("stopped"))
# print(lemmatize("goes"))