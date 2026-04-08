import re

def tokenize(sentence: str):

    # \b = word boundary
    # \w+ = one or more word characters (letters)
    #[.,!?] = punctuation tokens

    tokens = re.findall(r"\b\w+\b|[.,!?]", sentence)

    return tokens

# print(tokenize("He was running."))