def tokenize_telugu(sentence):
    """
    Tokenize Telugu text by splitting on spaces and handling punctuation.
    """
    # Simple space-based tokenization for Telugu
    # In production, you would use a more sophisticated tokenizer
    tokens = []
    current_token = ""
    
    for char in sentence:
        if char in " \t\n":
            if current_token:
                tokens.append(current_token)
                current_token = ""
        elif char in "।॥,.?!":  # Common Telugu punctuation
            if current_token:
                tokens.append(current_token)
                current_token = ""
            tokens.append(char)
        else:
            current_token += char
    
    if current_token:
        tokens.append(current_token)
    
    return tokens
