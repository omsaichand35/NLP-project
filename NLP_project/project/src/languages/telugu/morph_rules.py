from collections import defaultdict

# Learned suffix statistics (dataset-driven, rule-based)
SUFFIX_STATS_TELUGU = {}


def _parse_target_feats_telugu(feats_str):
    """Parse UD feature string into a dictionary."""
    parsed = {}
    
    if not feats_str or feats_str == "_":
        return parsed
    
    for item in feats_str.split("|"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key in {"Number", "Tense", "Aspect", "Gender", "Mood", "Voice"} and value:
            parsed[key] = value
    
    return parsed


def learn_suffix_stats_telugu(sentences, min_suffix_len=2, max_suffix_len=4, debug=False, top_n=10):
    """Learn suffix statistics from Telugu dataset."""
    counts = defaultdict(lambda: defaultdict(int))
    
    for sentence in sentences:
        for token in sentence:
            word = token.get("form", "")
            if not word:
                continue
            
            feats = _parse_target_feats_telugu(token.get("feats", "_"))
            if not feats:
                continue
            
            max_len = min(max_suffix_len, len(word))
            for size in range(min_suffix_len, max_len + 1):
                suffix = word[-size:]
                for key, value in feats.items():
                    pattern = f"{key}={value}"
                    counts[suffix][pattern] += 1
    
    suffix_stats = {sfx: dict(pattern_counts) for sfx, pattern_counts in counts.items()}
    
    if debug and suffix_stats:
        ranked = sorted(
            suffix_stats.items(),
            key=lambda kv: sum(kv[1].values()),
            reverse=True
        )[:top_n]
        print("Top learned Telugu suffix patterns:")
        for suffix, pattern_counts in ranked:
            print(f"  {suffix}: {pattern_counts}")
    
    return suffix_stats


def set_suffix_stats_telugu(suffix_stats):
    """Set the learned suffix statistics."""
    global SUFFIX_STATS_TELUGU
    SUFFIX_STATS_TELUGU = suffix_stats or {}


def predict_from_suffix_telugu(word, suffix_stats):
    """Predict morphological feature from learned suffix patterns."""
    if not word or not suffix_stats:
        return {}
    
    # Prefer longer suffixes first (4 -> 2) for specificity
    for size in range(4, 1, -1):
        if len(word) < size:
            continue
        
        suffix = word[-size:]
        pattern_counts = suffix_stats.get(suffix)
        if not pattern_counts:
            continue
        
        best_pattern = max(pattern_counts.items(), key=lambda kv: kv[1])[0]
        if "=" not in best_pattern:
            return {}
        
        key, value = best_pattern.split("=", 1)
        return {key: value}
    
    return {}


# Common Telugu function words (particles, postpositions, etc.)
FUNCTION_WORDS_TELUGU = {
    "ఆ", "ఈ", "ఉ", "ఆ",  # demonstratives
    "నీ", "అతని", "ఆమె", "వారి", "నా", "అతన్", "ఆమె",  # possessives
    "న", "కు", "నుండి", "కూడా", "కోసం", "వద్ద", "ద్వారా",  # postpositions
    "లేదా", "కానీ", "మరియు", "అయితే",  # conjunctions
    "చాలా", "చాలానీ", "అంత",  # adverbs
}

# Common Telugu words with known morphological features
EXCEPTIONS_TELUGU = {
    "నేను": {"Number": "Sing"},  # I (1st person singular)
    "నువ్వు": {"Number": "Sing"},  # You (2nd person singular)
    "ఆయన": {"Gender": "Masc", "Number": "Sing"},  # He (honorific)
    "ఆమె": {"Gender": "Fem", "Number": "Sing"},  # She
    "వారు": {"Number": "Plur"},  # They (plural/honorific)
    "అది": {"Number": "Sing"},  # It (singular neuter)
    "ఇవి": {"Number": "Plur"},  # These (plural)
}


def is_plural_telugu(word):
    """
    Detect plural forms in Telugu.
    Common plural suffixes: -లు, -ండ్లు, -ట్లు
    """
    if word.endswith("లు"):
        return True
    if word.endswith("ండ్లు"):
        return True
    if word.endswith("ట్లు"):
        return True
    return False


def is_verb_telugu(word):
    """
    Detect verb forms in Telugu.
    Common verb markers for various tenses and moods.
    """
    # Comprehensive Telugu verb suffixes
    verb_endings = [
        # Infinitive
        "తు", "ము",
        # Past tense markers
        "న", "ాడు", "ింది", "ాను", "ాసాను",
        # Present continuous
        "తున్న", "ుతున్న", "ుతుంది", "ుతున్నారు",
        # Future
        "బో", "బోను", "భ",
        # Subjunctive/Conditional
        "చు", "చూ", "గా",
    ]
    for ending in verb_endings:
        if word.endswith(ending):
            return True
    return False


def analyze_morph_telugu(token):
    """
    Analyze morphological features for Telugu words.
    Returns dictionary with features like Number, Tense, Aspect, Gender, etc.
    
    Priority:
    1. Exceptions (known words)
    2. Function words (return empty)
    3. Verbs (tense/aspect only)
    4. Nouns (number only if confident)
    """
    word = token
    features = {}
    
    # Skip empty strings
    if not word:
        return features
    
    # Check exceptions first
    if word in EXCEPTIONS_TELUGU:
        result = EXCEPTIONS_TELUGU[word].copy()
        # Filter to only UD standard features
        return {k: v for k, v in result.items() if k in {"Number", "Tense", "Aspect", "Gender", "Mood", "Voice"}}
    
    # Skip function words - return empty (no features)
    if word in FUNCTION_WORDS_TELUGU:
        return features
    
    # Detect if it's a verb first (before number assignment)
    if is_verb_telugu(word):
        # Detect tense and aspect for verbs
        # Past tense
        if word.endswith(("ాడు", "ింది", "ాను", "ాసాను")):
            features["Tense"] = "Past"
        
        # Present continuous
        elif word.endswith(("తున్న", "ుతుంది", "ుతున్నారు")):
            features["Tense"] = "Pres"
            features["Aspect"] = "Prog"
        
        # Future tense
        elif word.endswith(("బో", "భ")):
            features["Tense"] = "Fut"
        
        # Default to Present for other verbs
        else:
            features["Tense"] = "Pres"
        
        # NOTE: Don't assign Number for verbs unconditionally
        # Only assign if there's gender/person marking that implies plurality
        if word.endswith(("ారు", "ాము")):  # plural markers
            features["Number"] = "Plur"
    
    # For non-verbs, try to detect noun number
    else:
        # Detect plural for nouns ONLY if clear plural markers
        if is_plural_telugu(word):
            features["Number"] = "Plur"
        # Don't assign "Sing" unconditionally - let dataset defaults handle it
    
    # Detect gender (if possible from morphology)
    if word.endswith("ిక"):  # feminine adjective suffix
        features["Gender"] = "Fem"
    
    # Learned suffix fallback (never override handcrafted signals)
    for key, value in predict_from_suffix_telugu(word, SUFFIX_STATS_TELUGU).items():
        if key not in features:
            features[key] = value
    
    return features
