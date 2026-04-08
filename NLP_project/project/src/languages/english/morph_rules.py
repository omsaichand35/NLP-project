import re
from collections import defaultdict

# -------------------
# Function words (skip assigning features)
# -------------------
FUNCTION_WORDS = {
    "in", "on", "at", "the", "a", "an", "and", "of", "to", "for", "by",
    "with", "from", "as", "is", "was", "are", "were", "be", "been", "being"
}

FUNCTION_WORDS.update({
    "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "not", "very", "so", "too"
})

# -------------------
# Exception dictionary (dataset-driven truths)
# -------------------
EXCEPTIONS = {
    "is": {"Tense": "Pres"},
    "was": {"Tense": "Past"},
    "are": {"Tense": "Pres"},
    "were": {"Tense": "Past"},
    "has": {"Tense": "Pres"},
    "had": {"Tense": "Past"},
    "does": {"Tense": "Pres"},
    "studies": {"Tense": "Pres"}  # 3rd person singular verb
}


# -------------------
# Learned suffix statistics (dataset-driven, rule-based)
# -------------------
SUFFIX_STATS = {}


def _parse_target_feats(feats_str):
    parsed = {}

    if not feats_str or feats_str == "_":
        return parsed

    for item in feats_str.split("|"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key in {"Number", "Tense", "Aspect"} and value:
            parsed[key] = value

    return parsed


def learn_suffix_stats(sentences, min_suffix_len=2, max_suffix_len=4, debug=False, top_n=10):
    counts = defaultdict(lambda: defaultdict(int))

    for sentence in sentences:
        for token in sentence:
            word = token.get("form", "").lower()
            if not word.isalpha():
                continue

            feats = _parse_target_feats(token.get("feats", "_"))
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
        print("Top learned suffix patterns:")
        for suffix, pattern_counts in ranked:
            print(f"  {suffix}: {pattern_counts}")

    return suffix_stats


def set_suffix_stats(suffix_stats):
    global SUFFIX_STATS
    SUFFIX_STATS = suffix_stats or {}


def predict_from_suffix(word, suffix_stats):
    if not word or not suffix_stats:
        return {}

    lowered = word.lower()

    # Prefer longer suffixes first (4 -> 2) for specificity.
    for size in range(4, 1, -1):
        if len(lowered) < size:
            continue

        suffix = lowered[-size:]
        pattern_counts = suffix_stats.get(suffix)
        if not pattern_counts:
            continue

        best_pattern = max(pattern_counts.items(), key=lambda kv: kv[1])[0]
        if "=" not in best_pattern:
            return {}

        key, value = best_pattern.split("=", 1)
        return {key: value}

    return {}


# -------------------
# Plural detection
# -------------------
def is_plural(word):
    # studies, cities
    if re.search(r"ies$", word):
        return True

    # buses, boxes, classes
    if re.search(r"(ses|xes|zes|ches|shes)$", word):
        return True

    # simple plurals: cats
    if re.search(r"[a-z]{3,}s$", word):
        if word.endswith(("us", "is")):
            return False
        return True

    return False


# -------------------
# Verb detection (rule-based POS approximation)
# -------------------
def is_verb(word):
    if word.endswith("ed") or word.endswith("ing"):
        return True
    
    # "ies" is ambiguous: could be verb (studies, flies) or noun plural (universities, cities)
    # Only treat short "ies" words as verbs; longer ones are usually noun plurals
    if word.endswith("ies") and len(word) <= 7:
        return True
    
    return False


# -------------------
# Main Morph Analyzer
# -------------------
def analyze_morph(token):
    word = token.lower()
    features = {}

    # -------------------
    # Skip punctuation
    # -------------------
    if word in [".", ",", "!", "?"]:
        return features

    # -------------------
    # Exception handling
    # -------------------
    if word in EXCEPTIONS:
        return EXCEPTIONS[word]

    # -------------------
    # Skip function words
    # -------------------
    if word in FUNCTION_WORDS:
        return features

    # -------------------
    # VERB RULES
    # -------------------
    if is_verb(word):

        if word.endswith("ed"):
            features["Tense"] = "Past"

        elif word.endswith("ing"):
            features["Tense"] = "Pres"
            features["Aspect"] = "Prog"

        elif word.endswith("ies"):
            features["Tense"] = "Pres"

    # -------------------
    # NOUN RULES
    # -------------------
    else:
        # assign number ONLY if confident it's a noun
        if is_plural(word):
            features["Number"] = "Plur"
        elif re.search(r"[a-z]{4,}$", word):  # longer words more likely nouns
            features["Number"] = "Sing"

    # -------------------
    # GENDER
    # -------------------
    if word in ["he", "him"]:
        features["Gender"] = "Masc"
    elif word in ["she", "her"]:
        features["Gender"] = "Fem"

    # -------------------
    # Learned suffix fallback (never override handcrafted signals)
    # -------------------
    for key, value in predict_from_suffix(word, SUFFIX_STATS).items():
        if key not in features:
            features[key] = value

    return features