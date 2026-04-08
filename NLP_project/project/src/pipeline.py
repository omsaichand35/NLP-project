import sys
from pathlib import Path

# Ensure src package is discoverable when run directly or imported
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.tokenizer import tokenize
from src.core.feature_mapper import format_ud_feats

from src.languages.english.lemmatizer import lemmatize
from src.languages.english.morph_rules import analyze_morph

from src.languages.telugu.tokenizer import tokenize_telugu
from src.languages.telugu.lemmatizer import lemmatize_telugu
from src.languages.telugu.morph_rules import analyze_morph_telugu


def guess_pos(token):
    word = token.lower()

    if word in ["he", "she", "they"]:
        return "PRON"
    if word in [".", ",", "!", "?"]:
        return "PUNCT"
    if word.endswith("ing") or word.endswith("ed"):
        return "VERB"
    return "NOUN"


def process_sentence(sentence, lang="en"):
    if lang == "te":
        tokens = tokenize_telugu(sentence)
    else:
        tokens = tokenize(sentence)

    output_lines = []

    for i, token in enumerate(tokens, start=1):
        if lang == "te":
            lemma = lemmatize_telugu(token)
            feats = format_ud_feats(analyze_morph_telugu(token))
            pos = "NOUN"
        else:
            lemma = lemmatize(token)

            # Keep punctuation handling for English output.
            if token in [".", ",", "!", "?"]:
                feats = "_"
            else:
                feats = format_ud_feats(analyze_morph(token))

            pos = guess_pos(token)

        row = [
            str(i),       # ID
            token,        # FORM
            lemma,        # LEMMA
            pos,          # UPOS
            "_",          # XPOS
            feats,        # FEATS
            "0",          # HEAD (dummy)
            "root",       # DEPREL (dummy)
            "_",          # DEPS
            "_"           # MISC
        ]

        output_lines.append("\t".join(row))

    return "\n".join(output_lines)

print(process_sentence("She is doing her homework.", lang="en"))
print(process_sentence("పిల్లలు ఆడుతున్నారు", lang="te"))