from src.languages.english.lemmatizer import lemmatize
from src.languages.english.morph_rules import analyze_morph, learn_suffix_stats, set_suffix_stats
from src.languages.telugu.lemmatizer import lemmatize_telugu
from src.languages.telugu.morph_rules import analyze_morph_telugu, learn_suffix_stats_telugu, set_suffix_stats_telugu
from src.core.feature_mapper import format_ud_feats


def evaluate(sentences):
    set_suffix_stats(learn_suffix_stats(sentences))

    total = 0
    lemma_correct = 0
    feat_correct = 0

    for sentence in sentences:
        for token in sentence:
            word = token["form"]

            # skip punctuation
            if token["upos"] == "PUNCT":
                continue

            true_lemma = token["lemma"]
            true_feats = token["feats"]

            pred_lemma = lemmatize(word)
            pred_feats = format_ud_feats(analyze_morph(word))

            if pred_lemma == true_lemma:
                lemma_correct += 1

            if pred_feats == true_feats:
                feat_correct += 1

            total += 1

    print("Lemma Accuracy:", lemma_correct / total)
    print("Feature Accuracy:", feat_correct / total)

def evaluate_telugu(sentences):
    # Learn suffix patterns from the dataset
    set_suffix_stats_telugu(learn_suffix_stats_telugu(sentences))
    
    total = 0
    lemma_correct = 0
    feat_correct = 0

    for sentence in sentences:
        for token in sentence:
            word = token["form"]

            # skip punctuation
            if token["upos"] == "PUNCT":
                continue

            true_lemma = token["lemma"]
            true_feats = token["feats"]

            pred_lemma = lemmatize_telugu(word)
            pred_feats = format_ud_feats(analyze_morph_telugu(word))

            if pred_lemma == true_lemma:
                lemma_correct += 1

            if pred_feats == true_feats:
                feat_correct += 1

            total += 1

    print("Telugu Lemma Accuracy:", lemma_correct / total)
    print("Telugu Feature Accuracy:", feat_correct / total)