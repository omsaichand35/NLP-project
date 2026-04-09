import sys
from pathlib import Path
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from tabulate import tabulate # Assuming it might be available, otherwise I'll use simple formatting

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
sys.path.append(str(root_dir / "project"))
sys.path.append(str(root_dir / "innovation"))

# Import Project Baseline
from src.languages.english.lemmatizer import lemmatize as project_lemmatize
from src.languages.english.morph_rules import analyze_morph as project_analyze_morph
from src.core.feature_mapper import format_ud_feats

# Import Innovation (Adaptive Engine)
from innovation.core import Rule, extract_context
from innovation.engine import RuleEngine
from innovation.trainer import Trainer
from innovation.main import get_base_rules

def nltk_to_ud(word, tag):
    """Map NLTK Penn Treebank tags to UD features."""
    feats = []
    
    # Simple mapping
    if tag.startswith('V'): # Verb
        if tag == 'VBD': feats.append("Tense=Past")
        elif tag == 'VBG': feats.append("Tense=Pres|Aspect=Prog")
        elif tag in ['VBP', 'VBZ']: feats.append("Tense=Pres")
    elif tag == 'NNS': # Noun plural
        feats.append("Number=Plur")
    elif tag == 'NN': # Noun singular
        feats.append("Number=Sing")
    elif tag == 'JJR': feats.append("Degree=Cmp")
    elif tag == 'JJS': feats.append("Degree=Sup")
    elif tag == 'JJ': feats.append("Degree=Pos")
    elif tag == 'PRP': feats.append("PronType=Prs")
    
    return "|".join(feats) if feats else "_"

def get_nltk_result(sentence):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    
    results = []
    for word, tag in tagged:
        # NLTK lemmatizer needs pos for accuracy
        wn_pos = 'n'
        if tag.startswith('V'): wn_pos = 'v'
        elif tag.startswith('J'): wn_pos = 'a'
        elif tag.startswith('R'): wn_pos = 'r'
        
        lemma = lemmatizer.lemmatize(word, pos=wn_pos)
        ud_feats = nltk_to_ud(word, tag)
        results.append((lemma, ud_feats))
    return results

def get_project_result(sentence):
    from src.core.tokenizer import tokenize
    tokens = tokenize(sentence)
    results = []
    for token in tokens:
        lemma = project_lemmatize(token)
        feats = format_ud_feats(project_analyze_morph(token))
        results.append((lemma, feats))
    return results

class GlobalStats:
    def __init__(self):
        self.total_tokens = 0
        self.project_vs_innov_agreement = 0
        self.innov_vs_nltk_agreement = 0
        self.project_vs_nltk_agreement = 0
        self.discrepancies = []

def run_comparison(sentence, stats=None):
    print(f"\nAnalyzing Sentence: \"{sentence}\"")
    print("-" * 50)
    
    # Pre-tokenize once to keep everything aligned
    tokens = word_tokenize(sentence)
    
    # 1. Project Baseline
    project_res = get_project_result(sentence)
    # Ensure alignment with tokens
    project_res = project_res[:len(tokens)] 
    
    # 2. Innovation Engine (Briefly trained for context)
    innov_engine = get_base_rules("english")
    innov_tags, _ = innov_engine.predict(tokens)
    
    # 3. NLTK
    nltk_res = get_nltk_result(sentence)
    nltk_res = nltk_res[:len(tokens)]

    # Table data
    table_data = []
    for i in range(len(tokens)):
        word = tokens[i]
        p_l, p_f = project_res[i] if i < len(project_res) else ("-", "-")
        i_l, i_f = innov_tags[i] if i < len(innov_tags) else ("-", "-")
        n_l, n_f = nltk_res[i] if i < len(nltk_res) else ("-", "-")
        
        table_data.append([
            word,
            f"{p_l} ({p_f})",
            f"{i_l} ({i_f})",
            f"{n_l} ({n_f})"
        ])
        
        if stats:
            stats.total_tokens += 1
            if (p_l, p_f) == (i_l, i_f): stats.project_vs_innov_agreement += 1
            if (i_l, i_f) == (n_l, n_f): stats.innov_vs_nltk_agreement += 1
            if (p_l, p_f) == (n_l, n_f): stats.project_vs_nltk_agreement += 1
            
            # Catch interesting discrepancies for the analysis result
            if (i_l, i_f) != (n_l, n_f):
                stats.discrepancies.append((word, f"Innov: {i_l}/{i_f}", f"NLTK: {n_l}/{n_f}"))
    
    headers = ["Word", "Project (Baseline)", "Innovation (Adaptive)", "NLTK"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def generate_analysis_summary(stats):
    print("\n" + "="*50)
    print("           ANALYSIS RESULTS SUMMARY")
    print("="*50)
    
    if stats.total_tokens == 0: return

    summary = [
        ["Total Tokens Analyzed", stats.total_tokens],
        ["Project vs Innovation Agreement (%)", f"{(stats.project_vs_innov_agreement/stats.total_tokens)*100:.1f}%"],
        ["Innovation vs NLTK Agreement (%)", f"{(stats.innov_vs_nltk_agreement/stats.total_tokens)*100:.1f}%"],
        ["Shared Discrepancy Sample", "See below"]
    ]
    print(tabulate(summary, headers=["Metric", "Result"], tablefmt="grid"))
    
    print("\nTop 5 Notable Discrepancies (Learning Opportunities):")
    headers = ["Word", "Innovation Result", "NLTK Result (Reference)"]
    print(tabulate(stats.discrepancies[:10], headers=headers, tablefmt="simple"))

if __name__ == "__main__":
    try:
        from tabulate import tabulate
    except ImportError:
        def tabulate(data, headers, **kwargs):
            header_str = " | ".join(f"{h: <25}" for h in headers)
            print(header_str)
            print("-" * len(header_str))
            for row in data:
                print(" | ".join(f"{str(item): <25}" for item in row))
            return ""

    sentences = [
        "The children are playing better today.",
        "The criteria were surprisingly clear.",
        "The sheep were grazing in the field.",
        "He read the books that he had already read.",
        "The data are being processed."
    ]
    
    global_stats = GlobalStats()
    for s in sentences:
        run_comparison(s, stats=global_stats)
    
    generate_analysis_summary(global_stats)
