import sys
import os
from pathlib import Path

# Add the project root to sys.path so we can import your excellent base modules
sys.path.append(str(Path(__file__).resolve().parent.parent / "project"))

from src.languages.english.lemmatizer import lemmatize as eng_lemmatize
from src.languages.english.morph_rules import analyze_morph as eng_analyze_morph, set_suffix_stats, learn_suffix_stats
from src.languages.telugu.lemmatizer import lemmatize_telugu as tel_lemmatize
from src.languages.telugu.morph_rules import analyze_morph_telugu as tel_analyze_morph, set_suffix_stats_telugu, learn_suffix_stats_telugu
from src.core.feature_mapper import format_ud_feats

from core import Rule
from engine import RuleEngine
from trainer import Trainer

def get_base_rules(language="english") -> RuleEngine:
    engine = RuleEngine()
    
    if language == "english":
        # Wrap your superior project lemmatizer and morph analyzer into our Baseline Rule
        base_rule = Rule(
            name="legacy_project_baseline",
            condition=lambda ctx: True, # Always fires as the default baseline
            feat_output="_", # Dynamic, handled in overridden apply
            lemma_action=lambda w: eng_lemmatize(w),
            confidence=0.5,
            priority=0 # Lowest priority so specializations override it
        )
        
        # Override the apply function to use your dynamic morph analyzer directly
        def apply_override(context):
            pred_lemma = eng_lemmatize(context.word)
            pred_feats = format_ud_feats(eng_analyze_morph(context.word))
            return pred_lemma, pred_feats
            
        base_rule.apply = apply_override
        engine.add_rule(base_rule)
        
    elif language == "telugu":
        base_rule = Rule(
            name="legacy_project_baseline_telugu",
            condition=lambda ctx: True,
            feat_output="_",
            lemma_action=lambda w: tel_lemmatize(w),
            confidence=0.5,
            priority=0
        )
        
        def apply_override_telugu(context):
            pred_lemma = tel_lemmatize(context.word)
            pred_feats = format_ud_feats(tel_analyze_morph(context.word))
            return pred_lemma, pred_feats
            
        base_rule.apply = apply_override_telugu
        engine.add_rule(base_rule)

    return engine

def run_pipeline():
    # 3. Load Sample Data and Train
    data_file = Path(__file__).resolve().parent.parent / "project" / "data" / "raw" / "en_ewt-ud-train.conllu"
    
    trainer = Trainer(data_path=str(data_file))
    try:
        trainer.load_conllu()
        # Create a real Train/Test split to avoid data leakage
        all_data = trainer.sentences
        split_idx = int(len(all_data) * 0.8)
        train_dataset = all_data[:split_idx]
        test_dataset = all_data[split_idx:]
    except FileNotFoundError:
        print("CoNLL-U dataset not found. Using a dummy dataset for demonstration.")
        # Dummy data matching conllu structure
        dummy_data = [
            [("The", ("the", "Definite=Def|PronType=Art")), ("running", ("run", "VerbForm=Ger")), ("man", ("man", "Number=Sing")), ("is", ("be", "Mood=Ind|Tense=Pres|VerbForm=Fin")), ("fast", ("fast", "Degree=Pos"))],
            [("A", ("a", "Definite=Ind|PronType=Art")), ("running", ("running", "Number=Sing")), ("is", ("be", "Mood=Ind|Tense=Pres|VerbForm=Fin")), ("fun", ("fun", "Degree=Pos"))],
            [("She", ("she", "Case=Nom|PronType=Prs")), ("enjoyed", ("enjoy", "Tense=Past|VerbForm=Part")), ("the", ("the", "Definite=Def|PronType=Art")), ("running", ("running", "Number=Sing"))],
            [("I", ("I", "Case=Nom|PronType=Prs")), ("am", ("be", "Mood=Ind|Tense=Pres|VerbForm=Fin")), ("running", ("run", "VerbForm=Ger"))],
            [("Running", ("running", "Number=Sing")), ("shoes", ("shoe", "Number=Plur")), ("are", ("be", "Mood=Ind|Tense=Pres|VerbForm=Fin")), ("expensive", ("expensive", "Degree=Pos"))]
        ]
        train_dataset = dummy_data[:3]
        test_dataset = dummy_data[3:]
        
    print(f"Training on {len(train_dataset)} sentences, evaluating on {len(test_dataset)} sentences.")
    
    # --- English Setup ---
    # First prep your project's custom statistical suffix learners
    set_suffix_stats(learn_suffix_stats([
        # format your data into {"form": word, "feats": feats} just like your old project for the initial learner
        [{"form": w, "feats": f} for w, (l, f) in sentence] for sentence in train_dataset
    ]))
    
    trainer.sentences = train_dataset
    eng_engine = get_base_rules("english")
    trainer.train(eng_engine, epochs=5)
    
    print("\n=== ENGLISH EVALUATION (UNSEEN DATA) ===")
    trainer.evaluate(eng_engine, test_dataset, prefix="")

    # --- Telugu Setup ---
    # Load Telugu specific dataset
    telugu_file = Path(__file__).resolve().parent.parent / "project" / "data" / "te_mtg-ud-train.conllu"
    trainer_tel = Trainer(data_path=str(telugu_file))
    try:
        trainer_tel.load_conllu()
        all_tel_data = trainer_tel.sentences
        split_idx_t = int(len(all_tel_data) * 0.8)
        train_tel = all_tel_data[:split_idx_t]
        test_tel = all_tel_data[split_idx_t:]
    except FileNotFoundError:
        train_tel = train_dataset
        test_tel = test_dataset
    
    set_suffix_stats_telugu(learn_suffix_stats_telugu([
        [{"form": w, "feats": f} for w, (l, f) in sentence] for sentence in train_tel
    ]))
    
    trainer_tel.sentences = train_tel
    telugu_engine = get_base_rules("telugu") 
    trainer_tel.train(telugu_engine, epochs=5)
    
    print("\n=== TELUGU EVALUATION (UNSEEN DATA) ===")
    trainer_tel.evaluate(telugu_engine, test_tel, prefix="Telugu")

if __name__ == "__main__":
    run_pipeline()
