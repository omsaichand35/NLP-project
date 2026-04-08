from typing import List, Tuple
from collections import defaultdict
from core import Rule
from engine import RuleEngine
from learner import AdaptiveLearner

class Trainer:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.sentences: List[List[Tuple[str, str]]] = []
        
    def load_conllu(self):
        """
        Parses CoNLL-U format into List of sentences.
        Each sentence is a List of (token, (gold_lemma, gold_feat)).
        """
        with open(self.data_path, "r", encoding="utf-8") as file:
            current_sentence = []
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    if current_sentence:
                        self.sentences.append(current_sentence)
                        current_sentence = []
                    continue
                
                parts = line.split("\t")
                if len(parts) > 6 and not ("-" in parts[0] or "." in parts[0]):
                    word = parts[1]
                    lemma = parts[2]
                    feats = parts[5]
                    if feats == "_":
                        feats = parts[3] # Fallback to UPOS if feats are empty
                    current_sentence.append((word, (lemma, feats)))
                    
            if current_sentence:
                self.sentences.append(current_sentence)
                
    def train(self, engine: RuleEngine, epochs: int = 5):
        learner = AdaptiveLearner(engine)
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            lemma_correct = 0
            feat_correct = 0
            total = 0

            # Macro calc variables
            true_positives = defaultdict(int)
            false_positives = defaultdict(int)
            false_negatives = defaultdict(int)
            
            for sentence in self.sentences:
                words = [t[0] for t in sentence]
                gold_tags = [t[1] for t in sentence]
                
                predicted_tags, explain_logs = engine.predict(words)
                
                learner.evaluate_and_update(words, gold_tags, predicted_tags, explain_logs)
                
                for i in range(len(words)):
                    gold_lemma, gold_feat = gold_tags[i]
                    pred_lemma, pred_feat = predicted_tags[i]
                    
                    if gold_lemma == pred_lemma:
                        lemma_correct += 1
                        
                    if gold_feat == pred_feat:
                        feat_correct += 1
                        true_positives[gold_feat] += 1
                    else:
                        false_positives[pred_feat] += 1
                        false_negatives[gold_feat] += 1
                        
                    total += 1
            
            lemma_acc = (lemma_correct / total) * 100 if total > 0 else 0
            feat_acc = (feat_correct / total) * 100 if total > 0 else 0
            
            # overall F1 Macro approx
            macro_f1 = 0
            classes = set(true_positives.keys()) | set(false_positives.keys()) | set(false_negatives.keys())
            f1_scores = {}
            for c in classes:
                tp = true_positives[c]
                fp = false_positives[c]
                fn = false_negatives[c]
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_c = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores[c] = f1_c
            
            if classes:
                macro_f1 = sum(f1_scores.values()) / len(classes)
                
            print(f"Epoch {epoch + 1} - Lemma Acc: {lemma_acc:.2f}% | Feat Acc: {feat_acc:.2f}% | Feat F1: {macro_f1:.2f}")
            learner._prune_rules() # Extra prune pass after epoch
            
    def evaluate(self, engine: RuleEngine, sentences: List[List[Tuple[str, Tuple[str, str]]]], prefix: str = ""):
        lemma_correct = 0
        feat_correct = 0
        total = 0
        
        for sentence in sentences:
            words = [t[0] for t in sentence]
            gold_tags = [t[1] for t in sentence]
            
            predicted_tags, _ = engine.predict(words)
            
            for i in range(len(words)):
                gold_lemma, gold_feat = gold_tags[i]
                pred_lemma, pred_feat = predicted_tags[i]
                
                if gold_lemma == pred_lemma:
                    lemma_correct += 1
                    
                if gold_feat == pred_feat:
                    feat_correct += 1
                    
                total += 1

        lemma_acc = lemma_correct / total if total > 0 else 0
        feat_acc = feat_correct / total if total > 0 else 0
        
        if prefix:
            print(f"{prefix} Lemma Accuracy: {lemma_acc}")
            print(f"{prefix} Feature Accuracy: {feat_acc}")
        else:
            print(f"Lemma Accuracy: {lemma_acc}")
            print(f"Feature Accuracy: {feat_acc}")
