from typing import List, Dict, Optional, Tuple
import copy
from core import Rule, TokenContext, extract_context

class RuleEngine:
    def __init__(self, rules: List[Rule] = None, exception_dict: Dict[str, Tuple[str, str]] = None):
        self.rules = rules if rules is not None else []
        self.exception_dict = exception_dict if exception_dict is not None else {}
        self.default_feat = "_"
    
    def add_rule(self, rule: Rule):
        self.rules.append(rule)
        
    def set_exception(self, word: str, lemma: str, feats: str):
         self.exception_dict[word] = (lemma, feats)

    def predict(self, tokens: List[str]) -> Tuple[List[Tuple[str, str]], List[Dict]]:
        """
        Runs the rule engine over a sequence of tokens.
        Returns the output (lemma, feat) tuples, and a list of explainability logs per token.
        """
        tags = []
        logs = []
        
        for i, word in enumerate(tokens):
            context = extract_context(tokens, i, pos_tags=[t[1] for t in tags]) # use predicted feats as pos surrogate for now
            
            # Check exceptions first
            if word.lower() in self.exception_dict:
                 lemma, feat = self.exception_dict[word.lower()]
                 tags.append((lemma, feat))
                 logs.append({
                     "word": word,
                     "lemma": lemma,
                     "feat": feat,
                     "reason": "Exception Dictionary Override",
                     "rule_name": None,
                     "confidence": 1.0
                 })
                 continue
                 
            # Find matching rules
            matches = [rule for rule in self.rules if rule.matches(context)]
            
            if not matches:
                 tags.append((word, self.default_feat)) # lemma defaults to word
                 logs.append({
                     "word": word,
                     "lemma": word,
                     "feat": self.default_feat,
                     "reason": "Default fallback",
                     "rule_name": None,
                     "confidence": 0.0
                 })
                 continue
                 
            # Conflict Resolution: Sort by Priority then Confidence
            matches.sort(key=lambda r: (r.priority, r.confidence), reverse=True)
            best_rule = matches[0]
            
            pred_lemma, pred_feat = best_rule.apply(context)
            tags.append((pred_lemma, pred_feat))
            logs.append({
                 "word": word,
                 "lemma": pred_lemma,
                 "feat": pred_feat,
                 "reason": "Highest priority/confidence rule match",
                 "rule_name": best_rule.name,
                 "confidence": best_rule.confidence
            })
            
        return tags, logs
