from typing import List, Dict, Optional, Tuple
from core import Rule, TokenContext, extract_context

class AdaptiveLearner:
    def __init__(self, engine):
        self.engine = engine
        self.learning_rate = 0.05
        self.max_failures_threshold = 5  # Allow max 5 failures before pruning if conf is low
        self.confidence_threshold = 0.4  # Remove rules below 0.4
        self.specialized_rule_count = 0
        
    def evaluate_and_update(self, tokens: List[str], gold_tags: List[Tuple[str, str]], predicted_tags: List[Tuple[str, str]], explain_logs: List[Dict]):
        """
        Evaluate predictions, update rule confidences, prune weak rules, 
        and specialize failing rules. (gold/pred are tuples of (lemma, feats))
        """
        errors_by_rule = {}

        for i, (word, gold, pred, log) in enumerate(zip(tokens, gold_tags, predicted_tags, explain_logs)):
            rule_name = log.get("rule_name")
            
            # Always learn vocabulary exceptions for incorrect predictions, even if it's from default fallback
            if gold != pred:
                self._add_to_exceptions(word.lower(), gold[0], gold[1])

            if not rule_name:
                continue

            # Find rule instance
            rule = next((r for r in self.engine.rules if r.name == rule_name), None)
            if not rule:
                continue

            # A rule must get both lemma and feats correct to be fully "success"
            if gold == pred:
                # Correct prediction! Update confidence and success tracking.
                rule.successes += 1
                rule.confidence = min(1.0, rule.confidence + self.learning_rate)
            else:
                # Incorrect Prediction. Update failure tracking and lower confidence.
                rule.failures += 1
                rule.confidence = max(0.0, rule.confidence - self.learning_rate)
                
                # Store context of failure for specialization.
                context = extract_context(tokens, i, pos_tags=[t[1] for t in predicted_tags])
                if rule_name not in errors_by_rule:
                    errors_by_rule[rule_name] = []
                errors_by_rule[rule_name].append({
                    "context": context,
                    "gold_lemma": gold[0],
                    "gold_feat": gold[1]
                })

            # Specialization
        for rule_name, error_list in errors_by_rule.items():
             rule = next((r for r in self.engine.rules if r.name == rule_name), None)
             if len(error_list) >= 3 and rule: # if failing repeatedly in this epoch
                  self._specialize_rule(rule, error_list)

        # Prune weak rules
        self._prune_rules()

    def _add_to_exceptions(self, word: str, lemma: str, feat: str):
        # Simplistic heuristic: if the word is consistently failing across the epochs, add exception.
        self.engine.set_exception(word, lemma, feat)

    def _specialize_rule(self, base_rule: Rule, error_list: List[Dict]):
        """
        Create a more specific rule based on the context of false positive errors.
        """
        import uuid
        
        # Look for common context among errors. For example, common previous word.
        prev_words = [err["context"].prev_word for err in error_list if err["context"].prev_word]
        if not prev_words: return
        
        from collections import Counter
        most_common_prev, count = Counter(prev_words).most_common(1)[0]
        
        # If a specific previous word is a common factor in errors, build a specialized rule
        if count >= 2:
            subset_errs = [err for err in error_list if err["context"].prev_word == most_common_prev]
            gold_feat = subset_errs[0]["gold_feat"]
            # Specialized lemma action isn't handled as easily automatically; default to word
            
            def condition(ctx: TokenContext, prev=most_common_prev, base_cond=base_rule.condition):
                return base_cond(ctx) and (ctx.prev_word is not None and ctx.prev_word.lower() == prev.lower())
                
            new_rule = Rule(
                name=f"{base_rule.name}_specialized_prev_{most_common_prev}_{str(uuid.uuid4())[:4]}",
                condition=condition,
                feat_output=gold_feat,
                lemma_action=lambda w: subset_errs[0]["gold_lemma"], # hardcoded lemma override as fallback
                confidence=0.8,
                priority=base_rule.priority + 1 # Higher priority than base rule
            )
            self.engine.add_rule(new_rule)
            self.specialized_rule_count += 1
            print(f"SPECIALIZED: Created rule {new_rule.name}")

    def _prune_rules(self):
        """
        Remove rules with high failures or low confidence.
        """
        initial_count = len(self.engine.rules)
        pruned_rules = [
            r for r in self.engine.rules 
            if not(r.total_applications() > 5 and r.confidence < self.confidence_threshold)
        ]
        self.engine.rules = pruned_rules
        if len(pruned_rules) < initial_count:
            print(f"PRUNED {initial_count - len(pruned_rules)} rules.")
