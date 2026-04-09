from dataclasses import dataclass
from typing import Callable, Optional, Dict, List, Any, Tuple

@dataclass
class TokenContext:
    word: str
    prev_word: Optional[str] = None
    next_word: Optional[str] = None
    prev_pos: Optional[str] = None
    next_pos: Optional[str] = None
    dependency: Optional[str] = None
    is_capitalized: bool = False # NEW: Track case information

class Rule:
    def __init__(
            self,
            name: str,
            condition: Callable[[TokenContext], bool],
            feat_output: str,
            lemma_action: Callable[[str], str] = lambda w: w,
            confidence: float = 0.5,
            priority: int = 1
    ):
        self.name = name
        self.condition = condition
        self.feat_output = feat_output
        self.lemma_action = lemma_action
        self.confidence = confidence
        self.priority = priority
        self.successes = 0
        self.failures = 0
    
    def matches(self, context: TokenContext) -> bool:
        try:
            return self.condition(context)
        except Exception:
            return False
            
    def apply(self, context: TokenContext) -> Tuple[str, str]:
        """Returns (pred_lemma, pred_feats)"""
        return self.lemma_action(context.word), self.feat_output

    def total_applications(self):
        return self.successes + self.failures

    def hit_rate(self) -> float:
        total = self.total_applications()
        if total == 0:
            return 0.0
        return self.successes / total

    def __repr__(self):
        return f"Rule({self.name} -> {self.feat_output} | conf: {self.confidence:.2f}, hits: {self.successes}/{self.total_applications()})"

def extract_context(tokens: List[str], index: int, pos_tags: List[str] = None) -> TokenContext:
    """Helper to safely build TokenContext from a list of words."""
    word = tokens[index]
    prev_word = tokens[index - 1] if index > 0 else None
    next_word = tokens[index + 1] if index < len(tokens) - 1 else None
    
    prev_pos = None
    next_pos = None
    if pos_tags:
        prev_pos = pos_tags[index - 1] if index > 0 else None
        next_pos = pos_tags[index + 1] if index < len(pos_tags) - 1 else None

    return TokenContext(
        word=word,
        prev_word=prev_word,
        next_word=next_word,
        prev_pos=prev_pos,
        next_pos=next_pos,
        is_capitalized=word[0].isupper() if word else False
    )
