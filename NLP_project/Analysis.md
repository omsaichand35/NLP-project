# Comparative Analysis: Static vs. Adaptive Morphological Analyzer

This document outlines the architectural and functional differences between the baseline `project` and the new `innovation` implementation of the Morphological Analyzer. 

Both systems adhere strictly to the constraint of avoiding deep learning or neural network models, operating purely within the realm of **Symbolic AI and Rule-Based logic**. However, the `innovation` branch introduces an iterative, self-improving engine that significantly enhances accuracy without manual intervention.

---

## 1. Architectural Overview

### The Baseline (`project/`)
The original analyzer relies on a **static procedural pipeline**:
* **Hardcoded Dictionaries:** Uses fixed, pre-defined lookup tables for lemmas (`LEMMA_DICT`) and exceptions (`EXCEPTIONS`).
* **Static Rules:** Uses fixed `if/else` and Regex statements to match suffixes and strip characters.
* **One-Pass Execution:** The system takes a token, runs it through the static rules, and returns a result. It does not learn from its mistakes during execution.

### The Innovation (`innovation/`)
The new analyzer introduces an **Adaptive Rule Engine**:
* **Object-Oriented Rules:** Rules are encapsulated in a `Rule` class with metrics like `confidence`, `successes`, `failures`, and `priority`.
* **Iterative Learning (`AdaptiveLearner`):** Instead of a single pass, the engine runs through multiple epochs over a training dataset. It compares its predictions to Universal Dependencies (UD) gold labels and mathematically updates the rule engine's internal state.
* **Conflict Resolution:** Multiple rules can evaluate the same token. The engine dynamically picks the outcome using the highest `priority` and `confidence` score.

---

## 2. Major Changes & Innovations

The `innovation` branch introduces five major capabilities over the baseline project:

### A. Dynamic Rule Pruning
In the baseline setup, a bad or overly broad rule (like assuming any word ending in "s" is plural) will cause errors endlessly. 
* **Innovation:** The `AdaptiveLearner` tracks the hit-rate of every rule. If a rule's confidence drops below a specified threshold (e.g., 0.40) or has too many consecutive failures, the system **prunes (deletes)** the rule entirely from memory.

### B. Context-Aware Rule Specialization
The baseline looks at words in isolation (or with minimal global context). 
* **Innovation:** When the system gets a prediction wrong repeatedly, it looks at the `TokenContext` (e.g., the previous word, the next word). If a pattern emerges (e.g., the rule fails specifically when the previous word is "The"), the system algorithmically generates and injects a **new, specialized rule** combining the original rule condition with the context condition, boosting its priority.

### C. Automatic Exception Memorization
* **Baseline:** Exceptions had to be hand-coded by the developer (e.g., `"went": "go"`).
* **Innovation:** During the training epochs, any token prediction that fails but has a consistent gold label in the UD dataset is automatically added to an internal `ExceptionDictionary`. The system organically bootstraps its own vocabulary map from its failures.

### D. Explainability Logs
* **Baseline:** Returns the parsed Lemma and Features in a black-box style native return.
* **Innovation:** Every prediction is paired with an explainability log. The system outputs exactly **which rule fired**, **why it fired**, and the **confidence score** at the time of execution. This makes tracking edge cases highly transparent.

### E. Synergy with Baseline (Hybrid Bootstrapping)
Rather than abandoning the old logic, the `innovation` logic seamlessly wrapped the original `lemmatize` and `analyze_morph` scripts as **Priority 0 (Baseline) rules**. 
* Because of this, the new system starts exactly at the baseline's accuracy (~73% Lemma / 40% Feat) and strictly improves upon it during training by dynamically compiling overrides for the baseline's blind spots.

---

## 3. Performance Implications

Because the `innovation` pipeline runs an 80/20 Train/Test split:
1. **Training Phase:** It runs epochs over 80% of the dataset, weeding out bad rules, composing specialized context rules, and filling out its exception lexicon.
2. **Testing Phase:** It leverages its optimized rule-base against the **unseen** 20% of the data.

This eliminates Data Leakage (cheating via memorization of the test set) while naturally boosting accuracy over the baseline by mitigating repeating false-positives algorithmically.

## 4. Conclusion
The transition from `project/` to `innovation/` represents a shift from **Static Expert Systems** to **Adaptive Symbolic Systems**. By applying simple deterministic adjustments—scoring, counting context frequency, pruning, and caching—the innovation achieves machine-learning-style continuous improvement while satisfying the strict parameter to avoid deep learning.