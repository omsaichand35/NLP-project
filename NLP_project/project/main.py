from pathlib import Path

from src.dataset_reader import read_conllu
from src.evaluator import evaluate, evaluate_telugu

data_file = Path(__file__).resolve().parent / "data" / "raw" / "en_ewt-ud-train.conllu"
data = read_conllu(str(data_file))

print("=== ENGLISH EVALUATION ===")
evaluate(data[:50])

print("\n=== TELUGU EVALUATION ===")
evaluate_telugu(data[:50])