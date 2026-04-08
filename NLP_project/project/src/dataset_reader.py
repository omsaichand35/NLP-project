def read_conllu(file_path):
    sentences = []
    current_sentence = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            if line.startswith("#"):
                continue

            parts = line.split("\t")

            if len(parts) < 6:
                continue

            token_data = {
                "form": parts[1],
                "lemma": parts[2],
                "upos": parts[3],
                "feats": parts[5],
            }

            current_sentence.append(token_data)
    return sentences