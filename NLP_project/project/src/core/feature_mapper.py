def format_ud_feats(features):
    if not features:
        return "_"

    items = sorted(features.items())

    formatted = [f"{k}={v}" for (k, v) in items]

    return "|".join(formatted)

# print(format_ud_feats({"Number": "Plur"}))
# print(format_ud_feats({"Tense": "Pres", "Aspect": "Prog"}))
# print(format_ud_feats({}))

print(format_ud_feats({"Tense": "Pres", "Aspect": "Prog"}))