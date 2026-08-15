import os
import json
from difflib import SequenceMatcher

# ---------- top level ----------
TOKENIZED_FOLDER = "tokenizedSentences"
OUTPUT_BASE_FOLDER = "opium_sentences_complete"

CASE_INSENSITIVE = True
FUZZY_MATCHING = True
FUZZY_THRESHOLD = 0.85

TARGETS = {
    "opium": ["opium"],
    "opiate": ["opiate", "opiates"],
    "poppy": ["poppy", "poppies"]
}

# Explicitly exclude known bad fuzzy matches
EXCLUDED_FUZZY = {
    "opiate": {"pirates"},
    "poppy": {"puppies"}
}
# ---------------------------------


def normalize(token):
    return token.lower() if CASE_INSENSITIVE else token


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def match_type(token, targets, concept):
    """
    Determine how a token matches a target list.

    Returns:
        "exact"  → exact match
        "fuzzy"  → fuzzy match
        None     → no match
    """
    token_norm = normalize(token)

    # Step 1: exclude known bad fuzzy matches
    if concept in EXCLUDED_FUZZY:
        if token_norm in EXCLUDED_FUZZY[concept]:
            return None

    for target in targets:
        target_norm = normalize(target)

        # exact match
        if token_norm == target_norm:
            return "exact"

        # fuzzy match
        if FUZZY_MATCHING and similarity(token_norm, target_norm) >= FUZZY_THRESHOLD:
            return "fuzzy"

    return None


def process_file(input_path, targets, concept):
    with open(input_path, "r", encoding="utf-8") as f:
        sentences = json.load(f)

    exact_matches = []
    fuzzy_matches = []

    for s in sentences:
        tokens = s.get("tokens", [])

        exact_count = 0
        fuzzy_count = 0
        fuzzy_tokens = []

        for t in tokens:
            mtype = match_type(t, targets, concept)

            if mtype == "exact":
                exact_count += 1

            elif mtype == "fuzzy":
                fuzzy_count += 1
                fuzzy_tokens.append(t)

        if exact_count > 0:
            exact_matches.append({
                "sentence_id": s.get("sentence_id"),
                "text": s.get("text"),
                "tokens": tokens,
                "exact_count": exact_count
            })

        if fuzzy_count > 0:
            fuzzy_matches.append({
                "sentence_id": s.get("sentence_id"),
                "text": s.get("text"),
                "tokens": tokens,
                "fuzzy_count": fuzzy_count,
                "fuzzy_tokens": fuzzy_tokens
            })

    return exact_matches, fuzzy_matches


def save_results(output_folder, filename, payload):
    if not payload:
        return

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    os.makedirs(OUTPUT_BASE_FOLDER, exist_ok=True)

    for concept, forms in TARGETS.items():
        print(f"\nProcessing concept: {concept}")

        exact_dir = os.path.join(OUTPUT_BASE_FOLDER, concept, "exact")
        fuzzy_dir = os.path.join(OUTPUT_BASE_FOLDER, concept, "fuzzy")

        for filename in os.listdir(TOKENIZED_FOLDER):
            if not filename.endswith(".json"):
                continue

            input_path = os.path.join(TOKENIZED_FOLDER, filename)

            exact_matches, fuzzy_matches = process_file(
                input_path,
                forms,
                concept
            )

            if exact_matches:
                save_results(
                    exact_dir,
                    filename,
                    {
                        "source_file": filename,
                        "target_concept": concept,
                        "match_type": "exact",
                        "sentences": exact_matches
                    }
                )

            if fuzzy_matches:
                save_results(
                    fuzzy_dir,
                    filename,
                    {
                        "source_file": filename,
                        "target_concept": concept,
                        "match_type": "fuzzy",
                        "fuzzy_threshold": FUZZY_THRESHOLD,
                        "sentences": fuzzy_matches
                    }
                )

            if exact_matches or fuzzy_matches:
                print(
                    f"  {filename}: "
                    f"{len(exact_matches)} exact / {len(fuzzy_matches)} fuzzy sentences"
                )


if __name__ == "__main__":
    main()
