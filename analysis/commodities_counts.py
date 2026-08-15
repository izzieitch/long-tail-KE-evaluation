import os
import json

TOKENIZED_FOLDER = "tokenizedSentences"
TARGETS = ["sugar", "coffee", "tea", "metal", "cotton", "grain"]

counts = {word: 0 for word in TARGETS}

for filename in os.listdir(TOKENIZED_FOLDER):
    if not filename.endswith(".json"):
        continue

    with open(os.path.join(TOKENIZED_FOLDER, filename), "r", encoding="utf-8") as f:
        sentences = json.load(f)

    for sentence in sentences:
        for token in sentence.get("tokens", []):
            if token.lower() in counts:
                counts[token.lower()] += 1

print("\nWord counts:")
for word, count in counts.items():
    print(f"  {word}: {count}")