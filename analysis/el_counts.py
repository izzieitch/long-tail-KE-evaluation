import json
from collections import Counter
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE = Path("results/flair_ner_with_manual_linked.json")
TOP_N      = 20

# ── Load ──────────────────────────────────────────────────────────────────────
with open(INPUT_FILE, encoding="utf-8") as f:
    records = json.load(f)

print(f"Loaded {len(records):,} records from {INPUT_FILE}\n")

# ── Count ─────────────────────────────────────────────────────────────────────
entity_counter    = Counter()   # by raw text (e.g. "Canton")
wikipedia_counter = Counter()   # by resolved wikipedia title (e.g. "Guangzhou")
label_counter     = Counter()   # NER label distribution — LINKED only
unlinked_label_counter = Counter()  # NER label distribution — UNLINKED only
no_link_counter   = Counter()   # entities that couldn't be linked

for record in records:
    for ent in record.get("entities", []):
        text       = ent.get("text", "").strip()
        label      = ent.get("label", "UNKNOWN")
        wiki       = ent.get("wikipedia_title")
        link_score = ent.get("linking_score")

        if wiki and link_score is not None:
            # Successfully linked entity
            entity_counter[text]    += 1
            wikipedia_counter[wiki] += 1
            label_counter[label]    += 1
        else:
            # Entity detected but not linked to Wikipedia
            no_link_counter[text]        += 1
            unlinked_label_counter[label] += 1

# ── Report ────────────────────────────────────────────────────────────────────
total_entities = sum(entity_counter.values()) + sum(no_link_counter.values())
total_linked   = sum(entity_counter.values())

print("=" * 60)
print(f"  Total entity mentions      : {total_entities:,}")
print(f"  Successfully linked        : {total_linked:,}  ({100*total_linked/total_entities:.1f}%)")
print(f"  Not linked                 : {sum(no_link_counter.values()):,}")
print("=" * 60)

# ── Top 20 by raw surface text ─────────────────────────────────────────────
print(f"\nTop {TOP_N} most frequent linked entities (surface text):")
print("-" * 55)
for rank, (entity, count) in enumerate(entity_counter.most_common(TOP_N), 1):
    bar = "█" * min(count, 40)
    print(f"  {rank:>2}. {entity:<30} {count:>5}  {bar}")

# ── Top 20 by Wikipedia title (after disambiguation) ──────────────────────
print(f"\nTop {TOP_N} most frequent linked entities (Wikipedia title):")
print("-" * 55)
for rank, (wiki_title, count) in enumerate(wikipedia_counter.most_common(TOP_N), 1):
    bar = "█" * min(count, 40)
    print(f"  {rank:>2}. {wiki_title:<30} {count:>5}  {bar}")

# ── NER label breakdown ────────────────────────────────────────────────────
print(f"\nNER label distribution (linked entities only):")
print("-" * 40)
for label, count in label_counter.most_common():
    pct = 100 * count / total_linked
    print(f"  {label:<15} {count:>6}  ({pct:.1f}%)")

# ── Top unlinked (noisy / ambiguous) ──────────────────────────────────────
print(f"\nTop 10 most frequent UNLINKED entities:")
print("-" * 40)
for rank, (entity, count) in enumerate(no_link_counter.most_common(10), 1):
    print(f"  {rank:>2}. {entity:<30} {count:>5}")