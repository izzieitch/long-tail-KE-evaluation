import csv
import json
import re
from collections import Counter


JSON_PATH = "results/flair_ner_with_manual_linked.json"
CSV_PATH = "ner_analysis_output/entity_frequency_table.csv"

def normalize_text(text: str) -> str:
    """Collapse whitespace/newlines so OCR-mangled spacing doesn't fragment
    the same entity into multiple counts."""
    return re.sub(r"\s+", " ", text).strip().lower()
 
# Load bucket lookup: (norm_text, label) -> bucket
bucket_lookup = {}
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        bucket_lookup[(row["norm_text"], row["label"])] = row["bucket"]
 
# --- Labels to break down separately ---
TARGET_LABELS = {"GPE", "NORP", "PERSON"}
 
def merged_label(label: str) -> str:
    """Treat LOC as GPE for reporting purposes (they're merged into one category)."""
    return "GPE" if label == "LOC" else label

# Cross-tab: for every entity, record its bucket AND whether it was linked
bucket_link_counts = Counter()   # (bucket, "linked"/"unlinked") -> count
label_link_counts  = Counter()   # (label, "linked"/"unlinked") -> count, overall
label_bucket_link_counts = Counter()  # (label, bucket, "linked"/"unlinked") -> count
not_found = []
with open(JSON_PATH, encoding="utf-8") as f:
    data = json.load(f)
 
for record in data:
    for ent in record.get("entities", []):
        key = (normalize_text(ent["text"]), ent["label"])
        bucket = bucket_lookup.get(key, "not_found_in_csv")
        status = "linked" if ent.get("wikipedia_url") is not None else "unlinked"
 
        bucket_link_counts[(bucket, status)] += 1
        if bucket == "not_found_in_csv":
            not_found.append((ent["text"], ent["label"]))
 
        label = ent["label"]
        if label in TARGET_LABELS:
            label_link_counts[(label, status)] += 1
            label_bucket_link_counts[(label, bucket, status)] += 1
 
# Report: within each bucket, % linked vs % unlinked
buckets = sorted({b for b, _ in bucket_link_counts})
print("Linked vs. unlinked breakdown by bucket:\n")
for bucket in buckets:
    linked   = bucket_link_counts[(bucket, "linked")]
    unlinked = bucket_link_counts[(bucket, "unlinked")]
    total    = linked + unlinked
    print(f"{bucket} (n={total}):")
    print(f"  linked    {linked:6d}  ({linked / total * 100:.2f}%)")
    print(f"  unlinked  {unlinked:6d}  ({unlinked / total * 100:.2f}%)")
    print()
 
#if not_found:
 #   print(f"Entities not found in CSV ({len(not_found)}):")
 #  for text, label in not_found:
   #     print(f"  - {text!r}, {label!r}")
 
# Report: for GPE, NORP, PERSON — % linked overall, and % linked within head/tail
print("\n" + "=" * 50)
print("Linked % by entity type (GPE, NORP, PERSON)")
print("=" * 50)
for label in sorted(TARGET_LABELS):
    linked   = label_link_counts[(label, "linked")]
    unlinked = label_link_counts[(label, "unlinked")]
    total    = linked + unlinked
    if total == 0:
        print(f"\n{label}: no entities found")
        continue
 
    print(f"\n{label} (n={total}):")
    print(f"  overall linked   {linked:6d} / {total:6d}  ({linked / total * 100:.2f}%)")
 
    for bucket in ("head", "tail"):
        b_linked   = label_bucket_link_counts[(label, bucket, "linked")]
        b_unlinked = label_bucket_link_counts[(label, bucket, "unlinked")]
        b_total    = b_linked + b_unlinked
        if b_total == 0:
            print(f"  {bucket:8s} linked   n/a (no entities in this bucket)")
        else:
            print(f"  {bucket:8s} linked   {b_linked:6d} / {b_total:6d}  ({b_linked / b_total * 100:.2f}%)")
 
# ── Bar chart: Total, GPE, NORP, PERSON — head & tail bars, stacked linked/unlinked ──
import matplotlib.pyplot as plt
 
CHART_PATH = "ner_eval_results_analysis/top_EL_results.png"
 
categories = ["GPE", "NORP", "PERSON"]
 
def bucket_counts(label, bucket):
    """Return (unlinked_count, linked_count) for a given label ('Total' = all labels) and bucket."""
    if label == "Total":
        linked   = bucket_link_counts[(bucket, "linked")]
        unlinked = bucket_link_counts[(bucket, "unlinked")]
    else:
        linked   = label_bucket_link_counts[(label, bucket, "linked")]
        unlinked = label_bucket_link_counts[(label, bucket, "unlinked")]
    return unlinked, linked
 
# Colors: head = darker, tail = lighter
head_red,   head_green   = "#d62728", "#2e83c8"
tail_red,   tail_green   = "#ff9896", "#83c7ff"
 
x = range(len(categories))
width = 0.35
 
fig, ax = plt.subplots(figsize=(9, 6))
 
max_total = 0
for i, cat in enumerate(categories):
    head_unlinked, head_linked = bucket_counts(cat, "head")
    tail_unlinked, tail_linked = bucket_counts(cat, "tail")
    max_total = max(max_total, head_unlinked + head_linked, tail_unlinked + tail_linked)
 
    xh = i - width / 2
    xt = i + width / 2
 
    # Head bar (darker colors), stacked: unlinked (red) then linked (green)
    ax.bar(xh, head_unlinked, width, color=head_red, edgecolor="white")
    ax.bar(xh, head_linked, width, bottom=head_unlinked, color=head_green, edgecolor="white")
 
    # Tail bar (lighter colors), stacked: unlinked (red) then linked (green)
    ax.bar(xt, tail_unlinked, width, color=tail_red, edgecolor="white")
    ax.bar(xt, tail_linked, width, bottom=tail_unlinked, color=tail_green, edgecolor="white")
 
    # Labels under each pair of bars
    label_y = -max_total * 0.04
    ax.text(xh, label_y, "head", ha="center", va="top", fontsize=9)
    ax.text(xt, label_y, "tail", ha="center", va="top", fontsize=9)
 
ax.set_xticks(list(x))
ax.set_xticklabels(["GPE+LOC", "NORP", "PERSON"], fontsize=11, fontweight="bold")
ax.set_ylabel("Number of entities")
ax.set_title("Linked vs. Unlinked top entity types by bucket (head/tail)")
 
# Legend (manual, since bars are drawn per-category)
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=head_green, label="Linked (head)"),
    plt.Rectangle((0, 0), 1, 1, color=head_red,   label="Unlinked (head)"),
    plt.Rectangle((0, 0), 1, 1, color=tail_green, label="Linked (tail)"),
    plt.Rectangle((0, 0), 1, 1, color=tail_red,   label="Unlinked (tail)"),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
 
plt.tight_layout()
plt.savefig(CHART_PATH, dpi=150)
print(f"\nChart saved to {CHART_PATH}")