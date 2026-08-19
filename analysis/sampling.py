import json
import re
from pathlib import Path
 
import pandas as pd
import numpy as np
 
NER_RESULTS_FILE = "results/flair_ner_results.json"
OUTDIR = Path("sampling_output")
 
MIN_FREQ_HEAD = 1          # entities with freq > this are "head"
N_SHARED_HEAD = 25
N_SHARED_TAIL = 25
N_UNIQUE_HEAD_PER_ANNOTATOR = 75
N_UNIQUE_TAIL_PER_ANNOTATOR = 75
N_ANNOTATORS = 3
RANDOM_SEED = 42
 
# Additional balanced samples drawn from the pool of entities NOT used in the
# 3 annotator sheets above. Each size must be even (split 50/50 head/tail).
EXTRA_SAMPLE_SIZES = [200, 200, 300]
 
 
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
 
 
def load_mentions(path: Path) -> pd.DataFrame:
    """One row per entity mention, keeping sentence-level context fields
    (source_file, sent_id, ocr_quality, text) plus entity-level fields."""
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
 
    rows = []
    for record in records:
        for ent in record.get("entities", []):
            text = ent.get("text", "")
            label = ent.get("label")
            if not text or not label:
                continue
            rows.append(
                {
                    "source_file": record.get("source_file"),
                    "sent_id": record.get("sent_id"),
                    "ocr_quality": record.get("ocr_quality"),
                    "text": record.get("text"),          # sentence-level context text
                    "entity_text": text,                  # entity surface form
                    "norm_text": normalize_text(text),
                    "entity_label": label,
                    "start_pos": ent.get("start_pos"),
                    "end_pos": ent.get("end_pos"),
                    "score": ent.get("score"),
                }
            )
    return pd.DataFrame(rows)
 
 
def assign_bucket(mentions: pd.DataFrame, min_freq_head: int) -> pd.DataFrame:
    freq = (
        mentions.groupby("norm_text")
        .size()
        .reset_index(name="freq")
    )
    freq["bucket"] = freq["freq"].apply(lambda f: "head" if f > min_freq_head else "tail")
    return freq
 
 
def sample_entities(freq: pd.DataFrame, rng: np.random.Generator):
    """Draw the shared pool + per-annotator unique pools of entity keys
    (norm_text), for head and tail separately."""
    results = {}
    for bucket, n_shared, n_unique in [
        ("head", N_SHARED_HEAD, N_UNIQUE_HEAD_PER_ANNOTATOR),
        ("tail", N_SHARED_TAIL, N_UNIQUE_TAIL_PER_ANNOTATOR),
    ]:
        pool = freq.loc[freq["bucket"] == bucket, "norm_text"].tolist()
        needed = n_shared + n_unique * N_ANNOTATORS
        if len(pool) < needed:
            raise ValueError(
                f"Not enough unique '{bucket}' entities: need {needed}, have {len(pool)}. "
                f"Reduce sample sizes or lower N_ANNOTATORS."
            )
        chosen = rng.choice(pool, size=needed, replace=False)
        rng.shuffle(chosen)
 
        shared = list(chosen[:n_shared])
        remainder = list(chosen[n_shared:])
        per_annotator = [
            remainder[i * n_unique:(i + 1) * n_unique] for i in range(N_ANNOTATORS)
        ]
        results[bucket] = {"shared": shared, "per_annotator": per_annotator}
    return results
 
 
def pick_representative_mention(mentions: pd.DataFrame, norm_text: str, rng: np.random.Generator) -> pd.Series:
    """Pick one mention row at random to represent this entity (gives
    annotators a real sentence context to judge)."""
    candidates = mentions[mentions["norm_text"] == norm_text]
    idx = rng.integers(0, len(candidates))
    return candidates.iloc[idx]
 
 
def build_annotator_sheet(entity_keys_ordered, mentions, freq, rng, item_id_map):
    rows = []
    for norm_text in entity_keys_ordered:
        mention_row = pick_representative_mention(mentions, norm_text, rng)
        bucket = freq.loc[freq["norm_text"] == norm_text, "bucket"].iloc[0]
        row = {
            "item_id": item_id_map[norm_text],
            "source_file": mention_row["source_file"],
            "sent_id": mention_row["sent_id"],
            "ocr_quality": mention_row["ocr_quality"],
            "text": mention_row["text"],
            "entity_text": mention_row["entity_text"],
            "entity_label": mention_row["entity_label"],
            "bucket": bucket,
            # empty columns for the annotator to fill in
            "is_correct (Y/N)": "",
            "notes": "",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_sheet_from_mentions(mention_df: pd.DataFrame, item_ids):
    """Build an annotation sheet directly from sampled mention rows."""
    rows = []

    for _, mention in mention_df.iterrows():
        rows.append(
            {
                "item_id": item_ids[mention.name],  # use dataframe index as unique mention ID
                "source_file": mention["source_file"],
                "sent_id": mention["sent_id"],
                "ocr_quality": mention["ocr_quality"],
                "text": mention["text"],
                "entity_text": mention["entity_text"],
                "entity_label": mention["entity_label"],
                "bucket": mention["bucket"],
                "is_correct (Y/N)": "",
                "notes": "",
            }
        )

    return pd.DataFrame(rows)
 
 
def draw_balanced_extra_samples(head_mentions, tail_mentions, sizes, rng):
    """
    Draw balanced mention-level samples.

    Each mention can appear only once overall, but multiple mentions of the
    same entity are allowed.
    """

    head_mentions = head_mentions.copy()
    tail_mentions = tail_mentions.copy()

    samples = []

    for size in sizes:

        if size % 2 != 0:
            raise ValueError("Sample sizes must be even.")

        n_half = size // 2

        if len(head_mentions) < n_half or len(tail_mentions) < n_half:
            raise ValueError("Not enough remaining mentions.")

        chosen_head = head_mentions.sample(
            n=n_half,
            random_state=int(rng.integers(0, 1_000_000)),
            replace=False,
        )

        chosen_tail = tail_mentions.sample(
            n=n_half,
            random_state=int(rng.integers(0, 1_000_000)),
            replace=False,
        )

        sample = pd.concat([chosen_head, chosen_tail])
        sample = sample.sample(
            frac=1,
            random_state=int(rng.integers(0, 1_000_000)),
        )

        samples.append(sample)

        head_mentions = head_mentions.drop(chosen_head.index)
        tail_mentions = tail_mentions.drop(chosen_tail.index)

    return samples, head_mentions, tail_mentions
 
 
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
 
    mentions = load_mentions(Path(NER_RESULTS_FILE))
    print(f"Loaded {len(mentions)} entity mentions from {NER_RESULTS_FILE}")
 
    freq = assign_bucket(mentions, min_freq_head=MIN_FREQ_HEAD)
    print(freq["bucket"].value_counts())
 
    sampled = sample_entities(freq, rng)
 
    # assign a stable item_id to every sampled entity (shared + all unique)
    all_keys = (
        sampled["head"]["shared"] + sampled["tail"]["shared"]
        + [k for lst in sampled["head"]["per_annotator"] for k in lst]
        + [k for lst in sampled["tail"]["per_annotator"] for k in lst]
    )
    item_id_map = {key: f"item_{i:04d}" for i, key in enumerate(dict.fromkeys(all_keys))}
 
    # shared block: fixed order, identical across all 3 sheets (first 50 rows)
    shared_order = sampled["head"]["shared"] + sampled["tail"]["shared"]
    rng.shuffle(shared_order)  # single shuffle, reused identically for every annotator
 
    sheets = {}
    key_rows = []
 
    for a in range(N_ANNOTATORS):
        unique_head = sampled["head"]["per_annotator"][a]
        unique_tail = sampled["tail"]["per_annotator"][a]
        unique_order = unique_head + unique_tail
        rng.shuffle(unique_order)  # shuffled independently per annotator
 
        full_order = shared_order + unique_order  # first 50 = shared, rest = unique
        sheet_df = build_annotator_sheet(full_order, mentions, freq, rng, item_id_map)
        sheet_name = f"Annotator_{a + 1}"
        sheets[sheet_name] = sheet_df
 
        for pos, norm_text in enumerate(full_order):
            bucket = freq.loc[freq["norm_text"] == norm_text, "bucket"].iloc[0]
            freq_val = freq.loc[freq["norm_text"] == norm_text, "freq"].iloc[0]
            key_rows.append(
                {
                    "item_id": item_id_map[norm_text],
                    "norm_text": norm_text,
                    "bucket": bucket,
                    "freq": freq_val,
                    "sheet": sheet_name,
                    "row_position": pos + 1,
                    "is_shared_across_annotators": pos < (N_SHARED_HEAD + N_SHARED_TAIL),
                }
            )
 
# ------------------------------------------------------------------
# Build remaining MENTIONS
# ------------------------------------------------------------------

sampled_entity_keys = set(item_id_map.keys())

remaining_mentions = mentions[
    ~mentions["norm_text"].isin(sampled_entity_keys)
].copy()

bucket_lookup = freq.set_index("norm_text")["bucket"]
freq_lookup = freq.set_index("norm_text")["freq"]

remaining_mentions["bucket"] = remaining_mentions["norm_text"].map(bucket_lookup)
remaining_mentions["freq"] = remaining_mentions["norm_text"].map(freq_lookup)

remaining_head = remaining_mentions[
    remaining_mentions["bucket"] == "head"
]

remaining_tail = remaining_mentions[
    remaining_mentions["bucket"] == "tail"
]

extra_samples, leftover_head, leftover_tail = draw_balanced_extra_samples(
    remaining_head,
    remaining_tail,
    EXTRA_SAMPLE_SIZES,
    rng,
)

extra_sheets = {}

next_idx = len(item_id_map)

for i, sample_df in enumerate(extra_samples):

    mention_item_ids = {}

    for idx in sample_df.index:
        mention_item_ids[idx] = f"item_{next_idx:04d}"
        next_idx += 1

    sheet_name = f"Extra_Sample_{i + 1}"

    extra_sheet = build_sheet_from_mentions(
        sample_df,
        mention_item_ids,
    )

    extra_sheets[sheet_name] = extra_sheet

    print(
        f"{sheet_name}: {len(extra_sheet)} mentions "
        f"({(sample_df.bucket=='head').sum()} head, "
        f"{(sample_df.bucket=='tail').sum()} tail)"
    )

    for pos, (idx, row) in enumerate(sample_df.iterrows()):

        key_rows.append(
            {
                "item_id": mention_item_ids[idx],
                "norm_text": row["norm_text"],
                "bucket": row["bucket"],
                "freq": row["freq"],
                "sheet": sheet_name,
                "row_position": pos + 1,
                "is_shared_across_annotators": False,
            }
        )

# ------------------------------------------------------------------
# Remaining mention pool
# ------------------------------------------------------------------

remaining_df_mentions = pd.concat(
    [leftover_head, leftover_tail]
).sample(
    frac=1,
    random_state=int(rng.integers(0, 1_000_000)),
)

mention_item_ids = {}

for idx in remaining_df_mentions.index:
    mention_item_ids[idx] = f"item_{next_idx:04d}"
    next_idx += 1

remaining_df = build_sheet_from_mentions(
    remaining_df_mentions,
    mention_item_ids,
)

print(
    f"Remaining (unsampled) mention pool: {len(remaining_df)} mentions"
)

for pos, (idx, row) in enumerate(remaining_df_mentions.iterrows()):

    key_rows.append(
        {
            "item_id": mention_item_ids[idx],
            "norm_text": row["norm_text"],
            "bucket": row["bucket"],
            "freq": row["freq"],
            "sheet": "Remaining_Pool",
            "row_position": pos + 1,
            "is_shared_across_annotators": False,
        }
    )
 
    # --- write annotator-facing workbook (no bucket/freq columns -> blind) ---
    eval_path = OUTDIR / "evaluation_samples.xlsx"
    with pd.ExcelWriter(eval_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        for sheet_name, df in extra_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        remaining_df.to_excel(writer, sheet_name="Remaining_Pool", index=False)
    print(f"Saved annotator workbook -> {eval_path}")
 
    # --- write researcher-only key (bucket labels, for later stratified analysis) ---
    key_df = pd.DataFrame(key_rows)
    key_path = OUTDIR / "sampling_key_DO_NOT_SHARE.xlsx"
    key_df.to_excel(key_path, index=False)
    print(f"Saved sampling key (keep private) -> {key_path}")
 
    print("\nPer-annotator composition check:")
    for sheet_name, df in sheets.items():
        n = len(df)
        n_shared = (N_SHARED_HEAD + N_SHARED_TAIL)
        print(f"  {sheet_name}: {n} rows total ({n_shared} shared + {n - n_shared} unique)")
 
 
if __name__ == "__main__":
    main()
 