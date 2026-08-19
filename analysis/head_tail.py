import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

NER_RESULTS_FILE = "results/flair_ner_results.json"


def normalize_text(text: str) -> str:
    """Collapse whitespace/newlines so OCR-mangled spacing doesn't fragment
    the same entity into multiple counts."""
    return re.sub(r"\s+", " ", text).strip().lower()


def load_entities(path: Path, min_ocr_quality: float | None = None) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    rows = []
    for record in records:
        ocr_q = record.get("ocr_quality")
        if min_ocr_quality is not None and ocr_q is not None and ocr_q < min_ocr_quality:
            continue  # skip low-quality OCR sentences if filtering is enabled
        for ent in record.get("entities", []):
            text = ent.get("text", "")
            label = ent.get("label")
            if not text or not label:
                continue
            rows.append(
                {
                    "raw_text": text,
                    "norm_text": normalize_text(text),
                    "label": label,
                    "score": ent.get("score"),
                    "source_file": record.get("source_file"),
                    "sent_id": record.get("sent_id"),
                    "ocr_quality": ocr_q,
                }
            )
    return pd.DataFrame(rows)


def build_frequency_table(df: pd.DataFrame, key_cols=("norm_text", "label")) -> pd.DataFrame:
    """One row per unique entity (by key_cols), with mention count."""
    freq = (
        df.groupby(list(key_cols))
        .size()
        .reset_index(name="freq")
        .sort_values("freq", ascending=False)
        .reset_index(drop=True)
    )
    return freq


def assign_head_tail(
    freq: pd.DataFrame,
    method: str = "cumulative_mass",
    head_cutoff: float = 0.80,
    head_entity_pct: float = 0.20,
    min_freq_head: int = 10,
) -> pd.DataFrame:
    """Sort by freq desc and label each entity 'head' or 'tail'.

    method="cumulative_mass": head = smallest set of entities whose
        cumulative mention count reaches head_cutoff (e.g. 0.80) of all
        mentions. Head size adapts to how skewed the data actually is.

    method="entity_rank": head = the top head_entity_pct (e.g. 0.20) of
        *entities by rank*, regardless of what % of mentions that covers.
        This directly controls head size (the literal "top 20% of causes"
        framing) instead of letting it float based on skew.

    method="frequency_threshold": head = every entity with freq strictly
        greater than min_freq_head (e.g. >10 mentions). Simple, fixed,
        interpretable cutoff independent of corpus size or skew.
    """
    freq = freq.sort_values("freq", ascending=False).reset_index(drop=True)
    total = freq["freq"].sum()
    freq["cum_freq"] = freq["freq"].cumsum()
    freq["cum_pct"] = freq["cum_freq"] / total
    freq["rank"] = freq.index + 1

    if method == "cumulative_mass":
        freq["bucket"] = freq["cum_pct"].apply(lambda p: "head" if p <= head_cutoff else "tail")
    elif method == "entity_rank":
        n_head = max(1, int(round(len(freq) * head_entity_pct)))
        freq["bucket"] = ["head" if r <= n_head else "tail" for r in freq["rank"]]
    elif method == "frequency_threshold":
        freq["bucket"] = freq["freq"].apply(lambda f: "head" if f > min_freq_head else "tail")
    else:
        raise ValueError(f"Unknown method: {method}")

    return freq


def plot_pareto(freq: pd.DataFrame, out_path: Path):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(freq["rank"], freq["freq"], color="#4C72B0", width=1.0)
    ax1.set_xlabel("Entity rank (sorted by frequency, descending)")
    ax1.set_ylabel("Mention count", color="#4C72B0")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    ax2 = ax1.twinx()
    ax2.plot(freq["rank"], freq["cum_pct"] * 100, color="#C44E52", linewidth=2)
    ax2.axhline(80, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("Cumulative % of total mentions", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax2.set_ylim(0, 105)

    # mark the head/tail cutoff point
    cutoff_rank = freq[freq["bucket"] == "head"]["rank"].max()
    ax1.axvline(cutoff_rank, color="gray", linestyle="--", linewidth=1)
    ax1.text(
        cutoff_rank, ax1.get_ylim()[1] * 0.8,
        f" head/tail cutoff\n rank {cutoff_rank}",
        fontsize=9, color="gray"
    )

    plt.title("Entity frequency distribution (Pareto chart)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_elbow_detail(freq: pd.DataFrame, out_path: Path, zoom_max_rank: int = 500):
    """Detailed diagnostic for locating the head/tail knee point:
    (1) a zoomed, LINEAR-scale view of frequency vs. rank over the first
        `zoom_max_rank` entities, with fine gridlines, so small drops are
        visible instead of compressed by a log scale.
    (2) the marginal drop (freq[rank] - freq[rank+1]) over the same range,
        which turns a knee in the curve into a visible spike/peak, making
        the exact elbow rank easy to read off numerically.
    """
    zoomed = freq[freq["rank"] <= zoom_max_rank].copy()
    zoomed["marginal_drop"] = -zoomed["freq"].diff(-1)  # freq[r] - freq[r+1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # --- Panel 1: zoomed linear frequency curve ---
    ax1.plot(zoomed["rank"], zoomed["freq"], color="#4C72B0", linewidth=1.5, marker="o", markersize=2)
    ax1.set_ylabel("Mention count (linear scale)")
    ax1.set_title(f"Zoomed frequency curve (rank 1–{zoom_max_rank})")
    ax1.grid(True, which="both", axis="both", linestyle=":", linewidth=0.6)
    ax1.set_xticks(range(0, zoom_max_rank + 1, 50))

    # --- Panel 2: marginal drop (first difference) ---
    ax2.plot(zoomed["rank"], zoomed["marginal_drop"], color="#C44E52", linewidth=1.2)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Drop to next rank\n(freq[r] - freq[r+1])")
    ax2.set_xlabel("Entity rank")
    ax2.grid(True, which="both", axis="both", linestyle=":", linewidth=0.6)

    # annotate the single largest marginal drop as a candidate elbow point
    max_drop_row = zoomed.loc[zoomed["marginal_drop"].idxmax()]
    elbow_rank = int(max_drop_row["rank"])
    ax2.axvline(elbow_rank, color="gray", linestyle="--", linewidth=1)
    ax1.axvline(elbow_rank, color="gray", linestyle="--", linewidth=1)
    ax2.annotate(
        f"largest single-step drop\nat rank {elbow_rank}\n(freq {int(max_drop_row['freq'])} -> next)",
        xy=(elbow_rank, max_drop_row["marginal_drop"]),
        xytext=(elbow_rank + zoom_max_rank * 0.05, max_drop_row["marginal_drop"]),
        fontsize=8, color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return elbow_rank


def plot_head_entities_detail(freq: pd.DataFrame, out_path: Path, min_freq_head: int = 10, max_labels: int = 80):
    """Detailed bar chart of every entity in the head (freq > min_freq_head),
    individually labeled by entity text, sorted descending by frequency.

    If the head contains more than `max_labels` entities, x-axis text labels
    are thinned out (every Nth label shown) to stay readable, but every bar
    is still plotted.
    """
    head = freq[freq["bucket"] == "head"].sort_values("freq", ascending=False).reset_index(drop=True)

    n = len(head)
    fig_width = min(30, max(10, n * 0.18))  # cap width so huge head counts don't blow up the image
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    ax.bar(range(len(head)), head["freq"], color="#4C72B0", width=0.8)
    ax.set_ylabel("Mention count")
    ax.set_xlabel("Entity")
    ax.set_title(f"Head entities in detail (freq > {min_freq_head}), n = {len(head)}")
    ax.axhline(min_freq_head, color="gray", linestyle="--", linewidth=1)

    ax.set_xticks(range(len(head)))
    labels = head["norm_text"] if "norm_text" in head.columns else head.iloc[:, 0]
    step = max(1, len(head) // max_labels)
    tick_labels = [lbl if i % step == 0 else "" for i, lbl in enumerate(labels)]
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_type_composition(freq: pd.DataFrame, out_path: Path):
    comp = (
        freq.groupby(["bucket", "label"])["freq"]
        .sum()
        .reset_index()
    )
    pivot = comp.pivot(index="bucket", columns="label", values="freq").fillna(0)
    pivot = pivot.reindex(["head", "tail"])

    # normalize to % within each bucket for readability
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot_pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("% of mentions within bucket")
    ax.set_xlabel("")
    ax.set_title("Entity type composition: head vs. tail")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Entity type")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return pivot, pivot_pct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default=NER_RESULTS_FILE,
        help=f"Path to NER results JSON (default: {NER_RESULTS_FILE})"
    )
    parser.add_argument("--outdir", type=str, default="head-tail_analysis_output")
    parser.add_argument(
        "--bucket-method", choices=["cumulative_mass", "entity_rank", "frequency_threshold"],
        default="frequency_threshold",
        help="cumulative_mass: head = smallest set of entities covering --head-cutoff of mentions. "
             "entity_rank: head = top --head-entity-pct of entities by rank, regardless of mention share. "
             "frequency_threshold: head = entities with freq > --min-freq-head."
    )
    parser.add_argument("--head-cutoff", type=float, default=0.80,
                         help="Used when --bucket-method=cumulative_mass")
    parser.add_argument("--head-entity-pct", type=float, default=0.20,
                         help="Used when --bucket-method=entity_rank, e.g. 0.20 = top 20% of entities")
    parser.add_argument("--min-freq-head", type=int, default=1,
                         help="Used when --bucket-method=frequency_threshold, e.g. 1 = entities with freq > 1 are head")
    parser.add_argument(
        "--min-ocr-quality", type=float, default=None,
        help="Optional: drop entities from sentences below this OCR quality score"
    )
    parser.add_argument(
        "--key", choices=["text_and_label", "text_only"], default="text_and_label",
        help="Whether to treat the same text under different labels as distinct entities"
    )
    parser.add_argument(
        "--zoom-max-rank", type=int, default=500,
        help="Max entity rank to include in the zoomed elbow-detail plot"
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_entities(Path(args.input), min_ocr_quality=args.min_ocr_quality)
    print(f"Loaded {len(df)} entity mentions from {args.input}")

    key_cols = ("norm_text", "label") if args.key == "text_and_label" else ("norm_text",)
    freq = build_frequency_table(df, key_cols=key_cols)

    if "label" not in freq.columns:
        # if keyed by text only, recover a representative label per entity (majority)
        majority_label = df.groupby("norm_text")["label"].agg(lambda x: x.value_counts().idxmax())
        freq["label"] = freq["norm_text"].map(majority_label)

    freq = assign_head_tail(
        freq,
        method=args.bucket_method,
        head_cutoff=args.head_cutoff,
        head_entity_pct=args.head_entity_pct,
        min_freq_head=args.min_freq_head,
    )

    freq.to_csv(outdir / "entity_frequency_table.csv", index=False)
    print(f"Saved full frequency table -> {outdir / 'entity_frequency_table.csv'}")

    plot_pareto(freq, outdir / "pareto_chart.png")
    print(f"Saved Pareto chart -> {outdir / 'pareto_chart.png'}")

    elbow_rank = plot_elbow_detail(freq, outdir / "elbow_detail.png", zoom_max_rank=args.zoom_max_rank)
    print(f"Saved elbow-detail chart -> {outdir / 'elbow_detail.png'}")
    print(f"Largest single-step frequency drop detected at rank {elbow_rank} "
          f"(candidate head/tail knee point)")

    plot_head_entities_detail(freq, outdir / "head_entities_detail.png", min_freq_head=args.min_freq_head)
    print(f"Saved head-entities detail chart -> {outdir / 'head_entities_detail.png'}")

    pivot, pivot_pct = plot_type_composition(freq, outdir / "total_type_composition.png")
    print(f"Saved type composition chart -> {outdir / 'total_type_composition.png'}")

    print("\n--- Summary ---")
    n_head = (freq["bucket"] == "head").sum()
    n_tail = (freq["bucket"] == "tail").sum()
    print(f"Unique entities total: {len(freq)}")
    print(f"  Head: {n_head} unique entities ({n_head/len(freq)*100:.1f}% of entities, "
          f"{freq[freq['bucket']=='head']['freq'].sum()/freq['freq'].sum()*100:.1f}% of mentions)")
    print(f"  Tail: {n_tail} unique entities ({n_tail/len(freq)*100:.1f}% of entities, "
          f"{freq[freq['bucket']=='tail']['freq'].sum()/freq['freq'].sum()*100:.1f}% of mentions)")

    print("\nEntity type counts (mentions) per bucket:")
    print(pivot)

    print("\nSaved outputs in:", outdir.resolve())


if __name__ == "__main__":
    main()