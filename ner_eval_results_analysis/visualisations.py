import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_entity_label_by_bucket(eval_path: str, out_path: str = "/Users/izzie/Github/flair-KE-experiments/ner_eval_results_analysis/sampled_type_composition.png"):
    df = pd.read_excel(eval_path)

    comp = (
        df.groupby(["bucket", "entity_label"])
        .size()
        .reset_index(name="freq")
    )
    pivot = comp.pivot(index="bucket", columns="entity_label", values="freq").fillna(0)
    pivot = pivot.reindex(["head", "tail"])

    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot_pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("% of mentions within bucket")
    ax.set_xlabel("")
    ax.set_title("Sampled Entity type composition: head vs. tail")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Entity type")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return pivot, pivot_pct

pivot, pivot_pct = plot_entity_label_by_bucket("ner_eval_results_analysis/ner_eval_results_03aug.xlsx")
print(pivot_pct)

def count_and_rank_entity_labels(eval_path: str):
    df = pd.read_excel(eval_path)
    
    counts = (
        df["entity_label"]
        .value_counts()
        .reset_index()
    )
    counts.columns = ["entity_label", "count"]
    counts["rank"] = counts["count"].rank(ascending=False, method="min").astype(int)
    
    print(counts.to_string(index=False))
    return counts

counts = count_and_rank_entity_labels("ner_eval_results_analysis/ner_eval_results_03aug.xlsx")

def plot_top_entity_correctness(eval_path: str, out_path: str = "ner_eval_results_analysis/top_entity_correctness.png"):
    df = pd.read_excel(eval_path)

    # merge LOC into GPE before filtering
    df["entity_label"] = df["entity_label"].replace("LOC", "GPE")

    top_entities = ["CARDINAL", "GPE", "DATE", "NORP", "PERSON"]
    df_top = df[df["entity_label"].isin(top_entities)].copy()
    df_top["is_correct"] = df_top["is_correct (1/0)"].fillna(0).astype(int)

    grouped = (
        df_top.groupby(["entity_label", "bucket", "is_correct"])
        .size()
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = {
        ("head", "correct"):   "#2e83c8",  # dark blue
        ("head", "incorrect"): "#d62728",  # dark red
        ("tail", "correct"):   "#83c7ff",  # light blue
        ("tail", "incorrect"): "#ff9896",  # light red
    }

    bar_width = 0.35
    x = np.arange(len(top_entities))

    for i, (bucket, offset) in enumerate([("head", -0.5), ("tail", 0.5)]):
        for correct, label_suffix in [(1, "correct"), (0, "incorrect")]:
            counts = []
            bottoms = []
            for entity in top_entities:
                row = grouped[
                    (grouped["entity_label"] == entity) &
                    (grouped["bucket"] == bucket) &
                    (grouped["is_correct"] == correct)
                ]
                counts.append(row["count"].values[0] if not row.empty else 0)

                if correct == 0:
                    correct_row = grouped[
                        (grouped["entity_label"] == entity) &
                        (grouped["bucket"] == bucket) &
                        (grouped["is_correct"] == 1)
                    ]
                    bottoms.append(correct_row["count"].values[0] if not correct_row.empty else 0)
                else:
                    bottoms.append(0)

            ax.bar(
                x + offset * bar_width,
                counts,
                width=bar_width,
                bottom=bottoms,
                color=colors[(bucket, label_suffix)],
                label=f"{bucket.capitalize()} {label_suffix}" if i == 0 or bucket == "tail" else ""
            )

        for xi in x:
            ax.text(xi + offset * bar_width, -8, bucket, ha="center", va="top",
                    fontsize=8, color="grey")

    ax.set_xticks(x)
    ax.set_xticklabels(["CARDINAL", "GPE+LOC", "DATE", "NORP", "PERSON"])
    ax.set_ylabel("Count")
    #ax.set_xlabel("Entity type")
    ax.set_title("Top entity types: head vs. tail correctness")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Bucket / is_correct")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

plot_top_entity_correctness("ner_eval_results_analysis/ner_eval_results_03augv2.xlsx")