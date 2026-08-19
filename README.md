# FLAIR KE Evaluations

A knowledge extraction pipeline for historical travelogue texts about opium trade in 19th century China, performing NER and entity linking using FLAIR. The results are sampled, manually evaluated, and analysed according to the head / tail grouping of the entities.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install flair spacy pyinflect nltk matplotlib pandas
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

Download NLTK data:
```bash
python -c "import nltk; nltk.download('gutenberg'); nltk.download('words'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

All pipeline scripts should be run from the **repository root** with the venv active.

## Input data

Link to download all .txt files can be found in **corpus-info.xlsx**, along with the ID name of each travelogue. 
These .txt files should be used to populate a **travelogues_txt/** folder before running the pipeline.

| Folder | Description |
|--------|-------------|
| `travelogues_txt/` | Raw `.txt` travelogue files (gitignored). Populate your own! |
| `tokenizedSentences` | Tokenised sentences from all travelogues (gitignored). Run `pipeline/02_sentokenize.py` to populate it.|
| `opium_sentences_complete/` | Sentence JSON files organised by concept and match type|

## Pipeline

Run scripts in order:

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `pipeline/00_build_vocab.py` | NLTK corpora | `vocab/` | Builds expanded vocabulary for OCR quality scoring |
| `pipeline/01_ocr_quality.py` | `travelogues_txt/`, `vocab/` | printed report | Corpus-level OCR quality assessment |
| `pipeline/02_sentokenize.py` | `travelogues_txt/` | `tokenizedSentences/` | Sentence and word tokenisation |
| `pipeline/03_string_matching.py` | `tokenizedSentences/` | `opium_sentences_complete/` | Exact and fuzzy keyword matching for target concepts |
| `pipeline/04_per_sentence_ocr.py` | `opium_sentences_complete/`, `vocab/` | updates JSON in place | Adds `ocr_quality` score to each sentence |
| `pipeline/05_flair_ner.py` | `opium_sentences_complete/` | `results/flair_ner_results.json` | Named entity recognition with Flair |
| `pipeline/06_inject_manual.py` | `results/flair_ner_results.json`, `opium_sentences_complete/` | `results/flair_ner_with_manual.json` | Injects manual string matches as additional entities |
| `pipeline/07_entity_linking.py` | `results/flair_ner_with_manual.json` | `results/flair_ner_with_manual_linked.json` | Links entities to Wikipedia using Flair's entity linker |


## Analysis

Run from the repository root after the relevant pipeline steps are complete:

| Script | Input | Output |
|--------|-------|-------------|
| `analysis/ner_counts.py` | `results/flair_ner_results.json` | Entity label distribution |
| `analysis/el_counts.py` | `results/flair_ner_with_manual_linked.json` | Entity linking coverage and top linked entities |
| `head_tail.py` | `results/flair_ner_results.json` | `head-tail_analysis_output` |
| `el_counts.py` | `results/flair_ner_with_manual_linked.json` | Entity linking sucess rate grouped by head/tail |
| `sampling.py` | `results/flair_ner_results.json` | `sampling_output/evaluation_samples.xlsx` sampling strategy for manually evaluation and creates the first batch of evaluated samples  |
| `commodities_counts.py` | `tokenizedSentences` | Counts of other commodity target tokens |


## Repository structure

```
pipeline/                 pipeline scripts (00–08)
analysis/                 analysis and visualisation scripts
vocab/                    generated vocabulary files
results/                  pipeline outputs
head-tail_analysis_output grouping of NER results into head and tail based on entity frequency, including entity frequency tabel and visualisations
sampling_output           first batch of samples for manual evaluation
ner_eval_results_analysis contains the results of manual evaluation of ner sampled by head and tail, `ner_eval_results_14aug.xlsx`, visualisation scrpts and visualisation results
corpus-info.xlsx          corpus metadata
```

## AI statement
Claude Sonnet 4.6 and Copilot was used to assist in code development and drafting of this README file.