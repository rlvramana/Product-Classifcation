# Product Classification: Fashion vs Non-Fashion

This repository contains my part of the Shopper Categorization / Taxonomy project.

The overall company project aims to build a full product taxonomy.  
This repo focuses only on the first step:

Given a product text (title or description), classify it as  
0 = non-fashion  
1 = fashion

Once fashion products are separated, they can be passed into a second stage (rules or another model) to build a detailed fashion taxonomy.

Note: The product data is ambiguous and messy enough to make this a medium-complexity project.

---

The project has two main models:

1. **TF‑IDF + Logistic Regression baseline**
2. **BERT fine‑tuned classifier** (standard and class‑weighted versions)

Both models are evaluated on time‑aware splits and small manual QA samples, and then compared on **unlabelled real products**.

---

## 1. Folder structure

```text
Product-Classification/
├─ data/
│  ├─ raw/          # Original JSON / CSV files from the course (not in GitHub)
│  ├─ labels/       # Original label files from the course (not in GitHub)
│  ├─ interim/      # Small / intermediate artefacts (local)
│  └─ processed/    # Joined & cleaned parquet files + QA CSVs
├─ models/
│  ├─ bert_fashion/           # First BERT fine‑tune run
│  ├─ bert_fashion_weighted/  # BERT with class weights
│  └─ tfidf_logreg_fashion_baseline.joblib  # Saved logistic baseline
├─ notebooks/
│  ├─ 00_config_and_utils.ipynb
│  ├─ 01_labels_explore_and_normalize.ipynb
│  ├─ 02_join_json_and_labels.ipynb
│  ├─ 03_time_based_splits.ipynb
│  ├─ 04_tfidf_logreg_baseline.ipynb
│  ├─ 05_logreg_manual_QA.ipynb
│  ├─ 06_bert_finetune.ipynb
│  ├─ 06B_bert_finetune_weighted.ipynb
│  ├─ 07_bert_manual_QA_labelled.ipynb
│  ├─ 08_unlabelled_QA_logreg_vs_bert.ipynb
│  └─ 08B_unlabelled_QA_bert_weighted.ipynb
├─ src/
│  ├─ config.py
│  └─ preprocessing.py
└─ environment-fashion-env.yml
```

If some filenames differ slightly in your local copy, follow the same order.

---

## 2. Environment setup (Conda + pip)

We use a Conda environment file: `environment-fashion-env.yml` (shown in this repo).  
Conda creates the base environment and then `pip` installs the PyTorch / Hugging Face stack (as defined under the `pip:` section).

You can run these commands on **macOS** or **Windows** (assuming Conda is installed: Anaconda or Miniconda).

### 2.1 Create the environment

```bash
# From the root of the repo
conda env create -f environment-fashion-env.yml

# Activate
conda activate fashion-bert
```

If Conda says the env already exists:

```bash
conda activate fashion-bert
```

### 2.2 Start Jupyter Lab / Notebook

From the project root (with the env active):

```bash
jupyter lab
```

or

```bash
jupyter notebook
```

Then open the notebooks under `notebooks/` in the order listed below.

---

## 3. Raw data: where to put the files

The original course data is **not** stored in GitHub for size and privacy reasons.

You must copy the raw files from the shared course location into this project.  
Use this layout (you can adjust paths in `src/config.py` if needed):

```text
data/
├─ raw/
│  ├─ events/           # JSON / CSV clickstream per month (from course)
│  └─ products/         # Any product‑level reference files (if provided)
└─ labels/
   └─ FashionLabels.parquet (or equivalent label files from the course)
```

In `src/config.py` you will see constants like:

```python
PROJECT_ROOT = Path("...")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
LABELS_DIR = DATA_DIR / "labels"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
```

If your raw data lives somewhere else on disk, update these paths so they point to the correct folders.

---

## 4. Running the pipeline: notebook order

The idea is that we can re‑run everything from scratch.

Here is the recommended order and what each notebook does.

### 4.1 Pre‑processing and splits

1. `00_config_and_utils.ipynb`  
   - Sets up paths using `config.py`  
   - Common helper functions

2. `01_labels_explore_and_normalize.ipynb`  
   - Loads the label files  
   - Normalises label values (fashion vs non‑fashion)  
   - Saves cleaned label tables into `data/processed`

3. `02_join_json_and_labels.ipynb`  
   - Reads raw JSON / events  
   - Joins events with labels on product ID  
   - Normalises product text (`product_text_norm`)  
   - Writes per‑month joined parquet files into `data/processed`

4. `03_time_based_splits.ipynb`  
   - Combines per‑month product tables into a global product table  
   - Applies **time‑aware train/val/test split** to avoid data leakage  
   - Saves `products_with_splits.parquet` to `data/processed`

At this point you have a clean, deduplicated product‑level dataset with a `split` column (`train/val/test`).

### 4.2 Logistic regression baseline

5. `04_tfidf_logreg_baseline.ipynb`  
   - Builds a **TF‑IDF vectoriser** on `product_text_norm`  
   - Trains a logistic regression classifier on the train split  
   - Evaluates on train and validation using confusion matrices and PR curve  
   - Saves the fitted pipeline and tuned threshold to:
     - `models/tfidf_logreg_fashion_baseline.joblib`

6. `05_logreg_manual_QA.ipynb`  
   - Loads the saved logistic pipeline  
   - Samples products for manual QA  
   - Writes a CSV (e.g. `manual_qa_baseline_predictions.csv`) in `data/processed`  
   - You open that CSV, add a `manual_label` column, then re‑load it to compute QA accuracy.

### 4.3 BERT fine‑tuning (original and weighted)

7. `06_bert_finetune.ipynb`  
   - Converts the train/val/test product tables into Hugging Face `datasets.Dataset` objects  
   - Tokenises text using `bert-base-uncased` (max length 64)  
   - Fine‑tunes BERT with cross‑entropy loss  
   - Tracks train vs validation loss and saves the best model to:
     - `models/bert_fashion/`

8. `06B_bert_finetune_weighted.ipynb`  
   - Same as above, but uses **class weights** (higher weight for class 0 = non‑fashion)  
   - Saves the best weighted model to:
     - `models/bert_fashion_weighted/`

9. `07_bert_manual_QA_labelled.ipynb`  
   - Runs the BERT model(s) on a labelled QA sample  
   - Compares predictions to both dataset labels and your manual labels  
   - Produces confusion matrices and summary tables.

### 4.4 Unlabelled QA: comparing models on real products

10. `08_unlabelled_QA_logreg_vs_bert.ipynb`  
    - Samples ~100 **unlabelled** products from recent months  
    - Runs both logistic and the original BERT model on them  
    - Saves a CSV `manual_qa_unlabeled_logreg_vs_bert.csv` in `data/processed`  
    - You manually label the `manual_label` column, then re‑run the notebook to:
      - Compute accuracy / precision / recall for both models  
      - Plot side‑by‑side confusion matrices.

11. `08B_unlabelled_QA_bert_weighted.ipynb`  
    - Loads the same QA CSV  
    - Runs the **weighted BERT** model and compares three models side‑by‑side:
      - Logistic baseline  
      - Original BERT  
      - Weighted BERT  
    - Produces the final comparison plots used in the report.

---

## 5. Using the hosted BERT model (Hugging Face)

If you don’t want to fine‑tune BERT yourself, you can load my published model from Hugging Face.

Model repo (private): `rlvramana/fashion-bert-classification`

In any notebook (with `transformers` installed and internet access):

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_id = "rlvramana/fashion-bert-classification"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

def predict_fashion(text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits.softmax(dim=-1)[0]
    # probs[1] = probability of class 1 (fashion)
    return float(probs[1])

print(predict_fashion("women's black leather jacket"))
```

You can also plug this into the existing notebooks by replacing the local model‑loading code with the model ID above.

---

## 6. Hardware notes

- The **logistic regression** notebook is light and will run on almost any laptop.
- **BERT fine‑tuning** is heavier:
  - On Apple Silicon (M1/M2), we use the `mps` device if available.
  - On Windows laptops without GPU, training will still work on CPU, just slower.
- For very slow machines, you can:
  - Reduce the number of epochs in `06_bert_finetune*.ipynb`  
  - Or skip training completely and just use the hosted model.

---

## 7. Reproducibility checklist

To fully reproduce the project from scratch:

1. Clone the repo from GitHub.
2. Copy raw data into `data/raw` and label files into `data/labels`.
3. Create and activate the Conda env using `environment-fashion-env.yml`.
4. Run notebooks in the order described in section 4.
5. For QA notebooks, remember to fill in your own `manual_label` values in the exported CSVs.
6. Optionally, compare your results to the hosted BERT model.
