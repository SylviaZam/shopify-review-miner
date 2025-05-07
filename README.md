# shopify-review-miner

A standalone pipeline to clean, analyze, and extract insights from open‑text survey and review CSV files, powered by offline NLP and ready for AI integration. Inspired by professional workflows at Unmade, this repo lets you drop any CSV of post‑purchase reviews into `data/raw/`, run a single command, and receive:

* A Parquet file of cleaned and normalized responses (`data/processed/cleaned_reviews.parquet`).
* One CSV per survey question summarizing counts by sentiment and category (`data/processed/insights_<question>.csv`).

## 🚀 Features

* **Automatic cleaning:** lower-casing, punctuation stripping, whitespace collapsing.
* **Offline sentiment analysis:** VADER via NLTK provides positive/neutral/negative scores without external API.
* **Keyword-based categorization:** easily extensible topical buckets (e.g., flavor, price, packaging).
* **One-command pipeline:** ingest raw CSV, get processed outputs.
* **Extensible architecture:** switch to OpenAI or other LLMs by replacing `analyse_batch()`.

## 📋 Prerequisites

* Python 3.9+
* Git
* (Optional) An OpenAI API key for real LLM-driven analysis.

## 🛠️ Installation

```bash
# Clone the repo
git clone git@github.com:SylviaZam/shopify-review-miner.git
cd shopify-review-miner

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK VADER model (one-time)
python - << 'PY'
import nltk; nltk.download('vader_lexicon')
PY
```

## ⚙️ Usage

1. **Drop your raw CSV** into `data/raw/`. Columns should be survey questions and each row an open‑text answer.
2. **Run the pipeline:**

   ```bash
   python src/pipeline.py data/raw/your_file.csv
   ```
3. **Inspect outputs** in `data/processed/`:

   * `cleaned_reviews.parquet` (all answers + sentiment + category)
   * `insights_<question>.csv` (aggregated counts by sentiment + category)

## 🔧 Pipeline Overview

1. **Load & reshape:** pivot CSV to long format (`question`,`answer`).
2. **Normalize:** clean text via regex (`normalise()`).
3. **Analyze:** `analyse_batch()` applies VADER sentiment and keyword category.
4. **Aggregate & export:** save cleaned data and generate per‑question insight files.

## 📝 Customization

* **Sentiment thresholds:** adjust `score > 0.05` / `< -0.05` in `analyse_batch()`.
* **Add categories:** modify `CATEGORY_KEYWORDS` in `pipeline.py`.
* **Switch to LLM:** replace `analyse_batch()` with OpenAI or other model calls.

## 💡 Tips

* To skip data artifacts in Git, ensure `data/raw/` and `data/processed/` are in `.gitignore`.
* Alias the pipeline command in your shell for speed:

  ```bash
  alias reviews="python ~/Projects/shopify-review-miner/src/pipeline.py"
  ```

## 🤝 Contributing

Feel free to submit issues or pull requests for enhancements, bug fixes, or additional categorizations.

## 📄 License

MIT © Sylvia Zamora
