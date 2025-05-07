"""
pipeline.py
CLI: python src/pipeline.py data/raw/sample.csv
------------------------------------------------
Reads a survey CSV where each column is a question
and each row is a respondent’s open‑text answer.

Outputs:
• cleaned_reviews.parquet               (normalized text)
• insights_<question>.csv               (sentiment + categories + counts)
"""

from pathlib import Path
import pandas as pd
import re, json, os, sys, time, collections
import openai

# ------------------------- CONFIG --------------------------
openai.api_key = os.getenv("OPENAI_API_KEY", "sk‑demo")  # any dummy key is OK for dry‑run
BATCH = 20                           # how many rows per OpenAI call
SYSTEM = """You are a CX analyst.
For each answer, reply with compact JSON:
{"sentiment": one of ["positive","neutral","negative"],
 "category": <single word or short phrase that best describes the main topic>}"""
# ---------------------- HELPERS ----------------------------

def normalise(text: str) -> str:
    """lower‑case, strip punctuation, collapse spaces"""
    text = text.lower()
    text = re.sub(r"[^\\w\\s]", " ", text)
    return re.sub(r"\\s+", " ", text).strip()

def chunker(seq, size):
    for pos in range(0, len(seq), size):
        yield seq[pos : pos + size]

def analyse_batch(batch):
    """
    Offline stub: returns neutral sentiment and a generic 'misc' category
    for every answer so the rest of the pipeline can run without OpenAI.
    """
    return [{"sentiment": "neutral", "category": "misc"} for _ in batch]

# ------------------------- PIPELINE ------------------------

def run(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df.melt(var_name="question", value_name="answer")  # long format
    df["answer"] = df["answer"].astype(str).apply(normalise)
    df = df[df["answer"].str.len() > 0].reset_index(drop=True)

    # call OpenAI in batches
    results = []
    for batch in chunker(df["answer"].tolist(), BATCH):
        try:
            results += analyse_batch(batch)
        except Exception as e:
            print("⚠️ OpenAI error, backing off 10s", e)
            time.sleep(10)

    out = pd.concat([df, pd.DataFrame(results)], axis=1)

    # save cleaned answers parquet
    parquet_path = Path("data/processed/cleaned_reviews.parquet")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(parquet_path)
    print("✅ cleaned data →", parquet_path)

    # aggregate per question / category / sentiment
    for q, group in out.groupby("question"):
        summary = (
            group.groupby(["category", "sentiment"])
                 .size()
                 .reset_index(name="count")
                 .sort_values("count", ascending=False)
        )
                # build a safe file name without backslashes inside an f‑string
        safe_q = re.sub(r'[^A-Za-z0-9]+', '_', q)[:30]     # sanitize question text
        fname = Path("data/processed") / f"insights_{safe_q}.csv"
        summary.to_csv(fname, index=False)
        print("✅ insights →", fname)

# ------------------------- ENTRY ---------------------------
if __name__ == "__main__":
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/sample.csv")
    run(csv)
    