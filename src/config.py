from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)


class cfg:
    gcp_project = os.getenv("GCP_PROJECT", "demo-project")
    bq_dataset = os.getenv("BQ_DATASET", "shopify_reviews")
    processed = BASE_DIR / "data" / "processed"
