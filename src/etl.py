import pandas as pd
from google.cloud import bigquery
from pathlib import Path
from config import cfg


def csv_to_bq(csv_path: Path, brand: str):
    """Load raw CSV → parquet → BigQuery."""
    df = (
        pd.read_csv(csv_path)
          .assign(brand=brand)
          .rename(columns=str.lower)
    )
    df.to_parquet(cfg.processed / f"{brand}.parquet")
    client = bigquery.Client(project=cfg.gcp_project)
    client.load_table_from_dataframe(
        df, f"{cfg.bq_dataset}.raw_{brand}"
    ).result()
