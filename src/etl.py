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

    # ----------------- quick demo entry point -----------------
if __name__ == "__main__":
    """
    Usage:
        python src/etl.py data/raw/sample.csv
    This lets reviewers run a demo without BigQuery credentials.
    """
    import sys
    from pathlib import Path
    import pandas as pd

    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/sample.csv")
    csv_to_bq(csv_path, "demo")        # writes parquet to data/processed/
    print("✅  Loaded", len(pd.read_csv(csv_path)), "rows")

