import polars as pl
from src.data.split import deterministic_split
df = pl.read_parquet('data/pairs/ce_train.parquet') # dummy, just to see what happens
print(f"Loaded {len(df)} rows")
