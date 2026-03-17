import os
import json
from pathlib import Path
from datasets import Dataset, DatasetDict
import polars as pl
import argparse

def upload_dataset(data_dir: Path, manifest_dir: Path, repo_id: str, dry_run: bool = False):
    # Set HF flag as requested
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    
    # Load ONLY train and val. Never test.
    train_path = data_dir / "ce_train.parquet"
    val_path = data_dir / "ce_val.parquet"
    
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing train or val parquets in {data_dir}")
        
    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)
    
    # Convert Polars DataFrames to HF Datasets (pandas intermediation is most reliable)
    train_dataset = Dataset.from_pandas(train_df.to_pandas())
    val_dataset = Dataset.from_pandas(val_df.to_pandas())
    
    ds = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })
    
    print(f"Dataset assembled. Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    if not dry_run:
        print(f"Pushing to HF Hub: {repo_id}...")
        try:
            # We assume the environment is already authenticated (e.g. via huggingface-cli login
            # or HF_TOKEN env var). Private push is optional, defaults to public if not specified, 
            # but we can set private=True just in case, though the plan implies public usage for blog.
            ds.push_to_hub(repo_id, private=False)
            print("Successfully pushed to Hub.")
        except Exception as e:
            print(f"Warning: Push to hub failed. This is expected if HF_TOKEN is not set or repo access is denied. Error: {e}")
            if "HF_TOKEN" not in os.environ:
                print("\n[Mocking upload manifest creation for testing purposes since HF_TOKEN is missing]")
            else:
                raise e
            
    # Write manifest
    manifest = {
        "repo_id": repo_id,
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "test_rows": 0, # Intentionally excluded
        "status": "pushed" if not dry_run and "HF_TOKEN" in os.environ else "dry_run_or_mocked"
    }
    
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "upload_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Saved manifest to {manifest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/pairs"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--repo-id", type=str, default="jayshah5696/entity-resolution-ce-pairs-v2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    upload_dataset(args.data_dir, args.manifest_dir, args.repo_id, args.dry_run)
