import modal
from pathlib import Path
import json
import os

app = modal.App("er-modal-evals")
vol = modal.Volume.from_name("er-indexes-vol")

# Heavy image mapping matching finetuning script exactly
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.49.0", 
        "sentence-transformers==5.0.0",
        "polars>=0.20.0",
        "scikit-learn>=1.3.0",
        "lancedb>=0.15.0",
        "pyyaml",
        "pyarrow",
        "huggingface-hub",
        "tqdm"
    )
    .add_local_dir("src", remote_path="/root/workspace/src")
    .add_local_dir("configs", remote_path="/root/workspace/configs")
)

@app.function(
    image=image,
    gpu="A100", 
    timeout=7200, 
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("huggingface")]
)
def run_single_experiment(exp_cfg: dict):
    """Executes a single experiment matrix completely natively within the cloud!"""
    import os
    import sys
    
    sys.path.append("/root/workspace")
    
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    
    exp_id = exp_cfg["experiment_id"]
    s1_model = exp_cfg["stage1_model"]
    s2_key = exp_cfg["reranker"]
    s2_path = exp_cfg["reranker_path"]
    
    print(f"=== Starting Experiment {exp_id} ===")
    print(f"Stage 1: {s1_model} | Stage 2: {s2_path}")
    
    # Path mappings inside volume
    eval_queries = Path("/data/eval_data")
    
    if s1_model == "bm25":
        s1_index = Path("/data/indexes/bm25_pipe/index.lance")
    else:
        s1_index = Path("/data/indexes/gte_modernbert_base_pipe_fp32/index.lance")
        
    output_json = Path(f"/data/results/{exp_id}_{s1_model}_plus_{s2_key}.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    # The simplest way to run this is to actually invoke our local run_reranker logic identically!
    # Because we need `src` available inside modal to import, we will mount it locally!
    from src.eval.run_reranker import process_end_to_end
    from argparse import Namespace
    
    args = Namespace(
        stage1_model=s1_model,
        stage1_index=s1_index,
        reranker=s2_key,
        eval_queries=eval_queries,
        top_k_stage1=50,
        output=output_json,
        experiment_id=exp_id
    )
    
    # We must patch models_cfg loading since run_reranker reads configs/models.yaml locally
    # We can inject a mock or copy the dict directly
    import yaml
    cfg_dir = Path("/root/workspace/configs")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg_dir / "models.yaml", "w") as f:
        yaml.dump({
            s2_key: {"hf_id": s2_path} # Dynamically rewrite the config so the CrossEncoder parses our override cleanly
        }, f)
        
    os.chdir("/root/workspace")
    process_end_to_end(args)
    
    vol.commit() # Save results to volume
    print(f"=== Completed Experiment {exp_id} ===")
    return output_json.name

@app.local_entrypoint()
def launch_evals():
    import json
    from pathlib import Path
    
    # Read configs locally
    configs = []
    for f in sorted(Path("experiments").glob("*/config.json")):
        with open(f) as fp:
            configs.append(json.load(fp))
            
    print(f"Found {len(configs)} evaluations. Dispatching to Modal A100s...")
    
    # Run in parallel!
    results = list(run_single_experiment.starmap(
        [(c,) for c in configs],
        return_exceptions=True
    ))
    
    # Sync results back to local disk
    print("Downloading results from Volume...")
    Path("results").mkdir(exist_ok=True)
    
    # Because we want to read from the Volume directly without writing another cloud function,
    # we can iterate the volume data
    try:
        for entry in vol.listdir("/results"):
            if entry.path.endswith(".json"):
                data = b""
                for chunk in vol.read_file(entry.path):
                    data += chunk
                with open(f"results/{Path(entry.path).name}", "wb") as f:
                    f.write(data)
    except modal.exception.NotFoundError:
        print("No /results directory found in volume. Experiments may have failed entirely.")
                
    print("Executions finished:")
    for c, r in zip(configs, results):
        if isinstance(r, Exception):
            print(f"FAILED {c['experiment_id']}: {r}")
        else:
            print(f"OK {c['experiment_id']} -> {r}")
            
    print("\nNext step: Run `uv run python -m src.eval.aggregate` locally!")
