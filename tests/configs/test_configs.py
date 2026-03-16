import yaml
from pathlib import Path

def test_configs_exist_and_load():
    configs_dir = Path("configs")
    files = ["models.yaml", "training.yaml", "data.yaml", "eval.yaml.example"]
    for f in files:
        path = configs_dir / f
        assert path.exists(), f"Missing config file: {f}"
        with open(path, "r") as file:
            data = yaml.safe_load(file)
            assert data is not None, f"Empty config file: {f}"

def test_models_yaml():
    path = Path("configs/models.yaml")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    expected_keys = {"minilm_reranker", "gte_reranker", "bge_reranker_m3", "granite_reranker"}
    assert set(data.keys()) == expected_keys, f"Expected {expected_keys}, got {set(data.keys())}"
    
    expected_fields = {"hf_id", "license", "params", "context", "fine_tuned"}
    for model_key, model_data in data.items():
        assert set(model_data.keys()).issuperset(expected_fields), f"Missing fields in {model_key}"

def test_training_yaml():
    path = Path("configs/training.yaml")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        
    assert "training" in data
    assert "epochs" in data["training"]
    assert "batch_size" in data["training"]
    assert "lr" in data["training"]
    assert "seed" in data["training"]
    
    assert "loss" in data
    assert "phase1_loss" in data["loss"]
    assert "phase2_loss" in data["loss"]

def test_data_yaml():
    path = Path("configs/data.yaml")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        
    assert "sources" in data
    assert "ethnicity_distribution" in data
    assert "corruption_codes" in data

def test_eval_yaml_example():
    path = Path("configs/eval.yaml.example")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        
    assert "phase1" in data
    assert "index_root" in data["phase1"]
    assert "eval" in data
    assert "top_k_stage1" in data["eval"]
    assert "metrics" in data
