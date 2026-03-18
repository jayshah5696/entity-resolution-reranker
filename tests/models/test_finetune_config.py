import pytest
import yaml
from pathlib import Path

def test_finetune_config_has_targets():
    # Verify the two targets are inside models.yaml
    path = Path("configs/models.yaml")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        
    assert "gte_reranker" in data
    assert "granite_reranker" in data

def test_finetune_config_loss_params():
    path = Path("configs/training.yaml")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        
    assert "loss" in data
    assert data["loss"]["phase1_loss"] == "bce"
    assert data["loss"]["phase2_loss"] == "lambda_rank"
    
def test_modal_script_imports():
    # Ensure no import errors when loading the modal script
    # It must mock out modal environment or just import properly
    import src.models.finetune_modal
    assert hasattr(src.models.finetune_modal, "app")
    
def test_output_repo_names():
    # Convention is jayshah5696/er2-ce-{model_key}-ft
    from src.models.finetune_modal import get_repo_name
    assert get_repo_name("gte_reranker") == "jayshah5696/er2-ce-gte-reranker-ft"
    assert get_repo_name("granite_reranker") == "jayshah5696/er2-ce-granite-reranker-ft"
