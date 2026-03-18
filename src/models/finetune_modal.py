import os
import modal

app = modal.App("entity-resolution-ce-ft")

# Define the image mirroring Phase 1 structure (pinned deps to ensure compatibility and stability)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.1.0",
    "transformers==4.49.0", 
    "sentence-transformers==5.0.0",
    "datasets>=2.16.1",
    "polars>=0.20.0",
    "scikit-learn>=1.3.0",
    "huggingface-hub",
    "pyyaml"
)

def get_repo_name(model_key: str) -> str:
    return f"jayshah5696/er2-ce-{model_key.replace('_', '-')}-ft"

@app.function(
    image=image,
    gpu="A10G",
    timeout=86400, # 24 hours
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def finetune_one(model_key: str, dry_run: bool = False):
    import os
    import yaml
    from pathlib import Path
    from datasets import load_dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    from sentence_transformers.cross_encoder import CrossEncoderTrainer
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss, LambdaLoss
    from sentence_transformers import InputExample
    import torch
    
    # HF_HUB_DISABLE_XET is required per plan
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    
    # 1. Configuration
    repo_name = get_repo_name(model_key)
    
    # Loading raw dict instead of reading yaml inside modal image without mounting,
    # but we can mount the configs or simply hardcode the expected config lookup for the runner.
    # We will pass the hf_id manually here or expect the yaml to be mounted.
    # To keep the function simple and self-contained for the stub, we resolve it directly:
    hf_models = {
        "gte_reranker": "Alibaba-NLP/gte-reranker-modernbert-base",
        "granite_reranker": "ibm-granite/granite-3.0-reranker"
    }
    
    base_model = hf_models.get(model_key)
    if not base_model:
        raise ValueError(f"Unknown model_key: {model_key}")
        
    print(f"Starting Fine-Tuning for {model_key} on {base_model}...")
    
    if dry_run:
        print(f"Dry run successful for {model_key}. Model repo would be: {repo_name}")
        return
        
    # 2. Dataset Loading
    dataset = load_dataset("jayshah5696/entity-resolution-ce-pairs-v2")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    
    # 3. Model Initialization
    # Assuming mixed precision for speed on A10G
    model = CrossEncoder(base_model, num_labels=1, trust_remote_code=True)
    
    # 4. Evaluator setup
    val_samples = []
    for row in val_dataset:
        val_samples.append(InputExample(texts=[row["text_a"], row["text_b"]], label=float(row["label"])))
        
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_samples, name=model_key)
    
    # 5. Curriculum Trainer setup
    # The plan requested BCE (epochs 1-3) -> LambdaLoss (epochs 4-5)
    # We subclass CrossEncoderTrainer to override loss depending on the epoch.
    
    class CurriculumTrainer(CrossEncoderTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Check epoch from state
            current_epoch = self.state.epoch if self.state.epoch is not None else 0
            
            if current_epoch < 3.0:
                loss_fct = BinaryCrossEntropyLoss(model)
            else:
                loss_fct = LambdaLoss(model)
                
            loss = loss_fct(inputs, return_outputs=return_outputs)
            return loss

    # Ensure format matches what sentence-transformers expects
    def format_dataset(ds):
        return ds.map(lambda x: {"texts": [x["text_a"], x["text_b"]], "label": float(x["label"])}, remove_columns=ds.column_names)
        
    formatted_train = format_dataset(train_dataset)
    formatted_val = format_dataset(val_dataset)
    
    from transformers import TrainingArguments
    
    # Standard STv5 CrossEncoderTrainer args
    args = TrainingArguments(
        output_dir=f"./results_{model_key}",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True, # A10G supports fp16 natively
        seed=42,
    )
    
    trainer = CurriculumTrainer(
        model=model.model,
        args=args,
        train_dataset=formatted_train,
        eval_dataset=formatted_val,
        evaluator=evaluator,
    )
    
    print("Beginning Training Loop...")
    trainer.train()
    
    print(f"Pushing tuned model to Hub: {repo_name}...")
    model.push_to_hub(repo_name)
    print("Complete!")

@app.local_entrypoint()
def run_all():
    print("Dispatching GTE Reranker...")
    finetune_one.spawn(model_key="gte_reranker")
    print("Dispatching Granite Reranker...")
    finetune_one.spawn(model_key="granite_reranker")
    print("All tasks dispatched to Modal!")
