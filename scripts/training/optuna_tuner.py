import optuna
import os
import argparse
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from .trainer import WhisperFineTuner, Seq2SeqTrainingArguments
from datasets import load_from_disk # Example for loading pre-preprocessed data

def objective(trial, train_dataset, eval_dataset, base_output_dir):
    # 1. Suggest Hyperparameters (Informed by Augusta baseline)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.2) # CAUTION: Augusta "0.2 cap" rule
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4, 8])
    
    # Optional regularizers
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    
    output_dir = os.path.join(base_output_dir, f"trial_{trial.number}")
    
    training_args = {
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": 10,  # Or larger for final v1.3+ scaling
        "weight_decay": weight_decay,
        "logging_steps": 25,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "fp16": True,
        "predict_with_generate": True,
        "generation_max_length": 225,
    }

    # Integrate dropout into model
    tuner = WhisperFineTuner()
    tuner.model.config.dropout = dropout
    
    # 2. Train
    trainer = tuner.train(train_dataset, eval_dataset, output_dir, training_args)
    
    # 3. Return best metric (WER)
    metrics = trainer.evaluate()
    return metrics["eval_wer"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--study_name", type=str, default="whisper_tune")
    parser.add_argument("--db_url", type=str, help="SQLite URL, e.g., sqlite:///optuna.db")
    args = parser.parse_args()

    # Configure Optuna with Augusta settings
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.db_url,
        direction="minimize",
        sampler=TPESampler(multivariate=True), # Correlation-aware
        pruner=MedianPruner(n_warmup_steps=3, interval_steps=1),
        load_if_exists=True
    )
    
    # In a real scenario, you'd load pre-preprocessed train/eval datasets here
    # study.optimize(lambda trial: objective(trial, train_ds, eval_ds, "./out"), n_trials=args.trials)
