# ParleZeneise: Low-Resource Whisper Fine-Tuning

This repository contains a modular framework for fine-tuning OpenAI's Whisper model on low-resource languages (e.g., Genovese, Ladin, Sicilian). It leverages insights from the Augusta project, focusing on transcription accuracy and efficient hyperparameter management.

## 🚀 Project Goals
- **Transcription Focus**: Specialized fine-tuning for S2T (Speech-to-Text) in dialectal and minority languages.
- **Modular Data Handling**: Support for Common Voice, Hugging Face Hub datasets, and custom CSV/TSV structures.
- **Production Readiness**: Scripts for CTranslate2 quantization and model export.
- **Optimization Strategy**: Flexible training with both fixed hyperparameters and Bayesian optimization (Optuna).

## 📁 Directory Structure
- `scripts/datasets/`: Data loading logic for various formats.
- `scripts/training/`: Logic for trainer and Optuna tuning.
- `scripts/evaluation/`: Rigorous benchmarking (WER/CER with normalization).
- `scripts/conversion/`: Export and quantization (CT2).
- `configs/`: Hyperparameter search spaces and configurations.
- `sbatch/`: Slurm batch scripts for HPC environments.

## 🛠️ Usage

### 1. Environment Preparation
Ensure you have the necessary environment set up. You can use the provided \`.env.example\` to manage your configuration:
\\\`\\\`\\\`bash
cp .env.example .env
\\\`\\\`\\\`

### 2. Simple Fine-Tuning
Run the basic trainer with fixed parameters:
\`\`\`bash
python scripts/training/trainer.py --config configs/base_params.yaml
\`\`\`

### 2. Optuna Optimization
Launch a Bayesian search:
\`\`\`bash
python scripts/training/optuna_tuner.py --trials 20
\`\`\`

## 🧠 Key Learnings Integrated
- **Dropout Capping**: Strictly capped at 0.2 to prevent phonetic "wash-out" in low-resource settings.
- **Gradient Accumulation**: Support for large effective batch sizes on limited hardware.
- **Unified Normalization**: Rigorous WER/CER calculations using standardized token normalization.

