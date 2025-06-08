# Somnia Project: LLM Dataset Preparation, Tokenizer, and Training Pipeline

This repository contains the full pipeline for data preparation, custom tokenizer training, and transformer model training (TinyLLaMA-style) for the Somnia project, focused on narrative datasets (fairy tales and stories.) and developed as part of a CAS thesis.

## Key Features

- **Automatic download** of datasets from Kaggle.
- **Preprocessing**: cleaning, normalization, and chunking of texts.
- **Dataset preparation** in `.jsonl` format for training.
- **Custom tokenizer training** (Byte-Pair Encoding) with special token management.
- **Data analysis** with statistics and plots.
- **TinyLLaMA-style transformer model** with rotary embeddings, RMSNorm, KV caching, checkpointing, and text generation.
- **Automated hyperparameter optimization** with resume and progressive saving.
- **Advanced logging** to file and console.

## Project Structure

```
├── data/
│   ├── fetch/           # Scripts to download data from Kaggle
│   ├── processed/       # Scripts to prepare data for training
│   └── analysis/        # Data analysis and visualization
├── model/
│   ├── transformer/     # Model, config, dataset, tokenizer
│   ├── metrics.py       # Training metrics tracking and plotting
│   ├── train_model.py   # Main training loop
│   ├── train_tokenizer.py # Custom tokenizer training
│   ├── hyperparameter.py # Hyperparameter optimization
├── utility/
│   ├── logger.py        # Centralized logger
│   └── paths.py         # Path and directory management
├── main.py              # Pipeline entry point
├── requirements.txt     # Python dependencies
└── README.md
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cem94/somnia-project.git
   cd somnia-project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Kaggle credentials:**
   - Download your `kaggle.json` from your Kaggle account.
   - Place it in `~/.kaggle/kaggle.json`.

## Usage

### 1. Full pipeline (recommended)

Run the entire pipeline (data processing, tokenizer, model training):

```bash
python main.py
```

### 2. Download and prepare data

Download datasets from Kaggle:
```bash
python data/fetch/fetch_data.py
```

Prepare the dataset for training:
```bash
python data/processed/prepare_dataset.py
```

### 3. Data analysis

Generate statistics and plots:
```bash
python data/analysis/analysis.py
```

### 4. Custom tokenizer training

Train the tokenizer:
```bash
python model/train_tokenizer.py
```

### 5. Model training

Start model training (with or without hyperparameter search):
```bash
python model/hyperparameter.py
```
Or standard training only:
```bash
python model/train_model.py
```

## Checkpointing and Resume

- Checkpoints are saved periodically in `model/output/` or `model/hyperparameters/`.
- If interrupted (`Ctrl+C`), training automatically resumes from the last available checkpoint.

## Output

- **Model and checkpoints:** `model/output/` or `model/hyperparameters/`
- **Tokenizers:** `model/tokenizer/`
- **Plots and metrics:** `model/plots/`
- **Logs:** `logs/`

## About

This project implements a TinyLLaMA-style transformer (adapted from [train-tiny-llm](https://github.com/FareedKhan-dev/train-tiny-llm)) and is developed as part of a CAS thesis on efficient LLM training and data pipelines for narrative text.

## Contributing

Contributions and suggestions are welcome!  
Open an issue or pull request to propose improvements.

---

**Note:**  
Make sure you have enough disk space for data and models.  
For issues or questions, check the logs in `logs/` or open an issue.